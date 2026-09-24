"""FastAPI serving layer: `uvicorn api.main:app`.

Two read paths, deliberately different in cost:

- `GET /predictions/{customer_id}` serves the latest row `batch_score.py`
  already computed -- a single indexed DB read, no model in the loop.
- `POST /score/{customer_id}` scores on demand against the current
  scoring cohort's already-materialized `feature_snapshots` row (not a
  live re-run of the window-function SQL -- see
  `churn.models.inference`'s module docstring for why that logic is
  shared, not duplicated here).

Graceful degradation: the champion model is loaded once at startup: if
that fails (registry/MLflow unreachable), the API still starts --
`/predictions` and `/health` keep working off Postgres alone, and only
`/score` returns 503 until a model is loaded. Conversely, if Postgres is
unreachable, `/health` and `/predictions` report 503 rather than
crashing.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import mlflow
import pandas as pd
import structlog
from fastapi import FastAPI, HTTPException, Request, Response
from mlflow.tracking import MlflowClient
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from sqlalchemy import select, text
from sqlalchemy.exc import SQLAlchemyError

from api.schemas import HealthResponse, PredictionResponse
from churn.config import AppConfig, get_app_config
from churn.db.engine import get_engine
from churn.db.models import Prediction
from churn.db.session import get_session
from churn.models.inference import (
    ChampionModel,
    PredictionRow,
    current_champion_version,
    load_champion,
    risk_band,
    shap_top_features,
    upsert_predictions,
)
from churn.models.pipeline import load_selected_features
from churn.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

REQUEST_COUNT = Counter(
    "churn_api_requests_total", "Total API requests", ["method", "path", "status"]
)
REQUEST_LATENCY = Histogram(
    "churn_api_request_duration_seconds", "Request latency in seconds", ["method", "path"]
)
PREDICTION_PROBABILITY = Histogram(
    "churn_api_prediction_probability",
    "Churn probability served by /predictions and /score",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    cfg = get_app_config()
    app.state.cfg = cfg
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    try:
        app.state.champion = load_champion(cfg.mlflow.registry_model_name)
        logger.info("champion_loaded_at_startup", model_version=app.state.champion.version.version)
    except Exception:
        app.state.champion = None
        logger.warning("champion_load_failed_at_startup_serving_degraded", exc_info=True)
    yield


app = FastAPI(title="Churn Prediction API", lifespan=lifespan)


@app.middleware("http")
async def request_id_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    structlog.contextvars.bind_contextvars(request_id=request_id)
    try:
        response = await call_next(request)
    finally:
        structlog.contextvars.clear_contextvars()
    response.headers["X-Request-ID"] = request_id
    return response


@app.middleware("http")
async def metrics_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    # An unmatched route (a probe, a typo'd path) has no route *template*
    # to label with -- falling back to the raw request path would let
    # anyone generate unbounded Prometheus series just by hitting random
    # URLs. Found by an independent review actually doing that. A real
    # route's `path` is always its template (e.g. "/predictions/{customer_id}"),
    # never the substituted value, so cardinality stays bounded either way.
    route = request.scope.get("route")
    path = route.path if route is not None else "<unmatched>"
    REQUEST_COUNT.labels(method=request.method, path=path, status=response.status_code).inc()
    REQUEST_LATENCY.labels(method=request.method, path=path).observe(duration)
    return response


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health", response_model=HealthResponse)
def health(request: Request, response: Response) -> HealthResponse:
    db_ok = True
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
    except SQLAlchemyError:
        db_ok = False
        logger.warning("health_check_db_unreachable", exc_info=True)

    champion: ChampionModel | None = request.app.state.champion
    response.status_code = 200 if db_ok else 503
    return HealthResponse(
        status="ok" if db_ok else "degraded",
        database="ok" if db_ok else "unreachable",
        model="loaded" if champion is not None else "unavailable",
        model_version=champion.version.version if champion is not None else None,
    )


@app.get("/predictions/{customer_id}", response_model=PredictionResponse)
def get_prediction(customer_id: str) -> PredictionResponse:
    try:
        with get_session() as session:
            row = session.execute(
                select(Prediction)
                .where(Prediction.customer_id == customer_id)
                .order_by(Prediction.scored_at.desc())
                .limit(1)
            ).scalar_one_or_none()
    except SQLAlchemyError as exc:
        logger.warning("get_prediction_db_unreachable", exc_info=True)
        raise HTTPException(status_code=503, detail="database unreachable") from exc

    if row is None:
        raise HTTPException(
            status_code=404, detail=f"no prediction found for customer_id={customer_id!r}"
        )
    PREDICTION_PROBABILITY.observe(float(row.probability))
    return PredictionResponse.model_validate(row)


def _ensure_fresh_champion(request: Request) -> ChampionModel:
    """Reloads the champion if the registry's alias has moved since this
    process started -- otherwise a long-lived API process serves a
    stale, superseded model indefinitely (only a restart would pick up a
    promotion), and worse, an on-demand `/score` from the stale model
    could overwrite a newer batch-scored row with a newer `scored_at`,
    silently shadowing it. Found by an independent review, not by
    inspection. Failure here (registry unreachable) is not fatal: falls
    back to whatever's already loaded, consistent with this API's
    graceful-degradation design."""
    champion: ChampionModel | None = request.app.state.champion
    cfg: AppConfig = request.app.state.cfg
    try:
        client = MlflowClient()
        latest_version = current_champion_version(cfg.mlflow.registry_model_name, client)
        if champion is None or champion.version.version != latest_version:
            champion = load_champion(cfg.mlflow.registry_model_name, client=client)
            request.app.state.champion = champion
            logger.info("champion_reloaded", model_version=champion.version.version)
    except Exception:
        logger.warning("champion_freshness_check_failed", exc_info=True)
    if champion is None:
        raise HTTPException(status_code=503, detail="model unavailable")
    return champion


@app.post("/score/{customer_id}", response_model=PredictionResponse)
def score_customer(customer_id: str, request: Request) -> PredictionResponse:
    champion = _ensure_fresh_champion(request)
    cfg: AppConfig = request.app.state.cfg

    selected_features = load_selected_features(cfg)
    try:
        engine = get_engine()
        snapshot = pd.read_sql(
            text(
                "SELECT * FROM feature_snapshots "
                "WHERE customer_id = :cid AND reference_date = :rd AND feature_set_version = :v"
            ),
            engine,
            params={
                "cid": customer_id,
                "rd": cfg.data.snapshot_date,
                "v": cfg.feature_set.version,
            },
        )
    except SQLAlchemyError as exc:
        logger.warning("score_customer_db_unreachable", exc_info=True)
        raise HTTPException(status_code=503, detail="database unreachable") from exc

    if snapshot.empty:
        raise HTTPException(
            status_code=404,
            detail=f"no current feature snapshot for customer_id={customer_id!r}",
        )

    features = snapshot[selected_features]
    # Rounded *before* risk-banding, matching `predictions.probability`'s
    # NUMERIC(6,5) column -- banding the full-precision value and then
    # storing a rounded one meant the stored `risk_band` could disagree
    # with `risk_band(stored probability)` right at a boundary (e.g.
    # 0.300004 bands "medium" but rounds to 0.30000, which bands "low").
    # Found by an independent review, not by inspection.
    probability = round(float(champion.pipeline.predict_proba(features)[:, 1][0]), 5)
    band = risk_band(probability, cfg)
    shap_values = shap_top_features(champion, features)[0]

    row: PredictionRow = {
        "customer_id": customer_id,
        "reference_date": cfg.data.snapshot_date,
        "model_version": champion.version.version,
        "probability": probability,
        "risk_band": band,
        "shap_top3": shap_values,
    }
    try:
        with engine.begin() as conn:
            scored_at = upsert_predictions(conn, [row])[0]
    except SQLAlchemyError as exc:
        # Previously only the read above was guarded, so a DB outage
        # between the read and this write surfaced as a bare 500 instead
        # of the same 503 every other DB-dependent path returns. Found by
        # an independent review.
        logger.warning("score_customer_upsert_failed", exc_info=True)
        raise HTTPException(status_code=503, detail="database unreachable") from exc

    logger.info("customer_scored_on_demand", customer_id=customer_id, probability=probability)
    PREDICTION_PROBABILITY.observe(probability)
    return PredictionResponse(**row, scored_at=scored_at)
