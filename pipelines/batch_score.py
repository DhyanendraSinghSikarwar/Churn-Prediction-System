"""Entry point: `python pipelines/batch_score.py`.

Scores the current scoring cohort (`feature_snapshots` rows where
`reference_date == snapshot_date` -- the one cohort whose 30-day label
hasn't matured yet, see `churn.models.inference.scoring_cohort`) with the
registry's "champion" model, and upserts one row per customer into
`predictions` (risk band + top-3 SHAP features included).

Idempotent: re-running for the same `(customer_id, reference_date,
model_version)` overwrites that row (`ON CONFLICT ... DO UPDATE`) rather
than duplicating it -- the same idempotency pattern as ingestion/panel
building, just scoped to the registry's unique constraint instead of a
`--force` flag.
"""

from __future__ import annotations

import sys

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sqlalchemy import text

from churn.config import AppConfig, get_app_config
from churn.db.engine import get_engine
from churn.models.inference import (
    PredictionRow,
    load_champion,
    risk_band,
    scoring_cohort,
    shap_top_features,
    upsert_predictions,
)
from churn.models.pipeline import load_selected_features
from churn.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


def _load_scoring_features(cfg: AppConfig) -> pd.DataFrame:
    engine = get_engine()
    df = pd.read_sql(
        text("SELECT * FROM feature_snapshots WHERE feature_set_version = :v"),
        engine,
        params={"v": cfg.feature_set.version},
    )
    df["reference_date"] = pd.to_datetime(df["reference_date"])
    return scoring_cohort(df, cfg)


def run() -> None:
    cfg = get_app_config()
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)

    cohort = _load_scoring_features(cfg)
    if cohort.empty:
        logger.warning("scoring_cohort_empty", snapshot_date=str(cfg.data.snapshot_date))
        return
    logger.info(
        "scoring_cohort_loaded", rows=len(cohort), snapshot_date=str(cfg.data.snapshot_date)
    )

    selected_features = load_selected_features(cfg)
    features = cohort[selected_features]

    client = MlflowClient()
    champion = load_champion(cfg.mlflow.registry_model_name, client=client)
    logger.info(
        "champion_loaded",
        model_name=cfg.mlflow.registry_model_name,
        model_version=champion.version.version,
    )

    probabilities = champion.pipeline.predict_proba(features)[:, 1]
    shap_values = shap_top_features(champion, features)
    bands = [risk_band(p, cfg) for p in probabilities]

    rows: list[PredictionRow] = [
        {
            "customer_id": customer_id,
            "reference_date": reference_date.date(),
            "model_version": champion.version.version,
            "probability": float(probability),
            "risk_band": band,
            "shap_top3": shap,
        }
        for customer_id, reference_date, probability, band, shap in zip(
            cohort["customer_id"],
            cohort["reference_date"],
            probabilities,
            bands,
            shap_values,
            strict=True,
        )
    ]

    engine = get_engine()
    with engine.begin() as conn:
        scored = len(upsert_predictions(conn, rows))

    band_counts = pd.Series(bands).value_counts().to_dict()
    logger.info(
        "batch_scoring_complete",
        rows_scored=scored,
        model_version=champion.version.version,
        risk_band_counts=band_counts,
    )


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[union-attr]
    configure_logging()
    run()


if __name__ == "__main__":
    main()
