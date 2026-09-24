"""Shared inference helpers: load the registry's "champion" pipeline,
compute the scoring cohort, risk-band a probability, and extract
top-3 SHAP features per prediction. Single source of truth for
`pipelines/batch_score.py` and `api/main.py` -- neither re-implements
this inline, for the same reason `models/split.py`/`pipeline.py` exist
(see DECISIONS.md#shared-columntransformer-corrupted-notebook-04s-threshold-tuning-numbers).
"""

from __future__ import annotations

import datetime as dt
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import mlflow
import numpy as np
import pandas as pd
import shap
from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Connection

from churn.config import AppConfig
from churn.db.models import Prediction
from churn.models.registry import CHAMPION_ALIAS

BACKGROUND_ARTIFACT_PATH = "explainability/background_sample.csv"
BACKGROUND_SAMPLE_SIZE = 100


class PredictionRow(TypedDict):
    customer_id: str
    reference_date: dt.date
    model_version: str
    probability: float
    risk_band: str
    shap_top3: dict[str, float]


def scoring_cohort(panel: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
    """The complement of `models.split.modeling_panel`: the single cohort
    with `reference_date == snapshot_date`, i.e. "today" in the synthetic
    world -- the one whose `label_churn_next_30d` hasn't matured yet, and
    therefore the one production scoring actually runs against."""
    snapshot_ts = pd.Timestamp(cfg.data.snapshot_date)
    return panel[panel["reference_date"] == snapshot_ts].copy()


def risk_band(probability: float, cfg: AppConfig) -> str:
    bands = cfg.business.risk_bands
    if probability <= bands.low_max:
        return "low"
    if probability <= bands.medium_max:
        return "medium"
    return "high"


@dataclass
class ChampionModel:
    pipeline: Pipeline
    version: ModelVersion
    background: pd.DataFrame  # raw (pre-transform) rows, for the SHAP explainer


def log_background_sample(
    trainval_features: pd.DataFrame, random_state: int, n: int = BACKGROUND_SAMPLE_SIZE
) -> None:
    """Logs a random sample of raw (pre-transform) training rows as a run
    artifact, so serving code can build a SHAP `LinearExplainer` without
    ever reloading the full historical panel -- must be called inside the
    same `with mlflow.start_run():` block as `registry.log_pipeline_artifact`.

    Sample size matches Notebook 05's exact precedent (100 rows is
    shap's own `LinearExplainer` default and already a stable estimate of
    the feature means/covariance it subtracts off).
    """
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(trainval_features), size=min(n, len(trainval_features)), replace=False)
    sample = trainval_features.iloc[idx]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "background_sample.csv"
        sample.to_csv(path, index=False)
        mlflow.log_artifact(str(path), artifact_path="explainability")


def load_champion(model_name: str, client: MlflowClient | None = None) -> ChampionModel:
    """Loads the pipeline + background sample currently aliased "champion".

    Uses `models:/{name}@{alias}` URIs throughout -- resolved via the
    standard artifact-download path (same one `log_artifacts`/
    `download_artifacts` use), not mlflow 3's "Logged Model" REST API,
    which this project's pinned mlflow *server* (2.14.1) doesn't have.
    See DECISIONS.md#mlflow-client-3x-vs-server-2x-model-logging-incompatibility.
    """
    client = client or MlflowClient()
    version = client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)
    model_uri = f"models:/{model_name}@{CHAMPION_ALIAS}"
    pipeline: Pipeline = mlflow.sklearn.load_model(model_uri)

    assert version.run_id is not None, f"champion version {version.version} has no run_id"
    background_path = client.download_artifacts(version.run_id, BACKGROUND_ARTIFACT_PATH)
    background = pd.read_csv(background_path)

    return ChampionModel(pipeline=pipeline, version=version, background=background)


def shap_top_features(
    champion: ChampionModel, features: pd.DataFrame, top_n: int = 3
) -> list[dict[str, float]]:
    """Top-`top_n` SHAP features per row of `features`, keyed by the
    *transformed* (post-`ColumnTransformer`) feature name -- matches
    `predictions.shap_top3`'s shape exactly as prototyped in
    Notebook 05's "Top-3 SHAP features per prediction" section.

    Only supports the linear primary model (`LinearExplainer`) -- this
    project's champion is always `LogisticRegression`
    (DECISIONS.md#logistic-regression-beats-every-gbm-variant); a future
    tree-based champion would need a `TreeExplainer` branch here.
    """
    prep = champion.pipeline.named_steps["prep"]
    clf = champion.pipeline.named_steps["clf"]
    feature_names = list(prep.get_feature_names_out())

    background_transformed = prep.transform(champion.background)
    if hasattr(background_transformed, "toarray"):
        background_transformed = background_transformed.toarray()

    features_transformed = prep.transform(features)
    if hasattr(features_transformed, "toarray"):
        features_transformed = features_transformed.toarray()

    explainer = shap.LinearExplainer(clf, background_transformed)
    shap_values = explainer(features_transformed).values

    results = []
    for row in shap_values:
        order = np.argsort(-np.abs(row))[:top_n]
        results.append({feature_names[i]: round(float(row[i]), 4) for i in order})
    return results


def upsert_predictions(conn: Connection, rows: list[PredictionRow]) -> list[dt.datetime]:
    """Inserts scored rows into `predictions`, overwriting on a repeat
    `(customer_id, reference_date, model_version)` rather than
    duplicating it -- shared by `pipelines/batch_score.py` (bulk, one
    cohort at a time) and `api/main.py`'s on-demand rescoring endpoint
    (one row at a time). Returns each row's DB-assigned `scored_at`, in
    the same order as `rows`."""
    if not rows:
        return []
    insert_stmt = pg_insert(Prediction).values(list(rows))
    upsert_stmt = insert_stmt.on_conflict_do_update(
        constraint="uq_predictions_customer_ref_model",
        set_={
            "probability": insert_stmt.excluded.probability,
            "risk_band": insert_stmt.excluded.risk_band,
            "shap_top3": insert_stmt.excluded.shap_top3,
            "scored_at": text("now()"),
        },
    ).returning(Prediction.scored_at)
    result = conn.execute(upsert_stmt)
    return [row.scored_at for row in result]
