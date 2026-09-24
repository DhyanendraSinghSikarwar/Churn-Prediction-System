"""Entry point: `python pipelines/backfill_predictions.py`.

Scores every historical cohort (`models.split.modeling_panel` -- every
`feature_snapshots` row whose 30-day label has already matured, i.e.
every `reference_date` strictly before `snapshot_date`) with the
registry's current champion, and upserts each row into `predictions`
with its ground truth (`actual_churned`, straight from
`feature_snapshots.label_churn_next_30d`) already populated -- unlike
`batch_score.py`'s current-cohort rows, these never need to wait for a
label to mature, since it already has.

This is what gives `src/churn/monitoring/performance.py` something real
to report on: a multi-month history of "what would this exact champion
have predicted for this cohort, and did the customer actually churn."
It is a retrospective backtest, not a live monitoring signal -- most of
these cohorts were part of the champion's own training data. Every
cohort is labeled train/validation/test/post-test
(`models.split.dataset_split_label`) specifically so that distinction is
never lost downstream: a train-window cohort's optimistic, partially
in-sample numbers must never be read as if they were the held-out test
score (see DECISIONS.md#groupkfold-cv-score-is-not-comparable-to-forward-in-time-pr-auc
for the same discipline applied elsewhere).

Idempotent, same as `batch_score.py`: re-running overwrites existing
rows (`ON CONFLICT ... DO UPDATE`) rather than duplicating them.
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
    shap_top_features,
    upsert_predictions,
)
from churn.models.pipeline import load_selected_features
from churn.models.split import modeling_panel
from churn.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


def _load_historical_panel(cfg: AppConfig) -> pd.DataFrame:
    engine = get_engine()
    df = pd.read_sql(
        text("SELECT * FROM feature_snapshots WHERE feature_set_version = :v"),
        engine,
        params={"v": cfg.feature_set.version},
    )
    df["reference_date"] = pd.to_datetime(df["reference_date"])
    return modeling_panel(df, cfg)


def run() -> None:
    cfg = get_app_config()
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)

    panel = _load_historical_panel(cfg)
    if panel.empty:
        logger.warning("historical_panel_empty")
        return
    logger.info("historical_panel_loaded", rows=len(panel))

    selected_features = load_selected_features(cfg)
    features = panel[selected_features]

    client = MlflowClient()
    champion = load_champion(cfg.mlflow.registry_model_name, client=client)
    logger.info("champion_loaded", model_version=champion.version.version)

    # Rounded before banding -- see
    # DECISIONS.md#probability-must-be-rounded-before-risk-banding-not-after.
    probabilities = [round(float(p), 5) for p in champion.pipeline.predict_proba(features)[:, 1]]
    shap_values = shap_top_features(champion, features)
    bands = [risk_band(p, cfg) for p in probabilities]
    matured_at = pd.Timestamp.now("UTC").to_pydatetime()

    rows: list[PredictionRow] = [
        {
            "customer_id": customer_id,
            "reference_date": reference_date.date(),
            "model_version": champion.version.version,
            "probability": probability,
            "risk_band": band,
            "shap_top3": shap,
            "actual_churned": bool(label),
            "label_matured_at": matured_at,
        }
        for customer_id, reference_date, probability, band, shap, label in zip(
            panel["customer_id"],
            panel["reference_date"],
            probabilities,
            bands,
            shap_values,
            panel["label_churn_next_30d"],
            strict=True,
        )
    ]

    engine = get_engine()
    with engine.begin() as conn:
        scored = len(upsert_predictions(conn, rows))

    logger.info(
        "backfill_complete",
        rows_scored=scored,
        model_version=champion.version.version,
        cohorts=panel["reference_date"].nunique(),
    )


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[union-attr]
    configure_logging()
    run()


if __name__ == "__main__":
    main()
