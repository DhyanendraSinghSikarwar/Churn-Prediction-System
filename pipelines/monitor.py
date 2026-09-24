"""Entry point: `python pipelines/monitor.py`.

Computes two monitoring reports and logs them (structured, one line per
finding) -- the read-only observability step the Prefect flow
(`pipelines/flows.py`) runs after `batch_score.py`/`backfill_predictions.py`:

1. **Feature drift**: the training window's feature distribution
   (reference) vs. the current scoring cohort's (comparison), via
   `churn.monitoring.drift.drift_report`. Logs a warning for any feature
   whose PSI crosses `config.yaml`'s `psi_warning_threshold`/
   `psi_critical_threshold`.
2. **Delayed-label performance decay**: every already-matured
   `predictions` cohort scored by the *current* champion (populated by
   `backfill_predictions.py`), via
   `churn.monitoring.performance.cohort_performance_report` -- logged per
   cohort so a real PR-AUC/precision/recall trend over time is visible,
   not just today's number.

Read-only: computes and logs, never writes anything back to Postgres.
`dashboards/app.py` runs these exact same functions on demand for the
live dashboard view -- this script exists for the *scheduled* check
(structured logs an operator or log-based alerting can act on), not as
the only way to see the numbers.
"""

from __future__ import annotations

import sys

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sqlalchemy import text

from churn.config import AppConfig, get_app_config
from churn.db.engine import get_engine
from churn.models.inference import current_champion_version, scoring_cohort
from churn.models.pipeline import load_selected_features
from churn.models.split import modeling_panel, time_based_split
from churn.monitoring.drift import drift_report
from churn.monitoring.performance import cohort_performance_report
from churn.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


def _load_panel(cfg: AppConfig) -> pd.DataFrame:
    engine = get_engine()
    df = pd.read_sql(
        text("SELECT * FROM feature_snapshots WHERE feature_set_version = :v"),
        engine,
        params={"v": cfg.feature_set.version},
    )
    df["reference_date"] = pd.to_datetime(df["reference_date"])
    return df


def _load_matured_predictions(cfg: AppConfig, model_version: str) -> pd.DataFrame:
    engine = get_engine()
    df = pd.read_sql(
        text(
            "SELECT reference_date, probability, risk_band, actual_churned FROM predictions "
            "WHERE model_version = :v AND actual_churned IS NOT NULL"
        ),
        engine,
        params={"v": model_version},
    )
    df["reference_date"] = pd.to_datetime(df["reference_date"])
    return df


def _check_drift(panel: pd.DataFrame, cfg: AppConfig) -> None:
    current_cohort = scoring_cohort(panel, cfg)
    if current_cohort.empty:
        logger.warning("drift_check_skipped_no_current_cohort")
        return

    train_df, _, _ = time_based_split(modeling_panel(panel, cfg), cfg)
    selected_features = load_selected_features(cfg)
    results = drift_report(train_df, current_cohort, selected_features, cfg)

    for result in results:
        log = logger.warning if result.severity in {"moderate", "significant"} else logger.info
        log(
            "feature_drift",
            feature=result.feature,
            psi=round(result.psi, 4) if result.psi == result.psi else None,
            ks_statistic=result.ks_statistic,
            ks_pvalue=result.ks_pvalue,
            kl_divergence=(
                round(result.kl_divergence, 4)
                if result.kl_divergence == result.kl_divergence
                else None
            ),
            severity=result.severity,
        )


def _check_performance(cfg: AppConfig) -> None:
    client = MlflowClient()
    try:
        model_version = current_champion_version(cfg.mlflow.registry_model_name, client)
    except Exception:
        logger.warning("performance_check_skipped_champion_unavailable", exc_info=True)
        return

    predictions = _load_matured_predictions(cfg, model_version)
    if predictions.empty:
        logger.warning(
            "performance_check_skipped_no_matured_predictions", model_version=model_version
        )
        return

    for cohort in cohort_performance_report(predictions, cfg):
        logger.info(
            "cohort_performance",
            reference_date=str(cohort.reference_date),
            dataset_split=cohort.dataset_split,
            n=cohort.n,
            positive_rate=round(cohort.positive_rate, 4),
            pr_auc=round(cohort.pr_auc, 4) if cohort.pr_auc == cohort.pr_auc else None,
            roc_auc=round(cohort.roc_auc, 4) if cohort.roc_auc == cohort.roc_auc else None,
            precision_at_medium_plus=round(cohort.precision_at_medium_plus, 4),
            recall_at_medium_plus=round(cohort.recall_at_medium_plus, 4),
        )


def run() -> None:
    cfg = get_app_config()
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)

    panel = _load_panel(cfg)
    _check_drift(panel, cfg)
    _check_performance(cfg)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[union-attr]
    configure_logging()
    run()


if __name__ == "__main__":
    main()
