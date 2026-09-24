"""Entry point: `python pipelines/monitor.py`.

Computes two monitoring reports and logs them (structured, one line per
finding) -- the read-only observability step the Prefect flow
(`pipelines/flows.py`) runs after `batch_score.py`/`backfill_predictions.py`:

1. **Feature drift**: the training window's feature distribution
   (reference -- train+val, exactly what the champion was fit on, see
   `churn.models.split.trainval_split`) vs. the current scoring cohort's
   (comparison), via `churn.monitoring.drift.drift_report`. Logs a
   warning for any feature whose severity comes back "moderate",
   "significant", or "unknown" (the last meaning PSI couldn't even be
   computed -- e.g. a feature that's gone entirely null -- which is
   itself worth a human looking at, not quietly logged at info level).
2. **Delayed-label performance decay**: every already-matured
   `predictions` cohort scored by the *current* champion (populated by
   `backfill_predictions.py`), via
   `churn.monitoring.performance.cohort_performance_report` -- logged per
   cohort so a real PR-AUC/precision/recall trend over time is visible,
   not just today's number. Each cohort's `in_sample` flag is logged
   alongside its split label: train/validation cohorts were literally
   part of what the champion was fit on, so their numbers are optimistic
   by construction, not just "partially" so.

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

from churn.config import AppConfig, get_app_config
from churn.models.inference import scoring_cohort
from churn.models.pipeline import load_selected_features
from churn.models.split import modeling_panel, trainval_split
from churn.monitoring.drift import drift_report
from churn.monitoring.loaders import champion_version, load_matured_predictions, load_panel
from churn.monitoring.performance import cohort_performance_report
from churn.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

# Severities worth a human looking at -- "unknown" means the drift check
# itself couldn't run (e.g. a feature gone entirely null), which is at
# least as concerning as a measured shift, not less.
_WARN_SEVERITIES = {"moderate", "significant", "unknown"}


def _check_drift(panel: pd.DataFrame, cfg: AppConfig) -> None:
    current_cohort = scoring_cohort(panel, cfg)
    if current_cohort.empty:
        logger.warning("drift_check_skipped_no_current_cohort")
        return

    reference = trainval_split(modeling_panel(panel, cfg), cfg)
    selected_features = load_selected_features(cfg)
    results = drift_report(reference, current_cohort, selected_features, cfg)

    for result in results:
        log = logger.warning if result.severity in _WARN_SEVERITIES else logger.info
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
            reference_null_rate=round(result.reference_null_rate, 4),
            current_null_rate=round(result.current_null_rate, 4),
            severity=result.severity,
        )


def _check_performance(cfg: AppConfig) -> None:
    version = champion_version(cfg)
    if version is None:
        logger.warning("performance_check_skipped_champion_unavailable")
        return

    predictions = load_matured_predictions(cfg, version)
    if predictions.empty:
        logger.warning("performance_check_skipped_no_matured_predictions", model_version=version)
        return

    for cohort in cohort_performance_report(predictions, cfg):
        logger.info(
            "cohort_performance",
            reference_date=str(cohort.reference_date),
            dataset_split=cohort.dataset_split,
            in_sample=cohort.in_sample,
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

    panel = load_panel(cfg)
    _check_drift(panel, cfg)
    _check_performance(cfg)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[union-attr]
    configure_logging()
    run()


if __name__ == "__main__":
    main()
