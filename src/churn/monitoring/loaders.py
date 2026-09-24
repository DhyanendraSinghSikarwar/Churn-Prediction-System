"""Shared data-loading helpers for Phase 7's three monitoring callers
(`pipelines/monitor.py`, `dashboards/app.py`, `dashboards/report.py`) --
one implementation of "how do we load the panel / the current champion
version / predictions for a given scope," so the three presentation
layers can never quietly diverge on what they filter.

Found by an independent review: before this module existed, each of the
three callers had its own copy of this loading/filtering logic, and one
real fix -- filtering the "current cohort" query by the champion's
`model_version`, not just `reference_date` -- would otherwise have
needed applying identically in three places, with real risk of one
being missed. `predictions.model_version` is part of the row's unique
key precisely because a customer can have rows from more than one model
version; once a second promotion happens, a `reference_date`-only filter
double-counts every "current cohort" summary.
"""

from __future__ import annotations

import pandas as pd
from mlflow.tracking import MlflowClient
from sqlalchemy import text

from churn.config import AppConfig
from churn.db.engine import get_engine
from churn.models.inference import current_champion_version


def load_panel(cfg: AppConfig) -> pd.DataFrame:
    """The full `feature_snapshots` panel for the configured feature set,
    with `reference_date` already parsed to datetime64."""
    engine = get_engine()
    df = pd.read_sql(
        text("SELECT * FROM feature_snapshots WHERE feature_set_version = :v"),
        engine,
        params={"v": cfg.feature_set.version},
    )
    df["reference_date"] = pd.to_datetime(df["reference_date"])
    return df


def champion_version(cfg: AppConfig, client: MlflowClient | None = None) -> str | None:
    """The registry's current `champion` alias version, or `None` if the
    registry is unreachable or no champion exists yet -- callers degrade
    gracefully rather than crashing, consistent with `api/main.py`'s
    graceful-degradation design."""
    try:
        return current_champion_version(cfg.mlflow.registry_model_name, client or MlflowClient())
    except Exception:
        return None


def load_current_cohort_predictions(cfg: AppConfig, model_version: str) -> pd.DataFrame:
    """The current scoring cohort's `predictions` rows, filtered to
    `model_version` *in SQL* -- not `SELECT * FROM predictions` plus a
    client-side filter on `reference_date` alone, which double-counts
    every customer once a second model version has been scored against
    this cohort."""
    engine = get_engine()
    df = pd.read_sql(
        text("SELECT * FROM predictions WHERE reference_date = :rd AND model_version = :v"),
        engine,
        params={"rd": cfg.data.snapshot_date, "v": model_version},
    )
    df["reference_date"] = pd.to_datetime(df["reference_date"])
    return df


def load_matured_predictions(cfg: AppConfig, model_version: str) -> pd.DataFrame:
    """Every historical, already-matured `predictions` row for
    `model_version` -- filtered in SQL. `predictions` grows by a full
    historical backfill (tens of thousands of rows) every time a new
    champion is promoted and backfilled; `SELECT *` with no filter reads
    every past version's rows on every dashboard render."""
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
