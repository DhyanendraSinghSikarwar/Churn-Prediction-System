"""Modeling-panel filter and the time-based train/validation/test split.

Both were duplicated ad hoc across Notebooks 04-06 (Phase 4). Pulled out
here as the single source of truth so Phase 5's training pipeline and any
future notebook use exactly the same split logic, not a hand-copied
version that can drift.
"""

from __future__ import annotations

import pandas as pd
from dateutil.relativedelta import relativedelta

from churn.config import AppConfig


def modeling_panel(panel: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
    """Drops the reference_date == snapshot_date cohort: a censoring
    artifact of the synthetic churn-date generator (it can never place a
    churn_date after snapshot_date), not a real zero-hazard month -- see
    DECISIONS.md#last-cohort-has-a-structural-zero-positive-rate. Every
    training/evaluation step in this project uses this filtered panel,
    never the raw materialized one.

    `panel["reference_date"]` must already be a datetime64 column.
    """
    snapshot_ts = pd.Timestamp(cfg.data.snapshot_date)
    return panel[panel["reference_date"] < snapshot_ts].copy()


def time_based_split(
    panel: pd.DataFrame, cfg: AppConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits an already-filtered modeling panel into (train, val, test)
    by calendar-month cutoffs off `panel_start_date`, using
    `train_months`/`validation_months`/`test_months` from config.yaml.

    Because the modeling panel already excludes the censored last cohort
    (see `modeling_panel`), the nominal test window may come back shorter
    than `test_months` -- currently 2 cohorts, not 3. This isn't
    special-cased: it falls out naturally from filtering before
    splitting, and is exactly what Phase 4's notebooks found and
    documented (see PLAN.md's confirmed assumptions).
    """
    panel_start = pd.Timestamp(cfg.data.panel_start_date)
    train_end = panel_start + relativedelta(months=cfg.data.train_months)
    val_end = train_end + relativedelta(months=cfg.data.validation_months)
    test_end = val_end + relativedelta(months=cfg.data.test_months)

    train = panel[panel["reference_date"] < train_end]
    val = panel[(panel["reference_date"] >= train_end) & (panel["reference_date"] < val_end)]
    test = panel[(panel["reference_date"] >= val_end) & (panel["reference_date"] < test_end)]
    return train, val, test
