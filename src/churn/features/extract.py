"""Executes the point-in-time feature extraction query
(sql/features/build_feature_snapshot.sql) for a single reference_date.

The same function serves both training (called once per historical month
by build_panel.py) and inference (Phase 6: called once with
reference_date = today) — see PLAN.md Phase 0 and the query's own header
comment for the point-in-time-correctness contract.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Connection

from churn.config import PROJECT_ROOT

_FEATURE_QUERY_PATH = PROJECT_ROOT / "sql" / "features" / "build_feature_snapshot.sql"

# Columns that can be NULL (a brand-new customer with no billing/ticket
# history yet, or too little history for a slope/trend). pandas upcasts a
# nullable int column read back from Postgres to float64 (5 -> 5.0), which
# a downstream COPY into an INTEGER column then rejects — see
# DECISIONS.md#copy-based-bulk-load-with-two-layer-idempotency (Phase 1)
# for the same failure mode with satisfaction_score. Int64 keeps these as
# proper nullable integers instead.
_NULLABLE_INT_COLUMNS = ("recency_days_since_last_charge", "recency_days_since_last_ticket")


def extract_feature_snapshot(
    conn: Connection,
    reference_date: dt.date,
    horizon_days: int,
    feature_set_version: str,
) -> pd.DataFrame:
    query = text(_FEATURE_QUERY_PATH.read_text(encoding="utf-8"))
    df = pd.read_sql(
        query,
        conn,
        params={
            "reference_date": reference_date,
            "horizon_days": horizon_days,
            "feature_set_version": feature_set_version,
        },
    )
    for column in _NULLABLE_INT_COLUMNS:
        df[column] = df[column].astype("Int64")
    return df
