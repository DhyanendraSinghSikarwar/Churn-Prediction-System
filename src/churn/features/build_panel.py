"""Entry point: `python -m churn.features.build_panel [--force]`.

Materializes feature_snapshots for every monthly reference_date in
[panel_start_date, panel_end_date] (config.yaml), by calling
extract_feature_snapshot() once per month and bulk-inserting the result.
Idempotent per (reference_date, feature_set_version): a month already
populated is skipped unless --force, in which case its existing rows are
replaced rather than duplicated.
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import sys

import pandas as pd
from dateutil.relativedelta import relativedelta
from sqlalchemy import text
from sqlalchemy.engine import Connection

from churn.config import get_app_config
from churn.db.engine import get_engine
from churn.features.extract import extract_feature_snapshot
from churn.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

# Must match the column order build_feature_snapshot.sql SELECTs, and the
# FeatureSnapshot model's column list (excluding the DB-generated
# snapshot_id / created_at).
_FEATURE_COLUMNS = [
    "customer_id",
    "reference_date",
    "feature_set_version",
    "senior_citizen",
    "partner",
    "dependents",
    "contract_type",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "paperless_billing",
    "payment_method",
    "monthly_charges",
    "tenure_months",
    "tenure_bucket",
    "recency_days_since_last_charge",
    "recency_days_since_last_ticket",
    "frequency_charges_30d",
    "frequency_charges_60d",
    "frequency_charges_90d",
    "frequency_tickets_30d",
    "frequency_tickets_60d",
    "frequency_tickets_90d",
    "monetary_charges_30d",
    "monetary_charges_60d",
    "monetary_charges_90d",
    "cumulative_charges_to_date",
    "billing_amount_slope_90d",
    "billing_amount_mom_delta",
    "ticket_velocity_delta_30_60",
    "payment_failure_streak",
    "payment_failure_rate_180d",
    "label_churn_next_30d",
]


def _monthly_reference_dates(start: dt.date, end: dt.date) -> list[dt.date]:
    dates = []
    cursor = start
    while cursor <= end:
        dates.append(cursor)
        cursor = cursor + relativedelta(months=1)
    return dates


def _already_built(conn: Connection, reference_date: dt.date, feature_set_version: str) -> bool:
    result = conn.execute(
        text(
            "SELECT 1 FROM feature_snapshots "
            "WHERE reference_date = :rd AND feature_set_version = :v LIMIT 1"
        ),
        {"rd": reference_date, "v": feature_set_version},
    ).first()
    return result is not None


def _delete_existing(conn: Connection, reference_date: dt.date, feature_set_version: str) -> None:
    conn.execute(
        text(
            "DELETE FROM feature_snapshots WHERE reference_date = :rd AND feature_set_version = :v"
        ),
        {"rd": reference_date, "v": feature_set_version},
    )


def _insert_snapshot(conn: Connection, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    buffer = io.StringIO()
    df[_FEATURE_COLUMNS].to_csv(
        buffer, index=False, header=False, na_rep=r"\N", lineterminator="\n"
    )
    buffer.seek(0)
    raw_conn = conn.connection.driver_connection
    assert raw_conn is not None, "connection was closed before COPY could run"
    cols_sql = ", ".join(_FEATURE_COLUMNS)
    with (
        raw_conn.cursor() as cur,
        cur.copy(
            f"COPY feature_snapshots ({cols_sql}) FROM STDIN WITH (FORMAT csv, NULL '\\N')"
        ) as copy,
    ):
        copy.write(buffer.read())
    return len(df)


def run(force: bool = False) -> None:
    cfg = get_app_config()
    engine = get_engine()
    reference_dates = _monthly_reference_dates(cfg.data.panel_start_date, cfg.data.panel_end_date)

    total_rows = 0
    for reference_date in reference_dates:
        with engine.begin() as conn:
            exists = _already_built(conn, reference_date, cfg.feature_set.version)
            if exists and not force:
                logger.info("panel_month_skipped_already_built", reference_date=str(reference_date))
                continue
            if exists and force:
                _delete_existing(conn, reference_date, cfg.feature_set.version)

            df = extract_feature_snapshot(
                conn,
                reference_date=reference_date,
                horizon_days=cfg.data.prediction_horizon_days,
                feature_set_version=cfg.feature_set.version,
            )
            rows = _insert_snapshot(conn, df)
            total_rows += rows
            logger.info("panel_month_built", reference_date=str(reference_date), row_count=rows)

    logger.info("panel_build_complete", total_rows=total_rows, months=len(reference_dates))


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Materialize the monthly feature_snapshots panel")
    parser.add_argument("--force", action="store_true", help="Rebuild months already populated")
    args = parser.parse_args(sys.argv[1:])
    run(force=args.force)


if __name__ == "__main__":
    main()
