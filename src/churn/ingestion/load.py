"""Bulk-load the synthesized tables into Postgres via COPY, inside one
transaction, with a row-count assertion per table. This module does not
decide *whether* to load — that idempotency decision is the CLI's job
(see cli.py) — it only executes the load once called.
"""

from __future__ import annotations

import io

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Connection

from churn.utils.logging import get_logger

logger = get_logger(__name__)

# Explicit column order per table — must match the order values are
# written to the CSV buffer in _copy_dataframe, and must match the
# Postgres column list passed to COPY.
_COPY_COLUMNS: dict[str, list[str]] = {
    "customers": [
        "customer_id",
        "gender",
        "senior_citizen",
        "partner",
        "dependents",
        "signup_date",
        "source_row_hash",
    ],
    "subscriptions": [
        "customer_id",
        "contract_type",
        "phone_service",
        "multiple_lines",
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
        "start_date",
        "end_date",
    ],
    "billing_events": ["customer_id", "subscription_id", "event_date", "event_type", "amount"],
    "support_tickets": [
        "customer_id",
        "created_date",
        "resolved_date",
        "category",
        "priority",
        "satisfaction_score",
    ],
    "churn_labels": ["customer_id", "churned", "churn_date", "censored_as_of"],
}


def _copy_dataframe(conn: Connection, table: str, df: pd.DataFrame) -> int:
    columns = _COPY_COLUMNS[table]
    buffer = io.StringIO()
    df[columns].to_csv(buffer, index=False, header=False, na_rep=r"\N", lineterminator="\n")
    buffer.seek(0)

    # SQLAlchemy's Connection wraps the DBAPI connection; driver_connection
    # (SQLAlchemy 2.0+) gives the unwrapped psycopg3 connection so we can
    # use its native COPY API.
    raw_conn = conn.connection.driver_connection
    assert raw_conn is not None, "connection was closed before COPY could run"
    cols_sql = ", ".join(columns)
    with (
        raw_conn.cursor() as cur,
        cur.copy(f"COPY {table} ({cols_sql}) FROM STDIN WITH (FORMAT csv, NULL '\\N')") as copy,
    ):
        copy.write(buffer.read())

    return len(df)


def load_all(conn: Connection, tables: dict[str, pd.DataFrame]) -> dict[str, int]:
    """Load customers -> subscriptions -> (fetch generated subscription_ids)
    -> billing_events -> support_tickets -> churn_labels, in one
    transaction. Order matters: FK constraints require parents before
    children, and billing_events needs the DB-generated subscription_id
    that only exists after subscriptions are inserted.
    """
    counts: dict[str, int] = {}

    counts["customers"] = _copy_dataframe(conn, "customers", tables["customers"])
    counts["subscriptions"] = _copy_dataframe(conn, "subscriptions", tables["subscriptions"])

    sub_ids = pd.read_sql(text("SELECT subscription_id, customer_id FROM subscriptions"), conn)
    billing_events = tables["billing_events"].merge(sub_ids, on="customer_id", how="left")
    if billing_events["subscription_id"].isna().any():
        raise ValueError("Some billing_events rows failed to match a subscription_id")

    counts["billing_events"] = _copy_dataframe(conn, "billing_events", billing_events)
    counts["support_tickets"] = _copy_dataframe(conn, "support_tickets", tables["support_tickets"])
    counts["churn_labels"] = _copy_dataframe(conn, "churn_labels", tables["churn_labels"])

    for table, expected in counts.items():
        actual = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one()
        if actual != expected:
            raise ValueError(
                f"Row count assertion failed for {table}: expected {expected}, found {actual}"
            )
        logger.info("table_loaded", table=table, row_count=actual)

    return counts
