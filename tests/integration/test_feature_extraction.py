"""Integration test for sql/features/build_feature_snapshot.sql: real
Postgres (testcontainers), hand-crafted fixture rows with known dates, so
every returned feature can be checked against an exact expected value —
including an explicit point-in-time leakage regression check (a billing
event dated on/after reference_date must never be counted, even though
it exists in the table).

Requires Docker. Run with: uv run pytest -m integration
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator

import pandas as pd
import pytest
from alembic import command
from alembic.config import Config as AlembicConfig
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session
from testcontainers.community.postgres import PostgresContainer

from churn.db.models import BillingEvent, ChurnLabel, Customer, Subscription
from churn.features.extract import extract_feature_snapshot

REFERENCE_DATE = dt.date(2023, 4, 15)


@pytest.fixture(scope="module")
def migrated_engine() -> Iterator[Engine]:
    with PostgresContainer("postgres:16") as pg:
        url = pg.get_connection_url().replace("psycopg2", "psycopg")
        engine = create_engine(url, future=True)

        alembic_cfg = AlembicConfig("alembic.ini")
        alembic_cfg.set_main_option("sqlalchemy.url", url)
        command.upgrade(alembic_cfg, "head")

        yield engine
        engine.dispose()


def _subscription(
    customer_id: str, start: dt.date, end: dt.date | None = None, monthly_charges: float = 70.0
) -> Subscription:
    return Subscription(
        customer_id=customer_id,
        contract_type="Month-to-month",
        phone_service=True,
        multiple_lines="No",
        internet_service="Fiber optic",
        online_security="No",
        online_backup="No",
        device_protection="No",
        tech_support="No",
        streaming_tv="No",
        streaming_movies="No",
        paperless_billing=True,
        payment_method="Electronic check",
        monthly_charges=monthly_charges,
        start_date=start,
        end_date=end,
    )


@pytest.fixture(scope="module")
def fixture_data(migrated_engine: Engine) -> None:
    with Session(migrated_engine) as session:
        # --- Customer A: active (never churned). Billing history includes
        # one event deliberately dated AFTER reference_date — this is the
        # point-in-time leakage regression check: if the extraction query
        # ever weakens its `< :reference_date` filters to `<=` or drops
        # them, this fixture will make cumulative_charges_to_date and
        # recency_days_since_last_charge wrong, and the tests below fail.
        session.add(
            Customer(
                customer_id="A",
                gender="Female",
                senior_citizen=False,
                partner=False,
                dependents=False,
                signup_date=dt.date(2023, 1, 1),
                source_row_hash="a" * 64,
            )
        )
        sub_a = _subscription("A", dt.date(2023, 1, 1))
        session.add(sub_a)
        session.add(
            ChurnLabel(
                customer_id="A", churned=False, churn_date=None, censored_as_of=dt.date(2024, 6, 1)
            )
        )
        session.flush()
        for event_date, amount in (
            (dt.date(2023, 2, 1), 100.0),
            (dt.date(2023, 3, 1), 100.0),
            (dt.date(2023, 4, 1), 100.0),
        ):
            session.add(
                BillingEvent(
                    customer_id="A",
                    subscription_id=sub_a.subscription_id,
                    event_date=event_date,
                    event_type="charge",
                    amount=amount,
                )
            )
        # Deliberately AFTER reference_date (2023-04-15).
        session.add(
            BillingEvent(
                customer_id="A",
                subscription_id=sub_a.subscription_id,
                event_date=dt.date(2023, 4, 20),
                event_type="charge",
                amount=999.0,
            )
        )

        # --- Customer B: churns 2023-05-01 — within 30 days of
        # reference_date (2023-04-15) -> label should be True.
        session.add(
            Customer(
                customer_id="B",
                gender="Male",
                senior_citizen=False,
                partner=False,
                dependents=False,
                signup_date=dt.date(2023, 1, 1),
                source_row_hash="b" * 64,
            )
        )
        sub_b = _subscription("B", dt.date(2023, 1, 1), end=dt.date(2023, 5, 1))
        session.add(sub_b)
        session.add(
            ChurnLabel(
                customer_id="B",
                churned=True,
                churn_date=dt.date(2023, 5, 1),
                censored_as_of=dt.date(2024, 6, 1),
            )
        )
        session.flush()
        session.add(
            BillingEvent(
                customer_id="B",
                subscription_id=sub_b.subscription_id,
                event_date=dt.date(2023, 2, 1),
                event_type="charge",
                amount=50.0,
            )
        )
        session.add(
            BillingEvent(
                customer_id="B",
                subscription_id=sub_b.subscription_id,
                event_date=dt.date(2023, 2, 1),
                event_type="payment_failed",
                amount=50.0,
            )
        )

        # --- Customer C: already churned BEFORE reference_date -> must be
        # excluded from the cohort entirely.
        session.add(
            Customer(
                customer_id="C",
                gender="Male",
                senior_citizen=False,
                partner=False,
                dependents=False,
                signup_date=dt.date(2022, 1, 1),
                source_row_hash="c" * 64,
            )
        )
        session.add(_subscription("C", dt.date(2022, 1, 1), end=dt.date(2023, 3, 1)))
        session.add(
            ChurnLabel(
                customer_id="C",
                churned=True,
                churn_date=dt.date(2023, 3, 1),
                censored_as_of=dt.date(2024, 6, 1),
            )
        )

        # --- Customer D: signs up AFTER reference_date -> must be excluded.
        session.add(
            Customer(
                customer_id="D",
                gender="Female",
                senior_citizen=False,
                partner=False,
                dependents=False,
                signup_date=dt.date(2023, 5, 1),
                source_row_hash="d" * 64,
            )
        )
        session.add(_subscription("D", dt.date(2023, 5, 1)))
        session.add(
            ChurnLabel(
                customer_id="D", churned=False, churn_date=None, censored_as_of=dt.date(2024, 6, 1)
            )
        )

        session.commit()


@pytest.mark.integration
def test_cohort_excludes_already_churned_and_not_yet_signed_up(
    migrated_engine: Engine, fixture_data: None
) -> None:
    with migrated_engine.connect() as conn:
        df = extract_feature_snapshot(
            conn, reference_date=REFERENCE_DATE, horizon_days=30, feature_set_version="v1"
        )
    assert set(df["customer_id"]) == {"A", "B"}


@pytest.mark.integration
def test_point_in_time_correctness_excludes_future_billing_event(
    migrated_engine: Engine, fixture_data: None
) -> None:
    with migrated_engine.connect() as conn:
        df = extract_feature_snapshot(
            conn, reference_date=REFERENCE_DATE, horizon_days=30, feature_set_version="v1"
        )
    row = df.set_index("customer_id").loc["A"]
    # 3 charges of 100 precede reference_date; the 999 charge dated
    # 2023-04-20 exists in billing_events but must NOT be summed in.
    assert row["cumulative_charges_to_date"] == pytest.approx(300.0)
    assert row["recency_days_since_last_charge"] == 14  # 2023-04-15 minus 2023-04-01


@pytest.mark.integration
def test_tenure_and_frequency_windows(migrated_engine: Engine, fixture_data: None) -> None:
    with migrated_engine.connect() as conn:
        df = extract_feature_snapshot(
            conn, reference_date=REFERENCE_DATE, horizon_days=30, feature_set_version="v1"
        )
    row = df.set_index("customer_id").loc["A"]
    assert row["tenure_months"] == 3
    assert row["tenure_bucket"] == "0-6"
    assert row["frequency_charges_30d"] == 1  # only the 2023-04-01 charge
    assert row["frequency_charges_90d"] == 3  # all three prior charges


@pytest.mark.integration
def test_new_customer_with_no_billing_history_gets_nulls_not_errors(
    migrated_engine: Engine, fixture_data: None
) -> None:
    with migrated_engine.connect() as conn:
        df = extract_feature_snapshot(
            conn, reference_date=dt.date(2023, 1, 15), horizon_days=30, feature_set_version="v1"
        )
    # As of 2023-01-15, customer A (signed up 2023-01-01) has no billing
    # events yet — the first charge isn't generated until a month after
    # signup in the real synthetic generator, and none exist in this
    # fixture before 2023-02-01 either.
    row = df.set_index("customer_id").loc["A"]
    assert row["frequency_charges_30d"] == 0
    assert row["cumulative_charges_to_date"] == 0
    assert pd.isna(row["recency_days_since_last_charge"])
    assert pd.isna(row["billing_amount_slope_90d"])


@pytest.mark.integration
def test_label_true_only_when_churn_falls_within_horizon(
    migrated_engine: Engine, fixture_data: None
) -> None:
    with migrated_engine.connect() as conn:
        df = extract_feature_snapshot(
            conn, reference_date=REFERENCE_DATE, horizon_days=30, feature_set_version="v1"
        )
    indexed = df.set_index("customer_id")
    assert bool(indexed.loc["B", "label_churn_next_30d"]) is True
    assert bool(indexed.loc["A", "label_churn_next_30d"]) is False
