"""Integration test for churn.models.inference.upsert_predictions: real
Postgres (testcontainers), verifying the conflict-update behavior an
independent review flagged as untested -- specifically that an
unlabeled rescore (`batch_score.py`/`api/main.py`'s row shape) must
never clear an existing row's `actual_churned`/`label_matured_at`
(`backfill_predictions.py`'s row shape), and that a labeled upsert does
correctly set a previously-null label. A fresh INSERT writes the label
columns regardless of which branch is used, so only the ON CONFLICT
UPDATE path (exercised here by upserting the *same* key twice) actually
proves the `with_labels` branch selection is correct.

Requires Docker. Run with: uv run pytest -m integration
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator

import pytest
from alembic import command
from alembic.config import Config as AlembicConfig
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session
from testcontainers.community.postgres import PostgresContainer

from churn.db.models import Customer, Prediction
from churn.models.inference import PredictionRow, upsert_predictions

_CUSTOMER_ID = "1111-TEST"


@pytest.fixture(scope="module")
def migrated_engine() -> Iterator[Engine]:
    with PostgresContainer("postgres:16") as pg:
        url = pg.get_connection_url().replace("psycopg2", "psycopg")
        engine = create_engine(url, future=True)

        alembic_cfg = AlembicConfig("alembic.ini")
        alembic_cfg.set_main_option("sqlalchemy.url", url)
        command.upgrade(alembic_cfg, "head")

        with Session(engine) as session:
            session.add(
                Customer(
                    customer_id=_CUSTOMER_ID,
                    gender="Female",
                    senior_citizen=False,
                    partner=True,
                    dependents=False,
                    signup_date=dt.date(2020, 1, 1),
                    source_row_hash="x" * 64,
                )
            )
            session.commit()

        yield engine
        engine.dispose()


def _fetch(engine: Engine, reference_date: dt.date) -> Prediction:
    with Session(engine) as session:
        row = session.execute(
            select(Prediction).where(
                Prediction.customer_id == _CUSTOMER_ID,
                Prediction.reference_date == reference_date,
            )
        ).scalar_one()
        session.expunge(row)
        return row


@pytest.mark.integration
def test_unlabeled_rescore_does_not_clear_an_existing_label(migrated_engine: Engine) -> None:
    reference_date = dt.date(2023, 1, 1)
    labeled: PredictionRow = {
        "customer_id": _CUSTOMER_ID,
        "reference_date": reference_date,
        "model_version": "1",
        "probability": 0.5,
        "risk_band": "medium",
        "shap_top3": [{"feature": "tenure_months", "shap_value": 0.1}],
        "actual_churned": True,
        "label_matured_at": dt.datetime(2023, 2, 1, tzinfo=dt.UTC),
    }
    with migrated_engine.begin() as conn:
        upsert_predictions(conn, [labeled])
    assert _fetch(migrated_engine, reference_date).actual_churned is True

    unlabeled: PredictionRow = {
        "customer_id": _CUSTOMER_ID,
        "reference_date": reference_date,
        "model_version": "1",
        "probability": 0.6,
        "risk_band": "medium",
        "shap_top3": [{"feature": "tenure_months", "shap_value": 0.2}],
    }
    with migrated_engine.begin() as conn:
        upsert_predictions(conn, [unlabeled])

    row = _fetch(migrated_engine, reference_date)
    # The label must survive an unlabeled rescore -- never silently
    # cleared just because a later upsert's row shape didn't carry it.
    assert row.actual_churned is True
    assert float(row.probability) == pytest.approx(0.6)  # the rescore itself did apply


@pytest.mark.integration
def test_labeled_upsert_sets_a_previously_null_label(migrated_engine: Engine) -> None:
    reference_date = dt.date(2023, 2, 1)
    unlabeled: PredictionRow = {
        "customer_id": _CUSTOMER_ID,
        "reference_date": reference_date,
        "model_version": "1",
        "probability": 0.3,
        "risk_band": "low",
        "shap_top3": [{"feature": "tenure_months", "shap_value": -0.1}],
    }
    with migrated_engine.begin() as conn:
        upsert_predictions(conn, [unlabeled])
    assert _fetch(migrated_engine, reference_date).actual_churned is None

    labeled: PredictionRow = {
        **unlabeled,
        "actual_churned": False,
        "label_matured_at": dt.datetime(2023, 3, 1, tzinfo=dt.UTC),
    }
    with migrated_engine.begin() as conn:
        upsert_predictions(conn, [labeled])

    assert _fetch(migrated_engine, reference_date).actual_churned is False


@pytest.mark.integration
def test_mixing_labeled_and_unlabeled_rows_in_one_call_fails_loudly(
    migrated_engine: Engine,
) -> None:
    mixed: list[PredictionRow] = [
        {
            "customer_id": _CUSTOMER_ID,
            "reference_date": dt.date(2023, 3, 1),
            "model_version": "1",
            "probability": 0.4,
            "risk_band": "low",
            "shap_top3": [],
            "actual_churned": True,
            "label_matured_at": dt.datetime(2023, 4, 1, tzinfo=dt.UTC),
        },
        {
            "customer_id": _CUSTOMER_ID,
            "reference_date": dt.date(2023, 4, 1),
            "model_version": "1",
            "probability": 0.4,
            "risk_band": "low",
            "shap_top3": [],
        },
    ]
    with pytest.raises(Exception), migrated_engine.begin() as conn:  # noqa: B017, PT011
        upsert_predictions(conn, mixed)
