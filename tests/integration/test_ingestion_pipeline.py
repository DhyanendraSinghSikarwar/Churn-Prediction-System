"""Integration test: real Postgres (via testcontainers), real Alembic
migrations, real COPY-based load. Requires Docker.

Run with: uv run pytest -m integration
"""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd
import pytest
from alembic import command
from alembic.config import Config as AlembicConfig
from sqlalchemy import Engine, create_engine, text
from testcontainers.community.postgres import PostgresContainer

from churn.config import AppConfig, get_app_config
from churn.ingestion.load import load_all
from churn.ingestion.synthesize import synthesize_all


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


@pytest.fixture
def app_config() -> AppConfig:
    return get_app_config()


@pytest.mark.integration
def test_synthesize_and_load_round_trip(migrated_engine: Engine, app_config: AppConfig) -> None:
    raw = pd.DataFrame(
        {
            "customerID": [f"C{i:04d}" for i in range(20)],
            "gender": ["Male", "Female"] * 10,
            "SeniorCitizen": [0, 1] * 10,
            "Partner": ["Yes", "No"] * 10,
            "Dependents": ["No", "No"] * 10,
            "tenure": list(range(1, 21)),
            "PhoneService": ["Yes"] * 20,
            "MultipleLines": ["No"] * 20,
            "InternetService": ["Fiber optic"] * 20,
            "OnlineSecurity": ["No"] * 20,
            "OnlineBackup": ["No"] * 20,
            "DeviceProtection": ["No"] * 20,
            "TechSupport": ["No"] * 20,
            "StreamingTV": ["No"] * 20,
            "StreamingMovies": ["No"] * 20,
            "Contract": ["Month-to-month"] * 20,
            "PaperlessBilling": ["Yes"] * 20,
            "PaymentMethod": ["Electronic check"] * 20,
            "MonthlyCharges": [70.0] * 20,
            "Churn": ["Yes", "No"] * 10,
        }
    )

    tables = synthesize_all(raw, app_config, seed=123)

    with migrated_engine.begin() as conn:
        counts = load_all(conn, tables)

    assert counts["customers"] == 20
    assert counts["churn_labels"] == 20
    assert counts["subscriptions"] == 20
    assert counts["billing_events"] > 0

    with migrated_engine.connect() as conn:
        orphaned = conn.execute(
            text(
                "SELECT COUNT(*) FROM billing_events be "
                "LEFT JOIN customers c ON be.customer_id = c.customer_id "
                "WHERE c.customer_id IS NULL"
            )
        ).scalar_one()
        assert orphaned == 0
