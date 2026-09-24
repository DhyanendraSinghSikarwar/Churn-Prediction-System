import datetime as dt
from decimal import Decimal
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sqlalchemy.exc import OperationalError

from churn.config import AppConfig
from churn.models.inference import ChampionModel


@pytest.fixture
def cfg() -> AppConfig:
    return AppConfig.model_validate(
        {
            "data": {
                "snapshot_date": "2024-06-01",
                "panel_start_date": "2023-01-01",
                "panel_end_date": "2024-06-01",
                "prediction_horizon_days": 30,
                "train_months": 12,
                "validation_months": 3,
                "test_months": 3,
                "source": {
                    "primary_url": "https://example.com/a.csv",
                    "fallback_url": "https://example.com/b.csv",
                    "expected_row_count": 1,
                    "raw_cache_path": "data/raw/x.csv",
                },
            },
            "synthetic": {
                "random_seed": 42,
                "payment_failure": {
                    "base_rate": 0.02,
                    "electronic_check_uplift": 0.06,
                    "month_to_month_uplift": 0.03,
                },
                "support_tickets": {
                    "base_monthly_rate": 0.15,
                    "no_tech_support_uplift": 0.10,
                    "no_online_security_uplift": 0.05,
                    "pre_churn_window_days": 60,
                    "pre_churn_multiplier": 2.5,
                },
            },
            "feature_set": {"version": "v1"},
            "mlflow": {
                "tracking_uri": "http://localhost:5000",
                "experiment_name": "churn-prediction",
                "registry_model_name": "churn_classifier",
                "promotion_epsilon": 0.005,
            },
            "business": {
                "retention_offer_cost": 50.0,
                "assumed_ltv_horizon_months": 12,
                "offer_success_rate": 0.30,
                "risk_bands": {"low_max": 0.3, "medium_max": 0.7},
            },
            "monitoring": {
                "psi_warning_threshold": 0.1,
                "psi_critical_threshold": 0.25,
                "drift_bins": 10,
                "null_rate_delta_threshold": 0.1,
            },
        }
    )


@pytest.fixture
def client(cfg: AppConfig):
    with (
        patch("api.main.get_app_config", return_value=cfg),
        patch("api.main.load_champion", side_effect=RuntimeError("registry unreachable")),
        patch("api.main.mlflow.set_tracking_uri"),
    ):
        from api.main import app

        with TestClient(app) as test_client:
            yield test_client


def test_health_reports_ok_when_db_reachable(client: TestClient) -> None:
    with patch("api.main.get_engine") as mock_get_engine:
        connection = mock_get_engine.return_value.connect.return_value.__enter__.return_value
        connection.execute.return_value = None
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["database"] == "ok"
    # startup patched load_champion to raise -- degraded, not crashed.
    assert body["model"] == "unavailable"
    assert body["model_version"] is None


def test_health_reports_503_when_db_unreachable(client: TestClient) -> None:
    with patch("api.main.get_engine") as mock_get_engine:
        mock_get_engine.return_value.connect.side_effect = OperationalError("x", {}, Exception("x"))
        response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["database"] == "unreachable"


def test_get_prediction_returns_404_when_not_found(client: TestClient) -> None:
    with patch("api.main.get_session") as mock_get_session:
        session = MagicMock()
        session.execute.return_value.scalar_one_or_none.return_value = None
        mock_get_session.return_value.__enter__.return_value = session
        response = client.get("/predictions/9999-NOSUCH")
    assert response.status_code == 404


def test_get_prediction_returns_row_when_found(client: TestClient) -> None:
    row = MagicMock(
        customer_id="1234-ABCD",
        reference_date=dt.date(2024, 6, 1),
        model_version="1",
        # A real decimal.Decimal, matching what SQLAlchemy's Numeric(6,5)
        # column actually returns -- a prior version of this test used a
        # plain float and never exercised pydantic's Decimal-to-float
        # coercion path at all.
        probability=Decimal("0.42000"),
        risk_band="medium",
        shap_top3=[{"feature": "tenure_months", "shap_value": -0.12}],
        scored_at=dt.datetime(2024, 6, 1, 12, 0, tzinfo=dt.UTC),
    )
    with patch("api.main.get_session") as mock_get_session:
        session = MagicMock()
        session.execute.return_value.scalar_one_or_none.return_value = row
        mock_get_session.return_value.__enter__.return_value = session
        response = client.get("/predictions/1234-ABCD")
    assert response.status_code == 200
    body = response.json()
    assert body["customer_id"] == "1234-ABCD"
    assert body["risk_band"] == "medium"
    assert body["probability"] == 0.42
    assert body["shap_top3"] == [{"feature": "tenure_months", "shap_value": -0.12}]


def test_get_prediction_returns_503_when_db_unreachable(client: TestClient) -> None:
    with patch("api.main.get_session") as mock_get_session:
        mock_get_session.side_effect = OperationalError("x", {}, Exception("x"))
        response = client.get("/predictions/1234-ABCD")
    assert response.status_code == 503


def test_score_customer_returns_503_when_model_unavailable(client: TestClient) -> None:
    # fixture's load_champion raises at startup -- app.state.champion stays None.
    # Also patch the freshness check so this test never makes a real
    # network call to MLflow: the champion is unavailable either way.
    with patch(
        "api.main.current_champion_version", side_effect=RuntimeError("registry unreachable")
    ):
        response = client.post("/score/1234-ABCD")
    assert response.status_code == 503


def _toy_champion_model(version: str = "1") -> ChampionModel:
    preprocessor = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["contract_type"]),
            ("num", StandardScaler(), ["tenure_months"]),
        ]
    )
    pipeline = Pipeline([("prep", preprocessor), ("clf", LogisticRegression(random_state=0))])
    rng = np.random.default_rng(0)
    train = pd.DataFrame(
        {
            "contract_type": rng.choice(["Month-to-month", "One year", "Two year"], size=60),
            "tenure_months": rng.integers(1, 72, size=60),
        }
    )
    pipeline.fit(train, rng.integers(0, 2, size=60))
    background = pd.DataFrame(
        {
            "contract_type": ["Month-to-month", "One year", "Two year"] * 5,
            "tenure_months": list(range(1, 16)),
        }
    )
    return ChampionModel(
        pipeline=pipeline, version=MagicMock(version=version), background=background
    )


@pytest.fixture
def client_with_champion(cfg: AppConfig):
    champion = _toy_champion_model(version="1")
    with (
        patch("api.main.get_app_config", return_value=cfg),
        patch("api.main.load_champion", return_value=champion),
        patch("api.main.current_champion_version", return_value="1"),
        patch("api.main.mlflow.set_tracking_uri"),
        patch("api.main.load_selected_features", return_value=["contract_type", "tenure_months"]),
    ):
        from api.main import app

        with TestClient(app) as test_client:
            yield test_client, champion


def test_score_customer_succeeds_and_persists(client_with_champion) -> None:
    test_client, _ = client_with_champion
    snapshot = pd.DataFrame(
        {"contract_type": ["One year"], "tenure_months": [24], "customer_id": ["1234-ABCD"]}
    )
    with (
        patch("api.main.pd.read_sql", return_value=snapshot),
        patch("api.main.upsert_predictions", return_value=[dt.datetime(2024, 6, 1)]) as mock_upsert,
    ):
        response = test_client.post("/score/1234-ABCD")

    assert response.status_code == 200
    body = response.json()
    assert body["customer_id"] == "1234-ABCD"
    assert body["model_version"] == "1"
    assert 0.0 <= body["probability"] <= 1.0
    assert len(body["shap_top3"]) <= 3
    mock_upsert.assert_called_once()


def test_score_customer_returns_404_when_customer_not_in_current_cohort(
    client_with_champion,
) -> None:
    test_client, _ = client_with_champion
    with patch("api.main.pd.read_sql", return_value=pd.DataFrame()):
        response = test_client.post("/score/9999-NOSUCH")
    assert response.status_code == 404


def test_score_customer_reloads_champion_when_registry_promotes_a_new_version(
    client_with_champion,
) -> None:
    """Regression test for the stale-champion gap an independent review
    found: a long-lived API process never picked up a new champion
    without a restart, and could then overwrite a newer batch-scored row
    with a stale model's output. `/score` must reload before scoring if
    the registry alias has moved."""
    test_client, original_champion = client_with_champion
    new_champion = _toy_champion_model(version="2")
    snapshot = pd.DataFrame(
        {"contract_type": ["One year"], "tenure_months": [24], "customer_id": ["1234-ABCD"]}
    )
    with (
        patch("api.main.current_champion_version", return_value="2"),
        patch("api.main.load_champion", return_value=new_champion) as mock_load_champion,
        patch("api.main.pd.read_sql", return_value=snapshot),
        patch("api.main.upsert_predictions", return_value=[dt.datetime(2024, 6, 1)]),
    ):
        response = test_client.post("/score/1234-ABCD")

    assert response.status_code == 200
    assert response.json()["model_version"] == "2"
    mock_load_champion.assert_called_once()
    assert original_champion.version.version == "1"  # the original object is untouched


def test_metrics_endpoint_is_scrapeable(client: TestClient) -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert b"churn_api_requests_total" in response.content


def test_request_id_header_is_echoed(client: TestClient) -> None:
    response = client.get("/health", headers={"X-Request-ID": "test-req-1"})
    assert response.headers["X-Request-ID"] == "test-req-1"


def test_request_id_is_generated_when_absent(client: TestClient) -> None:
    response = client.get("/health")
    assert "X-Request-ID" in response.headers
