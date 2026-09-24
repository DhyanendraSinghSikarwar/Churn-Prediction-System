import datetime as dt
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from churn.config import AppConfig


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
        probability=0.42,
        risk_band="medium",
        shap_top3={"tenure_months": -0.12},
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


def test_get_prediction_returns_503_when_db_unreachable(client: TestClient) -> None:
    with patch("api.main.get_session") as mock_get_session:
        mock_get_session.side_effect = OperationalError("x", {}, Exception("x"))
        response = client.get("/predictions/1234-ABCD")
    assert response.status_code == 503


def test_score_customer_returns_503_when_model_unavailable(client: TestClient) -> None:
    # fixture's load_champion raises at startup -- app.state.champion stays None.
    response = client.post("/score/1234-ABCD")
    assert response.status_code == 503


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
