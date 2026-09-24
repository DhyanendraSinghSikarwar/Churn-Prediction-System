"""Unit tests for churn.monitoring.loaders: each loader's SQL
query/params, verified by mocking pd.read_sql/get_engine rather than a
real database -- particularly that the current-cohort and matured-
predictions loaders filter by `model_version`, the exact filter an
independent review found missing before these loaders existed (see
DECISIONS.md's Phase 7 post-review section,
"current-cohort-queries-must-filter-by-champion-model-version-in-sql")."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from churn.config import AppConfig
from churn.monitoring.loaders import (
    champion_version,
    load_current_cohort_predictions,
    load_matured_predictions,
)


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


def test_load_current_cohort_predictions_filters_by_reference_date_and_model_version(
    cfg: AppConfig,
) -> None:
    with (
        patch("churn.monitoring.loaders.get_engine"),
        patch("churn.monitoring.loaders.pd.read_sql") as mock_read_sql,
    ):
        mock_read_sql.return_value = pd.DataFrame({"reference_date": []})
        load_current_cohort_predictions(cfg, "3")

    params = mock_read_sql.call_args.kwargs["params"]
    assert params["v"] == "3"
    assert str(params["rd"]) == "2024-06-01"


def test_load_matured_predictions_filters_by_model_version_and_matured_label(
    cfg: AppConfig,
) -> None:
    with (
        patch("churn.monitoring.loaders.get_engine"),
        patch("churn.monitoring.loaders.pd.read_sql") as mock_read_sql,
    ):
        mock_read_sql.return_value = pd.DataFrame({"reference_date": []})
        load_matured_predictions(cfg, "3")

    params = mock_read_sql.call_args.kwargs["params"]
    assert params["v"] == "3"
    query_text = str(mock_read_sql.call_args.args[0])
    # the precondition cohort_performance_report's docstring relies on
    # its caller to enforce.
    assert "actual_churned IS NOT NULL" in query_text


def test_champion_version_returns_none_when_registry_unreachable(cfg: AppConfig) -> None:
    client = MagicMock()
    client.get_model_version_by_alias.side_effect = RuntimeError("unreachable")
    assert champion_version(cfg, client=client) is None


def test_champion_version_returns_the_resolved_version(cfg: AppConfig) -> None:
    client = MagicMock()
    client.get_model_version_by_alias.return_value = MagicMock(version="5")
    assert champion_version(cfg, client=client) == "5"
