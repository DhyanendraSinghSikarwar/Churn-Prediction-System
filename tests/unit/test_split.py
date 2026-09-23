import datetime as dt

import pandas as pd
import pytest

from churn.config import AppConfig
from churn.models.split import modeling_panel, time_based_split


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


def _panel_with_monthly_cohorts(start: dt.date, end: dt.date) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="MS")
    customer_ids = [f"C{i}" for i in range(len(dates))]
    return pd.DataFrame({"reference_date": dates, "customer_id": customer_ids})


def test_modeling_panel_drops_snapshot_date_cohort(cfg: AppConfig) -> None:
    panel = _panel_with_monthly_cohorts(dt.date(2024, 4, 1), dt.date(2024, 6, 1))
    filtered = modeling_panel(panel, cfg)
    assert pd.Timestamp("2024-06-01") not in filtered["reference_date"].to_numpy()
    assert len(filtered) == 2


def test_modeling_panel_keeps_earlier_cohorts(cfg: AppConfig) -> None:
    panel = _panel_with_monthly_cohorts(dt.date(2023, 1, 1), dt.date(2024, 6, 1))
    filtered = modeling_panel(panel, cfg)
    assert len(filtered) == len(panel) - 1


def test_time_based_split_boundaries(cfg: AppConfig) -> None:
    panel = _panel_with_monthly_cohorts(dt.date(2023, 1, 1), dt.date(2024, 5, 1))
    train, val, test = time_based_split(panel, cfg)

    assert train["reference_date"].max() == pd.Timestamp("2023-12-01")
    assert val["reference_date"].min() == pd.Timestamp("2024-01-01")
    assert val["reference_date"].max() == pd.Timestamp("2024-03-01")
    assert test["reference_date"].min() == pd.Timestamp("2024-04-01")
    assert test["reference_date"].max() == pd.Timestamp("2024-05-01")


def test_time_based_split_is_exhaustive_and_disjoint(cfg: AppConfig) -> None:
    panel = _panel_with_monthly_cohorts(dt.date(2023, 1, 1), dt.date(2024, 5, 1))
    train, val, test = time_based_split(panel, cfg)

    assert len(train) + len(val) + len(test) == len(panel)
    assert set(train["customer_id"]) & set(val["customer_id"]) == set()
    assert set(val["customer_id"]) & set(test["customer_id"]) == set()


def test_time_based_split_shrinks_when_censored_cohort_already_dropped(cfg: AppConfig) -> None:
    """The nominal test window is 3 months, but if the caller already
    dropped the last (censored) cohort via modeling_panel(), the test
    split naturally comes back with only 2 -- not special-cased, this is
    exactly what Phase 4's notebooks found (see PLAN.md)."""
    panel = _panel_with_monthly_cohorts(dt.date(2023, 1, 1), dt.date(2024, 6, 1))
    filtered = modeling_panel(panel, cfg)
    _, _, test = time_based_split(filtered, cfg)
    assert len(test) == 2
