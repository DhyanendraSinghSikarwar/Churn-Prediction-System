import datetime as dt
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.config import AppConfig
from churn.models.inference import ChampionModel, risk_band, scoring_cohort, shap_top_features


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


def test_scoring_cohort_keeps_only_the_snapshot_date_row(cfg: AppConfig) -> None:
    panel = _panel_with_monthly_cohorts(dt.date(2024, 4, 1), dt.date(2024, 6, 1))
    cohort = scoring_cohort(panel, cfg)
    assert len(cohort) == 1
    assert cohort["reference_date"].iloc[0] == pd.Timestamp("2024-06-01")


def test_scoring_cohort_is_the_exact_complement_of_modeling_panel(cfg: AppConfig) -> None:
    from churn.models.split import modeling_panel

    panel = _panel_with_monthly_cohorts(dt.date(2023, 1, 1), dt.date(2024, 6, 1))
    cohort = scoring_cohort(panel, cfg)
    modeling = modeling_panel(panel, cfg)
    assert len(cohort) + len(modeling) == len(panel)
    assert set(cohort["customer_id"]) & set(modeling["customer_id"]) == set()


@pytest.mark.parametrize(
    ("probability", "expected"),
    [
        (0.0, "low"),
        (0.3, "low"),
        (0.30001, "medium"),
        (0.7, "medium"),
        (0.999, "high"),
        (1.0, "high"),
    ],
)
def test_risk_band_boundaries(cfg: AppConfig, probability: float, expected: str) -> None:
    assert risk_band(probability, cfg) == expected


def _fitted_toy_pipeline() -> Pipeline:
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
    y = rng.integers(0, 2, size=60)
    pipeline.fit(train, y)
    return pipeline


def test_shap_top_features_returns_top_n_per_row() -> None:
    pipeline = _fitted_toy_pipeline()
    background = pd.DataFrame(
        {
            "contract_type": ["Month-to-month", "One year", "Two year"] * 5,
            "tenure_months": list(range(1, 16)),
        }
    )
    champion = ChampionModel(pipeline=pipeline, version=MagicMock(), background=background)

    features = pd.DataFrame(
        {"contract_type": ["Month-to-month", "Two year"], "tenure_months": [3, 60]}
    )
    result = shap_top_features(champion, features, top_n=2)

    assert len(result) == 2
    for row in result:
        assert len(row) <= 2
        assert all(isinstance(v, float) for v in row.values())
