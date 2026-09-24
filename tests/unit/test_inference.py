import datetime as dt
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.config import AppConfig
from churn.models.inference import (
    _UPSERT_CHUNK_SIZE,
    ChampionModel,
    PredictionRow,
    _warn_on_unseen_categories,
    risk_band,
    scoring_cohort,
    shap_top_features,
    upsert_predictions,
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


def _toy_champion() -> tuple[ChampionModel, Pipeline]:
    pipeline = _fitted_toy_pipeline()
    background = pd.DataFrame(
        {
            "contract_type": ["Month-to-month", "One year", "Two year"] * 5,
            "tenure_months": list(range(1, 16)),
        }
    )
    champion = ChampionModel(pipeline=pipeline, version=MagicMock(), background=background)
    return champion, pipeline


def test_shap_top_features_returns_top_n_per_row() -> None:
    champion, _ = _toy_champion()
    features = pd.DataFrame(
        {"contract_type": ["Month-to-month", "Two year"], "tenure_months": [3, 60]}
    )
    result = shap_top_features(champion, features, top_n=2)

    assert len(result) == 2
    for row in result:
        assert len(row) <= 2
        assert all(isinstance(entry["shap_value"], float) for entry in row)
        assert all(isinstance(entry["feature"], str) for entry in row)


def test_shap_top_features_returns_a_ranked_list_not_a_dict() -> None:
    """Regression test: a dict silently loses rank order on a JSONB
    round-trip (Postgres sorts JSONB keys by length, not insertion
    order) -- found by an independent review comparing the same row read
    back two different ways. `shap_top_features` must return an ordered
    list."""
    champion, _ = _toy_champion()
    features = pd.DataFrame({"contract_type": ["Two year"], "tenure_months": [60]})
    result = shap_top_features(champion, features, top_n=3)[0]

    assert isinstance(result, list)
    magnitudes = [abs(entry["shap_value"]) for entry in result]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_shap_top_features_aggregates_one_hot_columns_by_source_feature() -> None:
    """Regression test for the blocker an independent review found: the
    old implementation kept one-hot column names as-is, so a customer's
    "top reasons" could name contract types they don't have (e.g.
    `cat__contract_type_Two year` appearing for a Month-to-month
    customer). Aggregating by source feature means `contract_type` can
    appear at most once per row, never once per category."""
    champion, _ = _toy_champion()
    features = pd.DataFrame(
        {"contract_type": ["Month-to-month", "One year", "Two year"], "tenure_months": [3, 30, 60]}
    )
    results = shap_top_features(champion, features, top_n=3)

    for row in results:
        feature_names = [entry["feature"] for entry in row]
        assert len(feature_names) == len(set(feature_names)), "a feature appeared more than once"
        for name in feature_names:
            assert name in {"contract_type", "tenure_months"}
            assert not name.startswith("cat__")
            assert not name.startswith("num__")


def test_warn_on_unseen_categories_logs_but_does_not_raise() -> None:
    champion, pipeline = _toy_champion()
    features = pd.DataFrame({"contract_type": ["Three year"], "tenure_months": [12]})
    prep = pipeline.named_steps["prep"]

    with patch("churn.models.inference.logger") as mock_logger:
        _warn_on_unseen_categories(prep, features)

    mock_logger.warning.assert_called_once()
    args, kwargs = mock_logger.warning.call_args
    assert args[0] == "unseen_category_scored"
    assert kwargs["unseen_values"] == ["Three year"]


def test_warn_on_unseen_categories_silent_when_all_known() -> None:
    champion, pipeline = _toy_champion()
    features = pd.DataFrame({"contract_type": ["One year"], "tenure_months": [12]})
    prep = pipeline.named_steps["prep"]

    with patch("churn.models.inference.logger") as mock_logger:
        _warn_on_unseen_categories(prep, features)

    mock_logger.warning.assert_not_called()


def _prediction_row(customer_id: str) -> PredictionRow:
    return PredictionRow(
        customer_id=customer_id,
        reference_date=dt.date(2024, 6, 1),
        model_version="1",
        probability=0.5,
        risk_band="medium",
        shap_top3=[{"feature": "tenure_months", "shap_value": 0.1}],
    )


def test_upsert_predictions_chunks_large_batches() -> None:
    """Regression test: a single multi-row upsert statement is capped by
    Postgres's 65535-bound-parameter limit (6 columns/row -> ~10,922
    rows/statement). Found live by an independent review pushing a batch
    past that limit. `upsert_predictions` must issue multiple statements
    rather than one oversized one."""
    rows = [_prediction_row(f"C{i}") for i in range(_UPSERT_CHUNK_SIZE + 1)]
    conn = MagicMock()
    conn.execute.return_value = [MagicMock(scored_at=dt.datetime(2024, 6, 1))] * _UPSERT_CHUNK_SIZE

    upsert_predictions(conn, rows)

    assert conn.execute.call_count == 2  # one full chunk + a one-row remainder


def test_upsert_predictions_issues_one_statement_for_a_small_batch() -> None:
    rows = [_prediction_row("C1"), _prediction_row("C2")]
    conn = MagicMock()
    conn.execute.return_value = [MagicMock(scored_at=dt.datetime(2024, 6, 1))] * 2

    result = upsert_predictions(conn, rows)

    assert conn.execute.call_count == 1
    assert len(result) == 2
