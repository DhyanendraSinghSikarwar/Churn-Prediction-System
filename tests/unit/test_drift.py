import numpy as np
import pandas as pd
import pytest

from churn.config import AppConfig
from churn.monitoring.drift import (
    drift_report,
    kl_divergence,
    ks_statistic,
    population_stability_index,
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


def _numeric_series(rng: np.random.Generator, mean: float, n: int = 2000) -> pd.Series:
    return pd.Series(rng.normal(loc=mean, scale=1.0, size=n))


def test_psi_is_near_zero_for_identical_distributions() -> None:
    rng = np.random.default_rng(0)
    reference = _numeric_series(rng, mean=0.0)
    current = _numeric_series(rng, mean=0.0)
    assert population_stability_index(reference, current) < 0.02


def test_psi_is_large_for_a_clearly_shifted_distribution() -> None:
    rng = np.random.default_rng(0)
    reference = _numeric_series(rng, mean=0.0)
    current = _numeric_series(rng, mean=3.0)  # 3 std devs away
    assert population_stability_index(reference, current) > 0.25


def test_psi_works_for_categorical_features() -> None:
    reference = pd.Series(["A"] * 500 + ["B"] * 500)
    current_same = pd.Series(["A"] * 500 + ["B"] * 500)
    current_shifted = pd.Series(["A"] * 950 + ["B"] * 50)

    assert population_stability_index(reference, current_same) < 0.02
    assert population_stability_index(reference, current_shifted) > 0.25


def test_ks_statistic_rejects_null_for_a_clearly_shifted_distribution() -> None:
    rng = np.random.default_rng(0)
    reference = _numeric_series(rng, mean=0.0)
    current = _numeric_series(rng, mean=3.0)
    statistic, p_value = ks_statistic(reference, current)
    assert statistic > 0.5
    assert p_value < 0.01


def test_ks_statistic_does_not_reject_null_for_identical_distributions() -> None:
    rng = np.random.default_rng(0)
    reference = _numeric_series(rng, mean=0.0)
    current = _numeric_series(rng, mean=0.0)
    statistic, p_value = ks_statistic(reference, current)
    assert statistic < 0.1
    assert p_value > 0.05


def test_kl_divergence_is_near_zero_for_identical_distributions() -> None:
    rng = np.random.default_rng(0)
    reference = _numeric_series(rng, mean=0.0)
    current = _numeric_series(rng, mean=0.0)
    assert kl_divergence(reference, current) < 0.05


def test_kl_divergence_is_large_for_a_shifted_distribution() -> None:
    rng = np.random.default_rng(0)
    reference = _numeric_series(rng, mean=0.0)
    current = _numeric_series(rng, mean=3.0)
    assert kl_divergence(reference, current) > 0.5


def test_psi_handles_empty_series_without_raising() -> None:
    reference = pd.Series([], dtype=float)
    current = pd.Series([1.0, 2.0])
    assert np.isnan(population_stability_index(reference, current))


def test_psi_detects_a_shift_within_a_low_cardinality_dominant_bucket() -> None:
    """Regression test for a real blocker an independent review found:
    quantile-binning alone collapsed a skewed, low-cardinality count
    feature (~91% of rows at one value, matching this project's
    frequency_charges_90d) into ~2 buckets, with nearly all mass of both
    distributions landing in the same catch-all bucket -- so PSI read
    ~0.0 no matter how much `current` actually shifted within it. Exact
    per-value binning (triggered whenever `reference` has <= `bins`
    distinct values) must catch this."""
    rng = np.random.default_rng(0)
    reference = pd.Series(np.concatenate([np.zeros(910), rng.integers(1, 6, size=90)]).astype(int))
    # current has moved a third of the dominant-bucket mass elsewhere --
    # a real shift a 2-bucket quantile split would hide entirely.
    current = pd.Series(
        np.concatenate([np.zeros(600), np.full(300, 3), rng.integers(1, 6, size=100)]).astype(int)
    )
    assert len(reference.unique()) <= 10  # confirms this hits the low-cardinality path
    assert population_stability_index(reference, current) > 0.1


def test_psi_is_zero_for_an_unshifted_constant_reference() -> None:
    reference = pd.Series([3] * 1000)
    current = pd.Series([3] * 1000)
    assert population_stability_index(reference, current) < 0.01


def test_psi_detects_a_shift_away_from_a_constant_reference() -> None:
    reference = pd.Series([3] * 1000)
    current = pd.Series([3] * 500 + [7] * 500)
    assert population_stability_index(reference, current) > 0.25


def test_drift_report_flags_a_null_rate_jump_as_significant_even_when_psi_is_stable(
    cfg: AppConfig,
) -> None:
    """Regression test: every PSI/KS/KL computation drops nulls before
    comparing, so a feature that started going missing far more often
    looked "stable" purely because the *surviving* non-null values
    hadn't moved -- an independent review found a feature going from 0%
    to 100% null came back merely "unknown" (PSI undefined), the least
    visible outcome for the most serious data failure a feature can
    have."""
    rng = np.random.default_rng(0)
    non_null = _numeric_series(rng, mean=0.0, n=1000)
    reference = pd.DataFrame({"feature": non_null})
    current_values = non_null.copy()
    current_values.iloc[:600] = np.nan  # 60% newly null; the rest unchanged
    current = pd.DataFrame({"feature": current_values})

    report = drift_report(reference, current, ["feature"], cfg)

    assert report[0].severity == "significant"
    assert report[0].reference_null_rate == pytest.approx(0.0)
    assert report[0].current_null_rate == pytest.approx(0.6)


def test_drift_report_null_rates_are_stable_when_unchanged(cfg: AppConfig) -> None:
    rng = np.random.default_rng(0)
    reference = pd.DataFrame({"feature": _numeric_series(rng, mean=0.0)})
    current = pd.DataFrame({"feature": _numeric_series(rng, mean=0.0)})

    report = drift_report(reference, current, ["feature"], cfg)

    assert report[0].reference_null_rate == 0.0
    assert report[0].current_null_rate == 0.0


def test_drift_report_sorts_by_psi_descending_and_classifies_severity(cfg: AppConfig) -> None:
    rng = np.random.default_rng(0)
    reference = pd.DataFrame(
        {
            "stable_feature": _numeric_series(rng, mean=0.0),
            "shifted_feature": _numeric_series(rng, mean=0.0),
        }
    )
    current = pd.DataFrame(
        {
            "stable_feature": _numeric_series(rng, mean=0.0),
            "shifted_feature": _numeric_series(rng, mean=3.0),
        }
    )

    report = drift_report(reference, current, ["stable_feature", "shifted_feature"], cfg)

    assert [r.feature for r in report] == ["shifted_feature", "stable_feature"]
    assert report[0].severity == "significant"
    assert report[1].severity == "stable"
    assert report[0].ks_statistic is not None  # numeric feature gets a KS result


def test_drift_report_skips_ks_for_categorical_features(cfg: AppConfig) -> None:
    reference = pd.DataFrame({"contract_type": ["Month-to-month"] * 500 + ["One year"] * 500})
    current = pd.DataFrame({"contract_type": ["Month-to-month"] * 500 + ["One year"] * 500})

    report = drift_report(reference, current, ["contract_type"], cfg)

    assert report[0].ks_statistic is None
    assert report[0].ks_pvalue is None
