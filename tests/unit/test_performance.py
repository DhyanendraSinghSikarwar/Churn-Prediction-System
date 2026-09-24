import datetime as dt

import pandas as pd
import pytest

from churn.config import AppConfig
from churn.monitoring.performance import cohort_performance_report


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


def _predictions_row(
    reference_date: str, probability: float, risk_band: str, actual_churned: bool
) -> dict[str, object]:
    return {
        "reference_date": pd.Timestamp(reference_date),
        "probability": probability,
        # Stored risk_band -- deliberately allowed to disagree with what
        # cfg's *current* risk_bands would produce in some tests below,
        # since cohort_performance_report must recompute from
        # `probability`, never trust this column. See
        # test_cohort_performance_report_recomputes_risk_band_from_current_config.
        "risk_band": risk_band,
        "actual_churned": actual_churned,
    }


def test_cohort_performance_report_computes_precision_recall_at_risk_bands(
    cfg: AppConfig,
) -> None:
    predictions = pd.DataFrame(
        [
            _predictions_row("2023-02-01", 0.9, "high", True),  # true positive (flagged)
            _predictions_row("2023-02-01", 0.8, "high", False),  # false positive (flagged)
            _predictions_row("2023-02-01", 0.5, "medium", True),  # true positive (flagged)
            _predictions_row("2023-02-01", 0.1, "low", False),  # true negative (not flagged)
            _predictions_row("2023-02-01", 0.2, "low", True),  # false negative (not flagged)
        ]
    )

    report = cohort_performance_report(predictions, cfg)

    assert len(report) == 1
    cohort = report[0]
    assert cohort.n == 5
    assert cohort.positive_rate == pytest.approx(3 / 5)
    # flagged = {high, medium} = 3 rows, 2 of which are true positives
    assert cohort.precision_at_medium_plus == pytest.approx(2 / 3)
    # 3 actual positives, 2 of which were flagged
    assert cohort.recall_at_medium_plus == pytest.approx(2 / 3)


def test_cohort_performance_report_recomputes_risk_band_from_current_config(
    cfg: AppConfig,
) -> None:
    """Regression test: precision/recall must be computed from each
    row's `probability` against cfg's *current* risk_bands, never from
    the row's stored `risk_band` column (which reflects whatever
    thresholds were configured back when that row was scored). Here the
    stored band is deliberately wrong for every row -- if the
    implementation trusted it, every row would flag/not-flag backwards."""
    predictions = pd.DataFrame(
        [
            # probability 0.9 truly bands "high" under cfg (0.3/0.7);
            # stored band claims "low" -- must still be flagged.
            _predictions_row("2023-02-01", 0.9, "low", True),
            # probability 0.1 truly bands "low"; stored band claims
            # "high" -- must NOT be flagged.
            _predictions_row("2023-02-01", 0.1, "high", False),
        ]
    )

    report = cohort_performance_report(predictions, cfg)

    cohort = report[0]
    # Both rows classified correctly by recomputed band -> perfect
    # precision and recall, which the stored (backwards) bands would
    # have gotten completely wrong.
    assert cohort.precision_at_medium_plus == pytest.approx(1.0)
    assert cohort.recall_at_medium_plus == pytest.approx(1.0)


def test_cohort_performance_report_labels_dataset_split_correctly(cfg: AppConfig) -> None:
    predictions = pd.DataFrame(
        [
            _predictions_row("2023-02-01", 0.5, "medium", True),  # within train window
            _predictions_row("2024-02-01", 0.5, "medium", True),  # within validation window
            _predictions_row("2024-05-01", 0.5, "medium", True),  # within test window
        ]
    )

    report = cohort_performance_report(predictions, cfg)
    by_date = {r.reference_date: r for r in report}

    assert by_date[dt.date(2023, 2, 1)].dataset_split == "train"
    assert by_date[dt.date(2024, 2, 1)].dataset_split == "validation"
    assert by_date[dt.date(2024, 5, 1)].dataset_split == "test"


def test_cohort_performance_report_in_sample_flag_matches_what_the_champion_was_fit_on(
    cfg: AppConfig,
) -> None:
    """train.py fits the champion on train+val combined -- both those
    cohorts' numbers are in-sample; test (and any post-test) cohort
    numbers are not, even though test also mildly influenced which
    model got promoted (see performance.py's module docstring)."""
    predictions = pd.DataFrame(
        [
            _predictions_row("2023-02-01", 0.5, "medium", True),  # train
            _predictions_row("2024-02-01", 0.5, "medium", True),  # validation
            _predictions_row("2024-05-01", 0.5, "medium", True),  # test
        ]
    )

    report = cohort_performance_report(predictions, cfg)
    by_date = {r.reference_date: r for r in report}

    assert by_date[dt.date(2023, 2, 1)].in_sample is True
    assert by_date[dt.date(2024, 2, 1)].in_sample is True
    assert by_date[dt.date(2024, 5, 1)].in_sample is False


def test_cohort_performance_report_pr_auc_is_nan_when_only_one_class_present(
    cfg: AppConfig,
) -> None:
    predictions = pd.DataFrame(
        [
            _predictions_row("2023-02-01", 0.9, "high", False),
            _predictions_row("2023-02-01", 0.1, "low", False),
        ]
    )

    report = cohort_performance_report(predictions, cfg)

    assert report[0].pr_auc != report[0].pr_auc  # NaN != NaN
    assert report[0].roc_auc != report[0].roc_auc


def test_cohort_performance_report_sorts_chronologically(cfg: AppConfig) -> None:
    predictions = pd.DataFrame(
        [
            _predictions_row("2023-06-01", 0.5, "medium", True),
            _predictions_row("2023-02-01", 0.5, "medium", True),
            _predictions_row("2023-04-01", 0.5, "medium", True),
        ]
    )

    report = cohort_performance_report(predictions, cfg)

    assert [r.reference_date for r in report] == sorted(r.reference_date for r in report)
