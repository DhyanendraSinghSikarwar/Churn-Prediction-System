import datetime as dt

import numpy as np
import pandas as pd
import pytest

from churn.ingestion.synthesize import derive_customer_dates


@pytest.fixture
def raw_sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customerID": ["C1", "C2"],
            "tenure": [12, 24],
            "Churn": ["No", "Yes"],
        }
    )


def test_active_customer_signup_date_is_before_snapshot(raw_sample: pd.DataFrame) -> None:
    snapshot = dt.date(2024, 6, 1)
    rng = np.random.default_rng(42)
    dates = derive_customer_dates(raw_sample, snapshot, rng)

    active = dates[dates["customer_id"] == "C1"].iloc[0]
    assert active["signup_date"] < snapshot
    assert active["churn_date"] is None
    assert active["censored_as_of"] == snapshot


def test_churned_customer_has_signup_before_churn_date(raw_sample: pd.DataFrame) -> None:
    snapshot = dt.date(2024, 6, 1)
    rng = np.random.default_rng(42)
    dates = derive_customer_dates(raw_sample, snapshot, rng)

    churned = dates[dates["customer_id"] == "C2"].iloc[0]
    assert churned["churn_date"] is not None
    assert churned["signup_date"] < churned["churn_date"]
    assert churned["churn_date"] <= snapshot


def test_reproducible_given_same_seed(raw_sample: pd.DataFrame) -> None:
    snapshot = dt.date(2024, 6, 1)
    dates_a = derive_customer_dates(raw_sample, snapshot, np.random.default_rng(7))
    dates_b = derive_customer_dates(raw_sample, snapshot, np.random.default_rng(7))

    pd.testing.assert_frame_equal(dates_a, dates_b)
