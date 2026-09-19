import datetime as dt

import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from churn.features.validation import validate_feature_snapshot, validate_training_panel

_BASE_ROW: dict[str, object] = {
    "customer_id": "C0001",
    "reference_date": dt.date(2024, 1, 1),
    "feature_set_version": "v1",
    "senior_citizen": False,
    "partner": True,
    "dependents": False,
    "contract_type": "Month-to-month",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "online_backup": "No",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "No",
    "streaming_movies": "No",
    "paperless_billing": True,
    "payment_method": "Electronic check",
    "monthly_charges": 70.0,
    "tenure_months": 12,
    "tenure_bucket": "6-12",
    "recency_days_since_last_charge": 5,
    "recency_days_since_last_ticket": None,
    "frequency_charges_30d": 1,
    "frequency_charges_60d": 2,
    "frequency_charges_90d": 3,
    "frequency_tickets_30d": 0,
    "frequency_tickets_60d": 0,
    "frequency_tickets_90d": 0,
    "monetary_charges_30d": 70.0,
    "monetary_charges_60d": 140.0,
    "monetary_charges_90d": 210.0,
    "cumulative_charges_to_date": 840.0,
    "billing_amount_slope_90d": 0.1,
    "billing_amount_mom_delta": 0.0,
    "ticket_velocity_delta_30_60": 0,
    "payment_failure_streak": 0,
    "payment_failure_rate_180d": 0.0,
    "label_churn_next_30d": False,
}


def _df(**overrides: object) -> pd.DataFrame:
    row = {**_BASE_ROW, **overrides}
    return pd.DataFrame([row])


def test_valid_row_passes() -> None:
    validated = validate_feature_snapshot(_df())
    assert len(validated) == 1


def test_negative_monthly_charges_rejected() -> None:
    with pytest.raises(SchemaErrors):
        validate_feature_snapshot(_df(monthly_charges=-10.0))


def test_unknown_contract_type_rejected() -> None:
    with pytest.raises(SchemaErrors):
        validate_feature_snapshot(_df(contract_type="Lifetime"))


def test_unknown_tenure_bucket_rejected() -> None:
    with pytest.raises(SchemaErrors):
        validate_feature_snapshot(_df(tenure_bucket="0-100"))


def test_payment_failure_rate_out_of_range_rejected() -> None:
    with pytest.raises(SchemaErrors):
        validate_feature_snapshot(_df(payment_failure_rate_180d=1.5))


def test_negative_tenure_months_rejected() -> None:
    with pytest.raises(SchemaErrors):
        validate_feature_snapshot(_df(tenure_months=-1))


def test_training_panel_rejects_null_label() -> None:
    df = _df(label_churn_next_30d=None)
    with pytest.raises(ValueError, match="NULL label"):
        validate_training_panel(df)


def test_training_panel_rejects_mixed_feature_set_versions() -> None:
    df = pd.concat([_df(), _df(customer_id="C0002", feature_set_version="v2")], ignore_index=True)
    with pytest.raises(ValueError, match="multiple feature_set_version"):
        validate_training_panel(df)


def test_training_panel_accepts_clean_panel() -> None:
    df = pd.concat([_df(), _df(customer_id="C0002")], ignore_index=True)
    validated = validate_training_panel(df)
    assert len(validated) == 2
