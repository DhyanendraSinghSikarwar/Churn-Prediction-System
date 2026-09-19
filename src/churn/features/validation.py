"""Pandera schema contracts for pipeline data — the data-validation gate
referenced in PLAN.md Phase 5 ("Data validation gate using Great
Expectations or Pandera — fail loudly, never train on bad data").
Defined here in Phase 2, alongside the extraction query they validate;
Phase 5's train.py wires validate_feature_snapshot() into the actual
pipeline gate (extract -> validate -> engineer -> split -> train -> ...).

Decision: Pandera over Great Expectations — see
DECISIONS.md (Phase 0, tech-stack). Ranges/enums are checked strictly;
exact pandas dtypes are coerced rather than hard-failed on, since the
same DataFrame passes through psycopg (object-dtype dates, nullable
Int64 columns) and later through scikit-learn preprocessing, and dtype
pedantry here would just be noise — the business-rule violations are
what "fail loudly" is actually protecting against.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

_TENURE_BUCKETS = ("0-6", "6-12", "12-24", "24-48", "48+")
_CONTRACT_TYPES = ("Month-to-month", "One year", "Two year")


class FeatureSnapshotSchema(pa.DataFrameModel):
    """Contract for the output of
    src/churn/features/extract.py::extract_feature_snapshot — one row per
    (customer_id, reference_date)."""

    customer_id: str = pa.Field(nullable=False)
    feature_set_version: str = pa.Field(nullable=False)

    senior_citizen: bool = pa.Field(nullable=False)
    partner: bool = pa.Field(nullable=False)
    dependents: bool = pa.Field(nullable=False)
    contract_type: str = pa.Field(isin=_CONTRACT_TYPES)
    paperless_billing: bool = pa.Field(nullable=False)
    monthly_charges: float = pa.Field(ge=0)

    tenure_months: int = pa.Field(ge=0)
    tenure_bucket: str = pa.Field(isin=_TENURE_BUCKETS)

    frequency_charges_30d: int = pa.Field(ge=0)
    frequency_charges_60d: int = pa.Field(ge=0)
    frequency_charges_90d: int = pa.Field(ge=0)
    frequency_tickets_30d: int = pa.Field(ge=0)
    frequency_tickets_60d: int = pa.Field(ge=0)
    frequency_tickets_90d: int = pa.Field(ge=0)
    monetary_charges_30d: float = pa.Field(ge=0)
    monetary_charges_60d: float = pa.Field(ge=0)
    monetary_charges_90d: float = pa.Field(ge=0)
    # Replaces the leaky raw TotalCharges column — see
    # DECISIONS.md#total-charges-dropped-at-ingestion.
    cumulative_charges_to_date: float = pa.Field(ge=0)

    payment_failure_streak: int = pa.Field(ge=0)
    payment_failure_rate_180d: float | None = pa.Field(nullable=True, ge=0, le=1)

    # label_churn_next_30d is nullable in the live-inference case (label
    # horizon hasn't matured yet) but must always be present, non-null,
    # in a training panel — enforced by validate_training_panel() below,
    # not by this base schema, so the same schema serves both contexts.
    label_churn_next_30d: bool | None = pa.Field(nullable=True)

    class Config:
        strict = False  # extra columns (snapshot_id, reference_date, created_at, ...) are fine
        coerce = True


def validate_feature_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a feature_snapshot DataFrame (one reference_date's worth,
    or the whole materialized panel). Raises
    pandera.errors.SchemaErrors (collecting every violation, not just the
    first) if the contract is violated.
    """
    return FeatureSnapshotSchema.validate(df, lazy=True)


def validate_training_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Stricter contract for a panel about to be used for training: the
    label must be fully populated (no NULLs — see
    DECISIONS.md#label-lookahead-is-not-leakage; a NULL label means the
    horizon hadn't matured, which should never happen for a historical
    training snapshot) and every declared feature_set_version must match
    the one requested, i.e. no accidental mixing of two feature schema
    versions in one training run.
    """
    validated = validate_feature_snapshot(df)
    if validated["label_churn_next_30d"].isna().any():
        raise ValueError(
            "Training panel has rows with a NULL label — every historical "
            "snapshot should have a matured, known outcome. Check "
            "reference_date range against SNAPSHOT_DATE / horizon_days."
        )
    if validated["feature_set_version"].nunique() > 1:
        raise ValueError(
            "Training panel mixes multiple feature_set_version values — "
            "pin one version per training run."
        )
    return validated
