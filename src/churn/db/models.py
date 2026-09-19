"""SQLAlchemy ORM models — the single source of truth for the schema.

Alembic migrations are generated from this file (`alembic revision
--autogenerate`). `sql/ddl/schema.sql` is a generated, human-readable
mirror dumped straight from Alembic (`alembic upgrade head --sql`) — see
DECISIONS.md#ddl-generated-from-alembic-not-hand-duplicated. Never hand-
edit either of those to add a column; change it here first.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Numeric,
    SmallInteger,
    String,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Customer(Base):
    __tablename__ = "customers"

    customer_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    gender: Mapped[str] = mapped_column(String(10), nullable=False)
    senior_citizen: Mapped[bool] = mapped_column(nullable=False)
    partner: Mapped[bool] = mapped_column(nullable=False)
    dependents: Mapped[bool] = mapped_column(nullable=False)
    # Derived: snapshot_date - tenure (active) or churn_date - tenure
    # (churned) — never the raw `tenure` column itself. See
    # DECISIONS.md#leakage-mitigations.
    signup_date: Mapped[dt.date] = mapped_column(Date, nullable=False)
    # Hash of the raw source row this customer was derived from — lets an
    # audit trace any customer back to its exact input row.
    source_row_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    subscriptions: Mapped[list[Subscription]] = relationship(back_populates="customer")
    billing_events: Mapped[list[BillingEvent]] = relationship(back_populates="customer")
    support_tickets: Mapped[list[SupportTicket]] = relationship(back_populates="customer")
    churn_label: Mapped[ChurnLabel] = relationship(back_populates="customer", uselist=False)


class Subscription(Base):
    """One row per customer for now — see
    DECISIONS.md#synthetic-service-attributes-are-static-for-now. `end_date`
    is nullable (not enforced 1:1 with churn) so the schema can support a
    real plan-change history later without a breaking migration.
    """

    __tablename__ = "subscriptions"
    __table_args__ = (
        CheckConstraint(
            "contract_type IN ('Month-to-month', 'One year', 'Two year')",
            name="ck_subscriptions_contract_type",
        ),
        CheckConstraint("monthly_charges >= 0", name="ck_subscriptions_monthly_charges_nonneg"),
        CheckConstraint(
            "end_date IS NULL OR end_date >= start_date",
            name="ck_subscriptions_end_after_start",
        ),
        Index(
            "ux_subscriptions_active_customer",
            "customer_id",
            unique=True,
            postgresql_where=text("end_date IS NULL"),
        ),
    )

    subscription_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.customer_id"), nullable=False)
    contract_type: Mapped[str] = mapped_column(String(20), nullable=False)
    phone_service: Mapped[bool] = mapped_column(nullable=False)
    multiple_lines: Mapped[str] = mapped_column(String(20), nullable=False)
    internet_service: Mapped[str] = mapped_column(String(20), nullable=False)
    online_security: Mapped[str] = mapped_column(String(20), nullable=False)
    online_backup: Mapped[str] = mapped_column(String(20), nullable=False)
    device_protection: Mapped[str] = mapped_column(String(20), nullable=False)
    tech_support: Mapped[str] = mapped_column(String(20), nullable=False)
    streaming_tv: Mapped[str] = mapped_column(String(20), nullable=False)
    streaming_movies: Mapped[str] = mapped_column(String(20), nullable=False)
    paperless_billing: Mapped[bool] = mapped_column(nullable=False)
    payment_method: Mapped[str] = mapped_column(String(30), nullable=False)
    monthly_charges: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)
    start_date: Mapped[dt.date] = mapped_column(Date, nullable=False)
    end_date: Mapped[dt.date | None] = mapped_column(Date, nullable=True)

    customer: Mapped[Customer] = relationship(back_populates="subscriptions")


class BillingEvent(Base):
    """Synthetic monthly billing cycle events. `refund` is allowed by the
    CHECK constraint for schema completeness but the v1 generator does not
    produce any yet — see DECISIONS.md.
    """

    __tablename__ = "billing_events"
    __table_args__ = (
        CheckConstraint(
            "event_type IN ('charge', 'payment_failed', 'refund')",
            name="ck_billing_events_event_type",
        ),
        CheckConstraint("amount >= 0", name="ck_billing_events_amount_nonneg"),
        Index("ix_billing_events_customer_date", "customer_id", "event_date"),
    )

    event_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.customer_id"), nullable=False)
    subscription_id: Mapped[int] = mapped_column(
        ForeignKey("subscriptions.subscription_id"), nullable=False
    )
    event_date: Mapped[dt.date] = mapped_column(Date, nullable=False)
    event_type: Mapped[str] = mapped_column(String(20), nullable=False)
    amount: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)

    customer: Mapped[Customer] = relationship(back_populates="billing_events")


class SupportTicket(Base):
    __tablename__ = "support_tickets"
    __table_args__ = (
        CheckConstraint(
            "priority IN ('low', 'medium', 'high')", name="ck_support_tickets_priority"
        ),
        CheckConstraint(
            "satisfaction_score IS NULL OR satisfaction_score BETWEEN 0 AND 5",
            name="ck_support_tickets_satisfaction_range",
        ),
        CheckConstraint(
            "resolved_date IS NULL OR resolved_date >= created_date",
            name="ck_support_tickets_resolved_after_created",
        ),
        Index("ix_support_tickets_customer_date", "customer_id", "created_date"),
    )

    ticket_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.customer_id"), nullable=False)
    created_date: Mapped[dt.date] = mapped_column(Date, nullable=False)
    resolved_date: Mapped[dt.date | None] = mapped_column(Date, nullable=True)
    category: Mapped[str] = mapped_column(String(30), nullable=False)
    priority: Mapped[str] = mapped_column(String(10), nullable=False)
    satisfaction_score: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)

    customer: Mapped[Customer] = relationship(back_populates="support_tickets")


class ChurnLabel(Base):
    """Ground truth, derived once from the raw `Churn` column — never
    exposed as a feature itself. See DECISIONS.md#leakage-mitigations."""

    __tablename__ = "churn_labels"
    __table_args__ = (
        CheckConstraint(
            "(churned = false AND churn_date IS NULL) "
            "OR (churned = true AND churn_date IS NOT NULL)",
            name="ck_churn_labels_churn_date_consistency",
        ),
    )

    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.customer_id"), primary_key=True)
    churned: Mapped[bool] = mapped_column(nullable=False)
    churn_date: Mapped[dt.date | None] = mapped_column(Date, nullable=True)
    censored_as_of: Mapped[dt.date] = mapped_column(Date, nullable=False)

    customer: Mapped[Customer] = relationship(back_populates="churn_label")


class FeatureSnapshot(Base):
    """The training/inference panel: one row per (customer_id,
    reference_date). Phase 1 created only the keys and the label; this is
    the Phase 2 migration that adds the typed feature columns, computed by
    sql/features/build_feature_snapshot.sql — see
    DECISIONS.md#feature_snapshots-schema-staged-across-phases.

    This holds the full CANDIDATE feature set. Feature *selection* (mutual
    information / permutation importance / Boruta / correlation pruning,
    notebook 03) picks a subset of these columns for the model to actually
    train on — it does not remove columns from this table. Selection is a
    modeling-time decision, not a schema change.
    """

    __tablename__ = "feature_snapshots"
    __table_args__ = (
        UniqueConstraint(
            "customer_id",
            "reference_date",
            "feature_set_version",
            name="uq_feature_snapshots_customer_ref_version",
        ),
        Index("ix_feature_snapshots_reference_date", "reference_date"),
        CheckConstraint(
            "tenure_bucket IN ('0-6', '6-12', '12-24', '24-48', '48+')",
            name="ck_feature_snapshots_tenure_bucket",
        ),
        CheckConstraint("tenure_months >= 0", name="ck_feature_snapshots_tenure_months_nonneg"),
        CheckConstraint(
            "payment_failure_rate_180d IS NULL " "OR payment_failure_rate_180d BETWEEN 0 AND 1",
            name="ck_feature_snapshots_payment_failure_rate_range",
        ),
    )

    snapshot_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.customer_id"), nullable=False)
    reference_date: Mapped[dt.date] = mapped_column(Date, nullable=False)
    feature_set_version: Mapped[str] = mapped_column(String(20), nullable=False)

    # --- static / plan attributes as of reference_date (see
    # DECISIONS.md#synthetic-service-attributes-are-static-for-now — safe
    # at any reference_date >= signup_date) ---
    senior_citizen: Mapped[bool] = mapped_column(nullable=False)
    partner: Mapped[bool] = mapped_column(nullable=False)
    dependents: Mapped[bool] = mapped_column(nullable=False)
    contract_type: Mapped[str] = mapped_column(String(20), nullable=False)
    internet_service: Mapped[str] = mapped_column(String(20), nullable=False)
    online_security: Mapped[str] = mapped_column(String(20), nullable=False)
    online_backup: Mapped[str] = mapped_column(String(20), nullable=False)
    device_protection: Mapped[str] = mapped_column(String(20), nullable=False)
    tech_support: Mapped[str] = mapped_column(String(20), nullable=False)
    streaming_tv: Mapped[str] = mapped_column(String(20), nullable=False)
    streaming_movies: Mapped[str] = mapped_column(String(20), nullable=False)
    paperless_billing: Mapped[bool] = mapped_column(nullable=False)
    payment_method: Mapped[str] = mapped_column(String(30), nullable=False)
    monthly_charges: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)

    # --- tenure, recomputed as of reference_date — never the raw
    # `tenure` column (DECISIONS.md#leakage-mitigations) ---
    tenure_months: Mapped[int] = mapped_column(nullable=False)
    tenure_bucket: Mapped[str] = mapped_column(String(10), nullable=False)

    # --- RFM, all strictly event_date/created_date < reference_date ---
    recency_days_since_last_charge: Mapped[int | None] = mapped_column(nullable=True)
    recency_days_since_last_ticket: Mapped[int | None] = mapped_column(nullable=True)
    frequency_charges_30d: Mapped[int] = mapped_column(nullable=False)
    frequency_charges_60d: Mapped[int] = mapped_column(nullable=False)
    frequency_charges_90d: Mapped[int] = mapped_column(nullable=False)
    frequency_tickets_30d: Mapped[int] = mapped_column(nullable=False)
    frequency_tickets_60d: Mapped[int] = mapped_column(nullable=False)
    frequency_tickets_90d: Mapped[int] = mapped_column(nullable=False)
    monetary_charges_30d: Mapped[float] = mapped_column(Numeric(12, 2), nullable=False)
    monetary_charges_60d: Mapped[float] = mapped_column(Numeric(12, 2), nullable=False)
    monetary_charges_90d: Mapped[float] = mapped_column(Numeric(12, 2), nullable=False)
    # Replaces the leaky raw TotalCharges column — see
    # DECISIONS.md#total-charges-dropped-at-ingestion.
    cumulative_charges_to_date: Mapped[float] = mapped_column(Numeric(12, 2), nullable=False)

    # --- trend / velocity ---
    billing_amount_slope_90d: Mapped[float | None] = mapped_column(Numeric(10, 4), nullable=True)
    billing_amount_mom_delta: Mapped[float | None] = mapped_column(Numeric(10, 2), nullable=True)
    ticket_velocity_delta_30_60: Mapped[int] = mapped_column(nullable=False)
    payment_failure_streak: Mapped[int] = mapped_column(nullable=False)
    payment_failure_rate_180d: Mapped[float | None] = mapped_column(Numeric(5, 4), nullable=True)

    # --- label: churn within prediction_horizon_days of reference_date.
    # Looks FORWARD from reference_date — that's expected of a supervised
    # label, not a leak, as long as it never feeds a feature column above.
    # NULL means the horizon hadn't matured yet when this snapshot was
    # built (live inference); always populated for historical/training
    # snapshots in this dataset. ---
    label_churn_next_30d: Mapped[bool | None] = mapped_column(nullable=True)

    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        CheckConstraint("risk_band IN ('low', 'medium', 'high')", name="ck_predictions_risk_band"),
        CheckConstraint(
            "probability >= 0 AND probability <= 1", name="ck_predictions_probability_range"
        ),
        UniqueConstraint(
            "customer_id",
            "reference_date",
            "model_version",
            name="uq_predictions_customer_ref_model",
        ),
    )

    prediction_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.customer_id"), nullable=False)
    reference_date: Mapped[dt.date] = mapped_column(Date, nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    probability: Mapped[float] = mapped_column(Numeric(6, 5), nullable=False)
    risk_band: Mapped[str] = mapped_column(String(10), nullable=False)
    shap_top3: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    scored_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    # Backfilled once the 30-day label window matures — see
    # DECISIONS.md / PLAN.md monitoring section (delayed-label handling).
    actual_churned: Mapped[bool | None] = mapped_column(nullable=True)
    label_matured_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class IngestionRun(Base):
    """Idempotency ledger: one row per successfully-loaded dataset
    checksum. See DECISIONS.md#copy-based-bulk-load-with-two-layer-idempotency.
    """

    __tablename__ = "ingestion_runs"
    __table_args__ = (UniqueConstraint("sha256", name="uq_ingestion_runs_sha256"),)

    run_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    source_url: Mapped[str] = mapped_column(String(500), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    row_count: Mapped[int] = mapped_column(nullable=False)
    loaded_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
