"""Synthesise the temporal layer (signup dates, billing events, support
tickets, churn labels) on top of the static IBM Telco snapshot.

WHY THIS EXISTS: the source dataset is a single cross-sectional snapshot —
one terminal (tenure, Churn) pair per customer, no event history. This
module manufactures a plausible event history so the rest of the system
can do genuine point-in-time feature engineering instead of reading
pre-aggregated columns. Every distributional assumption below (payment-
failure rates, ticket rates, churn-date placement) is an explicit,
documented, seeded choice — not fitted to any real telecom data. See
DECISIONS.md (Phase 1 section) for the reasoning behind each one.

Randomness is threaded through as an explicit numpy.random.Generator
rather than a global seed (DECISIONS.md#numpy-Generator-not-global-seed),
so calling this twice with the same seed is guaranteed to reproduce the
same tables byte-for-byte.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from churn.config import AppConfig
from churn.utils.hashing import sha256_of_bytes

# How far back, before SNAPSHOT_DATE, churned customers' churn dates are
# assumed to be uniformly spread. Must comfortably cover the training
# panel window (18 months) so the panel has real churn events to learn
# from — see DECISIONS.md#churn-date-placement-for-synthetic-history.
_CHURN_LOOKBACK_MONTHS = 24

_TICKET_CATEGORIES = ("billing", "technical", "account", "other")
_TICKET_PRIORITIES = ("low", "medium", "high")
_TICKET_PRIORITY_WEIGHTS = (0.6, 0.3, 0.1)

_BOOL_YES_NO = {"Yes": True, "No": False}


def _to_bool(series: pd.Series) -> pd.Series:
    return series.map(_BOOL_YES_NO).astype(bool)


def derive_customer_dates(
    raw: pd.DataFrame, snapshot_date: dt.date, rng: np.random.Generator
) -> pd.DataFrame:
    """One row per customer: signup_date, churned, churn_date, censored_as_of.

    Active customers: signup_date = snapshot_date - tenure months (they are
    still a customer today, so "today" anchors the window).

    Churned customers: the raw dataset tells us HOW LONG they were a
    customer (tenure) before churning, but not WHEN. We assume churn dates
    are spread uniformly over the trailing _CHURN_LOOKBACK_MONTHS before
    snapshot_date, then back-compute signup_date = churn_date - tenure.
    Day-level jitter is always subtracted (never added) from the anchor
    date so signup_date can never land after the date it's computed
    relative to.
    """
    n = len(raw)
    churned = raw["Churn"].eq("Yes").to_numpy()
    tenure_months = raw["tenure"].to_numpy(dtype=int)

    day_jitter = rng.integers(0, 28, size=n)
    lookback_days = rng.integers(0, _CHURN_LOOKBACK_MONTHS * 30, size=n)

    signup_dates: np.ndarray = np.empty(n, dtype=object)
    churn_dates: np.ndarray = np.empty(n, dtype=object)

    for i in range(n):
        if churned[i]:
            churn_date = snapshot_date - dt.timedelta(days=int(lookback_days[i]))
            signup_ts = pd.Timestamp(churn_date) - pd.DateOffset(months=int(tenure_months[i]))
            signup_dates[i] = signup_ts.date() - dt.timedelta(days=int(day_jitter[i]))
            churn_dates[i] = churn_date
        else:
            signup_ts = pd.Timestamp(snapshot_date) - pd.DateOffset(months=int(tenure_months[i]))
            signup_dates[i] = signup_ts.date() - dt.timedelta(days=int(day_jitter[i]))
            churn_dates[i] = None

    return pd.DataFrame(
        {
            "customer_id": raw["customerID"].to_numpy(),
            "signup_date": signup_dates,
            "churned": churned,
            "churn_date": churn_dates,
            "censored_as_of": [snapshot_date] * n,
        }
    )


def build_customers_table(raw: pd.DataFrame, dates: pd.DataFrame) -> pd.DataFrame:
    row_hashes = [sha256_of_bytes(row.to_json().encode("utf-8")) for _, row in raw.iterrows()]
    return pd.DataFrame(
        {
            "customer_id": raw["customerID"].to_numpy(),
            "gender": raw["gender"].to_numpy(),
            "senior_citizen": raw["SeniorCitizen"].astype(bool).to_numpy(),
            "partner": _to_bool(raw["Partner"]).to_numpy(),
            "dependents": _to_bool(raw["Dependents"]).to_numpy(),
            "signup_date": dates["signup_date"].to_numpy(),
            "source_row_hash": row_hashes,
        }
    )


def build_subscriptions_table(raw: pd.DataFrame, dates: pd.DataFrame) -> pd.DataFrame:
    # NOTE: raw["TotalCharges"] is deliberately never read — see
    # DECISIONS.md#total-charges-dropped-at-ingestion.
    return pd.DataFrame(
        {
            "customer_id": raw["customerID"].to_numpy(),
            "contract_type": raw["Contract"].to_numpy(),
            "phone_service": _to_bool(raw["PhoneService"]).to_numpy(),
            "multiple_lines": raw["MultipleLines"].to_numpy(),
            "internet_service": raw["InternetService"].to_numpy(),
            "online_security": raw["OnlineSecurity"].to_numpy(),
            "online_backup": raw["OnlineBackup"].to_numpy(),
            "device_protection": raw["DeviceProtection"].to_numpy(),
            "tech_support": raw["TechSupport"].to_numpy(),
            "streaming_tv": raw["StreamingTV"].to_numpy(),
            "streaming_movies": raw["StreamingMovies"].to_numpy(),
            "paperless_billing": _to_bool(raw["PaperlessBilling"]).to_numpy(),
            "payment_method": raw["PaymentMethod"].to_numpy(),
            "monthly_charges": raw["MonthlyCharges"].astype(float).to_numpy(),
            "start_date": dates["signup_date"].to_numpy(),
            "end_date": dates["churn_date"].to_numpy(),
        }
    )


def build_churn_labels_table(dates: pd.DataFrame) -> pd.DataFrame:
    return dates[["customer_id", "churned", "churn_date", "censored_as_of"]].copy()


def generate_billing_events(
    dates: pd.DataFrame,
    subscriptions: pd.DataFrame,
    cfg: AppConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """One 'charge' event per monthly billing cycle from signup to
    churn/snapshot, plus stochastic 'payment_failed' events. Failure
    probability is elevated for Electronic-check and month-to-month
    customers — the two factors this dataset is known to associate with
    churn — so ticket/payment features have real signal to find later.
    """
    pf_cfg = cfg.synthetic.payment_failure
    merged = dates.merge(subscriptions, on="customer_id")

    rows: list[dict[str, object]] = []
    # itertuples() rows are typed by pandas-stubs as a union across every
    # column in the frame, not per-field — the type: ignores below are
    # that known stub imprecision, not a real type risk (each field's
    # actual runtime type is fixed by how `merged` was built above).
    for rec in merged.itertuples(index=False):
        cutoff = pd.Timestamp(rec.churn_date if rec.churned else cfg.data.snapshot_date)  # type: ignore[arg-type]
        cursor = pd.Timestamp(rec.signup_date) + pd.DateOffset(months=1)  # type: ignore[arg-type]

        fail_prob = pf_cfg.base_rate
        if rec.payment_method == "Electronic check":
            fail_prob += pf_cfg.electronic_check_uplift
        if rec.contract_type == "Month-to-month":
            fail_prob += pf_cfg.month_to_month_uplift
        fail_prob = min(fail_prob, 1.0)

        while cursor <= cutoff:
            amount = round(max(float(rec.monthly_charges) + rng.normal(0, 1.5), 0.0), 2)  # type: ignore[arg-type]
            rows.append(
                {
                    "customer_id": rec.customer_id,
                    "event_date": cursor.date(),
                    "event_type": "charge",
                    "amount": amount,
                }
            )
            if rng.random() < fail_prob:
                rows.append(
                    {
                        "customer_id": rec.customer_id,
                        "event_date": cursor.date(),
                        "event_type": "payment_failed",
                        "amount": amount,
                    }
                )
            cursor += pd.DateOffset(months=1)

    return pd.DataFrame(rows, columns=["customer_id", "event_date", "event_type", "amount"])


def generate_support_tickets(
    dates: pd.DataFrame,
    subscriptions: pd.DataFrame,
    cfg: AppConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Poisson-arrival ticket generation per active month. Rate is elevated
    for customers lacking TechSupport/OnlineSecurity add-ons, and spikes
    in the pre_churn_window_days before a known churn date — the
    "ticket-velocity spike before churn" signal Deliverable 3 asks
    feature engineering to extract.
    """
    st_cfg = cfg.synthetic.support_tickets
    merged = dates.merge(subscriptions, on="customer_id")

    rows: list[dict[str, object]] = []
    # See the matching comment in generate_billing_events above — same
    # pandas-stubs itertuples() imprecision, not a real type risk.
    for rec in merged.itertuples(index=False):
        cutoff = pd.Timestamp(rec.churn_date if rec.churned else cfg.data.snapshot_date)  # type: ignore[arg-type]
        cursor = pd.Timestamp(rec.signup_date)  # type: ignore[arg-type]

        base_rate = st_cfg.base_monthly_rate
        if rec.tech_support == "No":
            base_rate += st_cfg.no_tech_support_uplift
        if rec.online_security == "No":
            base_rate += st_cfg.no_online_security_uplift

        while cursor <= cutoff:
            rate = base_rate
            if rec.churned:
                days_to_churn = (cutoff - cursor).days
                if 0 <= days_to_churn <= st_cfg.pre_churn_window_days:
                    rate *= st_cfg.pre_churn_multiplier

            n_tickets = int(rng.poisson(rate))
            for _ in range(n_tickets):
                day_offset = int(rng.integers(0, 28))
                created = (cursor + pd.Timedelta(days=day_offset)).date()
                if created > cutoff.date():
                    continue

                days_of_slack = (cutoff.date() - created).days
                resolved_prob = 0.95 if days_of_slack > 14 else 0.5
                resolved = bool(rng.random() < resolved_prob)
                resolved_date = (
                    created + dt.timedelta(days=int(rng.integers(1, 15))) if resolved else None
                )
                satisfaction = int(rng.integers(1, 6)) if resolved else None
                category = _TICKET_CATEGORIES[int(rng.integers(0, len(_TICKET_CATEGORIES)))]

                rows.append(
                    {
                        "customer_id": rec.customer_id,
                        "created_date": created,
                        "resolved_date": resolved_date,
                        "category": category,
                        "priority": rng.choice(_TICKET_PRIORITIES, p=_TICKET_PRIORITY_WEIGHTS),
                        "satisfaction_score": satisfaction,
                    }
                )
            cursor += pd.DateOffset(months=1)

    tickets = pd.DataFrame(
        rows,
        columns=[
            "customer_id",
            "created_date",
            "resolved_date",
            "category",
            "priority",
            "satisfaction_score",
        ],
    )
    # A plain int column with None values upcasts to float64 (e.g. 4 ->
    # 4.0), which Postgres's smallint parser then rejects on COPY. The
    # nullable Int64 dtype keeps missing scores as <NA> without forcing a
    # float representation.
    tickets["satisfaction_score"] = tickets["satisfaction_score"].astype("Int64")
    return tickets


def synthesize_all(raw: pd.DataFrame, cfg: AppConfig, seed: int) -> dict[str, pd.DataFrame]:
    """Orchestrates the full synthetic layer. Single entry point used by
    both the ingestion CLI and tests."""
    rng = np.random.default_rng(seed)

    dates = derive_customer_dates(raw, cfg.data.snapshot_date, rng)
    customers = build_customers_table(raw, dates)
    subscriptions = build_subscriptions_table(raw, dates)
    churn_labels = build_churn_labels_table(dates)
    billing_events = generate_billing_events(dates, subscriptions, cfg, rng)
    support_tickets = generate_support_tickets(dates, subscriptions, cfg, rng)

    return {
        "customers": customers,
        "subscriptions": subscriptions,
        "churn_labels": churn_labels,
        "billing_events": billing_events,
        "support_tickets": support_tickets,
    }
