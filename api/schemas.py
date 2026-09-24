"""Response models for the serving API."""

from __future__ import annotations

import datetime as dt

from pydantic import BaseModel


class ShapFeatureResponse(BaseModel):
    feature: str
    shap_value: float

    model_config = {"from_attributes": True}


class PredictionResponse(BaseModel):
    customer_id: str
    reference_date: dt.date
    model_version: str
    probability: float
    risk_band: str
    # A rank-ordered list, not a dict -- JSONB doesn't preserve key
    # insertion order (it sorts by key length), which silently discarded
    # the "top-3, in order" ranking on every round-trip through
    # `predictions.shap_top3`. Found by an independent review comparing
    # the same row read back two different ways. See DECISIONS.md.
    shap_top3: list[ShapFeatureResponse]
    scored_at: dt.datetime

    model_config = {"from_attributes": True}


class HealthResponse(BaseModel):
    status: str  # "ok" | "degraded"
    database: str  # "ok" | "unreachable"
    model: str  # "loaded" | "unavailable"
    model_version: str | None
