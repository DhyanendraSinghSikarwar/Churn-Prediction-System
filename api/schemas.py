"""Response models for the serving API."""

from __future__ import annotations

import datetime as dt

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    customer_id: str
    reference_date: dt.date
    model_version: str
    probability: float
    risk_band: str
    shap_top3: dict[str, float]
    scored_at: dt.datetime

    model_config = {"from_attributes": True}


class HealthResponse(BaseModel):
    status: str  # "ok" | "degraded"
    database: str  # "ok" | "unreachable"
    model: str  # "loaded" | "unavailable"
    model_version: str | None
