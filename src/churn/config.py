"""Application configuration.

Secrets (`Settings`) come from the environment / `.env`. Business and
pipeline parameters (`AppConfig`) come from `config/config.yaml`. They are
deliberately kept apart — see DECISIONS.md#secrets-never-in-yaml — so
`config/config.yaml` is always safe to commit while `.env` never is.
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Secrets and environment-specific values."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "churn"
    db_user: str = "churn_app"
    db_password: str = ""

    mlflow_tracking_uri: str = "http://localhost:5000"

    minio_endpoint: str = "http://localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = ""

    app_env: str = "dev"
    log_level: str = "INFO"
    random_seed: int = 42

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )


class SourceConfig(BaseModel):
    primary_url: str
    fallback_url: str
    expected_row_count: int = Field(gt=0)
    raw_cache_path: str


class DataConfig(BaseModel):
    snapshot_date: date
    panel_start_date: date
    panel_end_date: date
    prediction_horizon_days: int = Field(gt=0)
    train_months: int = Field(gt=0)
    validation_months: int = Field(gt=0)
    test_months: int = Field(gt=0)
    source: SourceConfig

    @field_validator("panel_end_date")
    @classmethod
    def _panel_end_after_start(cls, v: date, info: ValidationInfo) -> date:
        start = info.data.get("panel_start_date")
        if start is not None and v <= start:
            raise ValueError("panel_end_date must be after panel_start_date")
        return v


class PaymentFailureConfig(BaseModel):
    base_rate: float = Field(ge=0, le=1)
    electronic_check_uplift: float = Field(ge=0, le=1)
    month_to_month_uplift: float = Field(ge=0, le=1)


class SupportTicketConfig(BaseModel):
    base_monthly_rate: float = Field(ge=0)
    no_tech_support_uplift: float = Field(ge=0)
    no_online_security_uplift: float = Field(ge=0)
    pre_churn_window_days: int = Field(gt=0)
    pre_churn_multiplier: float = Field(ge=1)


class SyntheticConfig(BaseModel):
    random_seed: int
    payment_failure: PaymentFailureConfig
    support_tickets: SupportTicketConfig


class FeatureSetConfig(BaseModel):
    version: str


class MLflowConfig(BaseModel):
    tracking_uri: str
    experiment_name: str
    registry_model_name: str
    promotion_epsilon: float = Field(ge=0)


class RiskBandsConfig(BaseModel):
    low_max: float = Field(gt=0, lt=1)
    medium_max: float = Field(gt=0, lt=1)

    @field_validator("medium_max")
    @classmethod
    def _medium_after_low(cls, v: float, info: ValidationInfo) -> float:
        low = info.data.get("low_max")
        if low is not None and v <= low:
            raise ValueError("medium_max must be greater than low_max")
        return v


class BusinessConfig(BaseModel):
    retention_offer_cost: float = Field(ge=0)
    assumed_ltv_horizon_months: int = Field(gt=0)
    offer_success_rate: float = Field(ge=0, le=1)
    risk_bands: RiskBandsConfig


class MonitoringConfig(BaseModel):
    psi_warning_threshold: float = Field(gt=0)
    psi_critical_threshold: float = Field(gt=0)
    drift_bins: int = Field(gt=1)
    # A feature's null rate changing by at least this much (a fraction,
    # e.g. 0.1 = 10 percentage points) is flagged "significant" on its
    # own, regardless of PSI on the non-null values -- see
    # monitoring/drift.py's module docstring.
    null_rate_delta_threshold: float = Field(gt=0, le=1)

    @field_validator("psi_critical_threshold")
    @classmethod
    def _critical_after_warning(cls, v: float, info: ValidationInfo) -> float:
        warning = info.data.get("psi_warning_threshold")
        if warning is not None and v <= warning:
            raise ValueError("psi_critical_threshold must be greater than psi_warning_threshold")
        return v


class AppConfig(BaseModel):
    """Business / pipeline configuration loaded from config/config.yaml.
    Contains no secrets — see Settings for those."""

    data: DataConfig
    synthetic: SyntheticConfig
    feature_set: FeatureSetConfig
    mlflow: MLflowConfig
    business: BusinessConfig
    monitoring: MonitoringConfig


def load_app_config(path: Path | str = PROJECT_ROOT / "config" / "config.yaml") -> AppConfig:
    with Path(path).open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return AppConfig.model_validate(raw)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def get_app_config() -> AppConfig:
    return load_app_config()
