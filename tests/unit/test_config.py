import pytest
from pydantic import ValidationError

from churn.config import BusinessConfig, MonitoringConfig, RiskBandsConfig, get_app_config


def test_business_config_rejects_offer_success_rate_above_one() -> None:
    with pytest.raises(ValidationError):
        BusinessConfig(
            retention_offer_cost=50.0,
            assumed_ltv_horizon_months=12,
            offer_success_rate=1.5,
            risk_bands=RiskBandsConfig(low_max=0.3, medium_max=0.7),
        )


def test_risk_bands_require_medium_greater_than_low() -> None:
    with pytest.raises(ValidationError):
        RiskBandsConfig(low_max=0.7, medium_max=0.3)


def test_monitoring_config_requires_critical_greater_than_warning() -> None:
    with pytest.raises(ValidationError):
        MonitoringConfig(psi_warning_threshold=0.25, psi_critical_threshold=0.1, drift_bins=10)


def test_real_config_yaml_loads_and_validates() -> None:
    cfg = get_app_config()
    assert cfg.data.panel_end_date > cfg.data.panel_start_date
    assert cfg.data.source.expected_row_count == 7043
    assert cfg.monitoring.psi_critical_threshold > cfg.monitoring.psi_warning_threshold
