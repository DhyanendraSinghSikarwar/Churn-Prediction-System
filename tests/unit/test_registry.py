from unittest.mock import MagicMock, patch

import pytest
from mlflow.exceptions import MlflowException

from churn.models.registry import current_champion_metric, register_and_maybe_promote


def _client_with_champion(metric_value: float | None) -> MagicMock:
    client = MagicMock()
    if metric_value is None:
        client.get_model_version_by_alias.side_effect = MlflowException("no champion yet")
    else:
        client.get_model_version_by_alias.return_value = MagicMock(run_id="run-champion")
        metrics = {"test_pr_auc": metric_value}
        client.get_run.return_value = MagicMock(data=MagicMock(metrics=metrics))
    return client


def test_current_champion_metric_returns_none_when_no_champion_exists() -> None:
    client = _client_with_champion(None)
    assert current_champion_metric(client, "churn_classifier", "test_pr_auc") is None


def test_current_champion_metric_returns_the_logged_value() -> None:
    client = _client_with_champion(0.42)
    assert current_champion_metric(client, "churn_classifier", "test_pr_auc") == 0.42


def test_current_champion_metric_raises_if_champion_version_has_no_run_id() -> None:
    client = MagicMock()
    client.get_model_version_by_alias.return_value = MagicMock(run_id=None)
    with pytest.raises(ValueError, match="no run_id"):
        current_champion_metric(client, "churn_classifier", "test_pr_auc")


@patch("churn.models.registry.mlflow.register_model")
def test_bootstrap_run_is_always_promoted(mock_register_model: MagicMock) -> None:
    mock_register_model.return_value = MagicMock(version="1")
    client = _client_with_champion(None)

    version, promoted = register_and_maybe_promote(
        model_name="churn_classifier",
        run_id="run-1",
        candidate_metric=0.10,
        metric_name="test_pr_auc",
        promotion_epsilon=0.005,
        client=client,
    )

    assert promoted is True
    client.set_registered_model_alias.assert_called_once_with("churn_classifier", "champion", "1")


@patch("churn.models.registry.mlflow.register_model")
def test_candidate_beating_champion_by_epsilon_is_promoted(mock_register_model: MagicMock) -> None:
    mock_register_model.return_value = MagicMock(version="2")
    client = _client_with_champion(0.100)

    version, promoted = register_and_maybe_promote(
        model_name="churn_classifier",
        run_id="run-2",
        candidate_metric=0.106,  # beats 0.100 + epsilon(0.005)
        metric_name="test_pr_auc",
        promotion_epsilon=0.005,
        client=client,
    )

    assert promoted is True
    client.set_registered_model_alias.assert_called_once_with("churn_classifier", "champion", "2")


@patch("churn.models.registry.mlflow.register_model")
def test_candidate_tying_champion_is_not_promoted(mock_register_model: MagicMock) -> None:
    """Regression case for Phase 5's second real training run: an
    identical (deterministic, seeded) rerun must not re-promote itself
    just because it matches the champion -- it must clear the epsilon
    margin, not just tie."""
    mock_register_model.return_value = MagicMock(version="2")
    client = _client_with_champion(0.100)

    version, promoted = register_and_maybe_promote(
        model_name="churn_classifier",
        run_id="run-2",
        candidate_metric=0.100,
        metric_name="test_pr_auc",
        promotion_epsilon=0.005,
        client=client,
    )

    assert promoted is False
    client.set_registered_model_alias.assert_not_called()


@patch("churn.models.registry.mlflow.register_model")
def test_candidate_below_epsilon_margin_is_not_promoted(mock_register_model: MagicMock) -> None:
    mock_register_model.return_value = MagicMock(version="2")
    client = _client_with_champion(0.100)

    version, promoted = register_and_maybe_promote(
        model_name="churn_classifier",
        run_id="run-2",
        candidate_metric=0.103,  # beats 0.100 but not by the full epsilon(0.005)
        metric_name="test_pr_auc",
        promotion_epsilon=0.005,
        client=client,
    )

    assert promoted is False
    client.set_registered_model_alias.assert_not_called()
