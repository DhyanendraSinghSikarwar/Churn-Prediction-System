"""MLflow Model Registry helpers: log a fitted sklearn Pipeline as a run
artifact, register it as a new model version, and promote it to the
"champion" alias if it clears the promotion gate.

Two deliberate departures from the obvious API, both forced by a real
version mismatch discovered while building this (see
DECISIONS.md#mlflow-client-3x-vs-server-2x-model-logging-incompatibility):

1. Uses `mlflow.sklearn.save_model()` (writes locally, no server call)
   plus `mlflow.log_artifacts()` (the long-stable artifact endpoint)
   instead of `mlflow.sklearn.log_model()`. This project's mlflow
   *client* (3.x, resolved to the latest at `uv add` time so shap's
   numpy>=2 requirement could be satisfied) is newer than the Docker
   *server* (pinned to 2.14.1 for an unrelated setuptools/pkg_resources
   reason). `log_model()` unconditionally routes through mlflow 3's new
   "Logged Model" REST API, which a 2.14.1 server doesn't have (a bare
   404) -- `save_model()` + `log_artifacts()` never touches that API.
2. Uses the alias-based registry API (`set_registered_model_alias` /
   `get_model_version_by_alias`), not the "stages" API
   (`transition_model_version_stage`) PLAN.md originally described --
   stages are deprecated in modern MLflow in favor of aliases, and both
   client and server here support aliases, so there's no reason to build
   against the deprecated surface.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import mlflow
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline

CHAMPION_ALIAS = "champion"


def log_pipeline_artifact(pipeline: Pipeline, artifact_path: str = "model") -> None:
    """Saves `pipeline` locally (MLmodel format) and uploads it as an
    artifact of the currently-active run. Must be called inside a
    `with mlflow.start_run():` block."""
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = Path(tmp) / artifact_path
        mlflow.sklearn.save_model(pipeline, path=str(model_dir))
        mlflow.log_artifacts(str(model_dir), artifact_path=artifact_path)


def current_champion_metric(
    client: MlflowClient, model_name: str, metric_name: str
) -> float | None:
    """Returns the metric value logged on the run that produced the
    model version currently aliased "champion", or None if no champion
    exists yet (bootstrap case: the very first training run)."""
    try:
        version = client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)
    except MlflowException:
        return None
    if version.run_id is None:
        raise ValueError(
            f"Champion model version {version.version} of {model_name!r} has no run_id -- "
            "cannot look up its metrics."
        )
    run = client.get_run(version.run_id)
    return run.data.metrics.get(metric_name)


def register_and_maybe_promote(
    model_name: str,
    run_id: str,
    candidate_metric: float,
    metric_name: str,
    promotion_epsilon: float,
    artifact_path: str = "model",
    client: MlflowClient | None = None,
) -> tuple[ModelVersion, bool]:
    """Registers the model artifact logged at `runs:/{run_id}/{artifact_path}`
    as a new version of `model_name`, and promotes it to the "champion"
    alias if there is no champion yet, or if `candidate_metric` beats the
    current champion's `metric_name` by at least `promotion_epsilon`.

    Returns (the new ModelVersion, whether it was promoted).
    """
    client = client or MlflowClient()
    model_uri = f"runs:/{run_id}/{artifact_path}"
    version = mlflow.register_model(model_uri, model_name)

    current = current_champion_metric(client, model_name, metric_name)
    promoted = current is None or candidate_metric >= current + promotion_epsilon
    if promoted:
        client.set_registered_model_alias(model_name, CHAMPION_ALIAS, version.version)
    return version, promoted
