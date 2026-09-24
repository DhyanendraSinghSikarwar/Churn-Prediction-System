"""Prefect orchestration: `uv run python pipelines/flows.py` runs the
full pipeline end to end -- ingest -> build panel -> train -> batch
score -> backfill historical predictions -> monitoring checks -- as one
Prefect flow with retries and structured run history (visible via
`prefect server start` + the Prefect UI, or `prefect flow-run logs`).

Each task shells out to the exact same standalone entry point this
project's Makefile/README already document (`uv run python pipelines/train.py`,
etc.) rather than importing and re-calling their `run()` functions
in-process -- see
DECISIONS.md#prefect-tasks-shell-out-to-the-existing-cli-scripts-not-in-process-imports
for why: every pipeline stage stays independently runnable exactly as
before, and this file's only job is sequencing/retries/observability,
never a second place pipeline logic lives.
"""

from __future__ import annotations

import subprocess
import sys

from prefect import flow, task
from prefect.logging import get_run_logger

_PYTHON = sys.executable


@task(retries=1, retry_delay_seconds=30)
def run_script(*args: str) -> None:
    logger = get_run_logger()
    logger.info("running: %s", " ".join([_PYTHON, *args]))
    result = subprocess.run([_PYTHON, *args], check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(args)!r} exited with code {result.returncode}")


@flow(name="churn-pipeline")
def churn_pipeline_flow(force_ingest: bool = False) -> None:
    """Runs every stage in dependency order. Each `run_script` call is a
    plain (non-`.submit()`ted) task call, which Prefect executes
    synchronously -- exactly the strict sequencing this pipeline needs
    (batch scoring depends on a registered champion, which depends on a
    built panel, which depends on ingested data)."""
    ingest_args = ["-m", "churn.ingestion.cli"]
    if force_ingest:
        ingest_args.append("--force")
    run_script(*ingest_args)
    run_script("-m", "churn.features.build_panel")
    run_script("pipelines/train.py")
    run_script("pipelines/batch_score.py")
    run_script("pipelines/backfill_predictions.py")
    run_script("pipelines/monitor.py")


if __name__ == "__main__":
    churn_pipeline_flow()
