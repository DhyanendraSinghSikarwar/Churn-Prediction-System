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

from churn.config import PROJECT_ROOT

_PYTHON = sys.executable


@task(retries=1, retry_delay_seconds=30)
def run_script(*args: str) -> None:
    """Runs `python <args>` with `cwd=PROJECT_ROOT` (a Prefect worker's
    own working directory is never guaranteed to be this project's root
    -- every child script's `.env` resolution and relative paths like
    `pipelines/train.py` depend on it being exactly this) and relays the
    child's full stdout/stderr into Prefect's own run logger, not just a
    one-line "running: ..." announcement -- found by an independent
    review: without this, Prefect's "structured run history" claim
    wasn't backed by anything beyond task pass/fail and timing; the
    actual JSON log lines each script already produces went nowhere
    Prefect could show them.

    `encoding="utf-8"` is explicit and required, not cosmetic: piping a
    child's output (rather than letting it print straight to a console)
    makes Python decode the captured bytes using *this* process's
    default encoding unless told otherwise, which on Windows is cp1252 --
    MLflow's own emoji stdout print (see CLAUDE.md's Gotchas) would raise
    a `UnicodeDecodeError` here at decode time, not just Windows'
    console-print-time `UnicodeEncodeError` doing this differently.
    """
    logger = get_run_logger()
    logger.info("running: %s", " ".join([_PYTHON, *args]))
    result = subprocess.run(
        [_PYTHON, *args],
        check=False,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.stdout:
        logger.info(result.stdout)
    if result.stderr:
        logger.warning(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(args)!r} exited with code {result.returncode}")


@flow(name="churn-pipeline")
def churn_pipeline_flow(force_ingest: bool = False) -> None:
    """Runs every stage in dependency order. Each `run_script` call is a
    plain (non-`.submit()`ted) task call, which Prefect executes
    synchronously -- exactly the strict sequencing this pipeline needs
    (batch scoring depends on a registered champion, which depends on a
    built panel, which depends on ingested data). A task that raises
    halts the flow immediately -- verified live: a deliberately-failing
    task never lets a later task run, and the flow ends in a Failed
    state, not Completed.

    The training task uses `retries=0`, not this file's usual
    `retries=1`: `pipelines/train.py` registers a new model version and
    may promote it *before* it can fail later (e.g. while writing the
    model card) -- a retry after that point would register a second,
    redundant version rather than safely redoing idempotent work. Found
    by an independent review.
    """
    ingest_args = ["-m", "churn.ingestion.cli"]
    if force_ingest:
        ingest_args.append("--force")
    run_script(*ingest_args)
    run_script("-m", "churn.features.build_panel")
    run_script.with_options(retries=0)("pipelines/train.py")
    run_script("pipelines/batch_score.py")
    run_script("pipelines/backfill_predictions.py")
    run_script("pipelines/monitor.py")


if __name__ == "__main__":
    churn_pipeline_flow()
