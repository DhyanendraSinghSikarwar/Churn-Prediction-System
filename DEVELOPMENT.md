# DEVELOPMENT.md

Step-by-step build log: every command actually run to get this repo from
an empty clone to its current state (Phases 1–5), in order, so it can be
reproduced from scratch. This is the "how to run it" doc — see
[PLAN.md](PLAN.md) for what each phase produced and the verified numbers,
[DECISIONS.md](DECISIONS.md) for why each non-obvious call was made, and
[CLAUDE.md](CLAUDE.md) for the condensed command/gotcha reference used
day to day. This file is maintained phase by phase as the project grows;
it will cover Phases 6–8 as they're built.

## Prerequisites

- Python via `uv` (no separate interpreter install needed — `uv` manages it)
- Docker Desktop (containers: postgres, pgadmin, minio, minio-init, mlflow)
- On Windows: Docker Desktop does not auto-start — launch it manually and
  wait for `docker info` to succeed before any `docker compose` step
- If the project directory lives inside OneDrive (or another cloud-sync
  folder), prefix every `uv sync`/`uv add`/`uv remove` with
  `UV_LINK_MODE=copy` — see CLAUDE.md's Gotchas for why

```bash
git clone <repo>
cd ml_proj
cp .env.example .env   # fill in real DB/MinIO passwords — never commit this file
```

---

## Phase 1 — Infra, DB schema, ingestion

```bash
uv sync && uv run pre-commit install

docker compose up -d postgres pgadmin minio minio-init mlflow
# on Windows, if this hangs/times out: launch Docker Desktop, wait for
# `docker info` to succeed, then retry

uv run alembic upgrade head          # creates schema from alembic/ migrations
uv run python -m churn.ingestion.cli # downloads IBM Telco dataset, synthesizes
                                      # dates, loads Postgres idempotently

uv run pytest tests/unit             # no Docker needed
uv run pytest                        # full suite incl. testcontainers (needs Docker)
```

Verify: 7,043 customers loaded, zero orphaned foreign keys, re-running
the ingestion CLI is a no-op (idempotent). Full numbers:
[PLAN.md#phase-1](PLAN.md).

## Phase 2 — SQL feature extraction + validation

```bash
uv run python -m churn.features.build_panel   # materializes feature_snapshots
```

This runs the parameterized queries in `sql/features/` across every
monthly `reference_date` in the configured panel window, then the
data-quality queries in `sql/validation/` against the result. No
separate CLI flag for validation — `build_panel` fails loudly if a
validation query returns a mismatch.

```bash
uv run pytest   # re-run full suite — includes the point-in-time
                 # reconciliation / leakage checks
```

Verify: 87,517-row panel, all validation queries clean, monthly churn
rate 1.27%–2.02%. Full numbers: [PLAN.md#phase-2](PLAN.md).

## Phase 3 — Notebooks: data audit, EDA, feature selection

```bash
UV_LINK_MODE=copy uv sync --group notebooks   # jupyter, nbconvert, ipykernel,
                                                # scipy, statsmodels, etc.
```

Run notebooks 01–03 in order (each depends on state/artifacts from the
previous one). Two equivalent ways to run them:

**Interactively** (recommended for a human reproducing this):
```bash
uv run jupyter lab
# open notebooks/01_data_audit.ipynb, Run All, repeat for 02, then 03
```

**Headlessly** (what CI / a script would do):
```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_audit.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_eda_statistics.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/03_feature_engineering.ipynb
```

On Windows, headless execution needs one extra fix or the kernel hangs
on startup (pyzmq's `add_reader` isn't supported on the default
`ProactorEventLoop`) — set this **before** nbconvert starts the kernel,
e.g. at the top of a wrapper script that then calls nbconvert's CLI
entrypoint:
```python
import asyncio
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
```
Also pass a longer startup timeout the first time (cold `ipykernel`
install): `--ExecutePreprocessor.startup_timeout=300`.

Every notebook's setup cell does `os.chdir(PROJECT_ROOT)` right after
importing it — required because `nbconvert` runs with cwd set to
`notebooks/`, and `Settings`' `.env` resolution is cwd-relative.

Verify: synthetic generator's statistical assumptions confirmed against
loaded data; independent leakage audit, 0 mismatches across 80,968
checks; 44 candidate features narrowed to 12, written to
`config/selected_features_v1.yaml`. Full numbers:
[PLAN.md#phase-3](PLAN.md).

## Phase 4 — Notebooks: modelling, evaluation, business impact

Depends on `config/selected_features_v1.yaml` from Phase 3. Same
execution options as Phase 3 (Jupyter Lab "Run All", or nbconvert
headless), run in order:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/04_modelling_experiments.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/05_evaluation_explainability.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/06_business_impact.ipynb
```

Verify: plain `LogisticRegression(class_weight="balanced")` beat every
`HistGradientBoostingClassifier` variant (test PR-AUC 0.1093 vs. 0.0593);
top-decile lift 4.5x; cost-based threshold analysis and A/B test sizing
in notebook 06. Full numbers: [PLAN.md#phase-4](PLAN.md).

> Note: `notebooks/` is exploration-only — nothing under `src/churn`
> imports from it. The production preprocessing+model pipeline lives in
> `src/churn/models/pipeline.py`, built in Phase 5 from what these
> notebooks found, not by importing the notebooks themselves.

## Phase 5 — Training pipeline + MLflow registry

Requires `mlflow` up (already started as part of `docker compose up -d`
above) and reachable at `http://localhost:5000`.

```bash
uv run python pipelines/train.py
```

This loads the panel, filters to the modeling window, validates it
(Pandera schema gate), splits train/val/test with `GroupKFold`-safe
time boundaries, fits `src/churn/models/pipeline.py`'s
`build_pipeline()`, evaluates on the held-out test set, logs the run to
MLflow, and registers+conditionally promotes the model via
`src/churn/models/registry.py`'s alias-based gate (`champion` alias,
promoted only if the candidate beats the current champion by
`promotion_epsilon`).

Run it twice to see the gate in action: the first run on an empty
registry always promotes (bootstraps the champion); an identical second
run should **not** promote, since it only ties rather than beats the
existing champion.

```bash
uv run pytest                                                         # 39/39 incl. Phase 5 unit tests
uv run ruff check src tests && uv run black --check src tests && uv run mypy src   # lint clean
```

Verify: first run PR-AUC 0.1093 (bootstraps champion, matches notebook
05 exactly); second identical run correctly not promoted. Full
verification: [PLAN.md#phase-5](PLAN.md).

---

## Common gotchas when reproducing this

These bite during the steps above specifically — see CLAUDE.md for the
full list:

- **OneDrive + `uv sync`**: hardlink install fails with `os error 396`.
  Always `UV_LINK_MODE=copy uv sync ...` if the repo is cloud-synced.
- **Docker Desktop not running**: `docker compose up` / any DB
  connection just times out. Check `docker info` first.
- **pandas 3.0's default string dtype**: any code checking
  `df[c].dtype == object` to detect string columns will silently
  misclassify them post-Phase-3 dependency versions — use
  `pd.api.types.is_numeric_dtype(...)` instead.
- **MLflow client (3.x) vs. server (pinned 2.14.1)**: `mlflow.sklearn.log_model()`
  404s against this server version — this repo's code already works
  around it (`registry.log_pipeline_artifact()`); don't reintroduce
  `log_model()` directly.
- **MLflow's emoji stdout print** crashes plain console scripts on
  Windows (`UnicodeEncodeError`, cp1252) — not an issue inside
  notebooks, but any new standalone script touching mlflow needs
  `sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")`
  before mlflow prints anything.
