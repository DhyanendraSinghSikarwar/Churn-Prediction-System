# CLAUDE.md

## What this is

Production-style customer churn prediction system (Postgres → idempotent
ingestion → SQL feature extraction → training → serving API → monitoring)
built on the IBM Telco Customer Churn dataset. **Synthetic-data demo** —
everything with a date on it is a seeded, documented synthetic layer, not
real telecom events; see DECISIONS.md. Currently Phase 6/8 done — see
PLAN.md for phase status, DECISIONS.md for the why behind every
non-obvious call.

## Commands

```bash
uv sync && uv run pre-commit install                    # setup
uv sync --group notebooks                                 # + jupyter/EDA-only libs (Phase 3+)
docker compose up -d postgres pgadmin minio minio-init mlflow
uv run alembic upgrade head                              # db up + migrate
uv run python -m churn.ingestion.cli                      # ingest
uv run python pipelines/train.py                          # train + register (Phase 5)
uv run python pipelines/batch_score.py                    # batch-score the current cohort (Phase 6)
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000     # serve the inference API (Phase 6)
uv run pytest                                             # full suite (needs Docker)
uv run pytest tests/unit                                  # unit only, no Docker
uv run ruff check src tests pipelines api && uv run black --check src tests pipelines api && uv run mypy src pipelines api   # lint
uv run ruff check --fix src tests pipelines api && uv run black src tests pipelines api                                     # format
```

Equivalent `make` targets exist (`make setup`, `make db-up`, `make test`, …)
but `make` isn't preinstalled on Windows — the commands above are exactly
what each target runs.

## Layout

- `sql/ddl/schema.sql` is generated from Alembic (`alembic upgrade head --sql`)
  — never hand-edit; `alembic/` is the schema source of truth.
- `sql/features/`, `sql/validation/` — hand-written parameterized SQL, not
  hidden behind an ORM query layer (deliberate, so window-function SQL
  stays reviewable — DECISIONS.md#tech-stack).
- `notebooks/` — exploration only; nothing under `src/churn` imports from it.
- `src/churn/models/` — `split.py`/`pipeline.py`/`registry.py`/`inference.py`:
  the single source of truth for the train/val/test split, the
  preprocessing+model `Pipeline` builder, MLflow registry helpers, and
  serving-side helpers (load the champion, risk-band, SHAP top-3,
  predictions upsert). `pipelines/train.py`, `pipelines/batch_score.py`,
  `api/main.py`, and any future notebook/pipeline must reuse these, never
  re-implement them inline — that duplication is exactly what caused
  Phase 4's shared-`ColumnTransformer` bug (DECISIONS.md).
- `pipelines/batch_score.py` scores the current cohort in bulk and writes
  to `predictions`; `api/main.py` serves those rows plus an on-demand
  single-customer rescore — both built on `src/churn/models/inference.py`,
  neither duplicates its logic.
- `dashboards/` — scaffolded, not yet built (Phase 7).

## Conventions

- Commits: `feat: Phase N - <description>` (Conventional-Commits-flavored,
  phase-numbered).
- Config split: `Settings` (secrets, from `.env`, via pydantic-settings,
  never committed) vs `AppConfig` (business/pipeline params, `config/*.yaml`,
  safe to commit, no secrets ever go here).
- Logging: structlog → stdlib → single-line JSON, used everywhere; no `print()`.
- Dependencies are added per-phase, not front-loaded (see the `notebooks`
  dependency group comment in `pyproject.toml`).

## Invariants

- No leakage: raw `tenure` and `TotalCharges` never enter `feature_snapshots`;
  every feature CTE filters strictly `< reference_date`, never `<=`
  (DECISIONS.md#leakage-mitigations).
- `sql/ddl/schema.sql` must stay a drift-free dump of the Alembic schema —
  regenerate, don't hand-edit.
- Model CV must use `GroupKFold(groups=customer_id)`, never plain
  `StratifiedKFold` — the panel has the same customer at multiple adjacent
  `reference_date`s, so a naive split leaks label information.
- `ruff`, `black --check`, `mypy --strict` are enforced-clean, not
  aspirational (verified as part of this bootstrap: all three pass).

## Gotchas

- `tests/integration/` spins up real containers via testcontainers — needs
  Docker running; `tests/unit/` does not.
- mypy is pinned to `python_version = "3.12"` even though `requires-python`
  floor is 3.11 — numpy's bundled stubs need 3.12 syntax to parse.
- No git remote is configured yet — this is a local-only repo so far.
- The project directory lives inside OneDrive sync. `uv sync` installs
  editable/large dep sets (e.g. `--group notebooks`) by hardlinking from
  its cache — OneDrive's cloud-file driver rejects that
  (`os error 396`) and the install fails partway through. Run with
  `UV_LINK_MODE=copy uv sync ...` (slower, but works); this also explains
  otherwise-mysterious multi-minute waits on plain `uv sync`/`pytest`.
- Notebooks executed via `jupyter nbconvert --execute` run with cwd set to
  `notebooks/`, not the project root — `Settings`' `.env` resolution is
  cwd-relative (see DECISIONS.md#nbconvert-cwd-breaks-env-file-resolution),
  so any script/notebook run from a non-root cwd needs an explicit
  `os.chdir(PROJECT_ROOT)` before touching `get_settings()`/`get_engine()`.
- `pandas>=2.2.1` currently resolves to pandas 3.0.x, whose default string
  dtype is no longer `object` — code checking `df[c].dtype == object` to
  detect string/categorical columns silently misclassifies them; use
  `pd.api.types.is_numeric_dtype(...)` instead (see
  DECISIONS.md#pandas-3-default-string-dtype-breaks-dtype-object-checks).
- On Windows, executing a notebook via `nbconvert` needs
  `asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())`
  set *before* nbconvert starts the kernel, or kernel startup hangs and
  times out (pyzmq's `add_reader` isn't supported on the default
  `ProactorEventLoop`).
- Docker Desktop does not auto-start on this machine — if `docker compose`
  or a DB connection times out, check `docker info` first and launch
  Docker Desktop if the daemon isn't up.
- `docker-compose.yml`'s `mlflow` service's `MLFLOW_S3_ENDPOINT_URL` is
  hardcoded to `http://minio:9000` — never derive it from `${MINIO_ENDPOINT}`
  (that var is host-facing, `http://localhost:9000`, for code running
  outside Docker; inside the mlflow container "localhost" means itself,
  not minio — see DECISIONS.md#mlflow-s3-endpoint-url-must-not-come-from-the-host-facing-env-var).
- MLflow prints an emoji ("🏃 View run...") on every run completion, which
  crashes a plain console script on Windows (`UnicodeEncodeError`, cp1252
  codepage) — doesn't affect notebooks (nbconvert's kernel protocol is
  UTF-8) but does affect any real script, including `pipelines/train.py`
  (which fixes it via `sys.stdout.reconfigure(encoding="utf-8", ...)` in
  `main()`, before mlflow prints anything — copy that pattern in any new
  script that touches mlflow directly).
- This project's mlflow *client* (3.x, latest at install time) is newer
  than the Docker mlflow *server* (pinned to 2.14.1). `mlflow.sklearn.log_model()`
  always fails against it (404 on mlflow 3's new "Logged Model" API,
  which 2.14.1 doesn't have) — use
  `churn.models.registry.log_pipeline_artifact()` instead
  (`save_model()` locally + `log_artifacts()` upload), and use the alias
  API (`set_registered_model_alias`/`get_model_version_by_alias`), not
  the deprecated stages API — see
  DECISIONS.md#mlflow-client-3x-vs-server-2x-model-logging-incompatibility.
- `sklearn`/`mlflow`/`shap` ship no `py.typed` marker — all three need a
  `[[tool.mypy.overrides]] ignore_missing_imports = true` entry (already
  added); a new untyped ML dependency will need the same treatment.
- `api/main.py` loads the champion model once at startup (FastAPI
  `lifespan`), not per-request — if MLflow/the registry is unreachable at
  startup, the API still comes up (`app.state.champion = None`);
  `/predictions` and `/health` keep working off Postgres alone, and only
  `POST /score/{customer_id}` 503s until a model loads. Don't "fix" this
  by making startup fail hard — that's the graceful-degradation behavior
  Phase 6 was built for.
- `fastapi`/`uvicorn`/`prometheus-client`/`httpx` were already present as
  transitive deps (via mlflow's own bundled server) before Phase 6 added
  them explicitly to `pyproject.toml` — don't assume a package importing
  cleanly means it's an intentional dependency; check `pyproject.toml`.
