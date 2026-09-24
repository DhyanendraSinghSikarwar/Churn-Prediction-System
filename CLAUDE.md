# CLAUDE.md

## What this is

Production-style customer churn prediction system (Postgres → idempotent
ingestion → SQL feature extraction → training → serving API → monitoring)
built on the IBM Telco Customer Churn dataset. **Synthetic-data demo** —
everything with a date on it is a seeded, documented synthetic layer, not
real telecom events; see DECISIONS.md. All 8 phases done — see PLAN.md
for phase status, DECISIONS.md for the why behind every non-obvious
call.

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
uv run python pipelines/backfill_predictions.py            # backtest every historical cohort (Phase 7)
uv run python pipelines/monitor.py                        # log drift + performance-decay checks (Phase 7)
uv run streamlit run dashboards/app.py                    # interactive monitoring dashboard (Phase 7)
uv run python dashboards/report.py                        # static monitoring report -> reports/ (Phase 7)
uv run python pipelines/flows.py                          # run the whole pipeline as one Prefect flow (Phase 7)
uv run pytest                                             # full suite (needs Docker)
uv run pytest tests/unit                                  # unit only, no Docker
uv run ruff check src tests pipelines api dashboards && uv run black --check src tests pipelines api dashboards && uv run mypy src pipelines api dashboards   # lint
uv run ruff check --fix src tests pipelines api dashboards && uv run black src tests pipelines api dashboards                                                # format
uv run pre-commit run --all-files                          # run pre-commit hooks manually (Phase 8)
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
- `src/churn/monitoring/` — `drift.py` (PSI/KS/KL between a reference and
  a comparison feature distribution) and `performance.py` (per-cohort
  classification metrics from matured `predictions` rows). Pure
  computation, no I/O — `pipelines/monitor.py` (logs findings),
  `dashboards/app.py` (interactive), and `dashboards/report.py` (static
  HTML) are the only three callers, and none of them persist results;
  see DECISIONS.md#monitoring-results-are-computed-on-demand-not-persisted.
- `pipelines/backfill_predictions.py` scores every *historical* cohort
  with the current champion and backfills `predictions.actual_churned`
  from the already-matured `feature_snapshots.label_churn_next_30d` — a
  retrospective backtest, not a live signal; see
  DECISIONS.md#backfill-predictions-scores-historical-cohorts-as-a-retrospective-backtest.
- `pipelines/flows.py` — a Prefect flow wrapping every pipeline stage as
  a task that shells out to that stage's own standalone CLI command
  (never an in-process import); see
  DECISIONS.md#prefect-tasks-shell-out-to-the-existing-cli-scripts-not-in-process-imports.
- `dashboards/` — `app.py` (Streamlit, `streamlit run dashboards/app.py`)
  and `report.py` (static HTML, writes to `reports/`, gitignored).

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
  Phase 6 was built for. `POST /score` *does* re-check the registry
  alias on every call and reloads if it moved
  (`_ensure_fresh_champion`) — a promotion made while the API is running
  is picked up without a restart; only `/health`/`/predictions` skip
  this check (deliberately, to stay cheap DB-only reads).
- `predictions.shap_top3` is a rank-ordered **list** of
  `{"feature": ..., "shap_value": ...}`, keyed by *original* feature
  name (SHAP values summed across a categorical feature's one-hot
  columns) — never a dict (JSONB doesn't preserve key order) and never
  a raw one-hot column name like `cat__contract_type_Two year` (an
  independent review found the original implementation did both wrong;
  see DECISIONS.md#shap-values-must-aggregate-to-source-feature-not-stay-one-hot-column-per-entry).
  Any new code touching `shap_top3` must go through
  `churn.models.inference.shap_top_features`, never re-derive it.
- `churn.models.inference.upsert_predictions` chunks at 2,000 rows per
  statement — Postgres caps bound parameters at 65,535/statement, and
  this table's 6 columns/row means one statement maxes out around
  10,922 rows. Don't build a new bulk-upsert path that skips this.
- `fastapi`/`uvicorn`/`prometheus-client`/`httpx` were already present as
  transitive deps (via mlflow's own bundled server) before Phase 6 added
  them explicitly to `pyproject.toml` — don't assume a package importing
  cleanly means it's an intentional dependency; check `pyproject.toml`.
- `predictions` rows come in two shapes: just-scored (`batch_score.py`/
  `api/main.py`, `actual_churned`/`label_matured_at` both NULL — label
  hasn't matured) and backfilled (`backfill_predictions.py`, both
  populated — a historical, already-known outcome). The *caller* must
  filter on `actual_churned IS NOT NULL` before passing rows to
  `cohort_performance_report()` — it's a precondition documented in that
  function's docstring, not something it enforces itself; every current
  caller does this via `churn.monitoring.loaders.load_matured_predictions`.
  Scoring the unmatured rows as if they had a label is a silent
  correctness bug, not a crash.
- `src/churn/monitoring/performance.py`'s per-cohort numbers are **not**
  directly comparable to Phase 5's pooled test-window PR-AUC (0.1093) —
  one number per calendar month vs. one number over the whole pooled
  test window are different statistics computed on different (though
  overlapping) samples. Both are legitimate; don't treat a mismatch
  between them as a bug.
- `pipelines/flows.py`'s Prefect tasks shell out via `subprocess.run`,
  not in-process imports — `pipelines/` has no `__init__.py`
  (`from pipelines.flows import run_script` doesn't work as a package
  import); run `uv run python pipelines/flows.py` directly, same as
  every other pipeline entry point.
- Streamlit's `use_container_width` is deprecated (removed after
  2025-12-31) — use `width="stretch"`/`width="content"` in any new
  `dashboards/app.py` code, not `use_container_width=True/False`.
- Testing a Streamlit app headlessly (no real browser) needs
  `streamlit.testing.v1.AppTest`, not `curl` — Streamlit only executes
  the script per-websocket-session, so hitting the server with `curl`
  proves the process is up, never that the script itself runs without
  exceptions or renders the expected elements. `AppTest.from_file(path)`
  resolves `path` relative to the *calling* script's location, not cwd —
  pass an absolute path if the caller isn't in the project root.
- `pipelines/flows.py` may print a `sqlite3.OperationalError: database
  is locked` traceback at startup (writing its `TELEMETRY_SESSION` row)
  — this is Prefect's own local ephemeral-server SQLite backend, not
  Postgres, and not this project's code; it doesn't fail the flow run
  (every task still runs, exit code 0). Harmless noise, same category as
  the mlflow emoji/cp1252 gotcha below — don't chase it as a bug.
- `.pre-commit-config.yaml`'s hooks are scoped with
  `files: ^(src|tests|pipelines|api|dashboards)/` — **never** widen that
  to a bare `types: [python]` with no path filter. Found the hard way:
  an unscoped version let `black` silently reformat two committed
  Alembic migration files on a plain `pre-commit run --all-files`
  verification pass (reverted via `git checkout --` before it was ever
  committed) — `alembic/versions/` is autogenerated and this project's
  own convention is to never hand-edit it; `notebooks/` is excluded from
  mypy for the same PEP-695-stub reason `pyproject.toml` documents.
  After adding any new top-level source directory, add it to this
  hook's `files:` pattern too, not just to CLAUDE.md's lint commands.
- No git remote is configured — `.github/workflows/ci.yml` is written
  and its commands verified locally, but has never actually run in
  GitHub Actions. Don't claim it as "passing CI"; it's untested against
  a real runner.
