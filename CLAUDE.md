# CLAUDE.md

## What this is

Production-style customer churn prediction system (Postgres → idempotent
ingestion → SQL feature extraction → training → serving API → monitoring)
built on the IBM Telco Customer Churn dataset. **Synthetic-data demo** —
everything with a date on it is a seeded, documented synthetic layer, not
real telecom events; see DECISIONS.md. Currently Phase 3/8 done — see
PLAN.md for phase status, DECISIONS.md for the why behind every
non-obvious call.

## Commands

```bash
uv sync && uv run pre-commit install                    # setup
uv sync --group notebooks                                 # + jupyter/sklearn/etc (Phase 3+)
docker compose up -d postgres pgadmin minio minio-init mlflow
uv run alembic upgrade head                              # db up + migrate
uv run python -m churn.ingestion.cli                      # ingest
uv run pytest                                             # full suite (needs Docker)
uv run pytest tests/unit                                  # unit only, no Docker
uv run ruff check src tests && uv run black --check src tests && uv run mypy src   # lint
uv run ruff check --fix src tests && uv run black src tests                        # format
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
- `pipelines/`, `api/`, `dashboards/` — scaffolded, not yet built (Phase 5–7).

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
