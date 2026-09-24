# Building This Project From Scratch — A Complete Guide

This is a from-zero build guide for this churn prediction system: an
empty folder to a working, tested, monitored, production-style ML
system, in the same order it was actually built, with the reasoning
behind every major decision. It assumes you have **this repository open
as a reference** for full source code — this document explains the
*why*, the *design*, the *schema*, the *exact commands*, and enough code
to understand every mechanism deeply; for the complete, final
implementation of any given file, open it in the repo.

Read this once end to end before writing code, then use it phase by
phase as you build. By the end you should be able to explain, from
memory, why every major component exists and what would break if you
removed it — which is the actual goal here, not just having the code
run.

> **How this differs from DEVELOPMENT.md.** DEVELOPMENT.md in this repo
> is a terse *reproduction* log — commands to replay to get an already-
> designed system running again. This document is a *design* guide — it
> explains why the system looks the way it does, what to build in what
> order, and the trade-offs behind each choice, so you could design an
> equivalent system yourself, not just re-run someone else's commands.

---

## Table of contents

0. [The problem, and why this architecture](#0-the-problem-and-why-this-architecture)
1. [Tools and environment setup](#1-tools-and-environment-setup)
2. [Project skeleton — what to create first](#2-project-skeleton--what-to-create-first)
3. [Phase 1 — Database schema + idempotent ingestion](#3-phase-1--database-schema--idempotent-ingestion)
4. [Phase 2 — SQL feature engineering](#4-phase-2--sql-feature-engineering)
5. [Phase 3 — EDA and feature selection](#5-phase-3--eda-and-feature-selection)
6. [Phase 4 — Modeling experiments and evaluation](#6-phase-4--modeling-experiments-and-evaluation)
7. [Phase 5 — Production training pipeline + MLflow registry](#7-phase-5--production-training-pipeline--mlflow-registry)
8. [Phase 6 — Serving API + batch scoring](#8-phase-6--serving-api--batch-scoring)
9. [Phase 7 — Monitoring + dashboard](#9-phase-7--monitoring--dashboard)
10. [Phase 8 — Tests, CI, and docs](#10-phase-8--tests-ci-and-docs)
11. [Full command reference, by phase](#11-full-command-reference-by-phase)
12. [Real bugs found along the way — interview material](#12-real-bugs-found-along-the-way--interview-material)
13. [Questions you should be able to answer cold](#13-questions-you-should-be-able-to-answer-cold)

---

## 0. The problem, and why this architecture

**The task**: build a customer churn prediction system that looks like
something a real ML team would ship — not a notebook that ends at
`model.fit()`. That means: a real database, a real feature pipeline
with no data leakage, a real training/registry workflow, a real serving
layer, and real monitoring — each piece independently runnable, tested,
and documented.

**The dataset problem, immediately.** The IBM Telco Customer Churn
dataset (7,043 rows) is a single **cross-sectional snapshot** — one row
per customer, no dates, no history. A churn *prediction* system needs a
customer's state *as of some past date* and whether they churned in the
following N days — you cannot get that from a snapshot. Two honest
choices:

1. Find a real dataset with genuine event history (rare, and licensing/
   privacy issues for a portfolio project), or
2. **Synthesize a temporal layer** on top of the static snapshot —
   generate plausible `signup_date`s, monthly `billing_events`,
   `support_tickets`, and a `churn_date` for churned customers, all from
   documented, seeded statistical assumptions (e.g., "electronic-check
   customers fail payments 6pp more often") — and be completely upfront
   that it's synthetic, not real telecom behavior.

This project takes option 2. **Say this explicitly, everywhere** — in
the README, in code comments, in your own explanation of the project.
Passing off synthetic data as real would be the one genuinely bad thing
to do here; being honest about it costs nothing and is what a real
engineer would do.

**Why phases, and why review gates.** The system is built in 8 ordered
phases (infra → features → EDA → modeling → production training →
serving → monitoring → hardening), each one depending on the previous.
Two disciplines carried through every phase, worth stating up front
because they are the actual point of the exercise:

- **Verify against real infrastructure, not just written.** Every claim
  ("the panel has 87,517 rows", "the API returns 503 when the DB is
  down") was checked by actually running it against a real Postgres/
  MLflow/Docker stack, not asserted from reading the code. Several real
  bugs were only found this way (see [§12](#12-real-bugs-found-along-the-way--interview-material)).
- **No leakage, ever.** A feature computed *as of* `reference_date` must
  never use information that only existed *after* `reference_date`.
  This is the single most important invariant in the whole system and
  gets its own detailed treatment in [§4](#4-phase-2--sql-feature-engineering).

---

## 1. Tools and environment setup

Install these **before** writing any code, and understand *why* each
one is here — you'll be asked "why not just use X" in an interview.

| Tool | Why this, not the alternative |
|---|---|
| **Python via `uv`** | `uv` (Astral) manages the Python interpreter *and* the virtualenv *and* dependency resolution/locking in one fast tool, replacing pip+venv+pip-tools or Poetry. `uv sync` reads `pyproject.toml` + `uv.lock` and gives you an exactly-reproducible environment. Chosen over Poetry mainly for speed and because it doesn't need a separate Python install step. |
| **Docker Desktop** | Postgres, pgAdmin, MinIO, and MLflow all run as containers via `docker-compose.yml`. This means the *exact same* infra spins up on any machine — no "works on my machine" from a locally-installed Postgres with a different version/config. |
| **Postgres** | The system of record for everything durable: raw ingested data, the feature panel, predictions, the MLflow backend store. Chosen over SQLite for real concurrent-write behavior and because window functions (needed for the feature SQL) are a first-class Postgres feature. |
| **MLflow** (tracking server + model registry) | Tracks every training run's params/metrics/artifacts, and provides a model registry with an alias-based "which version is live" pointer (`champion`). The tracking server's backend store is a database (this project points it at a Postgres schema); its artifact store is S3-compatible object storage. |
| **MinIO** | An S3-compatible object store, run locally in Docker, so MLflow's artifact store (`mlflow.sklearn.save_model()` output — pickled pipeline, conda env, etc.) doesn't need a real AWS account. In a real deployment you'd point this at real S3 instead; the code doesn't change. |
| **Alembic** | SQLAlchemy's migration tool — the database schema is defined once, in Python (`src/churn/db/models.py`), and Alembic generates/applies versioned migration scripts from it. This means the schema has a single source of truth and a real audit trail of how it changed, instead of hand-written `ALTER TABLE` scripts you have to remember to run in order. |
| **Pandera** | Runtime schema/statistical validation for pandas DataFrames — used as a *data validation gate* right before training (fail loudly on bad data rather than silently training on it). |
| **scikit-learn** | The modeling library — a `Pipeline` + `ColumnTransformer` is the one artifact that bundles preprocessing and the model together, so "the model" is never separated from "how to preprocess for it." |
| **SHAP** | Model explainability — `LinearExplainer` for the linear champion model, giving an exact (not approximate) per-prediction feature attribution. |
| **FastAPI + Uvicorn** | The serving layer. Chosen over Flask mainly for built-in request/response validation via Pydantic and native async support; Uvicorn is the ASGI server that actually runs it. |
| **Prometheus client** | Exposes `/metrics` in the format Prometheus scrapes — request counts, latencies, a histogram of served prediction probabilities. |
| **Streamlit** | The interactive monitoring dashboard — chosen for how fast you can turn a pandas DataFrame + a chart into a shareable web UI, with no separate frontend build step. |
| **Prefect** | Orchestrates the whole pipeline (ingest → build panel → train → score → backfill → monitor) as one flow, with retries and structured run history — instead of a cron job blindly calling scripts in sequence with no visibility into what failed. |
| **pytest + testcontainers** | `pytest` for the test runner; `testcontainers` spins up a *real*, throwaway Postgres in Docker for integration tests, so SQL correctness is tested against the real database engine, not a mock. |
| **ruff, black, mypy** | Lint (ruff), format (black), and strict static type-check (mypy) — run via `pre-commit` hooks and CI. |
| **structlog** | Structured logging — every log line is single-line JSON, everywhere, no `print()`. Makes logs greppable/parseable in a way human-formatted text isn't. |

**About DVC.** This project does **not** use DVC (Data Version Control),
even though it's a very common MLOps tool and you may be asked about
it. Worth understanding why it wasn't needed here, and when it would
be:

- DVC's core job is versioning large data/model files that don't belong
  in git, with pointers stored in git.
- Here, the raw dataset is small (7,043 rows, ~1MB) and re-downloaded
  idempotently by the ingestion script (with a SHA-256 idempotency
  check) rather than committed at all.
- Model *artifacts* are versioned by MLflow's model registry (every
  training run gets a new version; the artifact lives in MinIO/S3, not
  git) — this is DVC's job, done by MLflow instead.
- The *feature panel* lives in Postgres, queryable and versioned by
  `feature_set_version`, not as flat files.
- **When you'd actually reach for DVC**: large training datasets (GBs+)
  that don't fit in a database and need to be versioned alongside code
  (e.g., image/audio datasets), or a team that wants `git checkout` to
  also check out the matching data snapshot. Know this distinction — it
  shows you understand DVC's actual job rather than using it by reflex.

### Environment setup, in order

```bash
# 1. Install uv (see https://docs.astral.sh/uv/ for your OS)
# 2. Install Docker Desktop, make sure it's running (`docker info` should succeed)
# 3. Clone/create the project directory, then:

cd your-project-dir
git init
uv init --name churn-prediction --python 3.12   # creates pyproject.toml

# 4. Create .env.example (never commit the real .env)
```

`.env.example` — every secret/environment-specific value your code
needs, with placeholder values:

```bash
# --- Postgres (app database) ---
DB_HOST=localhost
DB_PORT=5432
DB_NAME=churn
DB_USER=churn_app
DB_PASSWORD=changeme

# --- pgAdmin ---
PGADMIN_EMAIL=admin@example.com
PGADMIN_PASSWORD=changeme

# --- MLflow backend store (separate database on the same Postgres instance) ---
MLFLOW_BACKEND_DB_NAME=mlflow
MLFLOW_BACKEND_DB_USER=mlflow_app
MLFLOW_BACKEND_DB_PASSWORD=changeme
MLFLOW_TRACKING_URI=http://localhost:5000

# --- MinIO (S3-compatible artifact store for MLflow) ---
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=changeme
MINIO_ENDPOINT=http://localhost:9000
MLFLOW_ARTIFACT_BUCKET=mlflow-artifacts

# --- App ---
APP_ENV=dev
LOG_LEVEL=INFO
RANDOM_SEED=42
```

```bash
cp .env.example .env   # then fill in real (even if just "dev-only") passwords
```

**Why two config systems (`.env` + a YAML file), not one.** This
project deliberately splits config into two kinds, and you should be
able to explain why:

- **Secrets / environment-specific values** (DB password, MinIO keys,
  which environment you're in) → `.env`, loaded via `pydantic-settings`,
  **never committed** (`.gitignore`'d).
- **Business / pipeline parameters** (train/val/test split sizes, risk
  band thresholds, synthetic-data generation rates, drift thresholds) →
  `config/config.yaml`, loaded via plain `pyyaml` + Pydantic validation,
  **safe to commit** — no secret ever goes here, by convention, checked
  by not putting a `db_password` field anywhere near this model.

Mixing these (e.g., a password in a committed YAML file) is a real,
common security mistake — keeping them in two physically separate
Pydantic models (`Settings` vs `AppConfig`) makes it structurally
awkward to make that mistake by accident.

---

## 2. Project skeleton — what to create first

Create this folder structure **before** writing meaningful code in any
of them — deciding the shape up front avoids a lot of "where does this
file even go" churn later:

```
.
├── .env.example              # committed template
├── .env                       # real secrets, gitignored
├── .gitignore
├── pyproject.toml             # dependencies, tool config (ruff/black/mypy/pytest)
├── docker-compose.yml         # postgres, pgadmin, minio, minio-init, mlflow
├── Makefile                   # thin wrappers around the real commands
├── README.md                  # what this is, quickstart, status
├── PLAN.md                    # phase-by-phase build plan + "done" status + verify numbers
├── DECISIONS.md               # decision log: situation → decision → rejected alternative → why
├── DEVELOPMENT.md             # step-by-step reproduction commands (optional but recommended)
├── config/
│   ├── config.yaml            # business/pipeline params (committed, no secrets)
│   ├── logging.yaml           # structlog/stdlib logging config
│   └── model_params.yaml      # hyperparameter search spaces, etc.
├── sql/
│   ├── ddl/                   # schema.sql, generated from Alembic — never hand-edit
│   ├── features/               # hand-written parameterized feature-extraction SQL
│   └── validation/             # data-quality check queries
├── alembic/
│   ├── env.py
│   └── versions/               # one file per migration, autogenerated + hand-reviewed
├── src/
│   └── churn/                  # the actual installable package
│       ├── config.py            # Settings (secrets) + AppConfig (business params)
│       ├── db/                  # engine, session, ORM models
│       ├── ingestion/           # download, synthesize, load
│       ├── features/            # extract.py, build_panel.py, validation.py
│       ├── models/              # split.py, pipeline.py, registry.py, inference.py
│       ├── monitoring/          # drift.py, performance.py, loaders.py
│       └── utils/               # hashing.py, logging.py
├── notebooks/                   # exploration only — nothing under src/ imports from here
├── pipelines/                   # entry-point scripts: train.py, batch_score.py, etc.
├── api/                         # FastAPI serving layer
├── dashboards/                  # Streamlit app + static report generator
├── tests/
│   ├── unit/                    # no Docker needed
│   └── integration/             # testcontainers — needs Docker
└── .github/workflows/ci.yml     # lint + test on push/PR
```

**Why this shape, specifically:**

- **`src/churn/` is the only importable package.** Everything reusable
  lives here. `pipelines/`, `api/`, `dashboards/` are thin *entry
  points* that import from `src/churn/` and add a CLI/HTTP/UI wrapper —
  they should contain as little real logic as possible. This is the
  single most important structural decision in the whole project:
  **shared logic lives in exactly one place**, imported everywhere it's
  needed, never copy-pasted between a notebook and a script and an API
  handler. (A real bug — a shared-preprocessor bug — happened *because*
  this rule was violated once in an early notebook; see [§12](#12-real-bugs-found-along-the-way--interview-material).)
- **`sql/ddl/` is generated, not hand-written.** The schema's single
  source of truth is `src/churn/db/models.py` (SQLAlchemy ORM classes);
  Alembic generates migrations from it, and `sql/ddl/schema.sql` is a
  dumped, read-only mirror for quick reference. Never hand-edit it.
- **`notebooks/` never gets imported from.** Notebooks are for
  exploration and producing evidence (a chart, a statistic, a selected-
  features list written to a YAML file) — the actual reusable logic
  they arrive at gets *rewritten* into `src/churn/` once it's proven
  out, not imported directly. This keeps the production code path free
  of notebook-specific hacks (global variables, `%matplotlib inline`,
  etc.).
- **Two-tier tests.** `tests/unit/` never touches Docker — fast, run
  constantly. `tests/integration/` spins up real containers via
  `testcontainers` for anything that needs to prove out against a real
  Postgres (SQL correctness, upsert conflict behavior). Keep the split
  strict; don't let a "unit" test secretly need Docker.

`pyproject.toml` skeleton (fill in dependencies phase by phase — see
"per-phase dependencies" note below):

```toml
[project]
name = "churn-prediction"
version = "0.1.0"
requires-python = ">=3.11,<3.13"
dependencies = []

[dependency-groups]
dev = ["pytest", "pytest-cov", "ruff", "black", "mypy", "pre-commit"]
notebooks = []   # jupyter/EDA-only libs, added in Phase 3

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/churn"]

[tool.ruff]
line-length = 100
[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM", "N"]

[tool.black]
line-length = 100

[tool.mypy]
strict = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --cov=src/churn --cov-report=term-missing"
```

**A real, non-obvious discipline worth adopting from day one: add
dependencies *per phase*, not all up front.** Don't `uv add` scikit-
learn/mlflow/shap/prefect/streamlit before you're actually about to
write the code that imports them. This keeps `pyproject.toml` an
honest, minimal record of what the code *actually needs right now* —
and it's a good interview answer too ("I add dependencies when the code
that needs them lands, not speculatively").

**Start `PLAN.md` and `DECISIONS.md` now, before Phase 1 code.**
`PLAN.md` is your phase-by-phase task list with a "done, verified with
these numbers" status per phase, updated as you go. `DECISIONS.md` is a
running log: every time you make a non-obvious call (why Postgres not
SQLite, why aliases not the deprecated MLflow "stages" API, why one
model type over another), write one entry: **situation → decision →
rejected alternative → why**. This is not busywork — it is *exactly*
the artifact that lets you explain, months later or in an interview,
*why* the system looks the way it does, not just *that* it works. Do
this discipline for real and you will genuinely never need to guess
"wait, why did I do it this way" again.

```bash
docker compose up -d postgres pgadmin minio minio-init mlflow
uv sync
uv run pre-commit install   # once .pre-commit-config.yaml exists (Phase 8, or set up early)
```

---

## 3. Phase 1 — Database schema + idempotent ingestion

### Design the schema first

Before writing any Python, design the ER schema on paper (or in
`src/churn/db/models.py` directly as SQLAlchemy classes — same thing).
The tables, and why each exists:

| Table | Purpose |
|---|---|
| `customers` | One row per customer — static attributes only (`gender`, `senior_citizen`, `partner`, `dependents`, derived `signup_date`). |
| `subscriptions` | Contract/service attributes, with a `start_date`/`end_date` so a customer's plan can change over time (even though this dataset only has one plan per customer — the schema supports change). |
| `billing_events` | Synthetic monthly charge events — the raw material for recency/frequency/monetary features. |
| `support_tickets` | Synthetic support interactions — feeds ticket-velocity features (a classic churn signal: spike in complaints before leaving). |
| `churn_labels` | Ground truth: did this customer churn, and if so, on what date. `censored_as_of` records the point up to which "not churned" is actually known, not assumed forever. |
| `feature_snapshots` | The **materialized monthly panel** — one row per (customer, reference_date), computed strictly from data available *before* reference_date. This is what training reads. |
| `predictions` | Every model-scored prediction — `probability`, `risk_band`, `shap_top3`, and (once matured) `actual_churned`/`label_matured_at` for monitoring. |
| `ingestion_runs` | An idempotency ledger — one row per successfully-loaded source file (by SHA-256), so re-running ingestion is a safe no-op. |

Write these as SQLAlchemy `DeclarativeBase` subclasses in
`src/churn/db/models.py`, with real constraints (`CheckConstraint` for
things like `risk_band IN ('low','medium','high')`, `ForeignKeyConstraint`,
`UniqueConstraint` on natural keys like `(customer_id, reference_date,
feature_set_version)`). Constraints in the schema, not just in
application code, are what actually prevent bad data — application code
has bugs; a `CHECK` constraint doesn't care.

### Set up Alembic

```bash
uv add sqlalchemy "psycopg[binary]" alembic pydantic pydantic-settings pyyaml structlog

uv run alembic init alembic
# edit alembic/env.py to import your Base/models and read the DB URL from Settings
uv run alembic revision --autogenerate -m "initial schema"
# review the generated migration by hand -- autogenerate reliably misses
# CHECK constraints; add any it missed manually
uv run alembic upgrade head
uv run alembic upgrade head --sql > sql/ddl/schema.sql   # the read-only mirror
```

### Write the ingestion pipeline

`src/churn/ingestion/download.py` — download the CSV with a primary +
fallback URL, cache locally, return `(path, sha256)`:

```python
def download_dataset(data_config: DataConfig) -> tuple[Path, str]:
    target = PROJECT_ROOT / data_config.source.raw_cache_path
    if target.exists():
        return target, sha256_of_file(target)
    for url in (data_config.source.primary_url, data_config.source.fallback_url):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
        except requests.RequestException:
            continue
        target.write_bytes(response.content)
        return target, sha256_of_file(target)
    raise DownloadError("both URLs failed")
```

`src/churn/ingestion/synthesize.py` — this is where the temporal layer
gets generated. The key discipline: **every synthetic assumption is a
named, seeded parameter in `config.yaml`**, not a magic number buried
in code:

```yaml
synthetic:
  random_seed: 42
  payment_failure:
    base_rate: 0.02
    electronic_check_uplift: 0.06     # electronic-check customers fail 6pp more often
    month_to_month_uplift: 0.03
  support_tickets:
    base_monthly_rate: 0.15
    pre_churn_window_days: 60
    pre_churn_multiplier: 2.5          # ticket rate 2.5x higher in the 60 days before churning
```

`src/churn/ingestion/cli.py` — the orchestration, with two independent
idempotency layers worth understanding:

1. **Filesystem**: the raw CSV is cached; don't re-download if present.
2. **Database**: `ingestion_runs` records the SHA-256 of every file
   that's been successfully loaded — re-running with the *same* file is
   a no-op (`--force` overrides this).

```python
def run(force: bool = False) -> None:
    cfg = get_app_config()
    path, sha256 = download_dataset(cfg.data)
    if not force and _already_loaded(sha256):
        return
    raw = pd.read_csv(path)
    _validate_row_count(raw, cfg.data.source.expected_row_count)  # refuse a changed source file
    tables = synthesize_all(raw, cfg, seed=cfg.synthetic.random_seed)
    with engine.begin() as conn:              # one transaction: all tables load or none do
        load_all(conn, tables)
        record_ingestion_run(conn, sha256)
```

The bulk load itself (`src/churn/ingestion/load.py`) should use
Postgres's `COPY` protocol, not row-by-row `INSERT` — orders of
magnitude faster for ~7,000+ rows, and worth knowing the reason if
asked ("`INSERT` pays a round-trip per row; `COPY` streams the whole
table in one pass").

```bash
uv run python -m churn.ingestion.cli
uv run python -m churn.ingestion.cli --force   # reload even if already loaded
```

**Verify for real**: query the database directly and check the
numbers, don't just trust that the script exited 0.

```bash
docker compose exec postgres psql -U churn_app -d churn -c "SELECT count(*) FROM customers;"
```

---

## 4. Phase 2 — SQL feature engineering

This phase is where the single most important invariant in the entire
project lives, and you should be able to explain it precisely:

### The leakage rule

> A feature computed for `reference_date` must only use data that
> existed **strictly before** `reference_date` — `< reference_date`,
> **never** `<= reference_date`.

Why `<`, not `<=`: if a customer paid a bill *on* `reference_date`
itself, that's the *same day* you're supposedly standing on to make a
prediction — in a real system, that event might not even be recorded
yet at the moment you'd actually run scoring. Using `<=` would let the
model "see the future" relative to its own prediction time, inflating
offline metrics in a way that would never hold up in production. This
is the textbook definition of **data leakage**, and it's the one class
of bug in an ML system that silently produces a model that looks great
in testing and fails in production, so treat it as sacred.

Write this filter into **every** feature-extraction CTE, and write an
adversarial test that specifically tries to catch a violation of it
(e.g., insert a billing event *on* `reference_date`, assert it's
excluded from a `frequency_charges_30d` feature computed for that same
`reference_date`).

### The feature panel

`sql/features/build_feature_snapshot.sql` — one parameterized query,
taking `:reference_date`, `:horizon_days`, `:feature_set_version`,
computing every feature for every customer *as of* that date in one
pass, using window functions:

```sql
-- recency: days since last billing event before reference_date
, recency_days_since_last_charge AS (
    SELECT customer_id,
           :reference_date - MAX(event_date) AS recency_days_since_last_charge
    FROM billing_events
    WHERE event_date < :reference_date        -- <-- the leakage rule, applied
    GROUP BY customer_id
)
-- frequency: count of tickets in the trailing 30/60/90 days before reference_date
, frequency_tickets_30d AS (
    SELECT customer_id, COUNT(*) AS frequency_tickets_30d
    FROM support_tickets
    WHERE created_date < :reference_date
      AND created_date >= :reference_date - INTERVAL '30 days'
    GROUP BY customer_id
)
```

The **label** looks *forward* from `reference_date` — this is expected
and correct for a supervised label, not a leak, as long as it never
feeds a feature column:

```sql
, label AS (
    SELECT customer_id,
           CASE WHEN churn_date IS NOT NULL
                 AND churn_date >= :reference_date
                 AND churn_date < :reference_date + (:horizon_days || ' days')::interval
                THEN true ELSE false END AS label_churn_next_30d
    FROM churn_labels
)
```

**Why hand-written SQL, not an ORM query layer.** Window-function-heavy
point-in-time logic like this is genuinely easier to read, review, and
get right as SQL than as a chain of ORM method calls — an ORM query
builder tends to obscure exactly the thing you most need to see clearly
here (which date bound applies to which join). This is a deliberate,
defensible choice worth being able to justify if asked "why not
SQLAlchemy Core/ORM for this."

`src/churn/features/build_panel.py` — runs the query once per monthly
`reference_date` in the configured window, with the same
already-built/`--force` idempotency pattern as ingestion, bulk-loading
via `COPY` again:

```bash
uv run python -m churn.features.build_panel
uv run python -m churn.features.build_panel --force
```

### Validate

`sql/validation/` — hand-written data-quality queries: no orphaned
foreign keys, no negative counts, no future-dated events, positive
class rate within an expected range. Run these as part of
`build_panel.py` itself (fail loudly on a bad panel, don't materialize
one you haven't checked) and as their own integration tests.

**A genuinely good extra step, worth doing even though it's not
required**: an *adversarial* point-in-time reconciliation check — for
a sample of (customer, reference_date) pairs, recompute a feature by
hand from raw events and assert it matches the SQL's output exactly.
This is the check that would catch an off-by-one in your leakage logic
that a simpler "does the query run" test wouldn't.

---

## 5. Phase 3 — EDA and feature selection

```bash
uv add --group notebooks jupyter nbconvert ipykernel scipy statsmodels \
  matplotlib seaborn imbalanced-learn optuna
uv run jupyter lab
```

Build these notebooks in order, each producing an artifact the next
phase needs:

1. **Data audit** — load the panel, check dtypes, nulls, ranges;
   confirm the synthetic generator's *statistical* assumptions actually
   landed in the data as configured (e.g., "electronic-check customers
   really do show a higher payment-failure rate") — don't just trust
   the generator code, measure the output.
2. **EDA / leakage audit** — an independent leakage check from a
   different angle than the SQL validation queries (e.g., a
   tenure-trajectory check: does a customer's `tenure_months` feature
   increase monotonically month over month the way it should?).
3. **Feature selection** — from all engineered candidates, narrow to a
   final set using **more than one method** and requiring agreement:
   mutual information, permutation importance, a Boruta-style procedure
   (compare each real feature against random "shadow" features),
   correlation pruning (drop the lower-MI half of any highly correlated
   pair). Write the final list, with the methodology, to
   `config/selected_features_v1.yaml` — this file is what Phase 5's
   training pipeline reads, so notebook exploration and production
   training can never silently disagree on what "the features" are.

**A real structural bug to know about and avoid**: if two model
pipelines in the same notebook are built from *the same* `ColumnTransformer`
instance (e.g., `preprocessor = ColumnTransformer(...)` defined once,
reused across `pipeline_a = Pipeline([("prep", preprocessor), ...])`
and `pipeline_b = Pipeline([("prep", preprocessor), ...])`), fitting the
second pipeline **silently corrupts the first's already-fitted state**
— they share the same mutable object. Always build a **fresh**
preprocessor per pipeline, via a factory function:

```python
def build_pipeline(cat_features, num_features, random_state) -> Pipeline:
    """A brand-new Pipeline with its own, never-shared ColumnTransformer,
    every call. Never pass in a shared preprocessor instance."""
    preprocessor = ColumnTransformer([...])
    return Pipeline([("prep", preprocessor), ("clf", ...)])
```

This exact bug happened once in this project (see [§12](#12-real-bugs-found-along-the-way--interview-material)) and corrupted a
threshold-tuning result — it's one of the best "tell me about a bug you
found and fixed" interview stories available here, because the *fix*
(a factory function, never a shared mutable object) generalizes far
beyond this one case.

---

## 6. Phase 4 — Modeling experiments and evaluation

### Why PR-AUC, not ROC-AUC, as the primary metric

With severe class imbalance (this panel: ~1.5–2% monthly churn rate),
ROC-AUC can look deceptively good because it's dominated by the
(easy-to-get-right) true-negative rate on the huge majority class.
**PR-AUC (average precision)** focuses on the trade-off that actually
matters for a retention campaign — of the customers you flag, how many
really churn, and of the customers who really churn, how many did you
catch. Report both, but make PR-AUC the number you optimize and compare
runs on.

### Why `GroupKFold`, never plain K-fold

The panel has the **same customer** appearing at multiple, adjacent
monthly `reference_date`s. A naive random/stratified K-fold split can
put "the same customer in March" in the train fold and "the same
customer in April" in the test fold — since a customer's features
barely change month to month, the model effectively gets to see a
near-duplicate of its own test example during training. That's a subtle
form of leakage specific to panel data, and `GroupKFold(groups=customer_id)`
is the fix — it guarantees every fold gets *entirely disjoint*
customers.

```python
from sklearn.model_selection import GroupKFold
cv = GroupKFold(n_splits=5)
for train_idx, val_idx in cv.split(X, y, groups=customer_id):
    ...
```

### Try more than one model family, honestly

Baseline `LogisticRegression(class_weight="balanced")`, then tuned GBMs
(`HistGradientBoostingClassifier` — faster to tune than the classic
`GradientBoostingClassifier`) with imbalance handling compared head to
head: plain, class-weighted, and SMOTE-resampled (`imblearn.pipeline.Pipeline`,
so resampling only ever happens during `.fit()`, never `.predict()`).
Use Optuna (`TPESampler`) for hyperparameter search, nested inside
`GroupKFold` on the training window only — never let the tuning process
see the test set.

**Report whichever model actually wins, even if it's the "boring" one.**
In this project the plain logistic regression beat every tuned GBM
variant on the held-out test PR-AUC. Reporting that honestly (instead
of quietly picking the GBM because it "should" be better) is exactly
the kind of judgment that's worth being able to describe in an
interview — it demonstrates you let the evidence decide, not a prior.

### Evaluation and business impact

- **Calibration**: reliability diagram + Brier score — does "70%
  predicted probability" actually mean ~70% of those customers churned?
  (Spoiler, with `class_weight="balanced"`: no — worth understanding
  *why not*, and saying so honestly rather than glossing over it.)
- **SHAP explainability**: `LinearExplainer` for the linear model
  (exact, not approximate, for a linear model), `TreeExplainer` for the
  GBM. Build a "top-3 features per prediction" extraction here as a
  prototype — Phase 6 will productionize it.
- **Cost-based decision threshold**: from real business economics
  (retention-offer cost, assumed customer LTV, offer success rate — all
  named, documented parameters in `config.yaml`, not hardcoded), compute
  the threshold that actually maximizes expected value, and compare it
  to whatever round-number threshold (e.g., 0.5) you might have
  defaulted to. Expect this to be an uncomfortable, informative number —
  in this project, the actual optimal threshold turned out to be very
  high (~0.94), which only flags a small fraction of customers a month.
  Report that, don't round it away.
- **A/B test sample-size / power analysis**: given the flagging rate at
  your chosen threshold, how long would a properly powered randomized
  trial take to accumulate enough volume to validate the model's real
  causal impact? (`statsmodels.stats.power.NormalIndPower`.) This is
  the kind of "how would you actually validate this in production"
  question interviewers love, and having a real number ready is much
  stronger than a hand-wave.

---

## 7. Phase 5 — Production training pipeline + MLflow registry

```bash
uv add scikit-learn mlflow
docker compose up -d mlflow   # if not already up
```

### Pull the notebook logic into `src/churn/models/`

This is the moment the "shared logic lives in exactly one place" rule
from §2 gets enforced for real. Three modules, each a single source of
truth:

- **`split.py`** — `modeling_panel()` (drop the structurally-censored
  current cohort), `time_based_split()` (train/val/test by calendar
  month), `trainval_split()`, `dataset_split_label()`. Every notebook
  and every pipeline script imports these — never hand-copies the
  boundary math.
- **`pipeline.py`** — `build_pipeline()` (the factory function from
  §5 — this is where that lesson gets encoded structurally, so the
  shared-preprocessor bug *cannot* recur), `categorical_and_numeric_features()`.
- **`registry.py`** — MLflow registry helpers (below).

### Data validation gate

Before any split or fit, validate the modeling panel with Pandera —
fail loudly, never silently train on bad data:

```python
import pandera as pa

schema = pa.DataFrameSchema({
    "label_churn_next_30d": pa.Column(bool, nullable=False),
    "tenure_months": pa.Column(int, pa.Check.ge(0)),
    # ...
})

def validate_training_panel(panel: pd.DataFrame) -> pd.DataFrame:
    return schema.validate(panel)   # raises SchemaError on violation
```

### Model promotion gate — the actual production-safety mechanism

Train, evaluate on the held-out test set, then only promote to the
registry's `champion` alias if the new candidate **beats the current
champion by at least `promotion_epsilon`** (a config value, e.g.
`0.005`) — not merely ties or barely edges it out, which would just be
registry-flapping on noise:

```python
def register_and_maybe_promote(model_name, run_id, candidate_metric, metric_name, promotion_epsilon, client):
    version = mlflow.register_model(f"runs:/{run_id}/model", model_name)
    current = current_champion_metric(client, model_name, metric_name)
    promoted = current is None or candidate_metric >= current + promotion_epsilon
    if promoted:
        client.set_registered_model_alias(model_name, "champion", version.version)
    return version, promoted
```

Use MLflow's **alias** API (`set_registered_model_alias`/
`get_model_version_by_alias`), not the deprecated "stages" API
(`transition_model_version_stage`) — aliases are the current, supported
way to say "this version is the one currently serving."

**Verify the gate for real, not just by reading the code**: run
training **twice**. The first run (no champion yet) should always
promote (bootstrap case). The second, *identical, seeded* run should
produce the same metric and correctly **not** promote, since it only
ties rather than beats the existing champion by epsilon. This is the
single most valuable verification step in this whole phase — it's easy
to write a promotion-gate function that *looks* right and passes a unit
test with a mocked client, while still being wrong against the real
registry.

```bash
uv run python pipelines/train.py
uv run python pipelines/train.py   # run again -- should NOT promote
```

A real, non-obvious wrinkle worth knowing: if your MLflow **client**
library version is newer than your MLflow **server** version (easy to
happen if the server is pinned in Docker for an unrelated reason and
the client resolves to "latest" via `uv add`), `mlflow.sklearn.log_model()`
can 404 against an older server's API. The workaround —
`mlflow.sklearn.save_model()` (writes locally, no server call) +
`mlflow.log_artifacts()` (the long-stable artifact upload endpoint) —
is worth understanding and having ready as an answer to "tell me about
a real infra incompatibility you debugged."

---

## 8. Phase 6 — Serving API + batch scoring

```bash
uv add fastapi "uvicorn[standard]" prometheus-client
uv add --group dev httpx   # for FastAPI's TestClient
```

### Design decision: precomputed serving + on-demand hybrid, not either alone

Because features come from panel-shaped, window-function SQL (computed
for a whole monthly cohort at once, not efficiently per-customer), the
right design is:

- **`pipelines/batch_score.py`** — scores the *entire* current cohort
  in bulk, writes to `predictions`. This is the default, fast serving
  path.
- **`GET /predictions/{customer_id}`** — reads whatever `batch_score.py`
  already computed. A single indexed DB read, no model in the request
  path at all.
- **`POST /score/{customer_id}`** — on-demand rescore, but only against
  the customer's *already-materialized* current-cohort feature row
  (never a live rerun of the window-function extraction query — that
  would duplicate the SQL feature logic in a second, per-customer form).

Both the batch script and the API call the exact same shared
`src/churn/models/inference.py` functions — `load_champion()`,
`risk_band()`, `shap_top_features()`, `upsert_predictions()` — never
duplicate scoring logic between them.

### SHAP top-3, done correctly

Productionizing Phase 4's SHAP prototype has one real trap: a
categorical feature becomes several one-hot columns after
preprocessing, and if you report "top-3 SHAP" by the *transformed*
column names, a customer's explanation can include one-hot columns for
categories they don't even have (e.g., a "Month-to-month" customer's
top reasons naming both "Two year" and "Month-to-month" contract-type
columns) — mathematically correct per-column, but nonsensical to a
human reader. **Aggregate SHAP values back to the original feature**
(sum across a feature's one-hot columns — exact for a linear model, not
an approximation) before ranking:

```python
def shap_top_features(champion, features, top_n=3):
    prep, clf = champion.pipeline.named_steps["prep"], champion.pipeline.named_steps["clf"]
    source_features = _transformed_name_to_source_feature(prep)  # e.g. cat__contract_type_X -> "contract_type"
    explainer = shap.LinearExplainer(clf, prep.transform(champion.background))
    shap_values = explainer(prep.transform(features)).values
    results = []
    for row in shap_values:
        aggregated: dict[str, float] = {}
        for source, value in zip(source_features, row, strict=True):
            aggregated[source] = aggregated.get(source, 0.0) + float(value)
        ranked = sorted(aggregated.items(), key=lambda kv: -abs(kv[1]))[:top_n]
        results.append([{"feature": k, "shap_value": round(v, 4)} for k, v in ranked])
    return results
```

Also store this as a **rank-ordered list**, not a dict — Postgres JSONB
does not preserve key insertion order (it sorts by key length), so a
dict silently loses the "top-3, in order" ranking on every round-trip
through the database.

### Graceful degradation and request-ID correlation

- Load the champion model **once at startup** (FastAPI `lifespan`), not
  per request. If the registry is unreachable at startup, the API
  should still come up — `/health` and `/predictions` (DB-only) keep
  working, and only `/score` (needs the model) returns 503 until one
  loads.
- **`/health`** checks DB connectivity for real (`SELECT 1`) with a
  short, explicit connect timeout on the engine — without one, a
  stopped/restarting database can leave a health check *hanging* for
  the OS's default TCP timeout (tens of seconds) instead of failing
  fast, which defeats the entire point of a liveness probe. This is
  worth testing for real: stop the database mid-request, confirm you
  get a fast `503`, not a hang.
- **Request-ID middleware**: generate/propagate an `X-Request-ID`
  header, bind it into your structured-logging context for the
  duration of the request, so every log line from one request can be
  grepped together.
- **Prometheus `/metrics`**: request counts and latency histograms by
  method + route *template* (e.g. `/predictions/{customer_id}`), never
  by the raw substituted path — labeling by the literal request path
  lets anyone generate unbounded metric cardinality just by hitting
  made-up URLs.

```bash
uv run python pipelines/batch_score.py
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000
curl localhost:8000/health
curl localhost:8000/predictions/<customer_id>
curl -X POST localhost:8000/score/<customer_id>
curl localhost:8000/metrics
```

---

## 9. Phase 7 — Monitoring + dashboard

```bash
uv add scipy prefect streamlit
```

### Feature drift: PSI, KS, KL divergence

Three complementary metrics comparing a **reference** distribution
(the training window) against a **comparison** distribution (the
current scoring cohort):

- **PSI (Population Stability Index)**: bin both distributions onto the
  same grid, `sum((cur% - ref%) * ln(cur% / ref%))`. Conventional
  reading: `<0.1` stable, `0.1–0.25` moderate, `>0.25` significant.
- **KS statistic** (numeric features only): the two-sample
  Kolmogorov-Smirnov test — a small p-value rejects "these came from the
  same distribution."
- **KL divergence**: how many extra nats it costs to encode the current
  distribution using a code built for the reference distribution.

**A real, non-obvious implementation trap**: naive quantile binning
(picking bin edges at the reference distribution's quantiles) silently
breaks on a **low-cardinality, skewed** numeric feature — e.g., a
count feature where 90%+ of values are the same small integer. Most of
the quantile points land on the same value, collapsing to just 1–2
bins, with nearly all the mass of *both* distributions in one giant
catch-all bucket — PSI reads ~0 regardless of how much the distribution
actually shifted within that bucket. **Fix**: fall back to exact
per-value bins whenever a numeric feature has few enough distinct
values (same binning categorical features already use):

```python
def _numeric_bin_proportions(reference, current, bins):
    if len(reference.unique()) <= bins:
        return _category_proportions(reference, current)   # exact per-value bins
    # ... quantile binning for genuinely continuous features
```

This is a fantastic "describe a subtle bug you found" interview story:
the code *ran*, produced a plausible-looking number, and was simply
*wrong* — the kind of bug that only surfaces when you deliberately
probe an edge case (a real, skewed feature), not from "does it run
without error."

**Also track null-rate separately.** Every PSI/KS/KL computation drops
nulls before comparing — meaning a feature that suddenly starts arriving
90% NULL would otherwise look "stable" on whatever non-null values
remain. Compute and flag a null-rate delta as its own significant-drift
signal, independent of PSI.

**Compare against train+val, not train-only.** Whatever window the
model was actually *fit* on (`pipelines/train.py`'s
`trainval_df = pd.concat([train_df, val_df])`) is the correct reference
distribution — comparing against a narrower slice both understates the
reference sample and, for a feature that drifts gradually even within
the training window (e.g., tenure, which mechanically rises as the same
customers age), can make a mundane trend look like a bigger "finding"
than it is.

### Delayed-label performance monitoring — the backtest

`predictions` only holds the *current* cohort until you also score
every **historical** cohort with the current champion and backfill its
already-known outcome (`feature_snapshots.label_churn_next_30d`, which
matured long ago for anything but the current month):

```python
# pipelines/backfill_predictions.py
panel = modeling_panel(load_panel(cfg), cfg)   # every historical cohort
probabilities = champion.pipeline.predict_proba(panel[selected_features])[:, 1]
# upsert rows with actual_churned already populated from the historical label
```

This is a **retrospective backtest**, not a live signal — most of these
cohorts were part of the champion's own training data — so **label
every cohort honestly** as `train`/`validation`/`test`/`post-test`
(matching the exact boundaries `time_based_split()` used) and add an
explicit `in_sample: bool` flag. Say this out loud in any dashboard/
report: a train-window cohort's performance number is optimistic *by
construction*, not "partially" so — don't blur that distinction.

Compute per-cohort PR-AUC/ROC-AUC/precision/recall (never trust a
row's *stored* `risk_band` for this — recompute the flag from the
row's `probability` against the *current* `risk_bands` config, so a
later config change can't silently mix threshold vintages):

```python
def cohort_performance_report(predictions: pd.DataFrame, cfg: AppConfig) -> list[CohortPerformance]:
    for reference_date, group in predictions.groupby("reference_date"):
        flagged = np.array([risk_band(float(p), cfg) in ("medium", "high") for p in group["probability"]])
        # PR-AUC/ROC-AUC/precision/recall from group["actual_churned"] vs. group["probability"]/flagged
```

### Dashboard + static report, orchestration

- `dashboards/app.py` (Streamlit) and `dashboards/report.py` (static
  HTML) render the *same* drift/performance computations — one shared
  data-loading module (`monitoring/loaders.py`), not two independently
  duplicated copies. (Two copies is exactly how a filtering bug — the
  "current cohort" query not filtering by champion `model_version` —
  can hide in two places at once and get missed by review.)
- Test a Streamlit app headlessly with `streamlit.testing.v1.AppTest`,
  not `curl` — Streamlit only executes the actual script per websocket
  session, so `curl`ing the server only proves the *process* is up,
  never that the script runs without an exception.
- `pipelines/flows.py` (Prefect) sequences the whole pipeline as one
  flow. Each task can either import and call the underlying Python
  function directly, or shell out to the project's own CLI entry point
  via `subprocess.run` — the latter keeps every stage exactly as
  independently runnable as it already is, at the cost of the
  orchestration layer needing to explicitly set `cwd` and capture/relay
  child output into its own logs (both easy to forget). A failing task
  must halt the flow — **verify this for real** by deliberately making
  one step fail and confirming later steps never run.

```bash
uv run python pipelines/backfill_predictions.py
uv run python pipelines/monitor.py
uv run streamlit run dashboards/app.py
uv run python dashboards/report.py
uv run python pipelines/flows.py
```

---

## 10. Phase 8 — Tests, CI, and docs

### Coverage: target real gaps, not a percentage

Look at your coverage report's *missing lines*, not just the top-line
percentage. Prioritize testing:
- Pure logic with no I/O (binning math, split boundaries, idempotency
  checks) — fast, no mocking needed.
- Anything with a real regression risk (a SQL parameter that must
  always be passed, a conflict-update path that must never clobber a
  column it shouldn't).

Don't chase 100% by heavily mocking thin orchestration wrappers
(`download → validate → synthesize → load`, wired together) — that
kind of glue is already covered by integration tests and live
verification, and heavy mocking there tests "did I call the mocks
right," not real behavior.

```bash
uv add --group dev pytest-cov
uv run pytest   # full suite; needs Docker for tests/integration/
uv run pytest tests/unit   # fast, no Docker
```

### pre-commit

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: ruff-check
        entry: uv run ruff check --fix --exit-non-zero-on-fix
        language: system
        types: [python]
        files: ^(src|tests|pipelines|api|dashboards)/
      - id: black
        entry: uv run black
        language: system
        types: [python]
        files: ^(src|tests|pipelines|api|dashboards)/
      - id: mypy
        entry: uv run mypy src pipelines api dashboards
        language: system
        types: [python]
        files: ^(src|pipelines|api|dashboards)/
        pass_filenames: false
```

Local hooks that shell out to `uv run ...`, rather than pinning
separate hook-repo versions — one place your tool versions live
(`pyproject.toml`), not two that can drift apart.

**A real lesson, worth deliberately re-learning yourself**: after
writing this config, run `uv run pre-commit run --all-files` and
actually **inspect `git status`/`git diff`** before trusting it — don't
just check the exit code. An unscoped `files:` filter (missing the
path restriction above) will happily reformat *every* `.py` file in the
repo, including autogenerated Alembic migrations that should never be
hand-styled. If that happens, `git checkout -- <path>` to revert before
committing anything, then fix the scope. This is the same "verify
against the real thing, not just the exit code" discipline as
everywhere else in this project, applied to the tooling itself.

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

### CI

```yaml
# .github/workflows/ci.yml
name: CI
on: { push: { branches: [main] }, pull_request: {}, workflow_dispatch: {} }
permissions: { contents: read }
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
        with: { version: "latest", python-version: "3.12" }
      - run: uv sync
      - run: uv run ruff check src tests pipelines api dashboards
      - run: uv run black --check src tests pipelines api dashboards
      - run: uv run mypy src pipelines api dashboards
      - run: uv run pytest
```

GitHub's `ubuntu-latest` runners ship Docker already, so the
testcontainers-based integration tests need no extra `services:` block
or secrets — the exact same command as local.

### README, finished

By this point you have enough of a finished system to write, honestly:
an architecture diagram (every arrow a real, verified data flow — a
Mermaid diagram renders directly on GitHub), a short curated
"decisions that shaped the system" summary linking into your full
decision log, a known-limitations section (collect every caveat you've
already found and documented along the way — don't invent new ones,
just gather them), and an operational runbook (restart steps, how to
retrain/promote, how to roll back a bad promotion, how to re-run one
failed pipeline stage, where the logs are).

---

## 11. Full command reference, by phase

```bash
# --- Setup ---
uv sync && uv run pre-commit install
docker compose up -d postgres pgadmin minio minio-init mlflow
uv run alembic upgrade head

# --- Phase 1 ---
uv run python -m churn.ingestion.cli

# --- Phase 2 ---
uv run python -m churn.features.build_panel

# --- Phase 3 (needs: uv sync --group notebooks) ---
uv run jupyter lab
# or headless: uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_*.ipynb

# --- Phase 4 --- (notebooks 04-06, same as above)

# --- Phase 5 ---
uv run python pipelines/train.py

# --- Phase 6 ---
uv run python pipelines/batch_score.py
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000

# --- Phase 7 ---
uv run python pipelines/backfill_predictions.py
uv run python pipelines/monitor.py
uv run streamlit run dashboards/app.py
uv run python dashboards/report.py
uv run python pipelines/flows.py

# --- Phase 8 / ongoing ---
uv run pytest
uv run ruff check src tests pipelines api dashboards
uv run black --check src tests pipelines api dashboards
uv run mypy src pipelines api dashboards
uv run pre-commit run --all-files
```

---

## 12. Real bugs found along the way — interview material

These are genuinely good interview answers to "tell me about a bug you
found and how you fixed it," because each one is subtle, has a clear
root cause, and generalizes into a real engineering principle. Know
each one well enough to explain the failure mode, how it was found, and
the fix, without notes:

1. **Shared, mutable `ColumnTransformer` across two pipelines** (§5).
   Fitting one pipeline silently corrupted another's already-fitted
   state, since both held a reference to the *same* preprocessor
   object. Fix: a factory function that returns a fresh preprocessor
   every call — never share a mutable stateful object across two
   things you intend to fit independently. Generalizes to: any time you
   see the same mutable object instance passed into two places that
   each mutate it.
2. **SHAP top-3 named one-hot dummy columns individually** (§8). A
   categorical feature's explanation could name a category the customer
   didn't even have, because it wasn't aggregated back to the original
   feature before ranking. Fix: sum SHAP values across a feature's
   one-hot columns (exact, for a linear model) before taking the
   top-N. Generalizes to: always check whether "the data as the model
   sees it" and "the data as a human should read it" are the same
   shape — they often aren't after preprocessing.
3. **A dict silently loses rank order through JSONB** (§8). Postgres
   JSONB doesn't preserve key insertion order (sorts by key length) —
   storing "top-3, in order" as a dict quietly loses the order on every
   read-back. Fix: store an ordered list of `{feature, value}`, not a
   dict.
4. **No DB connect timeout → a health check that hangs, not fails
   fast** (§8). A stopped/restarting Postgres container doesn't always
   refuse a new connection immediately — without an explicit
   `connect_timeout`, a request can hang for the OS's default TCP
   timeout (tens of seconds), defeating a liveness probe's whole
   purpose. Fix: set a short, explicit connect timeout on the shared
   DB engine. Found by actually stopping the database mid-request and
   timing the response, not by reading the code.
5. **Quantile-binned drift detection silently blind on a
   low-cardinality feature** (§9). The exact mechanism above (§9) — a
   good story specifically because the bug produced a plausible,
   not-obviously-wrong number (PSI ≈ 0), and was only caught by
   deliberately constructing a skewed test case, not by "does it run."
6. **An unscoped pre-commit hook silently reformatted committed,
   autogenerated files** (§10). Verifying tooling against the real repo
   (not just trusting a green exit code) caught it before it was ever
   committed. Generalizes to: "verify against the real thing" applies
   to your tooling and CI config, not just your application code.

---

## 13. Questions you should be able to answer cold

If you can answer all of these without looking anything up, you
genuinely understand this system — which is the actual point of
building it this way rather than just running someone else's commands:

- Why does every feature-extraction query filter `< reference_date`,
  never `<= reference_date`? What would go wrong with `<=`?
- Why `GroupKFold(groups=customer_id)` instead of plain K-fold here
  specifically? What data property makes plain K-fold wrong?
- Why is PR-AUC the primary metric, not ROC-AUC, given this class
  balance?
- What does the MLflow `champion` alias actually point at, and under
  what exact condition does a new training run get promoted to it?
- Why does `/score` read from an already-materialized feature row
  instead of recomputing features live? What would have to change if
  you needed true real-time feature computation?
- Why is `shap_top3` stored as an ordered list, not a dict? What would
  break if it were a dict?
- What happens if Postgres goes down while the API is running — trace
  the exact sequence, from the dropped connection to the client's
  response.
- Why is the drift reference window train+val, not train-only? What
  would a train-only comparison overstate, and why?
- Why does this project not use DVC? What would change your mind?
- Pick any one of the six bugs in §12 and explain it end to end: what
  was wrong, how it was found, what the fix was, and why that fix is
  the *general* right answer, not just a patch for that one case.
