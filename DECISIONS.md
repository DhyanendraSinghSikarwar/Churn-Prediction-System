# DECISIONS.md — Decision Log

Every non-obvious judgment call made while building this repo, in the
order made. Format: situation → decision → rejected alternative → why.
Linked from [PLAN.md](PLAN.md) at the point each decision applies.

---

## Phase 0 — Architecture & Planning (2026-09-17)

### rolling-monthly-panel-not-cross-sectional

**Situation:** The IBM Telco dataset is cross-sectional — one row per
customer, a single terminal `(tenure, Churn)` outcome. The hard
constraints demand a `reference_date`-parameterized query that serves
both training and inference, and a genuine time-aware split.

**Decision:** Build a rolling monthly scoring panel: one row per
`(customer_id, reference_date)` for every month the customer was an
active subscriber, label = churn within 30 days of `reference_date`.

**Rejected alternative:** One row per customer, split by
`label_date = signup_date + tenure`. Rejected because `signup_date` was
itself derived as `snapshot_date − tenure`, so for every *non-churned*
customer `label_date` collapses to the same value (`snapshot_date`). A
"time-based" split on that field would dump nearly all active customers
into the most-recent bucket and only early-churners into earlier
buckets — that's class-imbalance-by-bucket dressed up as a temporal
split, and it would silently violate "no leakage" while appearing to
satisfy "time-aware splits only."

**Why this is better:** the same parameterized SQL produces training rows
(historical months) and inference rows (`reference_date = today`); point-
in-time correctness becomes a concrete, testable property
(`event_date < reference_date` everywhere); and a real calendar split
falls out naturally (earliest months = train, latest = test).

### imbalance-is-designed-in

**Situation:** User explicitly required the resulting dataset to be
meaningfully imbalanced, and for imbalance-handling techniques to be
applied for real, not just mentioned.

**Decision:** The monthly-panel design (previous entry) produces this
automatically: the 26.6% churn figure in the raw dataset is churn-over-
the-whole-relationship (spread across up to 72 months of tenure), so the
*monthly* `churn_next_30d` positive rate in the panel will be a low
single-digit percentage. This will be measured and reported exactly in
notebook 02 (Phase 3), not assumed. Phase 3/4 notebooks are required to
empirically compare class weighting, SMOTE, and threshold tuning and
report which wins on PR-AUC, per the original spec — none of the three is
picked in advance.

**Why:** an imbalance that falls naturally out of a defensible temporal
design is more honest than injecting an arbitrary imbalance ratio by
undersampling the raw data.

### tech-stack

**Situation:** Hard constraints allow `uv` or `poetry`; Pandera or Great
Expectations; Airflow or Prefect. Machine has `uv` installed, not
`poetry`. Several other stack choices were open.

**Decisions:**
- `uv` (already installed; `poetry` isn't).
- SQLAlchemy Core (engine/session) + hand-written parameterized `.sql`
  files loaded by a thin repository layer, **not** a query-building ORM —
  rejected the ORM-query approach because it would hide the window-
  function SQL that Deliverable 3 is centrally about behind an
  abstraction; reviewers need to read real SQL.
- Pandera over Great Expectations — GE's context/suite ceremony is heavy
  for a project this size; Pandera gives equivalent schema + statistical
  checks as plain, unit-testable Python.
- MLflow backend store: a second database (`mlflow`) on the *same*
  Postgres instance, MinIO as the S3-compatible artifact store — rejected
  a second DB engine (sqlite) to avoid a second thing to operate; keeps
  `docker-compose.yml` to a small, coherent set of services.
- Orchestrator: **Prefect 2**, not Airflow — Airflow's webserver +
  scheduler + metadata-DB triad is heavy for 3 DAGs in a single-repo demo;
  Prefect is Python-native and runs locally without extra infrastructure.
  Flagged as a call to revisit if the org standardizes on Airflow.
- Drift statistics (PSI/KS/KL): hand-rolled in `src/churn/monitoring/`
  using scipy/numpy, not `evidently` — the deliverable names 3 specific
  statistics; a full monitoring framework is more dependency surface than
  needed, and hand-rolling keeps the exact alert thresholds fully ours to
  test.
- Integration test DB: `testcontainers-python`, not a dependency on a
  pre-started `docker compose` stack — makes integration tests self-
  contained and CI-portable.
- `feature_snapshots` feature columns: typed columns, not JSONB — lets
  Postgres and Pandera both enforce dtypes/ranges and keeps the table
  directly queryable from the dashboard and validation SQL. JSONB is kept
  only for `predictions.shap_top3`, where the shape is genuinely variable.
- Pre-commit/lint stack kept exactly as specified: `ruff`, `black`,
  `mypy`, `nbstripout`.

### leakage-mitigations

**Situation:** Several columns in the raw dataset are direct leakage
risks if used naively as features in the monthly panel.

**Decision:**
1. Raw `tenure` encodes the *final* relationship length — never enters
   `feature_snapshots`; every panel row recomputes
   `tenure_months_as_of_reference_date` from `signup_date` and
   `reference_date`.
2. Raw `TotalCharges ≈ MonthlyCharges × tenure` — a near-deterministic
   function of the leaky `tenure` column — is **dropped entirely at
   ingestion** (not loaded into any table). Its point-in-time-correct
   replacement is `cumulative_charges_to_date`, computed from
   `billing_events` filtered to `event_date < reference_date`.
3. The raw `Churn` column is the label source and must never appear as a
   feature; it is translated once into `churn_labels.churn_date` /
   `churned` and nowhere else.
4. Same customer appears at multiple adjacent `reference_date`s in the
   panel → plain `StratifiedKFold` would leak label information across
   the split via near-duplicate rows. CV during hyperparameter tuning
   uses `GroupKFold(groups=customer_id)`, nested inside the time-blocked
   training window.
5. Every windowed feature CTE uses strict `<` (never `<=`) against
   `reference_date`; enforced later by a `sql/validation/` query
   asserting `max(event_date) < reference_date` per snapshot.
6. Support-ticket resolution-time features must exclude tickets not yet
   resolved as of `reference_date` (an open ticket contributes to *count*
   features, never to *resolution-time* features computed as of a date
   before it resolved).

**Why documented here instead of just in code:** these are exactly the
mistakes that look correct in a notebook and fail silently in production;
writing them down means they get checked, not just remembered.

---

## Phase 1 — Infra + DB + Ingestion (2026-09-17)

### secrets-never-in-yaml

**Situation:** Hard constraint: "Config in YAML + Pydantic validation. No
hardcoded paths, thresholds, or credentials."

**Decision:** Split configuration into two Pydantic models: `Settings`
(secrets — DB/MinIO credentials, tracking URIs — read from `.env` /
process env via `pydantic-settings`, never committed) and `AppConfig`
(business/pipeline parameters — snapshot date, panel window, synthetic-
data assumptions, business economics — read from `config/config.yaml`,
safe to commit). `.env` is git-ignored; `.env.example` documents every
variable with no real values.

**Why:** keeps `config/config.yaml` fully committable and reviewable
(it's part of "what the model does"), while credentials stay out of
version control entirely, satisfying the constraint literally rather than
just mixing everything into one YAML with placeholder secrets.

### feature_snapshots-schema-staged-across-phases

**Situation:** Deliverable 2 asks for a `feature_snapshots` table now
(Phase 1), but the actual feature set is only finalized empirically in
notebook 03 (Phase 3) via mutual information / permutation importance /
Boruta / correlation pruning.

**Decision:** Phase 1's migration creates only the spine of
`feature_snapshots` — `customer_id`, `reference_date`,
`feature_set_version`, `label_churn_next_30d`, `created_at` — plus its
uniqueness constraint. The typed feature columns themselves are added by
a Phase 2/3 migration once the feature set is fixed.

**Rejected alternative:** guess the final column list now. Rejected
because it would either be wrong (churn through migrations to fix it) or
bias notebook 03's "empirically select features" step toward whatever was
guessed first.

### ddl-generated-from-alembic-not-hand-duplicated

**Situation:** Deliverable 1 wants both Alembic migrations (source of
truth, versioned) and a `sql/ddl/` directory with plain schema SQL.
Hand-maintaining both from the same mental model risks drift.

**Decision:** `sql/ddl/schema.sql` is generated by running
`alembic upgrade head --sql` (Alembic's offline SQL-emission mode)
against the real migration, not hand-transcribed. It is a read-only,
regenerate-on-change artifact; a header comment says so.

**Why:** guarantees `sql/ddl/schema.sql` can never drift from what
Alembic actually applies — the alternative (hand-writing both) is exactly
the kind of duplication that silently rots.

### churn-date-placement-for-synthetic-history

**Situation:** The raw dataset gives churned customers a `tenure`
(how long they were a customer before churning) but not a churn *date*
(how long ago, relative to `SNAPSHOT_DATE`, that happened). Both are
needed to place billing/ticket events on a real calendar.

**Decision:** Assume churn dates are spread **uniformly over the
trailing 24 months before `SNAPSHOT_DATE`**, then back-compute
`signup_date = churn_date − tenure_months`. Active customers instead
anchor on `SNAPSHOT_DATE` directly (`signup_date = SNAPSHOT_DATE −
tenure_months`), since they are still a customer today.

**Why 24 months, not e.g. 6 or 36:** it needs to comfortably cover the
18-month training/validation/test panel window (2023-01 … 2024-06) so
the panel contains enough realized churn events to learn from, while
still leaving a plausible minority of "old" churns that predate the
panel (realistic — a real system also carries customers who churned
before its current scoring window began). This is a synthetic-data
placement assumption, not derived from any real distribution, and is
called out again in the `synthesize.py` module docstring and in
`README.md`.

**Rejected alternative:** clustering all churn dates at `SNAPSHOT_DATE`
(simplest to code). Rejected — it would make every churner's tenure-
months-before-reference_date identical to their final raw `tenure`,
reopening exactly the tenure-leakage problem the panel design exists to
avoid.

### total-charges-dropped-at-ingestion

Raw `TotalCharges` is not loaded into any table (see
leakage-mitigations #2 above) — noted again here because it's a Phase 1
implementation decision, not just a Phase 0 risk flag: `synthesize.py`
never reads that column from the source CSV at all.

### synthetic-service-attributes-are-static-for-now

**Situation:** The source dataset gives only the *final* state of each
customer's plan (contract type, add-ons, etc.), not a change history.

**Decision:** `subscriptions` holds one row per customer for the whole
synthetic history (`start_date = signup_date`, `end_date = churn_date` or
`NULL`). Plan changes over time are not simulated.

**Why acceptable:** these attributes are then safe to use as features at
*any* `reference_date ≥ signup_date` without a leakage risk, which
simplifies Phase 2's feature SQL honestly rather than papering over an
unsimulated complexity. The schema (`end_date` nullable, not a 1:1
constraint) is deliberately left able to support real plan-change history
later — flagged as a demo simplification, not a design dead-end.

### copy-based-bulk-load-with-two-layer-idempotency

**Situation:** Hard constraint: "one-time idempotent ingestion script:
download → checksum verify → skip if already loaded → bulk COPY insert →
row count assertion."

**Decision:** Two independent idempotency layers: (1) filesystem — if
`data/raw/telco_customer_churn.csv` already exists, it is not
re-downloaded; (2) database — the file's SHA-256 is recorded in
`ingestion_runs` after a successful load, and the CLI checks that table
before synthesizing/loading anything, so re-running the script is a
no-op even if the local cache was deleted and re-downloaded fresh.
Loading itself uses psycopg3's `COPY ... FROM STDIN`, not row-by-row
`INSERT`, and every table's post-load count is asserted against the
number of rows actually written before the transaction commits.

### numpy-Generator-not-global-seed

**Decision:** The synthetic-data generator threads a single
`numpy.random.Generator` (`np.random.default_rng(seed)`) through every
function that needs randomness, rather than calling the legacy global
`np.random.seed(...)`.

**Why:** avoids global mutable random state leaking into anything else
that imports the module (tests, notebooks) and makes reproducibility a
property of the function call, not of import order.

### infra-issues-hit-during-verification

Phase 1 was actually run end-to-end (not just written and assumed
correct) — `uv sync`, full `docker compose up`, `alembic
revision --autogenerate` + `upgrade head` against real Postgres, a real
`python -m churn.ingestion.cli` load, and `pytest` including the
testcontainers integration test. Four real problems surfaced and were
fixed, not worked around:

1. **`minio/mc` on Docker Hub returns "pull access denied".** MinIO Inc.
   moved image distribution off Docker Hub; `quay.io/minio/minio` and
   `quay.io/minio/mc` are the current official source. Both images in
   `docker-compose.yml` now pull from `quay.io`.
2. **The MLflow server image crash-looped on `ModuleNotFoundError: No
   module named 'pkg_resources'`.** Recent `python:3.12-slim` no longer
   preinstalls `setuptools`, and MLflow 2.14 still imports
   `pkg_resources` at startup. Worse, plain `pip install setuptools`
   installs a version (81+) that has fully dropped `pkg_resources`.
   `docker/mlflow/Dockerfile` now pins `setuptools<81` explicitly.
3. **`alembic/env.py` silently ignored a programmatically-set
   `sqlalchemy.url`.** It unconditionally called
   `config.set_main_option("sqlalchemy.url", get_settings().database_url)`,
   so the integration test's `alembic_cfg.set_main_option(...)` pointing
   at the ephemeral testcontainer was overwritten with the real dev
   database's URL — the test appeared to migrate a database, but silently
   migrated the *wrong* one (a no-op, since it was already at head),
   leaving the actual target database schema-less. Fixed by only falling
   back to `Settings().database_url` when no URL was already configured
   (`if not config.get_main_option("sqlalchemy.url"):`). This is the kind
   of bug that hides successfully in a fast test suite — the migration
   command returned success, just against the wrong target — which is
   exactly why the integration test was worth running for real rather
   than trusting it by inspection.
4. **`support_tickets.satisfaction_score` COPY failed with
   `invalid input syntax for type smallint: "4.0"`.** A pandas column
   built from Python `int`/`None` values upcasts to `float64` once a
   `None` is present, so non-null scores serialize to CSV as `"4.0"`,
   which Postgres's `smallint` parser rejects. Fixed by casting the
   column to pandas' nullable `Int64` dtype before the CSV write. General
   lesson for this codebase: any nullable integer column headed for a
   COPY load needs `Int64`, not plain `int`/`float64` — worth checking
   again whenever a new such column is added.

All four are now fixed in the committed files, not left as manual
workaround steps — a fresh `docker compose up` + `alembic upgrade head`
+ ingestion run on a clean checkout should hit none of them.

### make-unavailable-on-this-machine

**Situation:** `make` is not installed on this Windows dev machine (only
`git bash` / PowerShell / Docker / `uv`).

**Decision:** The `Makefile` is still written in full per the hard
constraint (targets: `setup`, `db-up`, `ingest`, `train`, `score`,
`serve`, `test`, `lint`, `clean`) since it documents the canonical
commands and will work wherever `make` is present (CI, Linux/macOS dev
machines, WSL). On this machine, Phase 1 was verified by running the
`uv run ...` / `docker compose ...` commands each target wraps, directly.
`README.md` notes this for Windows users without `make` installed.
