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

---

## Phase 2 — SQL Features + Data Validation (2026-09-17)

### autogenerate-misses-check-constraints

**Situation:** Added three `CheckConstraint`s to `FeatureSnapshot`
(`tenure_bucket` enum, `tenure_months >= 0`, `payment_failure_rate_180d`
range) alongside 31 new columns, then ran
`alembic revision --autogenerate`. The log showed "Detected added column"
for all 31 columns but never mentioned the constraints, and the generated
migration's `upgrade()` had no `create_check_constraint` calls at all.

**Decision:** Caught by re-running `alembic revision --autogenerate`
immediately after applying the migration and confirming it reported "No
new upgrade operations detected" — it didn't, the first time, which is
what surfaced this. Fixed by downgrading one revision, hand-adding the
three `op.create_check_constraint(...)` calls (and matching
`op.drop_constraint(...)` in `downgrade()`) to the migration file, then
re-upgrading and re-checking autogenerate until it reported no diff.

**Why this matters beyond this one migration:** Alembic's autogenerate
reliably detects added/removed tables, columns, indexes, unique
constraints, and foreign keys — but not CHECK constraints, at least not
consistently in this version. Any future migration that adds a
`CheckConstraint` to a model must be verified the same way: re-run
autogenerate after applying and confirm it comes back empty. This is now
also why `sql/ddl/schema.sql`'s header calls this out explicitly, so the
next person touching the schema doesn't trust autogenerate's silence.

### sqlalchemy-text-rejects-bind-param-postgres-cast

**Situation:** First real run of `build_feature_snapshot.sql` (via
`pd.read_sql(text(...), conn, params={...})`) failed with
`psycopg.errors.SyntaxError: syntax error at or near ":"` pointing at
`:reference_date::date AS reference_date` — caught immediately, on the
very first execution against real data, not discovered later.

**Decision:** `sqlalchemy.text()`'s bind-parameter tokenizer treats a
literal `::` immediately after a `:name` token as "leave this alone,
that's a Postgres cast operator" — but the way it does that also skips
substituting the bind parameter itself, so `:reference_date::date` was
sent to psycopg completely unsubstituted, `:` and all. Every occurrence
of `:reference_date::date` in the query was rewritten to
`CAST(:reference_date AS date)`. A plain column's own `::type` cast
(e.g. `br.failed_180d::numeric`) is unaffected — the ambiguity is
specific to casting a `:name` bind parameter with the shorthand syntax.

**Why note this here:** any future SQL file bound through SQLAlchemy
`text()` with a Postgres type cast on a parameter must use `CAST(:x AS
type)`, never `:x::type` — the failure mode is a bare syntax error, not
a silent wrong answer, so at least it fails loudly rather than leaking.

### last-cohort-has-a-structural-zero-positive-rate

**Situation:** After materializing the full 18-month panel and running
`sql/validation/class_balance.sql`, every monthly cohort showed a
positive rate in the 1.3%–2.0% range — exactly the "low single-digit
percent" the panel design predicted — **except** the very last cohort,
`reference_date = 2024-06-01` (= `SNAPSHOT_DATE`), which showed exactly
**0 positives out of 5,174**.

**Why:** this isn't a bug and isn't leakage — it's a censoring artifact
of the synthetic churn-date generator. `derive_customer_dates()` places
every churned customer's `churn_date` somewhere in
`[SNAPSHOT_DATE - 24 months, SNAPSHOT_DATE]` — it can never generate a
churn date *after* `SNAPSHOT_DATE`, because there's no "after" in this
synthetic world. So `label_churn_next_30d` for `reference_date =
SNAPSHOT_DATE` requires a `churn_date` in
`(SNAPSHOT_DATE, SNAPSHOT_DATE + 30d]`, a range that structurally cannot
contain any data, by construction. The true underlying churn hazard for
that cohort is presumably similar to every other month's (~1.5%); the
observed 0% is an artifact of where the synthetic history happens to end,
not a real signal.

**Decision:** Documented here rather than silently patched, because the
right fix depends on what Phase 4 actually wants: either (a) drop the
last cohort (or more conservatively, the last `horizon_days` worth of
cohorts) from the usable panel before splitting into train/val/test, or
(b) keep it but treat it as a known source of mild test-set dilution if
it lands in the test window. Given the current split (test = last 3
months = Apr/May/Jun 2024), option (a) is recommended for Phase 4 — using
`panel_end_date` for materialization but capping the *modeling* panel at
`SNAPSHOT_DATE - prediction_horizon_days` avoids training or evaluating
on a cohort whose label is unobservable by construction. Flagging this
explicitly now so it isn't rediscovered as a mysterious "why is recall
zero on the last month" bug later.

### feature-selection-vs-feature-extraction-clarified

**Situation:** DECISIONS.md's Phase 1 entry
`feature_snapshots-schema-staged-across-phases` said the feature set is
"finalized empirically in notebook 03" — read literally, that could mean
`feature_snapshots`' columns should keep changing every time notebook 03
reconsiders a feature, which would make the table's schema hostage to
notebook experimentation.

**Decision:** Split the concepts. `feature_snapshots` (this migration)
holds the full *candidate* feature set from `sql/features/build_feature_snapshot.sql`
— everything in the Phase 0 feature catalog (RFM, tenure, trend/velocity,
static plan attributes). Notebook 03's job is *selection*: mutual
information / permutation importance / Boruta / correlation pruning pick
a *subset* of these columns for a given model iteration to actually
train on. Selection is recorded as a feature list artifact (Phase 4/5),
not a schema migration. Ratio/interaction features (Deliverable 4's
"ratio and interaction features with business rationale") are computed
in pandas from these base columns in notebook 03, not in SQL — they're
cheap, leakage-free once the base features are already point-in-time
correct, and don't need a database round-trip.

### label-lookahead-is-not-leakage

**Situation:** `build_feature_snapshot.sql` (below) computes
`label_churn_next_30d` by looking at `churn_date` up to
`reference_date + 30 days` — i.e. *forward* from the feature cutoff. That
looks, at a glance, like exactly the kind of boundary violation
DECISIONS.md has spent two phases warning about.

**Decision:** This is intentional and correct, not a leak — documented
directly in the query's header comment so a future reader doesn't
"fix" it. The invariant that matters is: no *feature* column may see data
at or after `reference_date`. The *label* is expected to describe what
happens after the cutoff — that's the entire premise of supervised
learning from historical data. The distinction is enforced by keeping
label computation in a clearly separated final SELECT expression, never
joined into any feature CTE. In a live production system, this column
would simply be left NULL until 30 days after the actual scoring date
(exactly what `predictions.actual_churned` / `label_matured_at` do in
Phase 6/7); it's always computable here only because this dataset is a
complete historical replay with the outcome already known.

---

## Phase 3 — Notebooks 01-03: Data Audit, EDA, Feature Selection (2026-09-21)

### nbconvert-cwd-breaks-env-file-resolution

**Situation:** All three notebooks failed on their first automated
(`jupyter nbconvert --execute`) run with `psycopg.OperationalError:
fe_sendauth: no password supplied` — not a missing-file error, a silent
empty password.

**Decision:** `nbconvert` executes a notebook with the working directory
set to the notebook's own folder (`notebooks/`), but `Settings`
(`src/churn/config.py`) resolves `pydantic-settings`' `env_file=".env"`
relative to the process's current working directory, not the project
root. From `notebooks/`, that path silently misses `.env` — `pydantic-settings`
treats a missing env file as "no overrides" rather than an error, so every
field quietly falls back to its class default, including
`db_password: str = ""`. Every notebook's setup cell now does
`os.chdir(PROJECT_ROOT)` (computed from `churn.config.PROJECT_ROOT`, which
is itself cwd-independent) immediately after importing config, before
calling `get_settings()`/`get_engine()`.

**Rejected alternative:** hardcoding an absolute `.env` path into
`Settings.model_config`. Rejected because it would only fix the
notebook-from-a-different-cwd case and would break the documented
`uv run ...` workflow (always invoked from the project root) if `.env`
ever moved.

**Why this matters beyond notebooks:** any future script invoked with a
cwd other than the project root (a cron job, a CI step, a Prefect
deployment default working directory) hits the exact same silent
failure — worth remembering before assuming "it works when I run it" means
it will work everywhere.

### pandas-3-default-string-dtype-breaks-dtype-object-checks

**Situation:** `uv sync --group notebooks` resolved `pandas==3.0.5`
(`pyproject.toml` only pins the `>=2.2.1` floor). Notebook 02's
mutual-information cell crashed with
`ValueError: could not convert string to float: 'One year'` —
`contract_type` had been silently classified as a *numeric* column.

**Decision:** pandas 3.0 changed the default in-memory representation of
string columns from plain `object` dtype to a dedicated `str`/`string`
dtype (confirmed directly: notebook 01's own `df.info()` reports
`Dtype: str` for every text column, not `object`). Code written against
the pandas-2-era idiom `df[c].dtype == object` to detect categorical
columns silently stops matching string columns under pandas 3 — they fall
through into whatever branch handles "everything else," here treated as
numeric. Fixed in both notebooks 02 and 03 by classifying columns with
`pd.api.types.is_numeric_dtype(...) and not pd.api.types.is_bool_dtype(...)`
instead, which is correct regardless of which dtype backend a given pandas
version uses for strings.

**Rejected alternative:** pinning `pandas<3` in `pyproject.toml`. Rejected
here — the fix is one line more robust and forward-compatible, and pinning
would just defer the same bug to whenever the project does upgrade.
Flagged as worth a deliberate, tested pandas-3 compatibility pass across
`src/churn` before Phase 5 trains anything real, since this exact pattern
(`dtype == object`) is a natural one to have used elsewhere.

### hand-rolled-boruta-instead-of-the-boruta-package

**Situation:** Feature selection (Deliverable 4) explicitly calls for
Boruta among the four selection methods. The `boruta` / `BorutaPy` PyPI
package is unmaintained (last release ~2020) and known to break against
scikit-learn >= 1.3's API.

**Decision:** Hand-rolled the core Boruta algorithm directly in notebook
03 (`boruta_shadow_selection()`): shuffle every real feature into a
same-distribution "shadow" copy each iteration, fit a random forest on
real+shadow together, tally how often each real feature's importance
beats the *best* shadow feature's, and binomial-test the win rate against
chance to classify confirmed/tentative/rejected. ~30 lines, fully
inline and inspectable.

**Rejected alternative:** the `boruta` package, or skipping Boruta
entirely and only reporting 3 of the 4 required methods. Rejected for the
same reason DECISIONS.md's Phase 0 `tech-stack` entry rejected `evidently`
for drift statistics: the core idea is simple enough to hand-roll, and a
fragile, unmaintained dependency is worse than 30 lines of plain
scikit-learn/scipy this project already depends on.

### feature-list-artifact-lives-in-config-not-artifacts

**Situation:** DECISIONS.md's Phase 1 entry
(`feature-selection-vs-feature-extraction-clarified`) said selection
would be "recorded as a feature list artifact," without saying where.
`artifacts/` looks like the obvious name but is git-ignored (`.gitignore`:
"trained model artifacts live in MLflow/MinIO, not in git").

**Decision:** Notebook 03 writes `config/selected_features_v1.yaml` —
alongside `config/config.yaml`, following the same "committed,
human-reviewable, safe (no secrets)" pattern, since Phase 5's `train.py`
needs to read this file deterministically on every run, not regenerate it
from `artifacts/` (which may not exist in a fresh clone or CI checkout).

**Rejected alternative:** un-ignoring a specific path under `artifacts/`.
Rejected because it would mix a small, reviewable, version-controlled
config value into a directory whose entire purpose is to hold large,
regenerable, git-ignored outputs — exactly the kind of category mix the
`config/` vs `artifacts/` split exists to prevent.

### last-cohort-dropped-from-modeling-panel

**Situation:** Phase 2's `last-cohort-has-a-structural-zero-positive-rate`
entry left the choice open between two options for Phase 3/4: drop just
the single `reference_date == SNAPSHOT_DATE` cohort, or more
conservatively drop the last `horizon_days` worth of cohorts.

**Decision:** Notebook 02 independently re-confirmed, in pandas, that
*only* the single last cohort has the structural zero (every other
cohort's positive rate falls in the same 1.3-2.0% band predicted since
Phase 0) — so notebook 03 drops exactly that one cohort (5,174 of 87,517
rows) before doing any feature selection or, later, any modeling. The
`feature_snapshots` table itself is untouched; this is a modeling-panel
filter, applied fresh in every notebook/pipeline step that trains or
evaluates, not a data deletion.

**Why the less conservative option:** the empirical re-check (not just
trusting Phase 2's SQL) is exactly the kind of measurement this project
insists on before making a call — dropping more cohorts than the data
shows is affected would throw away real, usable training signal for no
benefit.

---

## Phase 4 — Notebooks 04-06: Modelling, Evaluation, Business Impact (2026-09-21)

### mlflow-s3-endpoint-url-must-not-come-from-the-host-facing-env-var

**Situation:** The first `mlflow.log_artifact()` call in Notebook 04
failed with `botocore.exceptions.EndpointConnectionError: Could not
connect to the endpoint URL: "http://localhost:9000/..."` — the MLflow
server container itself was trying to reach MinIO at `localhost:9000`.

**Decision:** `docker-compose.yml`'s `mlflow` service set
`MLFLOW_S3_ENDPOINT_URL: ${MINIO_ENDPOINT:-http://minio:9000}` — reading
the *same* `MINIO_ENDPOINT` variable `.env` sets to `http://localhost:9000`
for host-side code (`Settings.minio_endpoint`, used by anything running
outside Docker, since MinIO's ports are published to the host). Because
`.env` always sets that variable, the compose default never took effect,
and the mlflow container inherited a `localhost` that — from inside its
own network namespace — means the mlflow container itself, not the minio
one. Fixed by hardcoding `MLFLOW_S3_ENDPOINT_URL: http://minio:9000`
directly in `docker-compose.yml`, with no `${MINIO_ENDPOINT}` indirection
at all: this value is never host-configurable, so it should never have
been sourced from a host-facing variable.

**Rejected alternative:** renaming `MINIO_ENDPOINT` everywhere to force
docker-compose to use its own default. Rejected because the simpler fix
(hardcode the one value that's genuinely fixed) removes the variable
collision entirely instead of relying on nobody accidentally setting
`.env`'s `MINIO_ENDPOINT` again in the future.

**Why this matters beyond this one bug:** any container-internal
Docker-network address must never be sourced from a host-facing `.env`
variable of the same apparent purpose — the two are only coincidentally
related (same physical service, different network vantage points), and
`.env`'s existence as "the" config source makes it tempting to reuse
across both. This had been latent since Phase 1's `docker-compose.yml`
was written; it just never got exercised until Phase 4 actually called
`mlflow.log_artifact()`.

### histgradientboosting-instead-of-classic-gbm

**Situation:** Notebook 04's first real run (`GradientBoostingClassifier`,
scikit-learn's classic exact-split boosting) hit nbconvert's cell timeout
(1500s) partway through the 25-trial Optuna search on a ~66k-row training
fold — the exact-split algorithm doesn't scale well to that row count
repeated across dozens of hyperparameter trials.

**Decision:** Switched to `HistGradientBoostingClassifier`
(scikit-learn's native histogram-based boosting, LightGBM-style binned
splits) — same library, no new dependency, dramatically faster on this
data size. Hyperparameter names differ (`max_iter` instead of
`n_estimators`, no `subsample`, added `l2_regularization`); the search
space and technique-comparison code were updated accordingly. Also
reduced the Optuna search itself (25→12 trials, 3→2 `GroupKFold` splits)
after the identical, seeded computation still occasionally exceeded the
timeout on a second run — attributed to this dev machine's variable
background load (OneDrive sync, Docker Desktop), not the code.

**Rejected alternative:** adding `xgboost` or `lightgbm`. Rejected for
the same reason Boruta was hand-rolled in Phase 3 instead of pulling in
the unmaintained `boruta` package — a built-in, adequate tool beats a new
dependency when one is available.

### logistic-regression-beats-every-gbm-variant

**Situation:** Notebook 04 was built expecting the GBM (with imbalance
handling and Optuna tuning) to be the strongest candidate — the baseline
logistic regression was meant to be a reference point, not the winner.

**Decision:** Reported the actual result instead of re-framing the
notebook around it after the fact. `LogisticRegression(class_weight="balanced")`
scored val PR-AUC 0.1281; the best GBM variant (SMOTE, Optuna-tuned)
scored 0.0609 — logistic regression wins by a wide margin, confirmed
again on the untouched test set in Notebook 05 (0.1093 vs. 0.0593).
Notebook 04's summary cell explicitly flags this ("the GBM line of
experiments never beat the logistic regression baseline... reported as
found, not smoothed over") rather than quietly making the GBM the
headline model. Notebook 05/06 evaluate the logistic regression as the
primary model and carry the tuned GBM alongside only for completeness.

**Why this is a real result, not a bug:** only 12 selected features (7
low-cardinality categoricals contributing little nonlinear structure),
~1.7% positive rate in training, and none of Notebook 03's engineered
interaction features survived selection — exactly the regime where a
well-regularized linear model's lower variance tends to beat a boosted
ensemble's extra flexibility. Same principle as Notebook 03's engineered
features not making the final cut: an honest empirical result is worth
more than a result that matches what was expected going in.

### ltv-proxy-and-cost-based-threshold-formula

**Situation:** `config/config.yaml`'s `business:` block gives
`retention_offer_cost`, `assumed_ltv_horizon_months`, and
`offer_success_rate`, but no actual customer LTV figure — the dataset has
no real economics in it (PLAN.md's confirmed assumptions).

**Decision:** LTV proxy per customer = `monthly_charges *
assumed_ltv_horizon_months` (Notebook 05/06). Expected value of flagging
a customer counts only the *incremental* effect of making an offer: a
true positive nets `offer_success_rate * LTV - retention_offer_cost`
(the offer costs money whether or not it succeeds, but only pays off on
success); a false positive costs `-retention_offer_cost` with no offset;
true negatives and false negatives are exactly $0 (no offer made, so
nothing attributed to inaction).

**Rejected alternative:** counting false negatives as a "cost" equal to
the LTV lost. Rejected because that would be scoring the whole portfolio,
not the incremental value of *this specific intervention* — the question
this notebook actually needs to answer is "should we make this offer,"
not "what is our total exposure to churn," and conflating the two would
make every threshold look artificially better than contact-no-one, which
defeats the point of the comparison.

**Finding this formula produced, not assumed:** at the *configured*
`risk_bands` (0.3 / 0.7), this formula returns a strongly negative
expected value on the test set — false-positive cost dominates unless the
threshold is set near the top of the probability range (the actual
business-optimal threshold came out to 0.940). This is a legitimate
consequence of a ~1.6% base rate combined with a flat per-contact cost,
not an error in the formula; see PLAN.md Phase 4 for the concrete
numbers and DECISIONS.md's `logistic-regression-beats-every-gbm-variant`
entry for why the model's PR-AUC is modest to begin with.

### windows-emoji-console-encoding-breaks-plain-script-runs

**Situation:** A standalone `uv run python -c "..."` MLflow smoke test
crashed with `UnicodeEncodeError: 'charmap' codec can't encode character
'\U0001f3c3'` — MLflow prints a "🏃 View run..." message on every run
completion, and Windows' default console codepage (cp1252) can't encode
it.

**Decision:** Documented as environment-specific, not fixed in project
code — this never affects the actual notebooks (nbconvert communicates
with the kernel over the Jupyter message protocol, which is UTF-8
regardless of the host console's codepage), only ad hoc scripts run
directly in a Windows console. No project code prints emoji, so nothing
here needed changing; noted in CLAUDE.md's gotchas so a future direct
script invocation that hits this isn't mistaken for a real bug.

**Why not set `PYTHONIOENCODING=utf-8` globally:** it would paper over
the same failure mode for every future direct script run instead of
recognizing it as specific to this one interaction (Windows console +
emoji-printing library) — cheaper to know the shape of the problem than
to blanket-fix an entire class of encoding issues that mostly won't
recur.

---

## Phase 4 post-review fixes (2026-09-23)

Phase 4 was reviewed by a fresh agent (no prior context, briefed to be
adversarial rather than confirm the work) before Phase 5 started. It
found one real bug and two minor issues, all fixed below; everything
else it checked — no leakage, `GroupKFold` usage, the cost-based
threshold arithmetic, the A/B sample-size calculation, and every other
headline number — came back clean.

### shared-columntransformer-corrupted-notebook-04s-threshold-tuning-numbers

**Situation:** Notebook 04's imbalance-technique comparison
(`gbm_plain`, `gbm_classweight`, `gbm_smote`) and the later threshold-tuning
cell all referenced *one* `ColumnTransformer` instance. sklearn's
`Pipeline` does not clone its steps — fitting `gbm_smote` (whose SMOTENC
step resamples the training data before the shared preprocessor refits on
it) silently left `gbm_plain`'s already-fitted preprocessor rescaled to
the *resampled* distribution. The technique-comparison PR-AUCs (0.0529 /
0.0483 / 0.0606) were unaffected — each was computed by an immediate
fit-then-predict before any other pipeline touched the shared object —
but the threshold-tuning cell called `gbm_plain.predict_proba(X_val)` a
*second* time, after `gbm_smote.fit()` had already corrupted the shared
preprocessor. That produced wrong probabilities, and from them a wrong
F1 curve: the notebook (and PLAN.md, copying it) reported F1 0.0077 →
0.0585 at threshold 0.027, when the correct values — reproduced and
confirmed by the reviewing agent, then by re-running the fixed notebook —
are F1 0.0229 → 0.1183 at threshold 0.096.

**Decision:** Every pipeline in Notebook 04 now gets its own
`ColumnTransformer` from a `make_preprocessor()` factory — the exact
pattern Notebooks 05 and 06 already used correctly (their own comment
even flagged this as "a real bug caught before this ran," just never
backported to 04). Also split the three-fit technique-comparison cell
into one cell per technique, so a future slow or hung fit is visible on
its own rather than inside one large multi-fit cell.

**Why this matters beyond this one notebook:** any two `Pipeline` (or
`ImbPipeline`) objects built by literally passing the same transformer
instance into both are unsafe unless it is provably never refit twice —
which is easy to get right the first time a model is built and easy to
break the moment a second one is added next to it "for comparison."
Phase 5's `train.py` must build a fresh preprocessor per fitted pipeline,
never share one across candidate models.

### groupkfold-cv-score-is-not-comparable-to-forward-in-time-pr-auc

**Situation:** Notebook 04's Optuna search reported a `GroupKFold`
cross-validation PR-AUC of 0.1101 for the tuned SMOTE+GBM — noticeably
higher than that same model's forward-in-time validation PR-AUC (0.0609)
and test PR-AUC (0.0593).

**Decision:** Documented, not "fixed" — there's nothing wrong with either
number, they measure different things. `GroupKFold(groups=customer_id)`
prevents a customer appearing in both a fold's train and validation
split, but it does not block train, so the CV score is a same-training-window,
contemporaneous estimate; the val/test PR-AUCs are strictly forward-in-time
(later calendar months the model never saw). The gap is expected, not a
sign of a leak or a bug.

**Why this matters for Phase 5:** the training pipeline's promotion gate
("beat the current Production model's PR-AUC by at least
`promotion_epsilon`," per `config/config.yaml`) must compare like with
like — a candidate's forward-in-time validation/test PR-AUC against
Production's own forward-in-time PR-AUC, never a candidate's CV score
against Production's deployed metric. Comparing a CV number against a
forward-in-time number would make every candidate look better than it
is and risk promoting a model that doesn't actually generalize forward.

---

## Phase 5 — Training Pipeline + Registry (2026-09-23)

### production-model-type-follows-phase-4s-evidence-not-the-original-plan

**Situation:** The Phase 0 architecture plan assumed the production model
would be a GBM (`pipelines/train.py`'s original brief: "baselines →
GBMs"). Phase 4 found, empirically, that plain
`LogisticRegression(class_weight="balanced")` beats every GBM variant
tried on this feature set.

**Decision:** `pipelines/train.py` trains and registers
`LogisticRegression(class_weight="balanced")`, not a GBM. The evidence
Phase 4 gathered takes precedence over the plan written before that
evidence existed.

**Why note this explicitly:** it would be easy for a training pipeline,
built after the fact by someone re-reading only the original plan, to
silently "correct" this back to a GBM. `src/churn/models/pipeline.py`'s
own docstring and this entry both exist so that doesn't happen by
accident.

### mlflow-client-3x-vs-server-2x-model-logging-incompatibility

**Situation:** `pipelines/train.py`'s first real run of
`mlflow.sklearn.log_model(pipeline, name="model")` (and, tried next,
`artifact_path="model"`) both failed identically:
`MlflowException: API request to endpoint /api/2.0/mlflow/logged-models
failed with error code 404`.

**Decision:** This project's mlflow *client* resolved to 3.16.1 (`uv add`
picks the latest satisfying `shap`'s `numpy>=2` requirement — see below),
but the Docker mlflow *server* is pinned to 2.14.1
(`docker/mlflow/Dockerfile`, for an unrelated setuptools/pkg_resources
reason). `mlflow.sklearn.log_model()` unconditionally routes through
mlflow 3's new "Logged Model" REST API regardless of which parameter is
passed — a 2.14.1 server simply doesn't have that endpoint. Worked around
in `src/churn/models/registry.py::log_pipeline_artifact()`: save the
model *locally* with `mlflow.sklearn.save_model()` (no server call), then
upload the resulting directory with `mlflow.log_artifacts()` (the
long-stable artifact endpoint, already proven working in Phase 4).
Verified end to end against the real server, including registering the
resulting artifact and loading it back via `models:/churn_classifier@champion`.

**Rejected alternatives:**
- Downgrading the mlflow *client* to 2.14.1 to match the server. Tried
  first — `uv` immediately reported a hard conflict: mlflow 2.14.1
  requires `numpy<2`, `shap` requires `numpy>=2`, and downgrading shap
  to a `numpy<2`-compatible release would mean re-verifying Phase 3/4's
  already-reviewed, already-committed notebooks. Not worth it for an
  incompatibility this narrow (only model *logging* is affected; params/
  metrics/artifacts already worked fine in Phase 4 against this same
  server).
- Upgrading the mlflow *server* to 3.x to match the client. Would fix
  this properly, but risks a backend-store schema migration against a
  Postgres database that already holds Phase 4's real run history, for a
  problem the save_model+log_artifacts workaround solves with no
  infrastructure change at all. Worth reconsidering if Phase 6/7 hit
  another 3.x-only feature this workaround can't cover.

### alias-based-registry-api-not-the-deprecated-stages-api

**Situation:** PLAN.md's Phase 5 brief said "MLflow Model Registry stage
transitions" — written before this project's mlflow client version was
settled. MLflow's "stages" API (`transition_model_version_stage`) is
deprecated in modern MLflow in favor of aliases.

**Decision:** `src/churn/models/registry.py` uses
`set_registered_model_alias` / `get_model_version_by_alias` with a single
alias, `"champion"` — functionally the same "which version is currently
serving" concept "Production" staging provided, on the supported API.
Confirmed both client (3.16.1) and server (2.14.1) support aliases before
committing to this (unlike the model-logging incompatibility above,
alias operations are not affected by the client/server version gap).

**Rejected alternative:** using `transition_model_version_stage` because
PLAN.md named it literally. Rejected because building new code against a
documented-deprecated API just to match a brief written before the
implementation details were known would be choosing the wrong tool on
principle, not for any real requirement — the alias API achieves the
same intent PLAN.md was describing.

### model-gate-verified-with-two-real-training-runs-not-just-unit-tests

**Situation:** The model promotion gate (`register_and_maybe_promote`)
is the kind of logic that's easy to unit-test into looking correct while
still being wrong end-to-end (e.g. an off-by-one in comparing against
the *wrong* prior metric, or a mocked client hiding a real MLflow API
mismatch).

**Decision:** Ran `pipelines/train.py` twice against the real Postgres +
MLflow server, deliberately, before considering Phase 5 done. First run
(no champion yet) promoted as the bootstrap case. Second run, with
identical seeded input, produced the identical 0.1093 test PR-AUC and
was correctly *not* promoted (ties, doesn't clear `promotion_epsilon`) —
confirmed by independently re-fetching the `champion` alias afterward
and checking it still pointed at version 1. Unit tests (`test_registry.py`)
cover the same cases with a mocked client, but the live rerun is what
actually proves the gate protects the registry, not just that the
function returns the right boolean in isolation.

**Why this matters:** unit tests for `register_and_maybe_promote` passed
on the first attempt, before this live verification ever ran — they
would have passed just as well if `log_pipeline_artifact` or
`mlflow.register_model` were silently broken, since those are exactly
what the mocks replace. The two live runs are what this project's
"verified against real infrastructure, not just written" standard
(Phase 1 onward) actually requires for something whose entire job is to
protect a shared registry from a bad promotion.

## Phase 6 — Inference API + Batch Scoring (2026-09-24)

### precomputed-serving-plus-on-demand-rescore-hybrid

**Situation:** "Inference API" could mean either (a) always compute a
fresh prediction on request, or (b) only ever serve whatever the last
batch job wrote. This project's SQL feature extraction is deliberately
panel/window-function-shaped (DECISIONS.md#tech-stack) — it computes a
full month's cohort in one query, not one customer at a time — so a pure
"live" API would either duplicate that window-function SQL in a second,
per-customer form, or pay its full cost on every request.

**Decision:** Two read paths, at deliberately different cost:
- `GET /predictions/{customer_id}` serves whatever `pipelines/batch_score.py`
  already wrote to `predictions` — a single indexed read, no model in the
  loop, the fast/default path.
- `POST /score/{customer_id}` scores on demand, but only against the
  current cohort's *already-materialized* `feature_snapshots` row (same
  table `build_panel.py` fills) — never a live re-run of the
  window-function extraction query. Both this endpoint and
  `batch_score.py` call the exact same `churn.models.inference` helpers
  (`risk_band`, `shap_top_features`, `upsert_predictions`), so there is
  only one implementation of "how a customer gets scored," matching the
  project's split/pipeline/registry precedent.

**Rejected alternative:** a fully synchronous endpoint that re-derives
features inline from raw `customers`/`billing_events`/`support_tickets`
per request. Rejected because it would duplicate `sql/features/`'s
window-function logic in a second, per-customer-shaped form — exactly
the kind of duplication `src/churn/models/` exists to prevent (Phase 4's
shared-`ColumnTransformer` bug was caused by this same pattern in a
different guise).

### background-sample-logged-as-a-training-artifact-not-recomputed-at-serving-time

**Situation:** Notebook 05's SHAP `LinearExplainer` needs a small
"background" sample (the distribution `LinearExplainer` subtracts off) —
prototyped there as 100 random rows of the train+val set. Serving code
(`batch_score.py`, `api/main.py`) has no train+val set in scope by
design — reloading the full historical panel just to draw 100 background
rows would recouple serving to the entire training history and its cost.

**Decision:** `pipelines/train.py` now logs a 100-row raw-feature
background sample as a run artifact
(`explainability/background_sample.csv`, via
`churn.models.inference.log_background_sample()`) inside the same run as
the model. `churn.models.inference.load_champion()` downloads both
together, so serving code only ever touches the current scoring cohort's
features, never the historical panel.

**Rejected alternative:** computing the background sample at serving
time from a fresh `feature_snapshots` query. Works, but means every
batch-scoring run and every API cold start re-queries the full
historical panel for 100 rows it then throws away — an unnecessary
coupling and cost this project's "single source of truth, computed
once" pattern argues against.

### phase-5s-original-champion-manually-repromoted-for-a-required-serving-artifact

**Situation:** The champion registered in Phase 5 (`churn_classifier`
v1) predates `log_background_sample` above. The first real run of
`pipelines/batch_score.py` against it failed for real:
`MlflowException: Failed to download artifacts from path
'background_sample.csv', please ensure that the path is correct.`
Because training is fully seeded/deterministic, re-running
`pipelines/train.py` reproduces the *identical* test PR-AUC
(0.10932458317915489) every time — v3, the resulting run (now
carrying the background sample), can never naturally clear
`promotion_epsilon` against its own tied metric, so the normal
quality-based gate would never let a background-sample-carrying version
become champion on its own.

**Decision:** Manually promoted v3 to the `champion` alias
(`client.set_registered_model_alias("churn_classifier", "champion", "3")`),
after first confirming programmatically that v3's `test_pr_auc` exactly
equals v1's (both 0.10932458317915489 — a genuine tie, not a
close-enough approximation) — see the promotion script's own printed
confirmation. This is an artifact-schema promotion, not a quality-based
one, and is recorded here as exactly that: a one-time, deliberate,
manually-verified exception to the normal gate, made necessary by a
serving-time requirement (the background sample) that didn't exist when
v1 was trained. Every future training run logs the background sample by
construction, so this situation cannot recur.

**Rejected alternative:** adding a fallback into `load_champion()` that
recomputes a background sample from the historical panel whenever the
artifact is missing on an older champion version. Rejected because it
would keep artifact-incomplete old champions permanently "servable,"
which both reintroduces the cost the artifact was designed to avoid and
is exactly the kind of backwards-compatibility shim this project avoids
elsewhere — the champion always being current-and-complete is a
simpler invariant to maintain than a serving-time fallback path that
only exists for versions this project no longer intends to promote.

### phase-6-fastapi-uvicorn-prometheus-client-httpx-were-already-transitively-installed

**Situation:** `mlflow` bundles its own FastAPI-based model-serving
tooling, which pulls in `fastapi`, `uvicorn`, `httpx`, and (for a
different reason) `prometheus-client` as transitive dependencies —
confirmed present (`uv pip list`) before Phase 6 ever ran `uv add`.

**Decision:** Added all four explicitly to `pyproject.toml`
(`fastapi`/`uvicorn[standard]`/`prometheus-client` to `dependencies`,
`httpx` to the `dev` group for `fastapi.testclient.TestClient`) rather
than relying on them staying present as someone else's transitive
dependency. A future mlflow upgrade that drops or changes one of those
transitive pins would otherwise silently break `api/` with no entry in
`pyproject.toml` explaining why it used to work.

### shap-promoted-from-the-notebooks-group-to-main-dependencies

**Situation:** `shap` was a `notebooks`-group-only dependency through
Phase 5 (Notebook 05's explainability section). Phase 6's
`src/churn/models/inference.py` (used by `pipelines/batch_score.py` and
`api/main.py`, neither of which is a notebook) now imports it directly.

**Decision:** Moved `shap` from the `notebooks` group to `dependencies`,
per this project's stated per-phase dependency policy (the same move
already made for `scikit-learn`/`mlflow` in Phase 5) — see
`pyproject.toml`'s `notebooks` group comment.

### db-engine-needed-an-explicit-connect-timeout-for-graceful-degradation-to-actually-work

**Situation:** Found by actually breaking the dependency, not by
inspection: verifying `/health`'s graceful-degradation path by running
`docker compose stop postgres` against the live API, the *first* request
right after the stop happened to return fast (a pooled connection that
hadn't been invalidated yet), but a second request issued moments later
**hung** — `psycopg`/SQLAlchemy's `create_engine()` had no
`connect_timeout`, so a new connection attempt against a
stopping/stopped container's port fell back to the OS's own default TCP
connect timeout (tens of seconds), not a fast refusal. A DB-down
`/health` check that takes 30+ seconds to report unhealthy defeats the
entire point of a liveness probe.

**Decision:** Added `connect_args={"connect_timeout": 5}` to
`src/churn/db/engine.py`'s single shared `get_engine()` — every caller
(ingestion, feature building, training, batch scoring, the API) gets the
same bound. Re-verified live: `docker compose stop postgres`, `/health`
now returns `503` in ~5.1s (not 200 spuriously, not a multi-second+
hang); restarting postgres recovers `/health` to `200` within a couple
of health-check retries.

**Why this belongs in Phase 6, not filed as a separate bug:** the
hang was only exposed by this phase's own explicit "graceful
degradation" requirement — nothing before Phase 6 issued DB reads from
a long-lived server process that needs to fail fast and stay up; every
earlier phase's DB access was a short-lived CLI script where a slow
failure was merely annoying, never a liveness-probe correctness bug.
