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
