# PLAN.md — Churn Prediction System

End-to-end, production-style customer churn prediction system: Postgres →
idempotent ingestion → SQL feature extraction → notebook experimentation →
training pipeline → serving API → monitoring.

This file tracks **what** is being built and its status. See
[DECISIONS.md](DECISIONS.md) for **why** each non-obvious choice was made —
every entry here that needed a judgment call links to a decision record
there.

Work proceeds one phase at a time; each phase stops for review before the
next starts.

## Confirmed assumptions (signed off 2026-09-17)

- `SNAPSHOT_DATE = 2024-06-01` — the fixed "today" of the synthetic world.
- Monthly scoring panel spans 18 cohorts, `2023-01-01` … `2024-06-01`:
  train = first 12 months, validation = next 3, test = last 3.
- Prediction target: `churn_next_30d` — will the customer churn within 30
  days of `reference_date`.
- Orchestrator: Prefect 2 (see DECISIONS.md — orchestrator-choice).
- Retention-offer economics (offer cost, assumed LTV horizon, offer success
  rate) are illustrative, explicitly-labeled assumptions in
  `config/config.yaml`, not derived from any real data — the dataset has no
  economics in it.
- Dataset source: IBM Telco Customer Churn CSV
  (`raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d`), with a
  verified same-size mirror on `treselle-systems/customer_churn_analysis`
  as a documented fallback if the primary goes dead.
- **The monthly panel is expected to be meaningfully imbalanced** — a low
  single-digit percent positive rate per month, well below the 26.6%
  *cumulative* churn rate reported for the raw dataset (that figure is
  churn-over-the-whole-relationship, not a monthly rate). Imbalance
  handling (class weights vs. SMOTE vs. threshold tuning) is mandatory and
  will be measured empirically, not assumed away. See
  [decisions.md#imbalance-is-designed-in](DECISIONS.md#imbalance-is-designed-in).

## Phases

### Phase 1 — Infra + DB + ingestion — **DONE**

- `docker-compose.yml`: Postgres 16, pgAdmin, MinIO (+ bucket bootstrap),
  MLflow tracking server (custom image, backed by Postgres + MinIO).
- Normalized schema (8 tables) as SQLAlchemy models, Alembic-migrated;
  `sql/ddl/schema.sql` is a generated, drift-free mirror (dumped straight
  from Alembic, never hand-edited).
- `src/churn/config.py`: Pydantic Settings (secrets, from `.env`) +
  Pydantic-validated YAML config (business/pipeline parameters).
- `src/churn/utils/logging.py`: structlog → stdlib logging → single-line
  JSON formatter, used everywhere (no `print()`).
- `src/churn/ingestion/`: idempotent download (filesystem cache + DB-level
  checksum skip), the synthetic temporal-layer generator (seeded, documented
  assumptions), and a COPY-based bulk loader with row-count assertions.
- Unit tests for the checksum helper, config validation, and the date-
  derivation logic; an integration test (testcontainers) exercising a real
  synthesize → load → FK-integrity round trip against a real Postgres.
- **Verify:** `uv sync`, `docker compose up -d postgres pgadmin minio minio-init mlflow`,
  `uv run alembic upgrade head`, `uv run python -m churn.ingestion.cli`,
  `uv run pytest`. All of this was actually run against real Postgres/
  MLflow/MinIO containers, not just written — see
  [DECISIONS.md#infra-issues-hit-during-verification](DECISIONS.md#infra-issues-hit-during-verification)
  for the four real bugs that surfaced and were fixed in the process.
  Result: 7,043 customers / 7,043 subscriptions / 238,200 billing events /
  50,066 support tickets / 7,043 churn labels loaded with zero orphaned
  FKs; churn class balance matches the known dataset (26.5% churned,
  1,869 customers, 1,376 of whose churn dates fall inside the 18-month
  panel window); re-running the CLI is a verified no-op (idempotent);
  `ruff check`, `black --check`, and `mypy --strict` all pass clean;
  `pytest` is 9/9 passing (unit + a real testcontainers integration test).

### Phase 2 — SQL features + data validation — **DONE**

- `sql/features/build_feature_snapshot.sql`: one parameterized
  (`reference_date`, `horizon_days`, `feature_set_version`) query — CTEs,
  `regr_slope`, `LAG`/`ROW_NUMBER` gaps-and-islands, conditional rolling
  aggregates — computing RFM, tenure, billing trend, ticket velocity, and
  payment-failure-streak features, plus the `label_churn_next_30d`
  target. Every feature CTE filters strictly `< reference_date`; the
  label deliberately looks forward (see
  DECISIONS.md#label-lookahead-is-not-leakage).
- Migration `18765fcc5439` adds 31 typed feature columns +
  `label_churn_next_30d` to `feature_snapshots` (deferred from Phase 1 on
  purpose — see DECISIONS.md). Autogenerate missed 3 new CHECK
  constraints; added by hand and verified by re-running autogenerate
  until it reported no diff (see
  DECISIONS.md#autogenerate-misses-check-constraints).
- `src/churn/features/extract.py` (`extract_feature_snapshot`) and
  `src/churn/features/build_panel.py` (CLI: `python -m
  churn.features.build_panel [--force]`) materialize the full monthly
  panel, idempotently per `(reference_date, feature_set_version)`.
- `src/churn/features/validation.py`: Pandera schema contracts
  (`validate_feature_snapshot`, `validate_training_panel`) — the data-
  validation gate Phase 5's `train.py` will wire in.
- `sql/validation/`: `nulls_and_duplicates.sql`,
  `referential_integrity.sql`, `range_checks.sql`, `class_balance.sql`,
  and `point_in_time_reconciliation.sql` (an adversarial re-check that
  independently recomputes two aggregates from source tables and diffs
  them against what's stored).
- **Verify — all run against the real, fully-loaded database:**
  - Full 18-month panel materialized: **87,517 rows**, cohort sizes
    4,573–5,242/month.
  - `nulls_and_duplicates.sql`, `referential_integrity.sql`,
    `range_checks.sql`: **zero rows** (clean) on every query.
  - `point_in_time_reconciliation.sql`: **zero mismatches** across all
    87,517 rows — independent proof of no leakage, not just an assertion
    in a comment.
  - `class_balance.sql`: monthly positive rate **1.27%–2.02%** (panel-wide
    1.55%), confirming the imbalance predicted in Phase 0/1 — see
    DECISIONS.md#imbalance-is-designed-in. One caveat found and
    documented: the very last cohort (`reference_date = SNAPSHOT_DATE`)
    structurally has 0 positives — a synthetic-data censoring artifact,
    not a bug (DECISIONS.md#last-cohort-has-a-structural-zero-positive-rate).
  - 5 new integration tests with hand-crafted fixtures and exact expected
    values, including a dedicated point-in-time leakage regression test
    (a billing event dated after `reference_date` that must never be
    counted) and a brand-new-customer null-handling test.
  - `ruff`, `black --check`, `mypy --strict`: clean. Full suite: **23/23
    tests passing**.

### Phase 3 — Notebooks 01–03 — **DONE**

- `01_data_audit.ipynb`: independent pandas-side audit of the raw CSV and
  the synthetic temporal layer against `config/config.yaml`'s documented
  generator assumptions — distinct from `sql/validation/*.sql` (which
  checks internal DB consistency, not "does the generator behave as
  documented").
- `02_eda_statistics.ipynb`: empirically confirms the panel's class
  imbalance (re-measured independently in pandas, not just trusted from
  Phase 2's SQL), a leakage audit independent of
  `point_in_time_reconciliation.sql`, univariate/bivariate EDA, and
  mutual-information/chi²/Mann-Whitney statistical tests.
- `03_feature_engineering.ipynb`: 5 ratio/interaction features with
  explicit business rationale; feature *selection* via mutual information,
  permutation importance, a hand-rolled Boruta shadow-feature procedure
  (see
  [DECISIONS.md#hand-rolled-boruta-instead-of-the-boruta-package](DECISIONS.md#hand-rolled-boruta-instead-of-the-boruta-package)),
  and Spearman correlation pruning — selected subset written to
  `config/selected_features_v1.yaml` for Phase 5 to consume (see
  DECISIONS.md#feature-selection-vs-feature-extraction-clarified).
- **Verify — all three notebooks executed end-to-end against the real,
  fully-loaded database, not hand-typed:**
  - Notebook 01: raw-CSV audit found one known, harmless quirk (11 blank
    `TotalCharges` rows, all `tenure == 0`, irrelevant since the column is
    dropped at ingestion); `customers` confirmed an exact carry-through of
    the raw CSV; `signup_date`/`churn_date` placement confirmed consistent
    with each customer's raw tenure; churn-date lookback confirmed
    consistent with the documented uniform-over-24-months placement (KS
    test p=0.11); empirical payment-failure rates matched the configured
    per-segment formula within sampling noise in every segment; the
    pre-churn ticket-velocity spike was empirically measured at **2.30×**
    (configured: 2.5×).
  - Notebook 02: panel-wide positive rate **1.55%**, matching Phase 2's
    SQL figure exactly; leakage audit clean — 0 mismatches across 80,968
    consecutive-row tenure-trajectory checks, brand-new customers
    confirmed to show zero recent-charge history on their first snapshot,
    no single feature dominates the label's mutual information; 28/33
    candidate features significant at α=0.01.
  - Notebook 03: dropped the structurally-censored `reference_date ==
    SNAPSHOT_DATE` cohort (5,174 rows) per
    [DECISIONS.md#last-cohort-has-a-structural-zero-positive-rate](DECISIONS.md#last-cohort-has-a-structural-zero-positive-rate),
    leaving an 82,343-row modeling panel; baseline RandomForest (grouped
    train/val split, no customer in both) scored **PR-AUC 0.1184** vs. a
    0.0169 random-guess baseline; Boruta: 27 confirmed / 3 tentative / 14
    rejected; 12/44 candidate features selected by ≥2-of-3 majority vote
    (none of the 5 engineered ratio/interaction features made the cut —
    an honest empirical result, not an error: the raw RFM/categorical
    features carried more independent signal by all three methods).
  - `ruff`, `black --check`, `mypy --strict` on `src`/`tests`: unaffected
    (this phase touched only `notebooks/` and added
    `config/selected_features_v1.yaml`).
  - Notebook hygiene checked (claude-kit `jupyter-hygiene`): no secrets in
    outputs, `execution_count` runs 1..n in every notebook, all randomness
    seeded from `cfg.synthetic.random_seed`. Outputs were kept (not
    stripped) — these notebooks are this phase's evidence, in the same
    spirit as Phase 1/2's verify sections; `nbstripout` remains a Phase 8
    item per DECISIONS.md's `tech-stack` entry, not forgotten.
  - Two real environment bugs found and fixed during this phase, not
    swept under the rug — see
    [DECISIONS.md#nbconvert-cwd-breaks-env-file-resolution](DECISIONS.md#nbconvert-cwd-breaks-env-file-resolution)
    and
    [DECISIONS.md#pandas-3-default-string-dtype-breaks-dtype-object-checks](DECISIONS.md#pandas-3-default-string-dtype-breaks-dtype-object-checks).

### Phase 4 — Notebooks 04–06 — Not started

- `04_modelling_experiments.ipynb` (baselines → GBMs, time-based CV,
  Optuna, MLflow logging), `05_evaluation_explainability.ipynb`
  (PR-AUC-first evaluation, calibration, cost-based threshold, SHAP),
  `06_business_impact.ipynb` (decile lift, campaign simulation, A/B design).

### Phase 5 — Training pipeline + registry — Not started

- `pipelines/train.py`, data validation gate, model gate against current
  Production PR-AUC, single sklearn `Pipeline` artifact, MLflow Model
  Registry stage transitions, model card.

### Phase 6 — Inference API + batch scoring — Not started

- `pipelines/batch_score.py`, `api/` (FastAPI), Prometheus metrics,
  request-ID correlation, graceful degradation.

### Phase 7 — Monitoring + dashboard — Not started

- `src/churn/monitoring/` (PSI/KS/KL drift, delayed-label performance
  decay), Prefect DAGs, `dashboards/` Streamlit app + static report
  generator.

### Phase 8 — Tests, CI, docs — Not started

- Coverage target (>80% on `src/`), pre-commit, GitHub Actions, README
  finalization, runbook.
