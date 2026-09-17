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

### Phase 2 — SQL features + data validation — Not started

- `sql/features/`: parameterized (`reference_date`) CTE/window-function
  queries — RFM, tenure buckets, billing trend slope, ticket velocity,
  payment-failure streaks. Every query documents how it prevents leakage.
- Migration adding the finalized typed feature columns to
  `feature_snapshots` (deferred from Phase 1 on purpose — see DECISIONS.md).
- `sql/validation/`: nulls, duplicates, referential integrity, range
  checks, class balance, and the point-in-time assertion
  (`max(event_date) < reference_date` per snapshot).
- Pandera schema contracts wired to the validation gate.

### Phase 3 — Notebooks 01–03 — Not started

- `01_data_audit.ipynb`, `02_eda_statistics.ipynb`,
  `03_feature_engineering.ipynb`. Empirical confirmation of the panel's
  class imbalance, leakage audit, feature selection.

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
