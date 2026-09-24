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
  train = first 12 months, validation = next 3, test = last 3 **as
  materialized**. The *modeling* panel used from Phase 3 on drops the
  structurally-censored last cohort (see
  [DECISIONS.md#last-cohort-dropped-from-modeling-panel](DECISIONS.md#last-cohort-dropped-from-modeling-panel)),
  so in every notebook from Phase 4 on, test is actually the last **2**
  cohorts (`2024-04-01`/`2024-05-01`, 10,328 rows) — this was flagged in
  Notebook 04's own markdown at the time but not reflected here until
  now.
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

### Phase 4 — Notebooks 04–06 — **DONE**

- `04_modelling_experiments.ipynb`: time-based train(12mo)/val(3mo)/test
  split on the modeling panel; baselines (`DummyClassifier(strategy="prior")`,
  `LogisticRegression(class_weight="balanced")`); an empirical comparison
  of class weighting vs. SMOTE (`SMOTENC`, categorical-aware) vs. plain,
  per
  [DECISIONS.md#imbalance-is-designed-in](DECISIONS.md#imbalance-is-designed-in);
  Optuna hyperparameter tuning of the winning GBM variant nested inside
  `GroupKFold(groups=customer_id)` on the training window only; every run
  logged to MLflow.
- `05_evaluation_explainability.ipynb`: refits on train+val, evaluates on
  the untouched test set; PR-AUC-first metrics, calibration (reliability
  diagram + Brier score), a cost-based decision threshold from
  `config/config.yaml`'s actual retention economics, and SHAP
  (`LinearExplainer` for the primary model, `TreeExplainer` for the GBM),
  including a top-3-features-per-prediction extraction matching
  `predictions.shap_top3`'s shape for Phase 6.
- `06_business_impact.ipynb`: decile lift / cumulative gains chart, a
  4-strategy campaign simulation, a sensitivity grid over the two
  illustrative economic assumptions, and a concretely-sized A/B test
  design (power analysis grounded in this model's actual flagging rate).
- **Verify — all three notebooks executed end-to-end, GBM baseline
  library switched from the classic `GradientBoostingClassifier` to
  scikit-learn's `HistGradientBoostingClassifier` mid-phase for tractable
  Optuna runtimes (see
  [DECISIONS.md#histgradientboosting-instead-of-classic-gbm](DECISIONS.md#histgradientboosting-instead-of-classic-gbm)):**
  - **Honest result, reported as found:** plain
    `LogisticRegression(class_weight="balanced")` beat every GBM variant
    tried, tuned or not — val PR-AUC **0.1281** vs. the best tuned GBM's
    **0.0609** (Notebook 04); confirmed on the held-out test set, **0.1093**
    vs. **0.0593** (Notebook 05). See
    [DECISIONS.md#logistic-regression-beats-every-gbm-variant](DECISIONS.md#logistic-regression-beats-every-gbm-variant).
  - SMOTE won among the three GBM imbalance-handling techniques (val
    PR-AUC 0.0606 vs. 0.0529 plain vs. 0.0483 class-weighted); threshold
    tuning separately improved the plain GBM's F1 from 0.0229 to 0.1183
    at the F1-optimal cutoff (0.096). (A post-review correction — see
    [DECISIONS.md#shared-columntransformer-corrupted-notebook-04s-threshold-tuning-numbers](DECISIONS.md#shared-columntransformer-corrupted-notebook-04s-threshold-tuning-numbers).)
  - Test-set decile lift: top decile churns at **7.07%** vs. 1.58%
    overall — a **4.5x** lift, concrete evidence the ranking is real even
    though PR-AUC is modest.
  - Cost-based threshold (from `config.yaml`'s actual retention
    economics): business-optimal threshold **0.940**, expected value
    **$1,577** on the test window. At the *configured* `risk_bands`
    (0.3 / 0.7) the same economics are **strongly net-negative**
    (-$195,812 / -$48,267) — the default risk bands need recalibrating
    before Phase 6 uses them for anything real.
  - A/B test design: at the business-optimal threshold the model flags
    only ~16 customers/month, so a properly powered randomized trial
    (1,089/arm) would take **~137 months** to accumulate — a genuine,
    unglamorous finding about this specific model/threshold combination,
    not glossed over.
  - A real Docker networking bug was found and fixed along the way (MLflow
    couldn't reach MinIO to log artifacts) — see
    [DECISIONS.md#mlflow-s3-endpoint-url-must-not-come-from-the-host-facing-env-var](DECISIONS.md#mlflow-s3-endpoint-url-must-not-come-from-the-host-facing-env-var).
  - Notebook hygiene checked (claude-kit `jupyter-hygiene`): no secrets,
    `execution_count` runs 1..n in every notebook, every source of
    randomness seeded. Outputs kept, same rationale as Phase 3.
  - Independently reviewed post-hoc (a fresh agent, no prior context,
    briefed to be adversarial) before Phase 5 started: found and this
    project fixed one real correctness bug — see
    [DECISIONS.md#shared-columntransformer-corrupted-notebook-04s-threshold-tuning-numbers](DECISIONS.md#shared-columntransformer-corrupted-notebook-04s-threshold-tuning-numbers)
    — plus a dead conditional in Notebook 05's risk-band finding (fixed)
    and a minor LTV double-counting caveat in Notebook 06's illustrative
    comparison rows (documented, doesn't affect the reported strategy).
    Confirmed clean: no leakage, correct `GroupKFold` usage, the
    cost-based-threshold arithmetic, the A/B sample-size calculation, and
    every other headline number in this section.

### Phase 5 — Training pipeline + registry — **DONE**

- `pipelines/train.py` trains the model type Phase 4's evidence actually
  supports — `LogisticRegression(class_weight="balanced")`, not the GBM
  the Phase 0 plan assumed — see
  [DECISIONS.md#production-model-type-follows-phase-4s-evidence-not-the-original-plan](DECISIONS.md#production-model-type-follows-phase-4s-evidence-not-the-original-plan).
  `src/churn/models/{split,pipeline,registry}.py` hold the reusable
  split/preprocessing/registry logic as the single source of truth (no
  longer duplicated per-notebook, closing the exact gap that caused
  Phase 4's shared-`ColumnTransformer` bug).
- **Data validation gate**: `validate_training_panel` (Pandera, fail
  loudly) runs on the modeling panel before any split or fit.
- **Model gate**: promotes to the registry's `champion` alias only if
  test PR-AUC beats the current champion's by >=
  `config.yaml`'s `promotion_epsilon` (0.005), or if there is no
  champion yet. Uses MLflow's alias API, not the deprecated "stages" API
  PLAN.md originally named — see
  [DECISIONS.md#alias-based-registry-api-not-the-deprecated-stages-api](DECISIONS.md#alias-based-registry-api-not-the-deprecated-stages-api).
- **Single sklearn Pipeline artifact**: one `Pipeline` (preprocessing +
  model) logged per run via `mlflow.sklearn.save_model` +
  `mlflow.log_artifacts` — not `mlflow.sklearn.log_model()`, which this
  project's mlflow *client* (3.x) can't use against its mlflow *server*
  (pinned to 2.14.1) — see
  [DECISIONS.md#mlflow-client-3x-vs-server-2x-model-logging-incompatibility](DECISIONS.md#mlflow-client-3x-vs-server-2x-model-logging-incompatibility).
- **Model card**: generated per run, logged as `model_card.md` on the
  run — model type and why, features and their Phase 3 provenance,
  training window, test metrics, Phase 4's known limitations (risk bands
  need recalibrating, A/B accumulation time, GroupKFold-vs-forward-PR-AUC
  caution), and the promotion outcome.
- **Verify — run against the real Postgres + MLflow server, twice, not
  just written:**
  - First run (bootstrap, no prior champion): panel validated at 82,343
    rows (matches Phase 3/4 exactly); split 57,021/14,994/10,328 train/val/test
    (matches Phase 4 exactly); test PR-AUC **0.1093** (matches Notebook 05
    exactly, confirming the production code reproduces the notebook
    result); registered as `churn_classifier` v1; promoted (bootstrap
    case, no champion to compare against).
  - Second run (identical, seeded, deterministic): produced the *same*
    0.1093 test PR-AUC; registered as v2; **correctly not promoted** (ties
    the champion, doesn't clear `promotion_epsilon`) — the `champion`
    alias was confirmed still pointing at v1 afterward. This is the gate
    actually working, not just present in the code.
  - Model card, registered versions, and the `champion` alias all
    independently re-fetched and verified readable/loadable after each
    run (`mlflow.sklearn.load_model("models:/churn_classifier@champion")`
    returns a working `sklearn.pipeline.Pipeline`).
  - `scikit-learn` and `mlflow` moved from the `notebooks` dependency
    group to `dependencies` now that `src/churn/models` actually imports
    them, per this project's own stated per-phase dependency policy
    (`pyproject.toml`'s `notebooks` group comment).
  - 6 new unit tests for `split.py`/`pipeline.py` (100% coverage) and 5
    for `registry.py`'s gate logic (mocked `MlflowClient`, no real
    server needed) — including a regression test reproducing the exact
    shared-`ColumnTransformer` bug class from Phase 4's review, and a
    regression test for the "identical rerun must not re-promote itself"
    case the second live run above exercised for real.
  - `ruff`, `black --check`, `mypy --strict` clean on `src` (added a
    `sklearn.*`/`mlflow.*` mypy override — neither ships a `py.typed`
    marker, first exercised here). Full suite: **39/39 tests passing**
    (unit + testcontainers integration).

### Phase 6 — Inference API + batch scoring — **DONE**

- `src/churn/models/inference.py` — the single source of truth
  `pipelines/batch_score.py` and `api/main.py` both build on: loads the
  registry's `champion` pipeline + a logged background sample
  (`load_champion`), computes the scoring cohort (the complement of
  Phase 5's `modeling_panel` — the `reference_date == snapshot_date` rows
  whose label hasn't matured), risk-bands a probability, extracts top-3
  SHAP features per row (`LinearExplainer`, matching Notebook 05's
  prototype exactly), and upserts into `predictions`.
- `pipelines/batch_score.py` scores the entire current cohort in bulk and
  writes one row per customer to `predictions`, idempotently (`ON
  CONFLICT (customer_id, reference_date, model_version) DO UPDATE`).
- `api/main.py` (FastAPI): two read paths at deliberately different cost
  — see
  [DECISIONS.md#precomputed-serving-plus-on-demand-rescore-hybrid](DECISIONS.md#precomputed-serving-plus-on-demand-rescore-hybrid):
  - `GET /predictions/{customer_id}` — serves the latest precomputed row,
    DB-only, no model in the loop.
  - `POST /score/{customer_id}` — on-demand rescore against the
    customer's already-materialized current-cohort feature row (never a
    live re-run of the window-function SQL).
  - `GET /health` — DB + model liveness.
  - `GET /metrics` — Prometheus (request count/latency by method+path,
    served-probability histogram).
  - Request-ID correlation: every request gets an `X-Request-ID`
    (echoed if the caller sent one, generated otherwise), bound into
    structlog's contextvars so every log line for that request carries
    it.
  - **Graceful degradation**: the champion model loads once at FastAPI
    startup, not per-request. If MLflow/the registry is unreachable at
    startup, the API still comes up — `/predictions` and `/health` keep
    working off Postgres alone, and only `/score` 503s until a model
    loads.
- **Verify — run against the real Postgres + MLflow server, including
  deliberately breaking things, not just the happy path:**
  - **A required re-promotion, found live:** the champion registered in
    Phase 5 (`churn_classifier` v1) predates the background-sample
    artifact SHAP serving needs — the first real `batch_score.py` run
    against it failed for real (`Failed to download artifacts from path
    'background_sample.csv'`). Re-ran `pipelines/train.py` (fully
    seeded/deterministic — reproduced the *identical* test PR-AUC,
    **0.10932458317915489**, exactly matching v1) to produce v3, which
    now carries the artifact; confirmed the tie programmatically, then
    manually promoted v3 to `champion` — an artifact-schema promotion,
    not a quality-based one, documented in full at
    [DECISIONS.md#phase-5s-original-champion-manually-repromoted-for-a-required-serving-artifact](DECISIONS.md#phase-5s-original-champion-manually-repromoted-for-a-required-serving-artifact).
  - **Batch scoring**: scored the real current cohort — **5,174**
    customers (`reference_date=2024-06-01`) — risk bands **low=2,841 /
    medium=1,745 / high=588**. Re-ran a second time: identical row count
    and distribution, confirmed via direct SQL (`count(*)=5174`,
    `count(DISTINCT customer_id)=5174`) that the upsert overwrote rather
    than duplicated.
  - **API, live**: `GET /health` → `200 {"status":"ok","database":"ok","model":"loaded","model_version":"3"}`;
    `GET /predictions/{known}` → `200` with real SHAP top-3 (e.g. the
    highest-risk customer, probability 0.97437, top feature
    `num__frequency_tickets_30d`); `GET /predictions/{unknown}` → `404`;
    `POST /score/{customer}` → recomputed **0.974369536252729** for the
    same customer batch-scored at 0.97437 (same model, same features —
    on-demand and batch scoring agree, as they must, since both call the
    same `inference.py` functions); request-ID echoed when supplied,
    generated when absent; `/metrics` scrapeable.
  - **Graceful degradation, live, not just unit-tested**: `docker compose
    stop postgres` while the API kept running → `/health` returned `503
    {"database":"unreachable",...}`; `docker compose start postgres` →
    `/health` recovered to `200` within a couple of retries. This exposed
    a real bug along the way — see next bullet.
  - **A real bug found and fixed via this same live test**: the DB
    engine had no connect timeout, so a request issued while Postgres
    was mid-restart *hung* (falling back to the OS's default TCP connect
    timeout) instead of failing fast — defeating the point of a health
    check. Fixed by adding `connect_args={"connect_timeout": 5}` to
    `src/churn/db/engine.py`'s shared engine; re-verified live,
    `/health` now bounds at ~5.1s instead of hanging. See
    [DECISIONS.md#db-engine-needed-an-explicit-connect-timeout-for-graceful-degradation-to-actually-work](DECISIONS.md#db-engine-needed-an-explicit-connect-timeout-for-graceful-degradation-to-actually-work).
  - **Dependencies**: `shap` moved from the `notebooks` group to
    `dependencies` (same per-phase policy as Phase 5's
    scikit-learn/mlflow move); `fastapi`/`uvicorn[standard]`/
    `prometheus-client` added explicitly (were already present
    transitively via mlflow's own bundled server — see DECISIONS.md);
    `httpx` added to the `dev` group for `fastapi.testclient.TestClient`.
  - 18 new unit tests: 9 for `inference.py` (scoring-cohort/risk-band
    boundary cases, plus a real — not mocked — SHAP computation on a toy
    fitted pipeline) and 9 for the API (health/predictions/score across
    the happy path, not-found, and DB-unreachable cases, request-ID
    behavior, `/metrics`). `ruff`, `black --check`, `mypy --strict` clean
    on `src`, `pipelines`, and `api` (lint/format/type-check commands
    widened to cover all three, matching the fact that `pipelines/`
    already existed since Phase 5 but was never actually added to the
    documented lint scope until now).
- **Independently reviewed post-hoc** (a fresh agent, no prior context,
  briefed to be adversarial, with Docker access to verify live rather
  than just read code) before Phase 7 started. Found and this project
  fixed one real blocker and eight further real issues, all verified
  live — see
  [DECISIONS.md#shap-values-must-aggregate-to-source-feature-not-stay-one-hot-column-per-entry](DECISIONS.md#shap-values-must-aggregate-to-source-feature-not-stay-one-hot-column-per-entry)
  for the full list:
  - **Blocker**: `shap_top3` named one-hot dummy columns individually
    (e.g. both `cat__contract_type_Two year` and
    `cat__contract_type_Month-to-month` for the same customer), reading
    as contradictory nonsense and affecting every one of the 5,174
    already-stored rows — this exact field is what Phase 7's dashboard
    will show. Fixed by aggregating SHAP values to source features
    before ranking (exact for a linear model), and switched storage
    from a dict (loses rank order on JSONB round-trip, also found live)
    to an ordered list.
  - A real scale limit (`upsert_predictions` broke above ~10,922 rows —
    Postgres's per-statement bound-parameter cap), now chunked.
  - A real freshness gap (the API never picked up a newly promoted
    champion without a restart, and could shadow a newer batch-scored
    row with a stale one) — now checked and reloaded on every `/score`
    call.
  - A TOCTOU-shaped bug in `load_champion` (resolved the champion alias
    twice), an inconsistent probability/risk_band rounding boundary, an
    inconsistent 500-vs-503 on a DB failure mid-`/score`, an unbounded
    Prometheus label cardinality on unmatched routes, and
    `batch_score.py` loading the full historical panel instead of
    filtering in SQL — all fixed and re-verified live.
  - Confirmed clean/ruled out (verified, not assumed): SQL injection
    surface, `RETURNING` ordering (doesn't matter here — every row in
    one statement shares its `now()`-based `scored_at`), `Decimal`→float
    coercion through the API, `partial`-failure atomicity of the batch
    upsert, and the two fixes from the phase's own first live
    verification pass (the champion re-promotion, the DB connect
    timeout).
  - Re-ran `pipelines/batch_score.py` against the real 5,174-customer
    cohort after every fix; risk band counts unchanged (low=2,841/
    medium=1,745/high=588, as expected — the model didn't change, only
    the serving/storage logic did); spot-checked `shap_top3` against
    real `contract_type` values to confirm the aggregation fix.
  - 12 new/changed unit tests covering the fixes (SHAP aggregation +
    ranking regression tests, upsert chunking, a successful `/score`
    path and a champion-reload path — neither existed before the
    review — and a real `decimal.Decimal` in the `/predictions` test
    fixture, which the original test's plain `float` never exercised).

### Phase 7 — Monitoring + dashboard — Not started

- `src/churn/monitoring/` (PSI/KS/KL drift, delayed-label performance
  decay), Prefect DAGs, `dashboards/` Streamlit app + static report
  generator.

### Phase 8 — Tests, CI, docs — Not started

- Coverage target (>80% on `src/`), pre-commit, GitHub Actions, README
  finalization, runbook.
