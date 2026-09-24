# DEVELOPMENT.md

Step-by-step build log: every command actually run to get this repo from
an empty clone to its current state (Phases 1–7), in order, so it can be
reproduced from scratch. This is the "how to run it" doc — see
[PLAN.md](PLAN.md) for what each phase produced and the verified numbers,
[DECISIONS.md](DECISIONS.md) for why each non-obvious call was made, and
[CLAUDE.md](CLAUDE.md) for the condensed command/gotcha reference used
day to day. This file is maintained phase by phase as the project grows;
it will cover Phase 8 once it's built.

## Prerequisites

- Python via `uv` (no separate interpreter install needed — `uv` manages it)
- Docker Desktop (containers: postgres, pgadmin, minio, minio-init, mlflow)
- On Windows: Docker Desktop does not auto-start — launch it manually and
  wait for `docker info` to succeed before any `docker compose` step
- If the project directory lives inside OneDrive (or another cloud-sync
  folder), prefix every `uv sync`/`uv add`/`uv remove` with
  `UV_LINK_MODE=copy` — see CLAUDE.md's Gotchas for why

```bash
git clone <repo>
cd ml_proj
cp .env.example .env   # fill in real DB/MinIO passwords — never commit this file
```

---

## Phase 1 — Infra, DB schema, ingestion

```bash
uv sync && uv run pre-commit install

docker compose up -d postgres pgadmin minio minio-init mlflow
# on Windows, if this hangs/times out: launch Docker Desktop, wait for
# `docker info` to succeed, then retry

uv run alembic upgrade head          # creates schema from alembic/ migrations
uv run python -m churn.ingestion.cli # downloads IBM Telco dataset, synthesizes
                                      # dates, loads Postgres idempotently

uv run pytest tests/unit             # no Docker needed
uv run pytest                        # full suite incl. testcontainers (needs Docker)
```

Verify: 7,043 customers loaded, zero orphaned foreign keys, re-running
the ingestion CLI is a no-op (idempotent). Full numbers:
[PLAN.md#phase-1](PLAN.md).

## Phase 2 — SQL feature extraction + validation

```bash
uv run python -m churn.features.build_panel   # materializes feature_snapshots
```

This runs the parameterized queries in `sql/features/` across every
monthly `reference_date` in the configured panel window, then the
data-quality queries in `sql/validation/` against the result. No
separate CLI flag for validation — `build_panel` fails loudly if a
validation query returns a mismatch.

```bash
uv run pytest   # re-run full suite — includes the point-in-time
                 # reconciliation / leakage checks
```

Verify: 87,517-row panel, all validation queries clean, monthly churn
rate 1.27%–2.02%. Full numbers: [PLAN.md#phase-2](PLAN.md).

## Phase 3 — Notebooks: data audit, EDA, feature selection

```bash
UV_LINK_MODE=copy uv sync --group notebooks   # jupyter, nbconvert, ipykernel,
                                                # scipy, statsmodels, etc.
```

Run notebooks 01–03 in order (each depends on state/artifacts from the
previous one). Two equivalent ways to run them:

**Interactively** (recommended for a human reproducing this):
```bash
uv run jupyter lab
# open notebooks/01_data_audit.ipynb, Run All, repeat for 02, then 03
```

**Headlessly** (what CI / a script would do):
```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_audit.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_eda_statistics.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/03_feature_engineering.ipynb
```

On Windows, headless execution needs one extra fix or the kernel hangs
on startup (pyzmq's `add_reader` isn't supported on the default
`ProactorEventLoop`) — set this **before** nbconvert starts the kernel,
e.g. at the top of a wrapper script that then calls nbconvert's CLI
entrypoint:
```python
import asyncio
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
```
Also pass a longer startup timeout the first time (cold `ipykernel`
install): `--ExecutePreprocessor.startup_timeout=300`.

Every notebook's setup cell does `os.chdir(PROJECT_ROOT)` right after
importing it — required because `nbconvert` runs with cwd set to
`notebooks/`, and `Settings`' `.env` resolution is cwd-relative.

Verify: synthetic generator's statistical assumptions confirmed against
loaded data; independent leakage audit, 0 mismatches across 80,968
checks; 44 candidate features narrowed to 12, written to
`config/selected_features_v1.yaml`. Full numbers:
[PLAN.md#phase-3](PLAN.md).

## Phase 4 — Notebooks: modelling, evaluation, business impact

Depends on `config/selected_features_v1.yaml` from Phase 3. Same
execution options as Phase 3 (Jupyter Lab "Run All", or nbconvert
headless), run in order:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/04_modelling_experiments.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/05_evaluation_explainability.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/06_business_impact.ipynb
```

Verify: plain `LogisticRegression(class_weight="balanced")` beat every
`HistGradientBoostingClassifier` variant (test PR-AUC 0.1093 vs. 0.0593);
top-decile lift 4.5x; cost-based threshold analysis and A/B test sizing
in notebook 06. Full numbers: [PLAN.md#phase-4](PLAN.md).

> Note: `notebooks/` is exploration-only — nothing under `src/churn`
> imports from it. The production preprocessing+model pipeline lives in
> `src/churn/models/pipeline.py`, built in Phase 5 from what these
> notebooks found, not by importing the notebooks themselves.

## Phase 5 — Training pipeline + MLflow registry

Requires `mlflow` up (already started as part of `docker compose up -d`
above) and reachable at `http://localhost:5000`.

```bash
uv run python pipelines/train.py
```

This loads the panel, filters to the modeling window, validates it
(Pandera schema gate), splits train/val/test with `GroupKFold`-safe
time boundaries, fits `src/churn/models/pipeline.py`'s
`build_pipeline()`, evaluates on the held-out test set, logs the run to
MLflow, and registers+conditionally promotes the model via
`src/churn/models/registry.py`'s alias-based gate (`champion` alias,
promoted only if the candidate beats the current champion by
`promotion_epsilon`).

Run it twice to see the gate in action: the first run on an empty
registry always promotes (bootstraps the champion); an identical second
run should **not** promote, since it only ties rather than beats the
existing champion.

```bash
uv run pytest                                                         # 39/39 incl. Phase 5 unit tests
uv run ruff check src tests && uv run black --check src tests && uv run mypy src   # lint clean
```

Verify: first run PR-AUC 0.1093 (bootstraps champion, matches notebook
05 exactly); second identical run correctly not promoted. Full
verification: [PLAN.md#phase-5](PLAN.md).

## Phase 6 — Inference API + batch scoring

Requires a champion already registered (Phase 5) that carries the
`explainability/background_sample.csv` artifact `pipelines/train.py` now
logs alongside the model. If you're reproducing this project fresh
(training for the first time under current code), that artifact is
already there and you can skip straight to batch scoring. If you're
resuming from a Phase-5-only snapshot whose champion predates this
artifact, re-run training once first — see the note below.

```bash
uv run python pipelines/batch_score.py
```

Scores the current cohort (`feature_snapshots` rows where
`reference_date == snapshot_date` — the one cohort whose 30-day label
hasn't matured, i.e. "today" in the synthetic world) against the
registry's `champion` model, and idempotently upserts one row per
customer (risk band + top-3 SHAP features) into `predictions`. Safe to
re-run — a repeat run overwrites existing rows rather than duplicating
them.

```bash
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Starts the serving API. Try it against the cohort `batch_score.py` just
scored:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/predictions/<a customer_id from feature_snapshots>
curl -X POST http://localhost:8000/score/<same customer_id>
curl http://localhost:8000/metrics
```

```bash
uv run pytest                                                                       # 57/57 incl. Phase 6 unit tests
uv run ruff check src tests pipelines api && uv run black --check src tests pipelines api && uv run mypy src pipelines api   # lint clean
```

Verify: real cohort of 5,174 customers scored, risk bands
low=2,841/medium=1,745/high=588; on-demand `/score` and batch scoring
agree on the same customer (both call the same
`src/churn/models/inference.py` functions); `/health` returns 503 (not a
hang or a crash) when Postgres is stopped, and recovers once it's back.
Full verification: [PLAN.md#phase-6](PLAN.md).

> **If your champion predates the background-sample artifact** (i.e. you
> trained under Phase-5-era code before Phase 6 added
> `log_background_sample`), `batch_score.py` will fail with `Failed to
> download artifacts from path 'background_sample.csv'`. Fix: re-run
> `uv run python pipelines/train.py` (deterministic — it will reproduce
> the same test PR-AUC and *not* auto-promote, since it only ties the
> existing champion), then manually confirm the tie and move the
> `champion` alias to the new version — see
> [DECISIONS.md#phase-5s-original-champion-manually-repromoted-for-a-required-serving-artifact](DECISIONS.md#phase-5s-original-champion-manually-repromoted-for-a-required-serving-artifact)
> for the exact script and reasoning. This is a one-time situation —
> every training run from Phase 6 onward logs the artifact by
> construction.

## Phase 7 — Monitoring + dashboard

Requires Phase 6's champion + `predictions` table already populated
(`batch_score.py` at least once).

```bash
uv run python pipelines/backfill_predictions.py
```

Scores every *historical* cohort (every `feature_snapshots` row before
`snapshot_date`, i.e. everything `batch_score.py`'s current-cohort run
doesn't touch) with the current champion, and backfills
`actual_churned` straight from the already-matured
`feature_snapshots.label_churn_next_30d` — this is a retrospective
backtest (most cohorts were part of the champion's own training data),
not a live signal, but it's what gives the next command something real
to report on. Idempotent, same as `batch_score.py`.

```bash
uv run python pipelines/monitor.py
```

Logs two reports: feature drift (training window vs. current cohort —
PSI/KS/KL per feature, warning-level if PSI crosses
`config.yaml`'s `psi_warning_threshold`/`psi_critical_threshold`) and
delayed-label performance decay (PR-AUC/ROC-AUC/precision/recall *per
historical cohort*, each labeled train/validation/test/post-test so an
in-sample cohort's numbers are never mistaken for held-out
performance). Read-only — computes fresh from Postgres + the registry
every run, never writes anything back.

```bash
uv run streamlit run dashboards/app.py
# open http://localhost:8501 -- three tabs: current cohort, feature
# drift, performance over time
```

```bash
uv run python dashboards/report.py
# writes reports/monitoring_report.html (gitignored) -- the same
# computations as monitor.py/app.py, captured as a shareable static file
```

```bash
uv run python pipelines/flows.py
# runs the whole pipeline (ingest -> build panel -> train -> batch
# score -> backfill -> monitor) as one Prefect flow, in order
```

```bash
uv run pytest                                                                                  # 92/92 incl. Phase 7 unit + integration tests
uv run ruff check src tests pipelines api dashboards && uv run black --check src tests pipelines api dashboards && uv run mypy src pipelines api dashboards   # lint clean
```

Verify: backfilled 82,343 rows across 17 cohorts; `tenure_months` came
back with real, moderate drift (PSI 0.1881, reference window = train+val,
the window the champion was actually fit on) between the training
window and the current cohort, every other feature stable; per-cohort
performance honestly showed lower PR-AUC on the genuine held-out test
cohorts (`in_sample=False`, 0.11–0.12) than on the in-sample training
cohorts (`in_sample=True`, 0.09–0.20); the full Prefect flow ran end to
end for real, each stage's output correctly feeding the next (a tied,
non-promoted retrain correctly left every downstream stage using the
untouched champion). Full verification: [PLAN.md#phase-7](PLAN.md).

> **Post-review correction**: an independent review before Phase 8
> found the original drift binning silently zeroed out PSI on
> low-cardinality features (`frequency_charges_90d` read PSI ~0 when it
> should have read ~0.05) and that the reference window was train-only
> instead of train+val — both fixed; the numbers above are post-fix. See
> [DECISIONS.md#drift-binning-must-fall-back-to-exact-values-for-low-cardinality-features](DECISIONS.md#drift-binning-must-fall-back-to-exact-values-for-low-cardinality-features)
> for the full list of what else that review found and fixed.

> **Testing `dashboards/app.py` headlessly**: `curl`ing the running
> Streamlit server only proves the process is up — Streamlit executes
> the actual script per websocket session, so that alone can't catch an
> exception in the render code. Use
> `streamlit.testing.v1.AppTest.from_file(<absolute path>)` (`.run()`,
> then inspect `.exception`/`.metric`/`.dataframe`/etc.) for a real
> headless functional check instead.

---

## Common gotchas when reproducing this

These bite during the steps above specifically — see CLAUDE.md for the
full list:

- **OneDrive + `uv sync`**: hardlink install fails with `os error 396`.
  Always `UV_LINK_MODE=copy uv sync ...` if the repo is cloud-synced.
- **Docker Desktop not running**: `docker compose up` / any DB
  connection just times out. Check `docker info` first.
- **pandas 3.0's default string dtype**: any code checking
  `df[c].dtype == object` to detect string columns will silently
  misclassify them post-Phase-3 dependency versions — use
  `pd.api.types.is_numeric_dtype(...)` instead.
- **MLflow client (3.x) vs. server (pinned 2.14.1)**: `mlflow.sklearn.log_model()`
  404s against this server version — this repo's code already works
  around it (`registry.log_pipeline_artifact()`); don't reintroduce
  `log_model()` directly.
- **MLflow's emoji stdout print** crashes plain console scripts on
  Windows (`UnicodeEncodeError`, cp1252) — not an issue inside
  notebooks, but any new standalone script touching mlflow needs
  `sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")`
  before mlflow prints anything.
- **A champion trained before Phase 6** has no background-sample
  artifact — `batch_score.py`/the API's `/score` endpoint will fail
  downloading it. See the note at the end of the Phase 6 section above.
- **DB connect timeout matters for the API specifically**: unlike the
  short-lived CLI scripts (ingest/build_panel/train/batch_score),
  `api/main.py` is a long-lived server process — `src/churn/db/engine.py`
  sets `connect_timeout=5` so a DB outage fails `/health` fast instead of
  hanging on the OS's default TCP timeout. If you change the shared
  engine, keep this — it's what makes graceful degradation actually
  graceful.
