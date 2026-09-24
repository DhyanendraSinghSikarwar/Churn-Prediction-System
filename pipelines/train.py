"""Entry point: `python pipelines/train.py`.

Trains the champion model type Phase 4 found empirically best --
`LogisticRegression(class_weight="balanced")` on the 12 features
Notebook 03 selected (see DECISIONS.md#logistic-regression-beats-every-gbm-variant
and config/selected_features_v1.yaml) -- gated by:

1. A data validation gate: `validate_training_panel` (fail loudly, never
   train on bad data), applied after dropping the structurally-censored
   last cohort.
2. A model gate: the new run is promoted to the registry's "champion"
   alias only if its test PR-AUC beats the current champion's by at
   least `config.yaml`'s `promotion_epsilon` -- or if there is no
   champion yet (the first-ever training run).

Every run logs to MLflow: params, metrics (test PR-AUC/ROC-AUC/top-decile
lift), the single fitted sklearn Pipeline (preprocessing + model, one
artifact), a background sample of raw training rows for Phase 6's SHAP
explainer (`churn.models.inference.log_background_sample`), and a model
card summarizing what was trained, on what, and why.
"""

from __future__ import annotations

import sys

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import average_precision_score, roc_auc_score
from sqlalchemy import text

from churn.config import AppConfig, get_app_config
from churn.db.engine import get_engine
from churn.features.validation import validate_training_panel
from churn.models.inference import log_background_sample
from churn.models.pipeline import (
    build_pipeline,
    categorical_and_numeric_features,
    load_selected_features,
)
from churn.models.registry import (
    current_champion_metric,
    log_pipeline_artifact,
    register_and_maybe_promote,
)
from churn.models.split import modeling_panel, time_based_split
from churn.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

_METRIC_NAME = "test_pr_auc"
_ARTIFACT_PATH = "model"


def _load_panel(cfg: AppConfig) -> pd.DataFrame:
    engine = get_engine()
    df = pd.read_sql(
        text("SELECT * FROM feature_snapshots WHERE feature_set_version = :v"),
        engine,
        params={"v": cfg.feature_set.version},
    )
    df["reference_date"] = pd.to_datetime(df["reference_date"])
    return df


def _top_decile_lift(proba: np.ndarray, y_true: np.ndarray) -> float:
    """Actual churn rate in the highest-predicted-risk decile, divided by
    the overall churn rate -- the same metric Notebook 06 reports."""
    order = np.argsort(-proba)
    decile_size = max(len(proba) // 10, 1)
    top_decile_idx = order[:decile_size]
    top_decile_rate = y_true[top_decile_idx].mean()
    overall_rate = y_true.mean()
    return float(top_decile_rate / overall_rate)


def _model_card(
    cfg: AppConfig,
    selected_features: list[str],
    train_rows: int,
    val_rows: int,
    test_rows: int,
    test_pr_auc: float,
    test_roc_auc: float,
    top_decile_lift: float,
    promoted: bool,
    champion_metric_before: float | None,
) -> str:
    promotion_line = (
        "No prior champion -- this run was promoted as the bootstrap champion."
        if champion_metric_before is None
        else (
            f"Beat the prior champion's {_METRIC_NAME}={champion_metric_before:.4f} "
            f"by >= promotion_epsilon={cfg.mlflow.promotion_epsilon} -- promoted."
            if promoted
            else (
                f"Did not beat the prior champion's {_METRIC_NAME}={champion_metric_before:.4f} "
                f"by promotion_epsilon={cfg.mlflow.promotion_epsilon} -- not promoted."
            )
        )
    )
    return f"""\
# Model Card -- churn_classifier

## Model type

`LogisticRegression(class_weight="balanced")` on a `ColumnTransformer`
(one-hot categoricals, scaled numerics). Chosen because Phase 4 found it
empirically beats every `HistGradientBoostingClassifier` variant tried
(class weighting, SMOTE, plain, each with Optuna-tuned hyperparameters)
on this feature set -- see DECISIONS.md#logistic-regression-beats-every-gbm-variant.
This is not the model type originally assumed in the Phase 0 plan; it is
the one the evidence supports.

## Features ({len(selected_features)})

{chr(10).join(f"- `{f}`" for f in selected_features)}

Selected in Phase 3 (`notebooks/03_feature_engineering.ipynb`) from 44
candidates via mutual information / permutation importance / a
hand-rolled Boruta procedure / correlation pruning -- see
`config/selected_features_v1.yaml` and DECISIONS.md's Phase 3 section.
None of Phase 3's engineered ratio/interaction features were selected;
this model uses only base `feature_snapshots` columns.

## Training data

- `feature_set_version`: `{cfg.feature_set.version}`
- Modeling panel: `feature_snapshots` with the `reference_date ==
  snapshot_date` cohort dropped (structural censoring artifact -- see
  DECISIONS.md#last-cohort-has-a-structural-zero-positive-rate).
- Split (calendar months off `panel_start_date`): train={train_rows} rows,
  val={val_rows} rows, test={test_rows} rows. Fit on train+val; test held
  out entirely until final evaluation.

## Test-set metrics

- PR-AUC: **{test_pr_auc:.4f}**
- ROC-AUC (secondary): {test_roc_auc:.4f}
- Top-decile lift: {top_decile_lift:.2f}x

## Known limitations (from Phase 4's Notebooks 05/06)

- At `config/config.yaml`'s actual retention economics, the
  business-optimal decision threshold came out around 0.94 -- the
  *configured* `risk_bands` (low_max=0.3, medium_max=0.7) would lose
  money on a real campaign at this model's current precision. Recalibrate
  `risk_bands` before Phase 6 uses them for anything real.
- At that safe threshold, the model flags a very small fraction of
  customers per month -- a properly powered A/B test to validate real
  causal impact would need a long accumulation window (Notebook 06
  estimated ~137 months at the observed flagging rate). Consider a lower,
  less conservative threshold if faster experimental validation is a
  priority, trading some precision for volume.
- `GroupKFold` cross-validation scores computed during hyperparameter
  search are *not* comparable to this forward-in-time test PR-AUC --see
  DECISIONS.md#groupkfold-cv-score-is-not-comparable-to-forward-in-time-pr-auc.
  Only ever compare this run's test/val PR-AUC against another run's
  test/val PR-AUC, never against a CV score.

## Promotion outcome

{promotion_line}
"""


def run() -> None:
    cfg = get_app_config()
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    panel = _load_panel(cfg)
    panel = modeling_panel(panel, cfg)
    panel = validate_training_panel(panel)
    logger.info("panel_validated", rows=len(panel))

    selected_features = load_selected_features(cfg)
    cat_features, num_features = categorical_and_numeric_features(panel, selected_features)

    train_df, val_df, test_df = time_based_split(panel, cfg)
    trainval_df = pd.concat([train_df, val_df])
    logger.info(
        "split_complete", train_rows=len(train_df), val_rows=len(val_df), test_rows=len(test_df)
    )

    features_trainval = trainval_df[selected_features]
    y_trainval = trainval_df["label_churn_next_30d"].astype(int)
    features_test = test_df[selected_features]
    y_test = test_df["label_churn_next_30d"].astype(int)

    pipeline = build_pipeline(cat_features, num_features, random_state=cfg.synthetic.random_seed)
    pipeline.fit(features_trainval, y_trainval)

    proba_test = pipeline.predict_proba(features_test)[:, 1]
    y_test_arr = y_test.to_numpy()
    test_pr_auc = average_precision_score(y_test_arr, proba_test)
    test_roc_auc = roc_auc_score(y_test_arr, proba_test)
    top_decile_lift = _top_decile_lift(proba_test, y_test_arr)
    logger.info(
        "model_trained",
        test_pr_auc=test_pr_auc,
        test_roc_auc=test_roc_auc,
        top_decile_lift=top_decile_lift,
    )

    client = MlflowClient()
    champion_metric_before = None
    try:
        champion_metric_before = current_champion_metric(
            client, cfg.mlflow.registry_model_name, _METRIC_NAME
        )
    except Exception:  # pragma: no cover - defensive: registry not reachable yet
        logger.warning("champion_lookup_failed", exc_info=True)

    with mlflow.start_run(run_name="train") as active_run:
        mlflow.set_tag("phase", "5-training-pipeline")
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("selected_features", selected_features)
        mlflow.log_param("train_rows", len(train_df))
        mlflow.log_param("val_rows", len(val_df))
        mlflow.log_param("test_rows", len(test_df))
        mlflow.log_metric(_METRIC_NAME, test_pr_auc)
        mlflow.log_metric("test_roc_auc", test_roc_auc)
        mlflow.log_metric("top_decile_lift", top_decile_lift)

        log_pipeline_artifact(pipeline, artifact_path=_ARTIFACT_PATH)
        log_background_sample(features_trainval, random_state=cfg.synthetic.random_seed)

        run_id = active_run.info.run_id

    version, promoted = register_and_maybe_promote(
        model_name=cfg.mlflow.registry_model_name,
        run_id=run_id,
        candidate_metric=test_pr_auc,
        metric_name=_METRIC_NAME,
        promotion_epsilon=cfg.mlflow.promotion_epsilon,
        artifact_path=_ARTIFACT_PATH,
        client=client,
    )

    card = _model_card(
        cfg,
        selected_features,
        len(train_df),
        len(val_df),
        len(test_df),
        test_pr_auc,
        test_roc_auc,
        top_decile_lift,
        promoted,
        champion_metric_before,
    )
    client.log_text(run_id, card, "model_card.md")

    logger.info(
        "training_complete",
        run_id=run_id,
        model_version=version.version,
        promoted=promoted,
        champion_metric_before=champion_metric_before,
        test_pr_auc=test_pr_auc,
    )


def main() -> None:
    # MLflow prints an emoji ("View run...") on every run completion; the
    # default Windows console codepage (cp1252) can't encode it and
    # crashes a plain console script (does not affect notebooks --
    # nbconvert's kernel protocol is UTF-8 regardless of console
    # codepage). Only needs to happen before mlflow actually prints
    # anything, not before it's imported.
    # sys.stdout/stderr are typed as TextIO, which doesn't declare
    # .reconfigure() (that's specific to the concrete io.TextIOWrapper
    # they actually are at runtime) -- not a real type risk.
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[union-attr]
    configure_logging()
    run()


if __name__ == "__main__":
    main()
