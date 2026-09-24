"""Builds the single sklearn Pipeline this project trains and registers.

Per DECISIONS.md#logistic-regression-beats-every-gbm-variant, Phase 4
empirically found `LogisticRegression(class_weight="balanced")` beats
every GradientBoosting/HistGradientBoosting variant tried on this
feature set (val PR-AUC 0.1281 vs. 0.0609 best tuned GBM; confirmed on
the held-out test set, 0.1093 vs. 0.0593) -- this is the model type
Phase 5 trains and registers, not a hardcoded assumption carried over
from the Phase 0 architecture plan.
"""

from __future__ import annotations

import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.config import PROJECT_ROOT, AppConfig


def load_selected_features(cfg: AppConfig) -> list[str]:
    """Loads the feature list Notebook 03 selected for this
    `feature_set_version` -- the single source of truth for both
    `pipelines/train.py` and `pipelines/batch_score.py`, so the two can
    never silently train and score on different feature sets."""
    path = PROJECT_ROOT / "config" / f"selected_features_{cfg.feature_set.version}.yaml"
    with path.open(encoding="utf-8") as fh:
        artifact = yaml.safe_load(fh)
    return [str(f) for f in artifact["selected_features"]]


def categorical_and_numeric_features(
    panel: pd.DataFrame, selected_features: list[str]
) -> tuple[list[str], list[str]]:
    """Splits `selected_features` into (categorical, numeric) by dtype.

    Uses `is_numeric_dtype`/`is_bool_dtype` rather than `dtype == object`
    -- pandas 3's default string dtype is no longer `object`, which makes
    that check silently misclassify string columns as numeric. See
    DECISIONS.md#pandas-3-default-string-dtype-breaks-dtype-object-checks
    (found and fixed the hard way, in Phase 3's notebooks).
    """
    cat_features = [
        c
        for c in selected_features
        if not (
            pd.api.types.is_numeric_dtype(panel[c]) and not pd.api.types.is_bool_dtype(panel[c])
        )
    ]
    num_features = [c for c in selected_features if c not in cat_features]
    return cat_features, num_features


def build_pipeline(cat_features: list[str], num_features: list[str], random_state: int) -> Pipeline:
    """Returns a brand-new Pipeline (with its own, never-shared
    ColumnTransformer) every call.

    Always call this once per model you intend to fit separately.
    Two `Pipeline`/`ImbPipeline` objects built by passing in the *same*
    `ColumnTransformer` instance corrupt each other's fitted state the
    moment the second one is fit -- a real bug this exact pattern caused
    in Phase 4 (see
    DECISIONS.md#shared-columntransformer-corrupted-notebook-04s-threshold-tuning-numbers).
    This factory exists specifically so that mistake can't happen here.
    """
    preprocessor = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
            ("num", StandardScaler(), num_features),
        ]
    )
    return Pipeline(
        [
            ("prep", preprocessor),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000, class_weight="balanced", random_state=random_state
                ),
            ),
        ]
    )
