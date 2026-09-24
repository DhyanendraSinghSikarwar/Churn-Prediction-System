"""Delayed-label performance monitoring: given `predictions` rows whose
`actual_churned` ground truth has matured (backfilled by
`pipelines/backfill_predictions.py`), computes classification metrics
*per cohort* (`reference_date`) rather than a single aggregate number --
so real performance drift over time is visible, not just one point-in-
time test score.

Every cohort is labeled train/validation/test/post-test
(`churn.models.split.dataset_split_label`), and `in_sample` records
whether that cohort's rows were part of what the champion was actually
*fit* on (train+val -- see `pipelines/train.py`'s `trainval_df`) so a
train/validation-window cohort's optimistic numbers can never be read as
if they were held-out performance -- the same discipline
DECISIONS.md#groupkfold-cv-score-is-not-comparable-to-forward-in-time-pr-auc
already applies elsewhere in this project. Note that even `in_sample=False`
(test) cohorts aren't perfectly clean: `pipelines/train.py`'s promotion
gate compares candidates by test PR-AUC, so the test window has itself
mildly influenced which model became champion -- only `post-test`
cohorts are untouched by training or promotion.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

from churn.config import AppConfig
from churn.models.inference import risk_band
from churn.models.split import dataset_split_label

_IN_SAMPLE_SPLITS = {"train", "validation"}


@dataclass
class CohortPerformance:
    reference_date: dt.date
    dataset_split: str  # "train" | "validation" | "test" | "post-test"
    in_sample: bool  # True for train/validation -- part of what the champion was fit on
    n: int
    positive_rate: float
    pr_auc: float  # NaN if the cohort has only one class present
    roc_auc: float  # NaN if the cohort has only one class present
    precision_at_medium_plus: float  # "flagged" = risk_band(probability, cfg) in {medium, high}
    recall_at_medium_plus: float


def cohort_performance_report(predictions: pd.DataFrame, cfg: AppConfig) -> list[CohortPerformance]:
    """`predictions` must already be filtered to rows with a matured
    label (`actual_churned IS NOT NULL`) -- an unmatured row has no
    ground truth to score against. One row per distinct `reference_date`
    present in `predictions`, sorted chronologically.

    "Flagged" (for precision/recall) is recomputed from each row's
    stored `probability` against `cfg`'s *current* `risk_bands` --
    deliberately not the row's stored `risk_band` column, which instead
    reflects whatever thresholds were configured back when that row was
    scored. If `config.yaml`'s `risk_bands` ever changes, a report run
    against old rows should describe what the *current* deployed
    thresholds would have done, not silently mix threshold vintages.
    """
    results = []
    for reference_date, group in predictions.groupby("reference_date"):
        y_true = group["actual_churned"].astype(int).to_numpy()
        y_score = group["probability"].astype(float).to_numpy()
        flagged = np.array([risk_band(float(p), cfg) in ("medium", "high") for p in y_score])

        single_class = y_true.sum() == 0 or y_true.sum() == len(y_true)
        pr_auc = float("nan") if single_class else average_precision_score(y_true, y_score)
        roc_auc = float("nan") if single_class else roc_auc_score(y_true, y_score)

        # groupby's key type is a broad Hashable per pandas-stubs, but at
        # runtime this column is always datetime64 (the caller converts
        # it via pd.to_datetime before grouping).
        ts = pd.Timestamp(cast(pd.Timestamp, reference_date))
        split = dataset_split_label(ts, cfg)
        results.append(
            CohortPerformance(
                reference_date=ts.date(),
                dataset_split=split,
                in_sample=split in _IN_SAMPLE_SPLITS,
                n=len(group),
                positive_rate=float(y_true.mean()),
                pr_auc=float(pr_auc),
                roc_auc=float(roc_auc),
                precision_at_medium_plus=float(precision_score(y_true, flagged, zero_division=0)),
                recall_at_medium_plus=float(recall_score(y_true, flagged, zero_division=0)),
            )
        )
    return sorted(results, key=lambda r: r.reference_date)
