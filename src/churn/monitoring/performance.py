"""Delayed-label performance monitoring: given `predictions` rows whose
`actual_churned` ground truth has matured (backfilled by
`pipelines/backfill_predictions.py`), computes classification metrics
*per cohort* (`reference_date`) rather than a single aggregate number --
so real performance drift over time is visible, not just one point-in-
time test score.

Every cohort is labeled train/validation/test/post-test
(`churn.models.split.dataset_split_label`) so a train-window cohort's
optimistic, partially in-sample numbers can never be read as if they
were the held-out test performance -- the same discipline
DECISIONS.md#groupkfold-cv-score-is-not-comparable-to-forward-in-time-pr-auc
already applies elsewhere in this project.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import cast

import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

from churn.config import AppConfig
from churn.models.split import dataset_split_label


@dataclass
class CohortPerformance:
    reference_date: dt.date
    dataset_split: str  # "train" | "validation" | "test" | "post-test"
    n: int
    positive_rate: float
    pr_auc: float  # NaN if the cohort has only one class present
    roc_auc: float  # NaN if the cohort has only one class present
    precision_at_medium_plus: float  # "flagged" = risk_band in {medium, high}
    recall_at_medium_plus: float


def cohort_performance_report(predictions: pd.DataFrame, cfg: AppConfig) -> list[CohortPerformance]:
    """`predictions` must already be filtered to rows with a matured
    label (`actual_churned IS NOT NULL`) -- an unmatured row has no
    ground truth to score against. One row per distinct `reference_date`
    present in `predictions`, sorted chronologically."""
    results = []
    for reference_date, group in predictions.groupby("reference_date"):
        y_true = group["actual_churned"].astype(int).to_numpy()
        y_score = group["probability"].astype(float).to_numpy()
        flagged = group["risk_band"].isin(["medium", "high"]).to_numpy()

        single_class = y_true.sum() == 0 or y_true.sum() == len(y_true)
        pr_auc = float("nan") if single_class else average_precision_score(y_true, y_score)
        roc_auc = float("nan") if single_class else roc_auc_score(y_true, y_score)

        # groupby's key type is a broad Hashable per pandas-stubs, but at
        # runtime this column is always datetime64 (the caller converts
        # it via pd.to_datetime before grouping).
        ts = pd.Timestamp(cast(pd.Timestamp, reference_date))
        results.append(
            CohortPerformance(
                reference_date=ts.date(),
                dataset_split=dataset_split_label(ts, cfg),
                n=len(group),
                positive_rate=float(y_true.mean()),
                pr_auc=float(pr_auc),
                roc_auc=float(roc_auc),
                precision_at_medium_plus=float(precision_score(y_true, flagged, zero_division=0)),
                recall_at_medium_plus=float(recall_score(y_true, flagged, zero_division=0)),
            )
        )
    return sorted(results, key=lambda r: r.reference_date)
