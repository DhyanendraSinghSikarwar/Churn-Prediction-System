"""Feature-drift detection: PSI, KS, and KL divergence between a
reference distribution (the training window's feature values) and a
comparison distribution (any later cohort -- most commonly the current
scoring cohort). Numeric features get all three; categorical features
get PSI and KL only (a two-sample KS test is defined for continuous
distributions, not category labels).

PSI/KL both need the two distributions binned onto a shared grid before
they can be compared. For numeric features that grid is quantile edges
derived from the *reference* distribution only (so drift is always
measured against a fixed reference, not a grid that itself shifts with
whatever's being compared); for categorical features it's simply the
union of category values seen in either distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from churn.config import AppConfig

# Avoids divide-by-zero / log(0) when a bin is empty in one distribution
# but not the other -- standard PSI/KL practice, not a magic number
# tuned to this data.
_EPSILON = 1e-6


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _numeric_bin_proportions(
    reference: pd.Series, current: pd.Series, bins: int
) -> tuple[np.ndarray, np.ndarray]:
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(reference.quantile(quantiles).to_numpy())
    if len(edges) < 2:
        # every reference value is identical -- one bin, still comparable.
        edges = np.array([reference.iloc[0] - _EPSILON, reference.iloc[0] + _EPSILON])
    edges[0] = -np.inf
    edges[-1] = np.inf
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    return ref_counts / len(reference), cur_counts / len(current)


def _category_proportions(
    reference: pd.Series, current: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    categories = sorted(set(reference.unique()) | set(current.unique()))
    ref_counts = reference.value_counts()
    cur_counts = current.value_counts()
    ref_props = np.array([ref_counts.get(c, 0) for c in categories]) / len(reference)
    cur_props = np.array([cur_counts.get(c, 0) for c in categories]) / len(current)
    return ref_props, cur_props


def _bin_proportions(
    reference: pd.Series, current: pd.Series, bins: int
) -> tuple[np.ndarray, np.ndarray]:
    if _is_numeric(reference):
        return _numeric_bin_proportions(reference, current, bins)
    return _category_proportions(reference, current)


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Standard PSI: sum((cur% - ref%) * ln(cur% / ref%)) over bins.
    <0.1 conventionally reads as stable, 0.1-0.25 a moderate shift worth
    watching, >0.25 a significant shift -- see config.yaml's
    `monitoring` section."""
    reference, current = reference.dropna(), current.dropna()
    if reference.empty or current.empty:
        return float("nan")
    ref_props, cur_props = _bin_proportions(reference, current, bins)
    ref_props, cur_props = ref_props + _EPSILON, cur_props + _EPSILON
    return float(np.sum((cur_props - ref_props) * np.log(cur_props / ref_props)))


def ks_statistic(reference: pd.Series, current: pd.Series) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test (numeric features only).
    Returns (statistic, p_value) -- a small p-value rejects the null
    hypothesis that both samples were drawn from the same distribution."""
    reference, current = reference.dropna(), current.dropna()
    if reference.empty or current.empty:
        return float("nan"), float("nan")
    result = stats.ks_2samp(reference.to_numpy(), current.to_numpy())
    return float(result.statistic), float(result.pvalue)


def kl_divergence(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """KL(current || reference): how many extra nats it costs to encode
    `current` using a code built for `reference`. 0 means identical
    distributions; larger values mean more divergence. Not symmetric --
    order matters, "current relative to reference" is the intended
    reading for drift monitoring."""
    reference, current = reference.dropna(), current.dropna()
    if reference.empty or current.empty:
        return float("nan")
    ref_props, cur_props = _bin_proportions(reference, current, bins)
    return float(stats.entropy(cur_props + _EPSILON, ref_props + _EPSILON))


# Shared with dashboards/app.py and dashboards/report.py, so a severity
# color only ever has one definition -- never re-picked independently by
# each presentation layer.
SEVERITY_COLORS = {"significant": "#f8d7da", "moderate": "#fff3cd"}


def _severity(psi: float, cfg: AppConfig) -> str:
    if np.isnan(psi):
        return "unknown"
    if psi >= cfg.monitoring.psi_critical_threshold:
        return "significant"
    if psi >= cfg.monitoring.psi_warning_threshold:
        return "moderate"
    return "stable"


@dataclass
class FeatureDrift:
    feature: str
    psi: float
    ks_statistic: float | None  # None for categorical features
    ks_pvalue: float | None
    kl_divergence: float
    severity: str  # "stable" | "moderate" | "significant" | "unknown" (PSI undefined)


def drift_report(
    reference: pd.DataFrame, current: pd.DataFrame, features: list[str], cfg: AppConfig
) -> list[FeatureDrift]:
    """One `FeatureDrift` per feature in `features`, sorted by PSI
    descending (worst drift first)."""
    results = []
    for feature in features:
        ref_col, cur_col = reference[feature], current[feature]
        psi = population_stability_index(ref_col, cur_col, cfg.monitoring.drift_bins)
        kl = kl_divergence(ref_col, cur_col, cfg.monitoring.drift_bins)
        ks_stat, ks_p = ks_statistic(ref_col, cur_col) if _is_numeric(ref_col) else (None, None)
        results.append(
            FeatureDrift(
                feature=feature,
                psi=psi,
                ks_statistic=ks_stat,
                ks_pvalue=ks_p,
                kl_divergence=kl,
                severity=_severity(psi, cfg),
            )
        )
    return sorted(results, key=lambda r: (np.isnan(r.psi), -r.psi if not np.isnan(r.psi) else 0))
