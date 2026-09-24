"""Streamlit dashboard: `uv run streamlit run dashboards/app.py`
(run from the project root, matching every other pipeline entry point).

Reads live from Postgres + the MLflow registry on every render -- no
separate "monitoring_runs" table, no stale snapshot (see
DECISIONS.md#monitoring-results-are-computed-on-demand-not-persisted).
Same `churn.monitoring` functions `pipelines/monitor.py` logs, rendered
interactively here instead, via the shared loaders in
`churn.monitoring.loaders` -- not a second, independent copy of the
panel/predictions-loading logic `dashboards/report.py` also needs (an
independent review found the original per-file copies had let a real
bug -- the current-cohort query not filtering by champion model_version
-- go unnoticed in two places at once).
"""

from __future__ import annotations

import mlflow
import pandas as pd
import streamlit as st

from churn.config import AppConfig, get_app_config
from churn.models.inference import scoring_cohort
from churn.models.pipeline import load_selected_features
from churn.models.split import modeling_panel, trainval_split
from churn.monitoring.drift import SEVERITY_COLORS, drift_report
from churn.monitoring.loaders import (
    champion_version,
    load_current_cohort_predictions,
    load_matured_predictions,
    load_panel,
)
from churn.monitoring.performance import cohort_performance_report

st.set_page_config(page_title="Churn Prediction Monitoring", layout="wide")


@st.cache_resource
def _cfg() -> AppConfig:
    cfg = get_app_config()
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    return cfg


@st.cache_data(ttl=60)
def _cached_panel(_cfg: AppConfig) -> pd.DataFrame:
    return load_panel(_cfg)


@st.cache_data(ttl=60)
def _cached_current_cohort_predictions(_cfg: AppConfig, version: str) -> pd.DataFrame:
    return load_current_cohort_predictions(_cfg, version)


@st.cache_data(ttl=60)
def _cached_matured_predictions(_cfg: AppConfig, version: str) -> pd.DataFrame:
    return load_matured_predictions(_cfg, version)


def _highlight_drift_severity(row: pd.Series) -> list[str]:
    color = SEVERITY_COLORS.get(str(row["severity"]))
    return [f"background-color: {color}" if color else ""] * len(row)


def _render_overview(cfg: AppConfig, version: str | None) -> None:
    if version is None:
        st.info("No champion available -- cannot load the current cohort's predictions.")
        return
    current = _cached_current_cohort_predictions(cfg, version)
    if current.empty:
        st.info("No predictions for the current cohort yet -- run `pipelines/batch_score.py`.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Customers scored", len(current))
    col2.metric("High risk", int((current["risk_band"] == "high").sum()))
    col3.metric("Mean probability", f"{current['probability'].astype(float).mean():.3f}")
    st.caption(
        "⚠️ These are the model's raw scores, not calibrated churn "
        'probabilities -- `class_weight="balanced"` training biases them well '
        "above the true ~1.6% base churn rate (see DECISIONS.md's Phase 4 "
        "section on risk_bands needing recalibration before real use)."
    )

    st.subheader("Risk band distribution")
    band_counts = current["risk_band"].value_counts().reindex(["low", "medium", "high"]).fillna(0)
    st.bar_chart(band_counts)

    st.subheader("Top 20 highest-risk customers")
    top = current.sort_values("probability", ascending=False).head(20)
    st.dataframe(top[["customer_id", "probability", "risk_band", "shap_top3"]], width="stretch")


def _render_drift(panel: pd.DataFrame, cfg: AppConfig) -> None:
    current_cohort = scoring_cohort(panel, cfg)
    if current_cohort.empty:
        st.info("No current cohort to compare against yet.")
        return

    reference = trainval_split(modeling_panel(panel, cfg), cfg)
    selected_features = load_selected_features(cfg)
    results = drift_report(reference, current_cohort, selected_features, cfg)

    df = pd.DataFrame([r.__dict__ for r in results])

    st.subheader("Feature drift: training window (train+val) vs. current cohort")
    st.caption(
        f"PSI >= {cfg.monitoring.psi_critical_threshold} flagged significant, "
        f">= {cfg.monitoring.psi_warning_threshold} flagged moderate. A feature's "
        f"null rate changing by >= {cfg.monitoring.null_rate_delta_threshold:.0%} "
        "is flagged significant on its own, regardless of PSI."
    )
    st.caption(
        "KS p-values assume independent samples; this panel's cohorts share "
        "mostly the same customers a few months apart, so a very small "
        "p-value here can reflect that repetition rather than a real shift -- "
        "treat it as a rough signal, not a rigorous test."
    )
    st.dataframe(df.style.apply(_highlight_drift_severity, axis=1), width="stretch")


def _render_performance(cfg: AppConfig, version: str | None) -> None:
    if version is None:
        st.info("No champion available -- cannot filter performance to the current model.")
        return

    matured = _cached_matured_predictions(cfg, version)
    if matured.empty:
        st.info(
            "No matured predictions for the current champion yet -- run "
            "`pipelines/backfill_predictions.py`."
        )
        return

    report = cohort_performance_report(matured, cfg)
    df = pd.DataFrame([r.__dict__ for r in report])

    st.subheader("PR-AUC / ROC-AUC by cohort")
    st.caption(
        "`in_sample=True` (train/validation) cohorts were part of what the "
        "champion was actually fit on -- their numbers are optimistic by "
        'construction, not just "partially" so. `in_sample=False` (test/'
        "post-test) cohorts are held-out, though test also mildly influenced "
        "which model got promoted (see performance.py's module docstring)."
    )
    st.line_chart(df.set_index("reference_date")[["pr_auc", "roc_auc"]])

    st.subheader("Precision / recall at the deployed risk bands, by cohort")
    st.line_chart(
        df.set_index("reference_date")[["precision_at_medium_plus", "recall_at_medium_plus"]]
    )

    st.dataframe(df, width="stretch")


def main() -> None:
    cfg = _cfg()
    st.title("Churn Prediction Monitoring")

    version = champion_version(cfg)
    if version is None:
        st.error("MLflow registry unreachable -- no champion model version available.")
    else:
        st.caption(f"Champion model version: **{version}**")

    panel = _cached_panel(cfg)

    tab_overview, tab_drift, tab_performance = st.tabs(
        ["Current cohort", "Feature drift", "Performance over time"]
    )
    with tab_overview:
        _render_overview(cfg, version)
    with tab_drift:
        _render_drift(panel, cfg)
    with tab_performance:
        _render_performance(cfg, version)


if __name__ == "__main__":
    main()
