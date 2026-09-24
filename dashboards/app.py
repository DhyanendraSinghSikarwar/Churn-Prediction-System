"""Streamlit dashboard: `uv run streamlit run dashboards/app.py`
(run from the project root, matching every other pipeline entry point).

Reads live from Postgres + the MLflow registry on every render -- no
separate "monitoring_runs" table, no stale snapshot (see
DECISIONS.md#monitoring-results-are-computed-on-demand-not-persisted).
Same `churn.monitoring` functions `pipelines/monitor.py` logs, rendered
interactively here instead.
"""

from __future__ import annotations

import mlflow
import pandas as pd
import streamlit as st
from mlflow.tracking import MlflowClient
from sqlalchemy import text

from churn.config import AppConfig, get_app_config
from churn.db.engine import get_engine
from churn.models.inference import current_champion_version, scoring_cohort
from churn.models.pipeline import load_selected_features
from churn.models.split import modeling_panel, time_based_split
from churn.monitoring.drift import SEVERITY_COLORS, drift_report
from churn.monitoring.performance import cohort_performance_report

st.set_page_config(page_title="Churn Prediction Monitoring", layout="wide")


@st.cache_resource
def _cfg() -> AppConfig:
    cfg = get_app_config()
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    return cfg


@st.cache_data(ttl=60)
def _load_panel(_cfg: AppConfig) -> pd.DataFrame:
    engine = get_engine()
    df = pd.read_sql(
        text("SELECT * FROM feature_snapshots WHERE feature_set_version = :v"),
        engine,
        params={"v": _cfg.feature_set.version},
    )
    df["reference_date"] = pd.to_datetime(df["reference_date"])
    return df


@st.cache_data(ttl=60)
def _load_predictions(_cfg: AppConfig) -> pd.DataFrame:
    engine = get_engine()
    df = pd.read_sql(text("SELECT * FROM predictions"), engine)
    df["reference_date"] = pd.to_datetime(df["reference_date"])
    return df


def _champion_version(cfg: AppConfig) -> str | None:
    try:
        return current_champion_version(cfg.mlflow.registry_model_name, MlflowClient())
    except Exception:
        return None


def _highlight_drift_severity(row: pd.Series) -> list[str]:
    color = SEVERITY_COLORS.get(str(row["severity"]))
    return [f"background-color: {color}" if color else ""] * len(row)


def _render_overview(predictions: pd.DataFrame, cfg: AppConfig) -> None:
    current = predictions[predictions["reference_date"] == pd.Timestamp(cfg.data.snapshot_date)]
    if current.empty:
        st.info("No predictions for the current cohort yet -- run `pipelines/batch_score.py`.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Customers scored", len(current))
    col2.metric("High risk", int((current["risk_band"] == "high").sum()))
    col3.metric("Mean probability", f"{current['probability'].astype(float).mean():.3f}")

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

    train_df, _, _ = time_based_split(modeling_panel(panel, cfg), cfg)
    selected_features = load_selected_features(cfg)
    results = drift_report(train_df, current_cohort, selected_features, cfg)

    df = pd.DataFrame([r.__dict__ for r in results])

    st.subheader("Feature drift: training window vs. current cohort")
    st.caption(
        f"PSI >= {cfg.monitoring.psi_critical_threshold} flagged significant, "
        f">= {cfg.monitoring.psi_warning_threshold} flagged moderate."
    )
    st.dataframe(df.style.apply(_highlight_drift_severity, axis=1), width="stretch")


def _render_performance(
    predictions: pd.DataFrame, champion_version: str | None, cfg: AppConfig
) -> None:
    if champion_version is None:
        st.info("No champion available -- cannot filter performance to the current model.")
        return

    matured = predictions[
        (predictions["model_version"] == champion_version) & predictions["actual_churned"].notna()
    ]
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
        "train/validation cohorts are (partially) in-sample -- only test/post-test cohorts "
        "are genuine held-out performance."
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

    champion_version = _champion_version(cfg)
    if champion_version is None:
        st.error("MLflow registry unreachable -- no champion model version available.")
    else:
        st.caption(f"Champion model version: **{champion_version}**")

    panel = _load_panel(cfg)
    predictions = _load_predictions(cfg)

    tab_overview, tab_drift, tab_performance = st.tabs(
        ["Current cohort", "Feature drift", "Performance over time"]
    )
    with tab_overview:
        _render_overview(predictions, cfg)
    with tab_drift:
        _render_drift(panel, cfg)
    with tab_performance:
        _render_performance(predictions, champion_version, cfg)


if __name__ == "__main__":
    main()
