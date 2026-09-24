"""Entry point: `python dashboards/report.py [--output PATH]`.

Renders a single self-contained static HTML report -- the same
monitoring computations `pipelines/monitor.py` logs and
`dashboards/app.py` renders interactively, captured as a shareable file
for someone without a running Streamlit server. Tables only (pandas
`.to_html()`/`.style` + inline CSS) -- no new charting dependency for
what's otherwise a demo/portfolio deliverable already well served by
tables and the interactive dashboard's charts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sqlalchemy import text

from churn.config import PROJECT_ROOT, AppConfig, get_app_config
from churn.db.engine import get_engine
from churn.models.inference import current_champion_version, scoring_cohort
from churn.models.pipeline import load_selected_features
from churn.models.split import modeling_panel, time_based_split
from churn.monitoring.drift import SEVERITY_COLORS, drift_report
from churn.monitoring.performance import cohort_performance_report
from churn.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

_DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "monitoring_report.html"

_STYLE = """
<style>
body { font-family: -apple-system, "Segoe UI", sans-serif; margin: 2rem; color: #1a1a1a; }
h1, h2 { border-bottom: 1px solid #ddd; padding-bottom: 0.3rem; }
table { border-collapse: collapse; width: 100%; margin-bottom: 1rem; font-size: 0.9rem; }
th, td { border: 1px solid #ddd; padding: 6px 10px; text-align: left; }
th { background: #f5f5f5; }
.caption { color: #555; font-size: 0.85rem; margin-top: -0.5rem; margin-bottom: 1.5rem; }
.metrics { display: flex; gap: 2rem; margin-bottom: 1.5rem; }
.metric { font-size: 1.4rem; font-weight: 600; }
.metric-label { font-size: 0.8rem; color: #555; font-weight: 400; }
</style>
"""


def _highlight_drift_severity(row: pd.Series) -> list[str]:
    color = SEVERITY_COLORS.get(str(row["severity"]))
    return [f"background-color: {color}" if color else ""] * len(row)


def _champion_version(cfg: AppConfig) -> str | None:
    try:
        return current_champion_version(cfg.mlflow.registry_model_name, MlflowClient())
    except Exception:
        logger.warning("report_champion_unavailable", exc_info=True)
        return None


def _overview_section(predictions: pd.DataFrame, cfg: AppConfig) -> str:
    current = predictions[predictions["reference_date"] == pd.Timestamp(cfg.data.snapshot_date)]
    if current.empty:
        return "<h2>Current cohort</h2><p>No predictions for the current cohort yet.</p>"

    high = int((current["risk_band"] == "high").sum())
    mean_p = current["probability"].astype(float).mean()
    top = current.sort_values("probability", ascending=False).head(20)
    top_table = top[["customer_id", "probability", "risk_band", "shap_top3"]].to_html(
        index=False, escape=True
    )

    def _metric(value: str, label: str) -> str:
        return (
            f'<div><div class="metric">{value}</div>'
            f'<div class="metric-label">{label}</div></div>'
        )

    metrics = "".join(
        [
            _metric(str(len(current)), "customers scored"),
            _metric(str(high), "high risk"),
            _metric(f"{mean_p:.3f}", "mean probability"),
        ]
    )
    return f"""
    <h2>Current cohort ({cfg.data.snapshot_date})</h2>
    <div class="metrics">{metrics}</div>
    <h3>Top 20 highest-risk customers</h3>
    {top_table}
    """


def _drift_section(panel: pd.DataFrame, cfg: AppConfig) -> str:
    current_cohort = scoring_cohort(panel, cfg)
    if current_cohort.empty:
        return "<h2>Feature drift</h2><p>No current cohort to compare against yet.</p>"

    train_df, _, _ = time_based_split(modeling_panel(panel, cfg), cfg)
    selected_features = load_selected_features(cfg)
    results = drift_report(train_df, current_cohort, selected_features, cfg)
    df = pd.DataFrame([r.__dict__ for r in results])
    table = df.style.apply(_highlight_drift_severity, axis=1).to_html(index=False)
    return f"""
    <h2>Feature drift: training window vs. current cohort</h2>
    <p class="caption">PSI &gt;= {cfg.monitoring.psi_critical_threshold} flagged significant,
    &gt;= {cfg.monitoring.psi_warning_threshold} flagged moderate.</p>
    {table}
    """


def _performance_section(
    predictions: pd.DataFrame, champion_version: str | None, cfg: AppConfig
) -> str:
    if champion_version is None:
        return "<h2>Performance over time</h2><p>No champion available.</p>"

    matured = predictions[
        (predictions["model_version"] == champion_version) & predictions["actual_churned"].notna()
    ]
    if matured.empty:
        return (
            "<h2>Performance over time</h2>"
            "<p>No matured predictions for the current champion yet -- run "
            "<code>pipelines/backfill_predictions.py</code>.</p>"
        )

    report = cohort_performance_report(matured, cfg)
    df = pd.DataFrame([r.__dict__ for r in report])
    table = df.to_html(index=False)
    return f"""
    <h2>Performance over time (champion v{champion_version})</h2>
    <p class="caption">train/validation cohorts are (partially) in-sample -- only test/post-test
    cohorts are genuine held-out performance.</p>
    {table}
    """


def build_report(cfg: AppConfig) -> str:
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    engine = get_engine()

    panel = pd.read_sql(
        text("SELECT * FROM feature_snapshots WHERE feature_set_version = :v"),
        engine,
        params={"v": cfg.feature_set.version},
    )
    panel["reference_date"] = pd.to_datetime(panel["reference_date"])

    predictions = pd.read_sql(text("SELECT * FROM predictions"), engine)
    predictions["reference_date"] = pd.to_datetime(predictions["reference_date"])

    champion_version = _champion_version(cfg)

    body = (
        _overview_section(predictions, cfg)
        + _drift_section(panel, cfg)
        + _performance_section(predictions, champion_version, cfg)
    )
    generated_at = pd.Timestamp.now("UTC").isoformat()
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Churn Prediction Monitoring Report</title>{_STYLE}</head>
<body>
<h1>Churn Prediction Monitoring Report</h1>
<p class="caption">Generated {generated_at} UTC</p>
{body}
</body></html>"""


def run(output: Path) -> None:
    cfg = get_app_config()
    html = build_report(cfg)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    logger.info("report_written", path=str(output))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[union-attr]
    configure_logging()
    parser = argparse.ArgumentParser(description="Render a static monitoring report")
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    args = parser.parse_args(sys.argv[1:])
    run(args.output)


if __name__ == "__main__":
    main()
