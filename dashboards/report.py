"""Entry point: `python dashboards/report.py [--output PATH]`.

Renders a single self-contained static HTML report -- the same
monitoring computations `pipelines/monitor.py` logs and
`dashboards/app.py` renders interactively, captured as a shareable file
for someone without a running Streamlit server. Tables only (pandas
`.to_html()`/`.style` + inline CSS) -- no new charting dependency for
what's otherwise a demo/portfolio deliverable already well served by
tables and the interactive dashboard's charts.

Loads through `churn.monitoring.loaders`, the same shared module
`dashboards/app.py` uses -- not an independent copy of the panel/
predictions-loading logic. See `dashboards/app.py`'s module docstring
for why that matters (a review found a real bug hiding in two
independently-duplicated copies of this loading logic at once).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlflow
import pandas as pd

from churn.config import PROJECT_ROOT, AppConfig, get_app_config
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


def _overview_section(cfg: AppConfig, version: str | None) -> str:
    if version is None:
        return "<h2>Current cohort</h2><p>No champion available.</p>"
    current = load_current_cohort_predictions(cfg, version)
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
    <p class="caption">⚠️ These are the model's raw scores, not calibrated
    churn probabilities -- class_weight="balanced" training biases them well above
    the true ~1.6% base churn rate (see DECISIONS.md's Phase 4 section on
    risk_bands needing recalibration before real use).</p>
    <h3>Top 20 highest-risk customers</h3>
    {top_table}
    """


def _drift_section(panel: pd.DataFrame, cfg: AppConfig) -> str:
    current_cohort = scoring_cohort(panel, cfg)
    if current_cohort.empty:
        return "<h2>Feature drift</h2><p>No current cohort to compare against yet.</p>"

    reference = trainval_split(modeling_panel(panel, cfg), cfg)
    selected_features = load_selected_features(cfg)
    results = drift_report(reference, current_cohort, selected_features, cfg)
    df = pd.DataFrame([r.__dict__ for r in results])
    table = df.style.apply(_highlight_drift_severity, axis=1).hide(axis="index").to_html()
    return f"""
    <h2>Feature drift: training window (train+val) vs. current cohort</h2>
    <p class="caption">PSI &gt;= {cfg.monitoring.psi_critical_threshold} flagged significant,
    &gt;= {cfg.monitoring.psi_warning_threshold} flagged moderate. A feature's null rate
    changing by &gt;= {cfg.monitoring.null_rate_delta_threshold:.0%} is flagged significant
    on its own, regardless of PSI.</p>
    <p class="caption">KS p-values assume independent samples; this panel's cohorts share
    mostly the same customers a few months apart, so a very small p-value here can reflect
    that repetition rather than a real shift -- treat it as a rough signal, not a rigorous
    test.</p>
    {table}
    """


def _performance_section(cfg: AppConfig, version: str | None) -> str:
    if version is None:
        return "<h2>Performance over time</h2><p>No champion available.</p>"

    matured = load_matured_predictions(cfg, version)
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
    <h2>Performance over time (champion v{version})</h2>
    <p class="caption">in_sample=True (train/validation) cohorts were part of what the
    champion was actually fit on -- their numbers are optimistic by construction, not just
    "partially" so. in_sample=False (test/post-test) cohorts are held-out, though test also
    mildly influenced which model got promoted (see performance.py's module docstring).</p>
    {table}
    """


def build_report(cfg: AppConfig) -> str:
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)

    panel = load_panel(cfg)
    version = champion_version(cfg)

    body = (
        _overview_section(cfg, version)
        + _drift_section(panel, cfg)
        + _performance_section(cfg, version)
    )
    generated_at = pd.Timestamp.now("UTC").isoformat()
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Churn Prediction Monitoring Report</title>{_STYLE}</head>
<body>
<h1>Churn Prediction Monitoring Report</h1>
<p class="caption">Generated {generated_at}</p>
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
