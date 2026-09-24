"""Shared inference helpers: load the registry's "champion" pipeline,
compute the scoring cohort, risk-band a probability, and extract
top-3 SHAP features per prediction. Single source of truth for
`pipelines/batch_score.py` and `api/main.py` -- neither re-implements
this inline, for the same reason `models/split.py`/`pipeline.py` exist
(see DECISIONS.md#shared-columntransformer-corrupted-notebook-04s-threshold-tuning-numbers).
"""

from __future__ import annotations

import ast
import datetime as dt
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import NotRequired, TypedDict

import mlflow
import numpy as np
import pandas as pd
import shap
from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Connection

from churn.config import AppConfig, get_app_config
from churn.db.models import Prediction
from churn.models.pipeline import load_selected_features
from churn.models.registry import CHAMPION_ALIAS
from churn.utils.logging import get_logger

logger = get_logger(__name__)

BACKGROUND_ARTIFACT_PATH = "explainability/background_sample.csv"
BACKGROUND_SAMPLE_SIZE = 100
# Postgres binds at most 65535 parameters per statement; each upserted row
# uses 6 (customer_id, reference_date, model_version, probability,
# risk_band, shap_top3), so one statement tops out around 10,922 rows --
# found by an independent review actually pushing a batch past that
# limit, not by inspection. Chunked well under that margin.
_UPSERT_CHUNK_SIZE = 2000


class ShapFeature(TypedDict):
    feature: str
    shap_value: float


class PredictionRow(TypedDict):
    customer_id: str
    reference_date: dt.date
    model_version: str
    probability: float
    risk_band: str
    shap_top3: list[ShapFeature]
    # Only present for a historical, already-matured cohort
    # (`pipelines/backfill_predictions.py`) -- absent for a just-scored,
    # label-not-matured-yet row (`pipelines/batch_score.py`, `api/main.py`).
    # A single call to `upsert_predictions` must be all-one-or-all-other;
    # see its docstring.
    actual_churned: NotRequired[bool]
    label_matured_at: NotRequired[dt.datetime]


def scoring_cohort(panel: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
    """The complement of `models.split.modeling_panel`: the single cohort
    with `reference_date == snapshot_date`, i.e. "today" in the synthetic
    world -- the one whose `label_churn_next_30d` hasn't matured yet, and
    therefore the one production scoring actually runs against."""
    snapshot_ts = pd.Timestamp(cfg.data.snapshot_date)
    return panel[panel["reference_date"] == snapshot_ts].copy()


def risk_band(probability: float, cfg: AppConfig) -> str:
    bands = cfg.business.risk_bands
    if probability <= bands.low_max:
        return "low"
    if probability <= bands.medium_max:
        return "medium"
    return "high"


@dataclass
class ChampionModel:
    pipeline: Pipeline
    version: ModelVersion
    background: pd.DataFrame  # raw (pre-transform) rows, for the SHAP explainer


def log_background_sample(
    trainval_features: pd.DataFrame, random_state: int, n: int = BACKGROUND_SAMPLE_SIZE
) -> None:
    """Logs a random sample of raw (pre-transform) training rows as a run
    artifact, so serving code can build a SHAP `LinearExplainer` without
    ever reloading the full historical panel -- must be called inside the
    same `with mlflow.start_run():` block as `registry.log_pipeline_artifact`.

    Sample size matches Notebook 05's exact precedent (100 rows is
    shap's own `LinearExplainer` default and already a stable estimate of
    the feature means/covariance it subtracts off).
    """
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(trainval_features), size=min(n, len(trainval_features)), replace=False)
    sample = trainval_features.iloc[idx]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "background_sample.csv"
        sample.to_csv(path, index=False)
        mlflow.log_artifact(str(path), artifact_path="explainability")


def current_champion_version(model_name: str, client: MlflowClient) -> str:
    """The registry's current `champion`-aliased version number, as a
    cheap metadata-only call -- used to detect a promotion without
    reloading the full model."""
    version: str = client.get_model_version_by_alias(model_name, CHAMPION_ALIAS).version
    return version


def load_champion(model_name: str, client: MlflowClient | None = None) -> ChampionModel:
    """Loads the pipeline + background sample currently aliased "champion".

    Resolves the alias *once* (`get_model_version_by_alias`), then loads
    the pipeline by that exact version number (`models:/{name}/{version}`),
    not by re-resolving the alias a second time (`models:/{name}@{alias}`)
    -- if a promotion landed in between the two calls, the pipeline and
    its metadata (`version.version`, `version.run_id`) would otherwise
    silently come from two different model versions. Found by an
    independent review, not by inspection.

    Uses version/run-relative artifact URIs throughout -- resolved via
    the standard artifact-download path (same one `log_artifacts`/
    `download_artifacts` use), not mlflow 3's "Logged Model" REST API,
    which this project's pinned mlflow *server* (2.14.1) doesn't have.
    See DECISIONS.md#mlflow-client-3x-vs-server-2x-model-logging-incompatibility.
    """
    client = client or MlflowClient()
    version = client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)
    if version.run_id is None:
        raise ValueError(f"champion version {version.version} of {model_name!r} has no run_id")

    model_uri = f"models:/{model_name}/{version.version}"
    pipeline: Pipeline = mlflow.sklearn.load_model(model_uri)

    background_path = client.download_artifacts(version.run_id, BACKGROUND_ARTIFACT_PATH)
    # keep_default_na=False: a category value that happens to read as "NA"/
    # "None"/"null" (pandas' default NA sentinels) would otherwise silently
    # become NaN instead of that literal string on this CSV round-trip.
    # None of this feature set's current categories collide with those
    # sentinels, but the background sample is training data whose
    # categories aren't validated against pandas' NA list, so don't rely
    # on that staying true.
    background = pd.read_csv(background_path, keep_default_na=False)

    run = client.get_run(version.run_id)
    _warn_if_feature_set_mismatch(run.data.params.get("selected_features"), model_name)

    return ChampionModel(pipeline=pipeline, version=version, background=background)


def _warn_if_feature_set_mismatch(logged_features_param: str | None, model_name: str) -> None:
    """`pipelines/train.py` logs the exact feature list it trained on as a
    string-repr param. If `config/selected_features_<version>.yaml` has
    since changed (e.g. a new `feature_set.version`) without retraining,
    scoring would silently run the champion against a different feature
    set than it was trained on -- this can't catch every such mismatch
    (the yaml could change without the version bumping), but it catches
    the common case cheaply, using data `train.py` already logs."""
    if logged_features_param is None:
        logger.warning("champion_run_missing_selected_features_param", model_name=model_name)
        return
    try:
        logged_features = ast.literal_eval(logged_features_param)
    except (ValueError, SyntaxError):
        logger.warning("champion_selected_features_param_unparseable", model_name=model_name)
        return
    current_features = load_selected_features(get_app_config())
    if list(logged_features) != current_features:
        logger.warning(
            "champion_feature_set_mismatch",
            model_name=model_name,
            trained_on=logged_features,
            configured_now=current_features,
        )


def _transformed_name_to_source_feature(prep: ColumnTransformer) -> list[str]:
    """Maps each of `prep.get_feature_names_out()`'s columns back to the
    *original* (pre-`ColumnTransformer`) feature that produced it, in the
    same order `get_feature_names_out()` returns them.

    `OneHotEncoder` expands one input column into `len(categories_[i])`
    output columns; every other transformer here (`StandardScaler`) is
    1:1. Built from `transformers_`/`categories_` directly rather than
    string-parsing the `"cat__<col>_<category>"` naming convention, so it
    can't be fooled by a category value that itself contains an
    underscore.
    """
    source_names: list[str] = []
    for name, transformer, columns in prep.transformers_:
        if name == "remainder":
            continue
        if hasattr(transformer, "categories_"):
            for col, categories in zip(columns, transformer.categories_, strict=True):
                source_names.extend([col] * len(categories))
        else:
            source_names.extend(columns)
    return source_names


def _warn_on_unseen_categories(prep: ColumnTransformer, features: pd.DataFrame) -> None:
    """`OneHotEncoder(handle_unknown="ignore")` scores a category it never
    saw in training as all-zero dummies -- a valid probability comes out,
    but with no SHAP signal from that feature and nothing else surfacing
    that the input was outside the training distribution. Logs a warning
    so this is at least visible/monitorable, found by an independent
    review setting an out-of-vocabulary category and observing the model
    silently score it anyway."""
    for _, transformer, columns in prep.transformers_:
        if not hasattr(transformer, "categories_"):
            continue
        for col, known_categories in zip(columns, transformer.categories_, strict=True):
            unseen = set(features[col].unique()) - set(known_categories)
            if unseen:
                logger.warning("unseen_category_scored", feature=col, unseen_values=sorted(unseen))


def shap_top_features(
    champion: ChampionModel, features: pd.DataFrame, top_n: int = 3
) -> list[list[ShapFeature]]:
    """Top-`top_n` SHAP features per row of `features`, aggregated back to
    *original* feature names (summed across a categorical feature's
    one-hot columns -- exact for a linear model, not an approximation)
    and returned in descending order of `|shap_value|`.

    Notebook 05's prototype kept the raw one-hot column names
    (e.g. `cat__contract_type_Two year`) instead of aggregating -- fine
    for that notebook's illustrative sample rows, but wrong as a stored,
    served "top reasons" field: a customer's top-3 could (and did, for
    every one-hot categorical feature) include one-hot columns for
    categories that customer doesn't have, reading as contradictory
    nonsense. Found by an independent review, not by inspection -- see
    DECISIONS.md.

    Only supports the linear primary model (`LinearExplainer`) -- this
    project's champion is always `LogisticRegression`
    (DECISIONS.md#logistic-regression-beats-every-gbm-variant); a future
    tree-based champion would need a `TreeExplainer` branch here.
    """
    prep = champion.pipeline.named_steps["prep"]
    clf = champion.pipeline.named_steps["clf"]
    source_features = _transformed_name_to_source_feature(prep)
    _warn_on_unseen_categories(prep, features)

    background_transformed = prep.transform(champion.background)
    if hasattr(background_transformed, "toarray"):
        background_transformed = background_transformed.toarray()

    features_transformed = prep.transform(features)
    if hasattr(features_transformed, "toarray"):
        features_transformed = features_transformed.toarray()

    explainer = shap.LinearExplainer(clf, background_transformed)
    shap_values = explainer(features_transformed).values

    results = []
    for row in shap_values:
        aggregated: dict[str, float] = {}
        for source, value in zip(source_features, row, strict=True):
            aggregated[source] = aggregated.get(source, 0.0) + float(value)
        ranked = sorted(aggregated.items(), key=lambda kv: -abs(kv[1]))[:top_n]
        results.append([ShapFeature(feature=k, shap_value=round(v, 4)) for k, v in ranked])
    return results


def _upsert_predictions_chunk(
    conn: Connection, rows: list[PredictionRow], with_labels: bool
) -> list[dt.datetime]:
    insert_stmt = pg_insert(Prediction).values(list(rows))
    set_ = {
        "probability": insert_stmt.excluded.probability,
        "risk_band": insert_stmt.excluded.risk_band,
        "shap_top3": insert_stmt.excluded.shap_top3,
        "scored_at": text("now()"),
    }
    if with_labels:
        # Only `pipelines/backfill_predictions.py` sends these -- a
        # historical cohort's label is already known (matured) at
        # backfill time, unlike `batch_score.py`/`api/main.py`'s
        # just-scored rows, which never touch these two columns.
        set_["actual_churned"] = insert_stmt.excluded.actual_churned
        set_["label_matured_at"] = insert_stmt.excluded.label_matured_at
    upsert_stmt = insert_stmt.on_conflict_do_update(
        constraint="uq_predictions_customer_ref_model", set_=set_
    ).returning(Prediction.scored_at)
    result = conn.execute(upsert_stmt)
    # Every row in one statement shares its `scored_at` (both the INSERT
    # default and the DO UPDATE both use `now()`, which is the
    # transaction's start time in Postgres) -- so ordering doesn't need to
    # match `rows`' order for this to be correct, only the count does.
    return [row.scored_at for row in result]


def upsert_predictions(conn: Connection, rows: list[PredictionRow]) -> list[dt.datetime]:
    """Inserts scored rows into `predictions`, overwriting on a repeat
    `(customer_id, reference_date, model_version)` rather than
    duplicating it -- shared by `pipelines/batch_score.py` (bulk, one
    cohort at a time), `api/main.py`'s on-demand rescoring endpoint (one
    row at a time), and `pipelines/backfill_predictions.py` (bulk,
    historical cohorts whose label has already matured).

    `rows` must be internally consistent: either every row includes
    `actual_churned`/`label_matured_at` (the backfill case) or none do
    (the live-scoring case) -- never a mix within one call. Determined
    once from `rows[0]`, not re-checked per row.

    Chunked at `_UPSERT_CHUNK_SIZE`: Postgres caps a single statement at
    65535 bound parameters, and this table's 6-8 columns per row means
    one statement maxes out around 8,000-10,922 rows -- comfortably
    above today's ~5,000-customer cohort, but a silent landmine for a
    larger one. Found by an independent review actually pushing a
    single-statement upsert past that limit (`OperationalError: number
    of parameters must be between 0 and 65535`), not by inspection. All
    chunks share the caller's transaction (`conn` is never committed
    here), so a failure partway through still rolls back everything, not
    just the failed chunk.
    """
    if not rows:
        return []
    with_labels = "actual_churned" in rows[0]
    scored_at: list[dt.datetime] = []
    for start in range(0, len(rows), _UPSERT_CHUNK_SIZE):
        chunk = rows[start : start + _UPSERT_CHUNK_SIZE]
        scored_at.extend(_upsert_predictions_chunk(conn, chunk, with_labels))
    return scored_at
