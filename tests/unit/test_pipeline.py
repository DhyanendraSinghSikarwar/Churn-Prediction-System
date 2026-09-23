import pandas as pd

from churn.models.pipeline import build_pipeline, categorical_and_numeric_features


def test_categorical_and_numeric_features_splits_by_dtype() -> None:
    panel = pd.DataFrame(
        {
            "contract_type": pd.array(["Month-to-month", "Two year"], dtype="string"),
            "tenure_months": [1, 24],
            "monthly_charges": [70.5, 20.0],
            "senior_citizen": [True, False],
        }
    )
    cat, num = categorical_and_numeric_features(
        panel, ["contract_type", "tenure_months", "monthly_charges", "senior_citizen"]
    )
    assert set(cat) == {"contract_type", "senior_citizen"}
    assert set(num) == {"tenure_months", "monthly_charges"}


def test_categorical_and_numeric_features_object_dtype_still_categorical() -> None:
    """Pandas 3's default string dtype is not `object` -- this must still
    classify plain object-dtype string columns correctly, not just the
    newer string dtype (see
    DECISIONS.md#pandas-3-default-string-dtype-breaks-dtype-object-checks)."""
    values = pd.Series(["Electronic check", "Mailed check"], dtype=object)
    panel = pd.DataFrame({"payment_method": values})
    cat, num = categorical_and_numeric_features(panel, ["payment_method"])
    assert cat == ["payment_method"]
    assert num == []


def test_build_pipeline_returns_a_fresh_preprocessor_every_call() -> None:
    """Regression test for the exact bug that corrupted Notebook 04's
    threshold-tuning numbers in Phase 4: two pipelines must never share
    one ColumnTransformer instance, because fitting one silently refits
    -- and so corrupts -- the other's already-fitted state. See
    DECISIONS.md#shared-columntransformer-corrupted-notebook-04s-threshold-tuning-numbers.
    """
    pipeline_a = build_pipeline(["cat_col"], ["num_col"], random_state=42)
    pipeline_b = build_pipeline(["cat_col"], ["num_col"], random_state=42)
    assert pipeline_a.named_steps["prep"] is not pipeline_b.named_steps["prep"]
    assert pipeline_a is not pipeline_b


def test_build_pipeline_fits_and_predicts() -> None:
    panel = pd.DataFrame(
        {
            "contract_type": ["Month-to-month", "Two year", "One year", "Month-to-month"] * 5,
            "tenure_months": [1, 24, 12, 3] * 5,
            "label": [1, 0, 0, 1] * 5,
        }
    )
    pipeline = build_pipeline(["contract_type"], ["tenure_months"], random_state=42)
    pipeline.fit(panel[["contract_type", "tenure_months"]], panel["label"])
    proba = pipeline.predict_proba(panel[["contract_type", "tenure_months"]])
    assert proba.shape == (len(panel), 2)
