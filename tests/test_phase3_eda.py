import pandas as pd

from src.data.analyzer import (
    calculate_skewness,
    detect_constant_columns,
    detect_high_cardinality,
    detect_high_correlations,
    detect_outliers,
    get_categorical_summary,
    get_correlation_matrix,
    get_duplicate_summary,
    get_missing_value_summary,
    get_numerical_summary,
    summarize_classification_target,
    summarize_regression_target,
)


def make_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [20, 30, 40, 100, 30],
            "income": [10.0, 20.0, 30.0, 100.0, 20.0],
            "segment": ["A", "B", "A", "C", "B"],
            "constant": [1, 1, 1, 1, 1],
            "target": [0, 1, 0, 1, 1],
        }
    )


def test_missing_value_summary() -> None:
    df = pd.DataFrame({"a": [1, None, 3, None], "b": [1, 2, 3, 4]})
    summary = get_missing_value_summary(df).set_index("Column")
    assert summary.loc["a", "Missing Count"] == 2
    assert summary.loc["a", "Missing %"] == 50.0


def test_duplicate_summary() -> None:
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    assert get_duplicate_summary(df) == {"count": 1, "percentage": 33.33}


def test_numerical_summary() -> None:
    summary = get_numerical_summary(pd.DataFrame({"value": [1, 2, 3]})).set_index("Feature")
    assert summary.loc["value", "Mean"] == 2.0
    assert summary.loc["value", "Median"] == 2.0
    assert summary.loc["value", "Min"] == 1.0
    assert summary.loc["value", "Max"] == 3.0


def test_categorical_summary() -> None:
    summary = get_categorical_summary(pd.DataFrame({"kind": ["a", "a", "b"]}), "kind").set_index("Category")
    assert summary.loc["a", "Count"] == 2
    assert summary.loc["a", "Percentage"] == 66.67


def test_categorical_summary_handles_missing_category_dtype_values() -> None:
    dataset = pd.DataFrame({"kind": pd.Series(pd.Categorical(["a", None, "b"]))})
    summary = get_categorical_summary(dataset, "kind").set_index("Category")
    assert summary.loc["<Missing>", "Count"] == 1


def test_correlation_and_high_correlation_detection() -> None:
    df = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6], "label": ["a", "b", "c"]})
    assert get_correlation_matrix(df).loc["x", "y"] == 1.0
    strong = detect_high_correlations(df)
    assert strong.iloc[0]["Feature A"] == "x"
    assert strong.iloc[0]["Feature B"] == "y"


def test_constant_and_high_cardinality_detection() -> None:
    df = make_dataset()
    assert detect_constant_columns(df) == ["constant"]
    high_cardinality = detect_high_cardinality(pd.DataFrame({"id": [f"id-{i}" for i in range(60)]}))
    assert high_cardinality.iloc[0]["Feature"] == "id"


def test_outlier_detection() -> None:
    outliers = detect_outliers(pd.DataFrame({"value": [1, 2, 3, 4, 100]})).set_index("Feature")
    assert outliers.loc["value", "Outlier Count"] == 1


def test_classification_target_summary() -> None:
    report = summarize_classification_target(pd.Series(["yes", "no", "yes", None], name="churn"))
    assert report["n_classes"] == 2
    assert report["missing_count"] == 1
    assert report["class_counts"].set_index("Class").loc["yes", "Percentage"] == 66.67


def test_regression_target_summary() -> None:
    report = summarize_regression_target(pd.Series([10.0, 20.0, 30.0, None], name="price"))
    assert report["count"] == 3
    assert report["missing_count"] == 1
    assert report["mean"] == 20.0
    assert report["median"] == 20.0
    assert report["min"] == 10.0
    assert report["max"] == 30.0


def test_no_numerical_columns_is_supported() -> None:
    assert get_numerical_summary(pd.DataFrame({"kind": ["a", "b"]})).empty
    assert get_correlation_matrix(pd.DataFrame({"kind": ["a", "b"]})).empty


def test_no_categorical_columns_is_supported() -> None:
    summary = get_categorical_summary(pd.DataFrame({"value": [1, 2]}), "value")
    assert summary.iloc[0]["Category"] == 1


def test_skewness_marks_strong_features() -> None:
    skewness = calculate_skewness(pd.DataFrame({"value": [1, 1, 1, 1, 100]})).set_index("Feature")
    assert bool(skewness.loc["value", "Strong Skewness"])
