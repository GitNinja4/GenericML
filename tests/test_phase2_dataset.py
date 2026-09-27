from io import StringIO

import pandas as pd
import pytest

from src.data.analyzer import detect_problem_type, get_column_types
from src.data.loader import load_csv
from src.data.validator import validate_dataset, validate_target


def test_load_csv_reads_csv() -> None:
    csv_data = StringIO("age,income,churn\n25,50000,1\n30,60000,0\n")
    df = load_csv(csv_data)
    assert list(df.columns) == ["age", "income", "churn"]
    assert df.shape == (2, 3)


def test_load_csv_rejects_duplicate_header_names() -> None:
    with pytest.raises(ValueError, match="Duplicate column names detected"):
        load_csv(StringIO("value,value,target\n1,2,0\n3,4,1\n"))


def test_load_csv_reports_empty_file() -> None:
    with pytest.raises(ValueError, match="empty"):
        load_csv(StringIO(""))


def test_detect_problem_type_classification() -> None:
    target = pd.Series(["yes", "no", "yes", "no"], name="churn")
    assert detect_problem_type(target) == "classification"


def test_detect_problem_type_regression() -> None:
    target = pd.Series([100.0, 120.0, 150.0, 170.0], name="price")
    assert detect_problem_type(target) == "regression"


def test_detect_problem_type_fractional_numeric_target_as_regression() -> None:
    target = pd.Series([0.5, 1.5, 2.5, 3.5], name="measurement")
    assert detect_problem_type(target) == "regression"


def test_detect_problem_type_arbitrary_integer_labels_as_classification() -> None:
    target = pd.Series([10, 20, 10, 20, 30, 30], name="coded_class")
    assert detect_problem_type(target) == "classification"


def test_validate_dataset_duplicate_columns_raises() -> None:
    df = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "a"])
    with pytest.raises(ValueError):
        validate_dataset(df)


def test_validate_target_single_class_raises() -> None:
    with pytest.raises(ValueError):
        validate_target(pd.Series(["yes", "yes", "yes"]), problem_type="classification")


def test_validate_target_rejects_nonnumeric_regression_target() -> None:
    with pytest.raises(ValueError, match="must be numeric"):
        validate_target(pd.Series(["low", "high"], name="target"), problem_type="regression")


def test_get_column_types_separates_numeric_and_categorical() -> None:
    df = pd.DataFrame({
        "age": [25, 30, 35],
        "gender": ["M", "F", "M"],
        "flag": [True, False, True],
    })
    info = get_column_types(df)
    assert "numeric" in info["age"]
    assert "categorical" in info["gender"]
    assert "boolean" in info["flag"]


def test_mixed_type_classification_target_renders_dataset_summary() -> None:
    from streamlit.testing.v1 import AppTest
    from pathlib import Path

    app = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=15)
    app.session_state["dataset"] = pd.DataFrame(
        {"feature": [1.0, 2.0, 3.0, 4.0], "target": [1, "A", 1, "B"]}
    )
    app.session_state["dataset_valid"] = True
    app.session_state["target_column"] = "target"
    app.run()

    assert not app.exception, app.exception
    assert app.session_state["problem_type"] == "classification"
