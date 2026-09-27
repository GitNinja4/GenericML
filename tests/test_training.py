import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from src.preprocessing.pipeline_builder import PreprocessingConfig, build_preprocessing_pipeline
from src.training.workflow import train_models


def classification_data() -> tuple[pd.DataFrame, pd.Series, object]:
    features, target = make_classification(n_samples=80, n_features=4, n_informative=3, n_redundant=0, random_state=7)
    X = pd.DataFrame(features, columns=[f"feature_{index}" for index in range(4)])
    y = pd.Series(target, name="target")
    config = PreprocessingConfig(selected_features=list(X.columns), problem_type="classification", scaling_strategy="StandardScaler")
    return X, y, build_preprocessing_pipeline(config, list(X.columns), [])


def regression_data() -> tuple[pd.DataFrame, pd.Series, object]:
    features, target = make_regression(n_samples=80, n_features=4, random_state=7)
    X = pd.DataFrame(features, columns=[f"feature_{index}" for index in range(4)])
    y = pd.Series(target, name="target")
    config = PreprocessingConfig(selected_features=list(X.columns), problem_type="regression", scaling_strategy="StandardScaler")
    return X, y, build_preprocessing_pipeline(config, list(X.columns), [])


def test_classification_training_creates_complete_pipeline() -> None:
    X, y, preprocessor = classification_data()
    result = train_models(X, y, preprocessor, ["Logistic Regression", "Random Forest"], "classification", 0.2, 19)
    assert set(result["trained_models"]) == {"Logistic Regression", "Random Forest"}
    for metadata in result["trained_models"].values():
        assert metadata["pipeline"].named_steps["preprocessor"] is not preprocessor
        assert metadata["pipeline"].named_steps["model"] is not None
        assert metadata["status"] == "trained"
    assert len(result["X_train"]) == 64
    assert len(result["X_test"]) == 16


def test_regression_training_does_not_stratify() -> None:
    X, y, preprocessor = regression_data()
    result = train_models(X, y, preprocessor, ["Linear Regression", "Ridge"], "regression", 0.25, 19)
    assert set(result["trained_models"]) == {"Linear Regression", "Ridge"}
    assert len(result["X_test"]) == 20
    predictions = result["trained_models"]["Ridge"]["pipeline"].predict(result["X_test"])
    assert predictions.shape == (20,)


def test_training_uses_training_only_preprocessing_fit() -> None:
    X = pd.DataFrame({"value": [1.0, 2.0, np.nan, 4.0, 5.0, 1000.0]})
    y = pd.Series([0, 1, 0, 1, 0, 1], name="target")
    config = PreprocessingConfig(selected_features=["value"], problem_type="classification", numeric_missing_strategy="Median")
    preprocessor = build_preprocessing_pipeline(config, ["value"], [])
    result = train_models(X, y, preprocessor, ["Logistic Regression"], "classification", 0.33, 2)
    fitted = result["trained_models"]["Logistic Regression"]["pipeline"]
    imputer = fitted.named_steps["preprocessor"].named_steps["preprocessor"].named_transformers_["numeric"].named_steps["imputer"]
    train_values = result["X_train"]["value"].dropna()
    assert imputer.statistics_[0] == train_values.median()
    assert imputer.statistics_[0] != X["value"].median()


def test_remove_outliers_filters_training_rows_without_misalignment() -> None:
    X = pd.DataFrame(
        {
            "value": [1000.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            "group": ["A", "B"] * 5,
        }
    )
    y = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0], name="target")
    config = PreprocessingConfig(
        selected_features=["value", "group"],
        target_column="target",
        problem_type="regression",
        outlier_enabled=True,
        outlier_action="Remove",
        encoding_strategy="One-Hot Encoding",
    )
    preprocessor = build_preprocessing_pipeline(config, ["value"], ["group"])

    result = train_models(
        X,
        y,
        preprocessor,
        ["Ridge"],
        "regression",
        test_size=0.2,
        random_state=42,
        preprocessing_config=config.to_dict(),
    )

    assert result["outlier_rows_removed"] == 1
    assert len(result["X_train"]) == len(result["y_train"]) == 7
    assert len(result["X_test"]) == len(result["y_test"]) == 2
    numeric_pipeline = result["trained_models"]["Ridge"]["pipeline"].named_steps["preprocessor"].named_steps["preprocessor"].named_transformers_["numeric"]
    assert not getattr(numeric_pipeline, "named_steps", {})

def test_all_model_failures_return_per_model_error_records() -> None:
    X = pd.DataFrame({"value": [-8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0]})
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1], name="target")
    config = PreprocessingConfig(selected_features=["value"], problem_type="classification")
    preprocessor = build_preprocessing_pipeline(config, ["value"], [])

    result = train_models(X, y, preprocessor, ["Multinomial Naive Bayes"], "classification", 0.25, 42)

    assert not result["trained_models"]
    assert result["training_results"]["Multinomial Naive Bayes"]["status"] == "failed"
    assert result["training_results"]["Multinomial Naive Bayes"]["error"]


def test_training_isolates_model_failure_from_successful_models() -> None:
    X = pd.DataFrame({"value": [-8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0]})
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1], name="target")
    config = PreprocessingConfig(selected_features=["value"], problem_type="classification")
    preprocessor = build_preprocessing_pipeline(config, ["value"], [])

    result = train_models(
        X,
        y,
        preprocessor,
        ["Logistic Regression", "Multinomial Naive Bayes"],
        "classification",
        0.25,
        42,
    )

    assert result["training_results"]["Logistic Regression"]["status"] == "trained"
    assert result["training_results"]["Multinomial Naive Bayes"]["status"] == "failed"
    assert set(result["trained_models"]) == {"Logistic Regression"}


def test_training_records_estimator_warnings() -> None:
    X, y, preprocessor = classification_data()
    result = train_models(
        X,
        y,
        preprocessor,
        ["Logistic Regression"],
        "classification",
        model_configs={"Logistic Regression": {"max_iter": 1}},
    )

    warnings = result["training_results"]["Logistic Regression"]["warnings"]
    assert warnings
    assert any("converge" in warning.casefold() for warning in warnings)


def test_small_classification_split_reports_stratification_error() -> None:
    X = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
    y = pd.Series([0, 1, 1], name="target")
    config = PreprocessingConfig(selected_features=["value"], problem_type="classification")
    preprocessor = build_preprocessing_pipeline(config, ["value"], [])
    with pytest.raises(ValueError, match="Stratified classification split failed"):
        train_models(X, y, preprocessor, ["Logistic Regression"], "classification", 0.5, 42)


def test_training_validation_errors() -> None:
    X, y, preprocessor = classification_data()
    with pytest.raises(ValueError, match="at least one model"):
        train_models(X, y, preprocessor, [], "classification")
    with pytest.raises(ValueError, match="preprocessing"):
        train_models(X, y, None, ["Logistic Regression"], "classification")
    with pytest.raises(ValueError, match="Test size"):
        train_models(X, y, preprocessor, ["Logistic Regression"], "classification", 1.0)


def test_training_rejects_misaligned_feature_and_target_indices() -> None:
    X = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0]}, index=[0, 1, 2, 3])
    y = pd.Series([0, 1, 0, 1], index=[10, 11, 12, 13], name="target")
    config = PreprocessingConfig(selected_features=["value"], problem_type="classification")
    preprocessor = build_preprocessing_pipeline(config, ["value"], [])

    with pytest.raises(ValueError, match="indices must match"):
        train_models(X, y, preprocessor, ["Logistic Regression"], "classification")


def test_training_rejects_missing_target_values_before_model_fitting() -> None:
    X = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0]})
    y = pd.Series([0, 1, np.nan, 1], name="target")
    config = PreprocessingConfig(selected_features=["value"], problem_type="classification")
    preprocessor = build_preprocessing_pipeline(config, ["value"], [])

    with pytest.raises(ValueError, match="target contains 1 missing"):
        train_models(X, y, preprocessor, ["Logistic Regression"], "classification")


def test_training_rejects_nonnumeric_regression_target() -> None:
    X = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0]})
    y = pd.Series(["low", "medium", "high", "higher"], name="target")
    config = PreprocessingConfig(selected_features=["value"], problem_type="regression")
    preprocessor = build_preprocessing_pipeline(config, ["value"], [])

    with pytest.raises(ValueError, match="regression target must be numeric"):
        train_models(X, y, preprocessor, ["Ridge"], "regression")
