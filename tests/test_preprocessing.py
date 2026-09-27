import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from src.preprocessing.encoding import create_encoder
from src.preprocessing.feature_selection import create_feature_selector
from src.preprocessing.missing_values import create_categorical_imputer, create_numeric_imputer
from src.preprocessing.outliers import IQRTransformer
from src.preprocessing.pipeline_builder import (
    PreprocessingConfig,
    build_preprocessing_pipeline,
    get_processed_feature_names,
    validate_preprocessing_config,
)
from src.preprocessing.scaling import create_scaler


def test_missing_value_strategies() -> None:
    assert isinstance(create_numeric_imputer("Mean"), SimpleImputer)
    assert isinstance(create_numeric_imputer("Median"), SimpleImputer)
    assert isinstance(create_numeric_imputer("Constant", 7), SimpleImputer)
    assert create_numeric_imputer("None") is None
    assert isinstance(create_categorical_imputer("Most Frequent"), SimpleImputer)
    assert isinstance(create_categorical_imputer("Constant", "unknown"), SimpleImputer)
    assert create_categorical_imputer("None") is None


def test_encoding_strategies_and_unknown_categories() -> None:
    encoder = create_encoder("One-Hot Encoding")
    encoder.fit(pd.DataFrame({"kind": ["A", "B"]}))
    transformed = encoder.transform(pd.DataFrame({"kind": ["unseen"]}))
    assert transformed.shape == (1, 2)
    assert create_encoder("Ordinal Encoding").handle_unknown == "use_encoded_value"
    assert create_encoder("None") is None


def test_scaling_strategies() -> None:
    assert isinstance(create_scaler("StandardScaler"), StandardScaler)
    assert isinstance(create_scaler("MinMaxScaler"), MinMaxScaler)
    assert isinstance(create_scaler("RobustScaler"), RobustScaler)
    assert create_scaler("None") is None


def test_feature_selection_scoring_and_validation() -> None:
    assert create_feature_selector("classification", "SelectKBest", 1).score_func.__name__ == "mutual_info_classif"
    assert create_feature_selector("regression", "SelectKBest", 1).score_func.__name__ == "mutual_info_regression"
    with pytest.raises(ValueError):
        create_feature_selector("classification", "SelectKBest", 0)


def test_configuration_validation() -> None:
    base = PreprocessingConfig(selected_features=["age"], target_column="target", problem_type="classification")
    validate_preprocessing_config(base, ["age", "target"], "target")
    with pytest.raises(ValueError, match="at least one"):
        validate_preprocessing_config(PreprocessingConfig(problem_type="classification"), ["age"], "target")
    with pytest.raises(ValueError, match="target"):
        validate_preprocessing_config(
            PreprocessingConfig(selected_features=["target"], problem_type="classification"), ["age", "target"], "target"
        )
    with pytest.raises(ValueError, match="K"):
        validate_preprocessing_config(
            PreprocessingConfig(selected_features=["age"], feature_selection_enabled=True, feature_selection_k=2, problem_type="classification"),
            ["age"],
            "target",
        )


def make_mixed_data() -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame(
        {
            "age": [20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
            "income": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
            "kind": ["A", "B", "A", "B", "A", "B"],
            "excluded": [1, 1, 1, 1, 1, 1],
        }
    )
    return X, pd.Series([0, 1, 0, 1, 0, 1], name="target")


def test_pipeline_fit_transform_and_excluded_columns() -> None:
    X, y = make_mixed_data()
    config = PreprocessingConfig(
        selected_features=["age", "income", "kind"],
        excluded_features=["excluded"],
        problem_type="classification",
        encoding_strategy="One-Hot Encoding",
        scaling_strategy="StandardScaler",
    )
    pipeline = build_preprocessing_pipeline(config, ["age", "income"], ["kind"])
    pipeline.fit(X[config.selected_features], y)
    transformed = pipeline.transform(X[config.selected_features])
    names = get_processed_feature_names(pipeline)
    assert transformed.shape[0] == len(X)
    assert "excluded" not in names
    assert any("kind" in name for name in names)


def test_pipeline_works_with_numerical_only_data() -> None:
    X = pd.DataFrame({"age": [1.0, 2.0, 3.0], "income": [10.0, 20.0, 30.0]})
    config = PreprocessingConfig(selected_features=["age", "income"], problem_type="regression", scaling_strategy="MinMaxScaler")
    pipeline = build_preprocessing_pipeline(config, ["age", "income"], [])
    assert pipeline.fit_transform(X, pd.Series([1.0, 2.0, 3.0])).shape == (3, 2)


def test_pipeline_works_with_categorical_only_data() -> None:
    X = pd.DataFrame({"kind": ["A", "B", "A"]})
    config = PreprocessingConfig(selected_features=["kind"], problem_type="classification", encoding_strategy="Ordinal Encoding")
    pipeline = build_preprocessing_pipeline(config, [], ["kind"])
    assert pipeline.fit_transform(X, pd.Series([0, 1, 0])).shape == (3, 1)


def test_feature_selection_pipeline_for_regression() -> None:
    X = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6], "b": [6, 5, 4, 3, 2, 1], "c": [1, 1, 2, 2, 3, 3]})
    y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    config = PreprocessingConfig(selected_features=["a", "b", "c"], problem_type="regression", feature_selection_enabled=True, feature_selection_k=2)
    pipeline = build_preprocessing_pipeline(config, ["a", "b", "c"], [])
    transformed = pipeline.fit_transform(X, y)
    assert transformed.shape == (6, 2)


def test_leakage_safe_fit_uses_training_statistics_only() -> None:
    train = pd.DataFrame({"value": [1.0, 2.0, np.nan]})
    test = pd.DataFrame({"value": [100.0]})
    config = PreprocessingConfig(selected_features=["value"], problem_type="regression", numeric_missing_strategy="Median")
    pipeline = build_preprocessing_pipeline(config, ["value"], [])
    pipeline.fit(train, pd.Series([1.0, 2.0, 3.0]))
    assert pipeline.named_steps["preprocessor"].named_transformers_["numeric"].named_steps["imputer"].statistics_[0] == 1.5
    assert pipeline.transform(test)[0, 0] == 100.0


def test_iqr_transformer_clip_and_remove() -> None:
    data = pd.DataFrame({"value": [1.0, 2.0, 3.0, 100.0]})
    clipper = IQRTransformer("clip").fit(data)
    assert clipper.transform(data)["value"].max() < 100.0
    remover = IQRTransformer("remove").fit(data)
    assert len(remover.transform(data)) == 3
    assert remover.get_support_mask().tolist() == [True, True, True, False]
