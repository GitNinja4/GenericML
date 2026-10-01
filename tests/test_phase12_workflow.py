from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from src.evaluation.evaluator import evaluate_pipeline
from src.prediction.predictor import predict_with_pipeline
from src.prediction.schema import build_prediction_schema
from src.preprocessing.pipeline_builder import PreprocessingConfig, build_preprocessing_pipeline
from src.training.splitter import split_dataset
from src.training.workflow import train_models
from src.tuning.optuna_optimizer import tune_pipeline
from src.tuning.search_spaces import build_search_space


def _classification_data(rows: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "income": [float(index + 1) for index in range(rows)],
            "region": ["north", "south", "west", "east"] * (rows // 4),
            "target": [index % 2 for index in range(rows)],
        }
    )


def _regression_data(rows: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": np.arange(rows, dtype=float),
            "target": np.arange(rows, dtype=float) * 2.0 + 1.0,
        }
    )


def _config(problem_type: str, features: list[str], categorical: bool = False) -> PreprocessingConfig:
    return PreprocessingConfig(
        numeric_missing_strategy="Median",
        categorical_missing_strategy="Most Frequent" if categorical else "None",
        encoding_strategy="One-Hot Encoding" if categorical else "None",
        selected_features=features,
        target_column="target",
        problem_type=problem_type,
    )


def test_classification_cv_smote_nc_trains_using_training_partition_only() -> None:
    dataset = _classification_data()
    X_train, X_test, y_train, y_test = split_dataset(
        dataset[["income", "region"]], dataset["target"], "classification", test_size=0.2, random_state=42, stratified=True
    )
    config = _config("classification", ["income", "region"], categorical=True)
    preprocessing = build_preprocessing_pipeline(config, ["income"], ["region"])

    result = train_models(
        X_train,
        y_train,
        preprocessing,
        ["Logistic Regression"],
        "classification",
        preprocessing_config=config.to_dict(),
        cv_folds=3,
        random_state=42,
        shuffle=True,
        validation_strategy="K-Fold Cross-Validation",
        sampling_method="SMOTENC",
    )

    assert result["training_results"]["Logistic Regression"]["status"] == "trained"
    assert result["training_results"]["Logistic Regression"]["validation_metrics_available"] is True
    assert len(X_test) > 0
    assert result["trained_models"]["Logistic Regression"]["pipeline"] is not None
    classification_evaluation = evaluate_pipeline(
        result["trained_models"]["Logistic Regression"]["pipeline"], X_test, y_test, "classification"
    )
    assert set(("accuracy", "precision", "recall", "f1", "confusion_matrix")).issubset(classification_evaluation)

    numeric_config = _config("classification", ["income"])
    numeric_pipeline = build_preprocessing_pipeline(numeric_config, ["income"], [])
    numeric_result = train_models(
        X_train[["income"]],
        y_train,
        numeric_pipeline,
        ["Logistic Regression"],
        "classification",
        preprocessing_config=numeric_config.to_dict(),
        validation_strategy="None",
        sampling_method="SMOTE",
    )
    assert numeric_result["training_results"]["Logistic Regression"]["status"] == "trained"


def test_preprocessing_fit_does_not_learn_from_test_sentinel() -> None:
    X_train = pd.DataFrame({"income": [10.0, 20.0, 30.0], "region": ["north", "south", "north"]})
    X_test = pd.DataFrame({"income": [999999.0], "region": ["secret-test-category"]})
    config = _config("regression", ["income", "region"], categorical=True)
    pipeline = build_preprocessing_pipeline(config, ["income"], ["region"])

    pipeline.fit(X_train, pd.Series([1.0, 2.0, 3.0]))
    numeric_imputer = pipeline.named_steps["preprocessor"].named_transformers_["numeric"].named_steps["imputer"]
    categorical_encoder = pipeline.named_steps["preprocessor"].named_transformers_["categorical"].named_steps["encoder"]

    assert float(numeric_imputer.statistics_[0]) == 20.0
    assert "secret-test-category" not in categorical_encoder.categories_[0]


def test_regression_evaluation_and_schema_prediction_use_final_pipeline() -> None:
    dataset = _regression_data()
    X_train, X_test, y_train, y_test = split_dataset(
        dataset[["feature"]], dataset["target"], "regression", test_size=0.2, random_state=42, stratified=False
    )
    config = _config("regression", ["feature"])
    preprocessing = build_preprocessing_pipeline(config, ["feature"], [])
    result = train_models(
        X_train,
        y_train,
        preprocessing,
        ["Linear Regression"],
        "regression",
        preprocessing_config=config.to_dict(),
        validation_strategy="K-Fold Cross-Validation",
        cv_folds=3,
        random_state=42,
        shuffle=True,
    )
    pipeline = result["trained_models"]["Linear Regression"]["pipeline"]
    evaluation = evaluate_pipeline(pipeline, X_test, y_test, "regression")
    schema = build_prediction_schema(dataset, "target", ["feature"])
    prediction = predict_with_pipeline(pipeline, {"feature": 100.0}, schema, "regression")

    assert set(("mae", "mse", "rmse", "r2")).issubset(evaluation)
    assert evaluation["sample_count"] == len(X_test)
    assert isinstance(prediction["prediction"], float)


def test_tuning_uses_complete_training_pipeline_and_no_holdout_argument() -> None:
    dataset = _regression_data()
    X_train, _, y_train, _ = split_dataset(
        dataset[["feature"]], dataset["target"], "regression", test_size=0.2, random_state=42, stratified=False
    )
    config = _config("regression", ["feature"])
    preprocessing = build_preprocessing_pipeline(config, ["feature"], [])
    trained = train_models(
        X_train,
        y_train,
        preprocessing,
        ["Linear Regression"],
        "regression",
        preprocessing_config=config.to_dict(),
        validation_strategy="None",
        random_state=42,
    )
    pipeline = trained["trained_models"]["Linear Regression"]["pipeline"]
    tuned = tune_pipeline(
        pipeline,
        X_train,
        y_train,
        build_search_space("Linear Regression", "regression"),
        "Grid Search",
        cv=3,
        scoring="r2",
        random_state=42,
    )

    assert tuned["best_estimator"] is not None
    assert tuned["candidate_count"] >= 1
    assert "fit_intercept" in tuned["best_params"]


def test_new_dataset_upload_resets_downstream_state() -> None:
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py").run(timeout=30)
    first = b"feature,target\n1,2\n2,4\n3,6\n4,8\n"
    second = b"feature,target\n10,1\n20,2\n30,3\n40,4\n"
    app.file_uploader[0].upload("first.csv", first).run(timeout=30)
    app.session_state["selected_models"] = ["Linear Regression"]
    app.session_state["trained_models"] = {"stale": object()}
    app.file_uploader[0].upload("second.csv", second).run(timeout=30)

    assert app.session_state["dataset_name"] == "second.csv"
    assert app.session_state["selected_models"] == []
    assert app.session_state["trained_models"] == {}
    assert app.session_state["split_completed"] is False
