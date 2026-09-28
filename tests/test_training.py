import inspect

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.preprocessing.pipeline_builder import PreprocessingConfig, build_preprocessing_pipeline
from src.training.smote import TrainingDataImputer
import src.training.workflow as training_workflow
import src.training.trainer as training_trainer
from src.training.workflow import train_models


def classification_data() -> tuple[pd.DataFrame, pd.Series, object]:
	features, target = make_classification(n_samples=80, n_features=4, n_informative=3, n_redundant=0, random_state=7)
	X_train = pd.DataFrame(features, columns=[f"feature_{index}" for index in range(4)])
	y_train = pd.Series(target, name="target")
	config = PreprocessingConfig(selected_features=list(X_train.columns), problem_type="classification", scaling_strategy="StandardScaler")
	return X_train, y_train, build_preprocessing_pipeline(config, list(X_train.columns), [])


def regression_data() -> tuple[pd.DataFrame, pd.Series, object]:
	features, target = make_regression(n_samples=80, n_features=4, random_state=7)
	X_train = pd.DataFrame(features, columns=[f"feature_{index}" for index in range(4)])
	y_train = pd.Series(target, name="target")
	config = PreprocessingConfig(selected_features=list(X_train.columns), problem_type="regression", scaling_strategy="StandardScaler")
	return X_train, y_train, build_preprocessing_pipeline(config, list(X_train.columns), [])


def test_training_fits_final_pipeline_and_reports_fold_metrics() -> None:
	X_train, y_train, preprocessor = classification_data()
	result = train_models(
		X_train,
		y_train,
		preprocessor,
		["Logistic Regression", "Random Forest"],
		"classification",
		cv_folds=5,
		random_state=19,
		scoring_metric="accuracy",
		validation_strategy="K-Fold Cross-Validation",
	)
	assert set(result["trained_models"]) == {"Logistic Regression", "Random Forest"}
	assert "X_test" not in result and "y_test" not in result
	for model_name, model_state in result["trained_models"].items():
		assert model_state["pipeline"].named_steps["preprocessor"] is not preprocessor
		assert model_state["status"] == "trained"
		assert model_state["parameters"] == model_state["pipeline"].named_steps["model"].get_params(deep=False)
		assert model_state["training_time_seconds"] > 0
		assert len(model_state["cv_metrics"]["accuracy"]["fold_scores"]) == 5
		assert model_state["cv_configuration"]["cv_folds"] == 5
		assert result["training_results"][model_name]["training_metrics"]["accuracy"] >= 0


def test_regression_training_reports_regression_metrics() -> None:
	X_train, y_train, preprocessor = regression_data()
	result = train_models(
		X_train,
		y_train,
		preprocessor,
		["Linear Regression", "Ridge Regression"],
		"regression",
		cv_folds=4,
		random_state=19,
		scoring_metric="r2",
		validation_strategy="K-Fold Cross-Validation",
	)
	assert set(result["trained_models"]) == {"Linear Regression", "Ridge Regression"}
	for model_result in result["training_results"].values():
		assert set(model_result["training_metrics"]) == {"mae", "mse", "rmse", "r2"}
		assert set(model_result["cv_metrics"]) == {"mae", "mse", "rmse", "r2"}
		assert len(model_result["cv_metrics"]["r2"]["fold_scores"]) == 4


def test_classification_training_reports_only_training_metrics() -> None:
	X_train, y_train, preprocessor = classification_data()
	result = train_models(
		X_train,
		y_train,
		preprocessor,
		["Logistic Regression"],
		"classification",
		cv_folds=4,
		scoring_metric="f1",
		validation_strategy="K-Fold Cross-Validation",
	)
	model_result = result["training_results"]["Logistic Regression"]
	assert set(model_result["training_metrics"]) == {"accuracy", "precision", "recall", "f1"}
	assert set(model_result["cv_metrics"]) == {"accuracy", "precision", "recall", "f1"}
	assert len(model_result["cv_metrics"]["f1"]["fold_scores"]) == 4
	assert model_result["scoring_metric"] == "f1"


def test_preprocessing_fits_only_on_supplied_training_rows() -> None:
	X_train = pd.DataFrame({"value": [1.0, 2.0, np.nan, 4.0, 5.0]})
	y_train = pd.Series([0, 1, 0, 1, 0], name="target")
	config = PreprocessingConfig(selected_features=["value"], problem_type="classification", numeric_missing_strategy="Median")
	preprocessor = build_preprocessing_pipeline(config, ["value"], [])
	result = train_models(X_train, y_train, preprocessor, ["Logistic Regression"], "classification", cv_folds=2)
	fitted = result["trained_models"]["Logistic Regression"]["pipeline"]
	imputer = fitted.named_steps["preprocessor"].named_steps["preprocessor"].named_transformers_["numeric"].named_steps["imputer"]
	assert imputer.statistics_[0] == X_train["value"].median()


def test_remove_outliers_learns_bounds_only_from_training_data() -> None:
	X_train = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 5.0, 1000.0]})
	y_train = pd.Series([0, 1, 0, 1, 0, 1], name="target")
	config = PreprocessingConfig(
		selected_features=["value"],
		problem_type="classification",
		outlier_enabled=True,
		outlier_action="Remove",
	)
	preprocessor = build_preprocessing_pipeline(config, ["value"], [])
	result = train_models(
		X_train,
		y_train,
		preprocessor,
		["Logistic Regression"],
		"classification",
		preprocessing_config=config.to_dict(),
		cv_folds=2,
	)
	model_result = result["training_results"]["Logistic Regression"]
	assert model_result["outlier_rows_removed"] >= 0
	assert model_result["final_fit_samples"] == len(X_train) - model_result["outlier_rows_removed"]


def test_training_cannot_accept_a_test_partition() -> None:
	assert "X_test" not in inspect.signature(train_models).parameters
	assert "y_test" not in inspect.signature(train_models).parameters


def test_preprocessing_fits_inside_each_cv_training_fold(monkeypatch: pytest.MonkeyPatch) -> None:
	class FitRowRecorder(BaseEstimator, TransformerMixin):
		fit_sizes: list[int] = []

		def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FitRowRecorder":
			type(self).fit_sizes.append(len(X))
			return self

		def transform(self, X: pd.DataFrame) -> pd.DataFrame:
			return X

	X_train = pd.DataFrame({"value": np.arange(20, dtype=float)})
	y_train = pd.Series([0, 1] * 10, name="target")
	FitRowRecorder.fit_sizes.clear()

	def build_recording_pipeline(*args: object, **kwargs: object) -> Pipeline:
		return Pipeline([("preprocessor", FitRowRecorder()), ("model", LogisticRegression())])

	monkeypatch.setattr(training_workflow, "build_model_pipeline", build_recording_pipeline)
	config = PreprocessingConfig(selected_features=["value"], problem_type="classification")
	preprocessor = build_preprocessing_pipeline(config, ["value"], [])

	train_models(
		X_train,
		y_train,
		preprocessor,
		["Logistic Regression"],
		"classification",
		cv_folds=4,
		random_state=11,
		validation_strategy="K-Fold Cross-Validation",
	)

	assert FitRowRecorder.fit_sizes == [15, 15, 15, 15, 20]


def test_all_model_failures_return_per_model_error_records() -> None:
	X_train = pd.DataFrame({"value": [-8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0]})
	y_train = pd.Series([0, 1, 0, 1, 0, 1, 0, 1], name="target")
	config = PreprocessingConfig(selected_features=["value"], problem_type="classification")
	preprocessor = build_preprocessing_pipeline(config, ["value"], [])
	result = train_models(
		X_train,
		y_train,
		preprocessor,
		["Logistic Regression"],
		"classification",
		model_configs={"Logistic Regression": {"C": 0}},
		cv_folds=2,
	)
	assert not result["trained_models"]
	assert result["training_results"]["Logistic Regression"]["status"] == "failed"
	assert result["training_results"]["Logistic Regression"]["error"]


def test_training_isolates_model_failure_from_successful_models() -> None:
	X_train = pd.DataFrame({"value": [-8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0]})
	y_train = pd.Series([0, 1, 0, 1, 0, 1, 0, 1], name="target")
	config = PreprocessingConfig(selected_features=["value"], problem_type="classification")
	preprocessor = build_preprocessing_pipeline(config, ["value"], [])
	result = train_models(
		X_train,
		y_train,
		preprocessor,
		["Logistic Regression", "Support Vector Machine (SVM)"],
		"classification",
		model_configs={"Support Vector Machine (SVM)": {"C": 0}},
		cv_folds=2,
	)
	assert result["training_results"]["Logistic Regression"]["status"] == "trained"
	assert result["training_results"]["Support Vector Machine (SVM)"]["status"] == "failed"
	assert set(result["trained_models"]) == {"Logistic Regression"}


def test_training_records_estimator_warnings() -> None:
	X_train, y_train, preprocessor = classification_data()
	result = train_models(
		X_train,
		y_train,
		preprocessor,
		["Logistic Regression"],
		"classification",
		model_configs={"Logistic Regression": {"max_iter": 1}},
	)
	warnings = result["training_results"]["Logistic Regression"]["warnings"]
	assert warnings
	assert any("converge" in warning.casefold() for warning in warnings)


def test_classification_cv_rejects_too_few_rows_per_class() -> None:
	X_train = pd.DataFrame({"value": range(6)})
	y_train = pd.Series([0, 0, 0, 1, 1, 1], name="target")
	config = PreprocessingConfig(selected_features=["value"], problem_type="classification")
	preprocessor = build_preprocessing_pipeline(config, ["value"], [])
	with pytest.raises(ValueError, match="at least 4 training rows"):
		train_models(
			X_train,
			y_train,
			preprocessor,
			["Logistic Regression"],
			"classification",
			cv_folds=4,
			validation_strategy="K-Fold Cross-Validation",
		)


def test_training_validation_errors() -> None:
	X_train, y_train, preprocessor = classification_data()
	with pytest.raises(ValueError, match="at least one model"):
		train_models(X_train, y_train, preprocessor, [], "classification")
	with pytest.raises(ValueError, match="preprocessing"):
		train_models(X_train, y_train, None, ["Logistic Regression"], "classification")
	with pytest.raises(ValueError, match="at least 2 folds"):
		train_models(
			X_train,
			y_train,
			preprocessor,
			["Logistic Regression"],
			"classification",
			cv_folds=1,
			validation_strategy="K-Fold Cross-Validation",
		)
	with pytest.raises(ValueError, match="not available"):
		train_models(X_train, y_train, preprocessor, ["Ridge Regression"], "classification")


def test_training_rejects_misaligned_feature_and_target_indices() -> None:
	X_train = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0]}, index=[0, 1, 2, 3])
	y_train = pd.Series([0, 1, 0, 1], index=[10, 11, 12, 13], name="target")
	config = PreprocessingConfig(selected_features=["value"], problem_type="classification")
	preprocessor = build_preprocessing_pipeline(config, ["value"], [])
	with pytest.raises(ValueError, match="row-aligned"):
		train_models(X_train, y_train, preprocessor, ["Logistic Regression"], "classification")


def test_training_rejects_missing_target_values() -> None:
	X_train = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0]})
	y_train = pd.Series([0, 1, np.nan, 1], name="target")
	config = PreprocessingConfig(selected_features=["value"], problem_type="classification")
	preprocessor = build_preprocessing_pipeline(config, ["value"], [])
	with pytest.raises(ValueError, match="target contains missing"):
		train_models(X_train, y_train, preprocessor, ["Logistic Regression"], "classification")


def test_training_rejects_nonnumeric_regression_target() -> None:
	X_train = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0]})
	y_train = pd.Series(["low", "medium", "high", "higher"], name="target")
	config = PreprocessingConfig(selected_features=["value"], problem_type="regression")
	preprocessor = build_preprocessing_pipeline(config, ["value"], [])
	with pytest.raises(ValueError, match="must be numeric"):
		train_models(X_train, y_train, preprocessor, ["Ridge Regression"], "regression", cv_folds=2)


def test_training_without_validation_fits_multiple_models_and_marks_cv_unavailable() -> None:
	X_train, y_train, preprocessor = classification_data()
	result = train_models(
		X_train,
		y_train,
		preprocessor,
		["Logistic Regression", "Random Forest"],
		"classification",
	)
	assert set(result["trained_models"]) == {"Logistic Regression", "Random Forest"}
	assert result["cv_configuration"] is None
	for model_name, model_result in result["training_results"].items():
		assert model_result["status"] == "trained"
		assert model_result["validation_metrics_available"] is False
		assert model_result["cv_metrics"] is None
		assert model_result["training_metrics"]
		assert result["trained_models"][model_name]["validation_configuration"]["validation_strategy"] == "None"


def test_smote_pipeline_trains_numerical_classification_features_only() -> None:
	X_train, y_train, _ = classification_data()
	original_X = X_train.copy(deep=True)
	original_y = y_train.copy(deep=True)
	config = PreprocessingConfig(selected_features=list(X_train.columns), problem_type="classification")
	preprocessor = build_preprocessing_pipeline(config, list(X_train.columns), [])
	result = train_models(
		X_train,
		y_train,
		preprocessor,
		["Logistic Regression"],
		"classification",
		validation_strategy="None",
		sampling_method="SMOTE",
		preprocessing_config=config.to_dict(),
	)
	model = result["trained_models"]["Logistic Regression"]
	assert model["status"] == "trained"
	assert model["pipeline"].steps[1][0] == "sampler"
	assert model["pipeline"].steps[2][0] == "model"
	assert model["sampling_method"] == "SMOTE"
	pd.testing.assert_frame_equal(X_train, original_X)
	pd.testing.assert_series_equal(y_train, original_y)


def test_smotenc_is_fitted_inside_each_cv_training_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
	row_count = 80
	X_train = pd.DataFrame(
		{
			"numeric": np.linspace(0, 1, row_count),
			"category": np.where(np.arange(row_count) % 2, "odd", "even"),
		}
	)
	y_train = pd.Series([0] * 60 + [1] * 20, name="target")
	original_X = X_train.copy(deep=True)
	original_y = y_train.copy(deep=True)
	config = PreprocessingConfig(
		selected_features=["numeric", "category"],
		problem_type="classification",
		categorical_missing_strategy="Most Frequent",
	)
	preprocessor = build_preprocessing_pipeline(config, ["numeric"], ["category"])
	base_smotenc = training_trainer.SMOTENC

	class RecordingSMOTENC(base_smotenc):
		fit_resample_sizes: list[int] = []

		def fit_resample(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
			type(self).fit_resample_sizes.append(len(X))
			return super().fit_resample(X, y)

	RecordingSMOTENC.fit_resample_sizes.clear()
	monkeypatch.setattr(training_trainer, "SMOTENC", RecordingSMOTENC)
	result = train_models(
		X_train,
		y_train,
		preprocessor,
		["Logistic Regression"],
		"classification",
		cv_folds=4,
		sampling_method="SMOTENC",
		preprocessing_config=config.to_dict(),
		validation_strategy="K-Fold Cross-Validation",
	)
	model = result["trained_models"]["Logistic Regression"]
	assert model["status"] == "trained"
	assert model["pipeline"].steps[1][0] == "sampler"
	assert model["pipeline"].steps[2][0] == "preprocessor"
	assert model["pipeline"].named_steps["sampler"].categorical_features_ == [1]
	assert RecordingSMOTENC.fit_resample_sizes == [60, 60, 60, 60, 80]
	assert len(model["cv_metrics"]["accuracy"]["fold_scores"]) == 4
	assert model["sampling_method"] == "SMOTENC"
	assert model["preprocessing_config"] == config.to_dict()
	pd.testing.assert_frame_equal(X_train, original_X)
	pd.testing.assert_series_equal(y_train, original_y)


def test_oversampling_rejects_cv_folds_with_too_few_training_class_rows() -> None:
	X_train = pd.DataFrame({"value": np.arange(6, dtype=float)})
	y_train = pd.Series([0, 0, 0, 1, 1, 1], name="target")
	config = PreprocessingConfig(selected_features=["value"], problem_type="classification")
	preprocessor = build_preprocessing_pipeline(config, ["value"], [])
	with pytest.raises(ValueError, match="at least two samples per class"):
		train_models(
			X_train,
			y_train,
			preprocessor,
			["Logistic Regression"],
			"classification",
			cv_folds=2,
			sampling_method="SMOTE",
			validation_strategy="K-Fold Cross-Validation",
		)


def test_sampling_imputer_uses_categorical_strategy_for_boolean_columns() -> None:
	X_train = pd.DataFrame({"flag": pd.Series([True, False, pd.NA], dtype="boolean")})
	imputer = TrainingDataImputer(categorical_strategy="Most Frequent")
	transformed = imputer.fit_transform(X_train)
	assert transformed["flag"].isna().sum() == 0
	assert transformed["flag"].isin([True, False]).all()