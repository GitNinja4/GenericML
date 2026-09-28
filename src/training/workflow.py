"""Leakage-safe user-selected model training workflow."""

from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score
from sklearn.model_selection import KFold, StratifiedKFold

from src.models.registry import get_available_models
from src.preprocessing.outliers import IQRTransformer
from src.training.splitter import split_dataset
from src.training.trainer import build_model_pipeline
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _remove_training_outliers(
	X_train: pd.DataFrame,
	y_train: pd.Series,
	preprocessing_config: dict[str, Any] | None,
) -> tuple[pd.DataFrame, pd.Series, int]:
	"""Filter training rows with IQR bounds learned only from the training split."""
	config = preprocessing_config or {}
	if not config.get("outlier_enabled") or config.get("outlier_action") != "Remove":
		return X_train, y_train, 0

	numeric_columns = X_train.select_dtypes(include="number").columns.tolist()
	if not numeric_columns:
		return X_train, y_train, 0

	filter_transformer = IQRTransformer(
		"remove",
		float(config.get("lower_multiplier", 1.5)),
		float(config.get("upper_multiplier", 1.5)),
	)
	filter_transformer.fit(X_train[numeric_columns])
	filter_transformer.transform(X_train[numeric_columns])
	keep_mask = filter_transformer.get_support_mask()
	removed_rows = int((~keep_mask).sum())
	if removed_rows == len(X_train):
		raise ValueError("IQR outlier removal excluded every training row. Adjust the outlier settings or split.")
	return X_train.iloc[keep_mask].copy(), y_train.iloc[keep_mask].copy(), removed_rows


def _metric_values(
	y_true: pd.Series,
	y_predicted: np.ndarray,
	problem_type: str,
	estimator: Any,
	X: pd.DataFrame,
) -> dict[str, float]:
	if problem_type == "regression":
		mse = float(mean_squared_error(y_true, y_predicted))
		return {
			"mae": float(mean_absolute_error(y_true, y_predicted)),
			"mse": mse,
			"rmse": float(np.sqrt(mse)),
			"r2": float(r2_score(y_true, y_predicted)),
		}

	metrics = {
		"accuracy": float(accuracy_score(y_true, y_predicted)),
		"precision": float(precision_score(y_true, y_predicted, average="weighted", zero_division=0)),
		"recall": float(recall_score(y_true, y_predicted, average="weighted", zero_division=0)),
		"f1": float(f1_score(y_true, y_predicted, average="weighted", zero_division=0)),
	}
	return metrics


def _summarize_fold_metrics(fold_metrics: list[dict[str, float]]) -> dict[str, dict[str, Any]]:
	metric_names = fold_metrics[0].keys()
	return {
		name: {
			"mean": float(np.mean([fold[name] for fold in fold_metrics])),
			"std": float(np.std([fold[name] for fold in fold_metrics], ddof=0)),
			"fold_scores": [float(fold[name]) for fold in fold_metrics],
		}
		for name in metric_names
	}


def _validate_scoring_metric(problem_type: str, scoring_metric: str) -> None:
	if problem_type == "classification":
		valid_metrics = {"accuracy", "precision", "recall", "f1"}
	else:
		valid_metrics = {"mae", "mse", "rmse", "r2"}
	if scoring_metric not in valid_metrics:
		raise ValueError(f"Scoring metric '{scoring_metric}' is not available for {problem_type}.")


def _training_diagnostic(
	training_score: float,
	validation_score: float | None,
	metric_name: str,
	problem_type: str,
) -> dict[str, str]:
	if validation_score is None:
		return {
			"status": "unavailable",
			"message": "Validation was not performed. Generalization and overfitting cannot be reliably assessed from training metrics alone.",
		}
	lower_is_better = metric_name in {"mae", "mse", "rmse"}
	performance_gap = validation_score - training_score if lower_is_better else training_score - validation_score
	if lower_is_better:
		scale = max(abs(training_score), abs(validation_score), 1e-12)
		possible_overfit = performance_gap / scale >= 0.2
	else:
		possible_overfit = performance_gap >= 0.1
	if possible_overfit:
		return {
			"status": "possible_overfitting",
			"message": "Training performance is noticeably stronger than cross-validation performance.",
		}
	if problem_type == "classification" and training_score < 0.6 and validation_score < 0.6:
		return {
			"status": "possible_underfitting",
			"message": "Both training and validation scores are relatively low; possible underfitting.",
		}
	if metric_name == "r2" and training_score < 0.3 and validation_score < 0.3:
		return {
			"status": "possible_underfitting",
			"message": "Both training and validation R² scores are relatively low; possible underfitting.",
		}
	return {
		"status": "consistent",
		"message": "Training and validation performance are relatively consistent.",
	}


def train_models(
	X_train: pd.DataFrame,
	y_train: pd.Series,
	preprocessing_pipeline: Any,
	selected_models: list[str],
	problem_type: str,
	model_configs: dict[str, dict[str, Any]] | None = None,
	preprocessing_config: dict[str, Any] | None = None,
	cv_folds: int = 5,
	random_state: int = 42,
	shuffle: bool = True,
	scoring_metric: str | None = None,
	validation_strategy: str = "None",
	sampling_method: str = "Disabled",
) -> dict[str, Any]:
	"""Validate and fit models using only the protected training partition.

	Every fold receives a fresh sampler/preprocessing/model pipeline. The final
	fitted pipeline is then trained on all of X_train/y_train. This function has
	no test data parameters, so holdout rows cannot enter fitting, CV, or metrics.
	"""
	if preprocessing_pipeline is None:
		raise ValueError("Please configure and apply preprocessing before training models.")
	if not selected_models:
		raise ValueError("Please select at least one model.")
	if problem_type not in {"classification", "regression"}:
		raise ValueError("Problem type must be classification or regression.")
	if validation_strategy not in {"None", "K-Fold Cross-Validation"}:
		raise ValueError("Validation strategy must be None or K-Fold Cross-Validation.")
	use_cv = validation_strategy == "K-Fold Cross-Validation"
	if use_cv and (not isinstance(cv_folds, int) or cv_folds < 2):
		raise ValueError("Cross-validation requires at least 2 folds.")
	if not isinstance(random_state, int):
		raise ValueError("Random state must be an integer.")
	if X_train.empty or len(y_train) != len(X_train) or not X_train.index.equals(y_train.index):
		raise ValueError("Training features and target must be non-empty and row-aligned.")
	if not X_train.index.is_unique:
		raise ValueError("Training data must have unique row indices for cross-validation.")
	if y_train.isna().any():
		raise ValueError("Training target contains missing values.")
	available_models = set(get_available_models(problem_type))
	invalid_models = sorted(set(selected_models) - available_models)
	if invalid_models:
		raise ValueError(f"Models are not available for {problem_type}: {invalid_models}")
	if problem_type == "classification":
		class_counts = y_train.value_counts()
		if len(class_counts) < 2:
			raise ValueError("Classification training requires at least two target classes.")
		if use_cv and class_counts.min() < cv_folds:
			raise ValueError(f"Each class needs at least {cv_folds} training rows for stratified cross-validation.")
		if sampling_method != "Disabled" and class_counts.min() < 2:
			raise ValueError("Oversampling requires at least two training samples in every class.")
	if problem_type == "regression" and not pd.api.types.is_numeric_dtype(y_train.dtype):
		raise ValueError("Regression training target must be numeric.")
	if use_cv and len(X_train) < cv_folds:
		raise ValueError(f"Number of CV folds ({cv_folds}) cannot exceed training rows ({len(X_train)}).")
	if sampling_method != "Disabled" and problem_type != "classification":
		raise ValueError("SMOTE and SMOTENC are available only for classification training.")
	if scoring_metric is None:
		scoring_metric = "accuracy" if problem_type == "classification" else "r2"
	_validate_scoring_metric(problem_type, scoring_metric)

	if use_cv:
		if problem_type == "classification":
			fold_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=shuffle, random_state=random_state if shuffle else None)
		else:
			fold_splitter = KFold(n_splits=cv_folds, shuffle=shuffle, random_state=random_state if shuffle else None)
		fold_indices = list(fold_splitter.split(X_train, y_train))
		if problem_type == "classification" and sampling_method != "Disabled":
			for train_indices, _ in fold_indices:
				fold_class_counts = y_train.iloc[train_indices].value_counts()
				if fold_class_counts.empty or int(fold_class_counts.min()) < 2:
					raise ValueError(
						"Each CV training fold must contain at least two samples per class to use SMOTE or SMOTENC. "
						"Reduce the number of folds or disable oversampling."
					)
	else:
		fold_indices = []
	categorical_columns = X_train.select_dtypes(include=["object", "category", "string", "bool"]).columns.tolist()
	categorical_features = [X_train.columns.get_loc(column) for column in categorical_columns]
	trained_models: dict[str, dict[str, Any]] = {}
	training_results: dict[str, dict[str, Any]] = {}
	configs = model_configs or {}
	config = preprocessing_config or {}
	class_distribution = None
	if problem_type == "classification":
		class_counts = y_train.value_counts()
		class_distribution = [
			{
				"class": str(class_name),
				"count": int(count),
				"percentage": float(count / len(y_train) * 100),
			}
			for class_name, count in class_counts.items()
		]
	logger.info("Training started for %s model(s) with validation strategy %s", len(selected_models), validation_strategy)

	for model_name in selected_models:
		started = time.perf_counter()
		captured_warnings: list[warnings.WarningMessage] = []
		warning_messages: list[str] = []
		try:
			fold_validation_metrics: list[dict[str, float]] = []
			fold_training_metrics: list[dict[str, float]] = []
			with warnings.catch_warnings(record=True) as captured_warnings:
				warnings.simplefilter("always")
				for train_indices, validation_indices in fold_indices:
					X_fold_train = X_train.iloc[train_indices]
					y_fold_train = y_train.iloc[train_indices]
					X_fold_validation = X_train.iloc[validation_indices]
					y_fold_validation = y_train.iloc[validation_indices]
					X_fold_fit, y_fold_fit, _ = _remove_training_outliers(X_fold_train, y_fold_train, config)
					if len(X_fold_fit) < 1:
						raise ValueError("Outlier removal excluded every row in a CV training fold.")
					fold_pipeline = build_model_pipeline(
						model_name,
						problem_type,
						preprocessing_pipeline,
						configs.get(model_name),
						random_state,
						sampling_method=sampling_method,
						categorical_features=categorical_features,
						y_train=y_fold_fit,
						preprocessing_config=config,
					)
					fold_pipeline.fit(X_fold_fit, y_fold_fit)
					fold_training_metrics.append(
						_metric_values(y_fold_fit, fold_pipeline.predict(X_fold_fit), problem_type, fold_pipeline, X_fold_fit)
					)
					fold_validation_metrics.append(
						_metric_values(
							y_fold_validation,
							fold_pipeline.predict(X_fold_validation),
							problem_type,
							fold_pipeline,
							X_fold_validation,
						)
					)
			X_fit, y_fit, outlier_rows_removed = _remove_training_outliers(X_train, y_train, config)
			if X_fit.empty:
				raise ValueError("Outlier removal excluded every training row.")
			pipeline = build_model_pipeline(
				model_name,
				problem_type,
				preprocessing_pipeline,
				configs.get(model_name),
				random_state,
				sampling_method=sampling_method,
				categorical_features=categorical_features,
				y_train=y_fit,
				preprocessing_config=config,
			)
			with warnings.catch_warnings(record=True) as final_fit_warnings:
				warnings.simplefilter("always")
				pipeline.fit(X_fit, y_fit)
			captured_warnings.extend(final_fit_warnings)
			warning_messages = [str(item.message) for item in captured_warnings]
			duration = round(time.perf_counter() - started, 4)
			parameters = pipeline.named_steps["model"].get_params(deep=False)
			training_metrics = _metric_values(y_fit, pipeline.predict(X_fit), problem_type, pipeline, X_fit)
			cv_metrics = _summarize_fold_metrics(fold_validation_metrics) if use_cv else None
			cv_training_metrics = _summarize_fold_metrics(fold_training_metrics) if use_cv else None
			validation_score = cv_metrics[scoring_metric]["mean"] if use_cv else None
			diagnostic = _training_diagnostic(training_metrics[scoring_metric], validation_score, scoring_metric, problem_type)
			validation_method = "Stratified K-Fold" if problem_type == "classification" else "K-Fold"
			validation_configuration = {
				"validation_strategy": validation_strategy,
				"validation_method": validation_method if use_cv else None,
				"cv_folds": cv_folds if use_cv else None,
				"shuffle": shuffle if use_cv else None,
				"random_state": random_state,
				"scoring_metric": scoring_metric if use_cv else None,
			}
			model_result = {
				"status": "trained",
				"training_time_seconds": duration,
				"final_fit_samples": len(X_fit),
				"outlier_rows_removed": outlier_rows_removed,
				"training_metrics": training_metrics,
				"cv_training_metrics": cv_training_metrics,
				"cv_metrics": cv_metrics,
				"validation_metrics_available": use_cv,
				"classification_average": "weighted" if problem_type == "classification" else None,
				"class_distribution": class_distribution,
				"diagnostic": diagnostic,
				"scoring_metric": scoring_metric,
				"validation_strategy": validation_strategy,
				"validation_method": validation_method if use_cv else None,
				"cv_folds": cv_folds if use_cv else None,
				"cv_configuration": validation_configuration if use_cv else None,
				"validation_configuration": validation_configuration,
				"parameters": parameters,
				"preprocessing_config": dict(config),
				"sampling_method": sampling_method,
				"warnings": warning_messages,
			}
			trained_models[model_name] = {
				"pipeline": pipeline,
				"model_name": model_name,
				"problem_type": problem_type,
				"parameters": parameters,
				"training_metrics": training_metrics,
				"cv_training_metrics": cv_training_metrics,
				"cv_metrics": cv_metrics,
				"validation_metrics_available": use_cv,
				"classification_average": "weighted" if problem_type == "classification" else None,
				"class_distribution": class_distribution,
				"diagnostic": diagnostic,
				"training_time_seconds": duration,
				"cv_configuration": model_result["cv_configuration"],
				"validation_configuration": validation_configuration,
				"preprocessing_config": dict(config),
				"sampling_method": sampling_method,
				"status": "trained",
			}
			training_results[model_name] = model_result
			logger.info("Training completed for %s in %.4f seconds", model_name, duration)
		except Exception as exc:  # model-specific failures should not hide other results
			duration = round(time.perf_counter() - started, 4)
			warning_messages = [str(item.message) for item in captured_warnings]
			training_results[model_name] = {
				"status": "failed",
				"training_time_seconds": duration,
				"parameters": configs.get(model_name, {}),
				"validation_strategy": validation_strategy,
				"validation_configuration": {
					"validation_strategy": validation_strategy,
					"validation_method": "Stratified K-Fold" if problem_type == "classification" and use_cv else "K-Fold" if use_cv else None,
					"cv_folds": cv_folds if use_cv else None,
					"shuffle": shuffle if use_cv else None,
					"random_state": random_state,
					"scoring_metric": scoring_metric if use_cv else None,
				},
				"preprocessing_config": dict(config),
				"sampling_method": sampling_method,
				"class_distribution": class_distribution,
				"classification_average": "weighted" if problem_type == "classification" else None,
				"scoring_metric": scoring_metric,
				"error": str(exc),
				"warnings": warning_messages,
			}
			logger.exception("Model training failed for %s", model_name)

	if not trained_models:
		logger.error("All %s selected model(s) failed to train", len(selected_models))
	logger.info("Training workflow completed: %s succeeded, %s failed", len(trained_models), len(selected_models) - len(trained_models))
	return {
		"trained_models": trained_models,
		"training_results": training_results,
		"selected_models": list(selected_models),
		"model_configs": configs,
		"cv_configuration": {
			"validation_method": "Stratified K-Fold" if problem_type == "classification" else "K-Fold",
			"cv_folds": cv_folds,
			"shuffle": shuffle,
			"random_state": random_state,
			"scoring_metric": scoring_metric,
		} if use_cv else None,
		"validation_configuration": {
			"validation_strategy": validation_strategy,
			"cv_folds": cv_folds if use_cv else None,
			"shuffle": shuffle if use_cv else None,
			"random_state": random_state,
			"scoring_metric": scoring_metric if use_cv else None,
		},
		"preprocessing_config": dict(config),
		"sampling_method": sampling_method,
		"class_distribution": class_distribution,
		"classification_average": "weighted" if problem_type == "classification" else None,
		"problem_type": problem_type,
	}


train_selected_models = train_models
