"""Leakage-safe user-selected model training workflow."""

from __future__ import annotations

import time
import warnings
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.models.registry import get_available_models
from src.preprocessing.outliers import IQRTransformer
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


def split_dataset(
	X: pd.DataFrame,
	y: pd.Series,
	problem_type: str,
	test_size: float = 0.2,
	random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
	"""Split raw features and target before any preprocessing is fitted."""
	if not 0 < test_size < 1:
		raise ValueError("Test size must be greater than 0 and less than 1.")
	if not isinstance(random_state, int):
		raise ValueError("Random state must be an integer.")
	if X.empty or len(y) == 0:
		raise ValueError("The dataset must contain at least one row and one feature.")
	if len(X) != len(y):
		raise ValueError("Feature and target row counts must match.")
	if problem_type not in {"classification", "regression"}:
		raise ValueError("Problem type must be classification or regression.")
	missing_targets = int(y.isna().sum())
	if missing_targets:
		raise ValueError(f"The target contains {missing_targets} missing value(s). Remove rows with missing targets before training.")
	if problem_type == "classification" and y.nunique(dropna=True) < 2:
		raise ValueError("Classification requires at least two target classes.")
	if problem_type == "regression" and not pd.api.types.is_numeric_dtype(y.dtype):
		raise ValueError("The regression target must be numeric.")
	stratify = y if problem_type == "classification" else None
	try:
		return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)
	except ValueError as exc:
		if problem_type == "classification":
			raise ValueError(f"Stratified classification split failed: {exc}") from exc
		raise ValueError(f"Train/test split failed: {exc}") from exc


def train_models(
	X: pd.DataFrame,
	y: pd.Series,
	preprocessing_pipeline: Any,
	selected_models: list[str],
	problem_type: str,
	test_size: float = 0.2,
	random_state: int = 42,
	model_configs: dict[str, dict[str, Any]] | None = None,
	preprocessing_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
	"""Train selected models using one shared split and unfitted pipeline clones.

	The split happens before any complete pipeline is fitted. Each model receives
	a fresh clone of the Phase 4 preprocessing pipeline, so test rows never fit
	an imputer, encoder, scaler, selector, outlier transformer, or estimator.
	"""
	if preprocessing_pipeline is None:
		raise ValueError("Please configure and apply preprocessing before training models.")
	if not selected_models:
		raise ValueError("Please select at least one model.")
	available_models = set(get_available_models(problem_type))
	invalid_models = sorted(set(selected_models) - available_models)
	if invalid_models:
		raise ValueError(f"Models are not available for {problem_type}: {invalid_models}")

	X_train, X_test, y_train, y_test = split_dataset(X, y, problem_type, test_size, random_state)
	X_train, y_train, outlier_rows_removed = _remove_training_outliers(X_train, y_train, preprocessing_config)
	trained_models: dict[str, dict[str, Any]] = {}
	training_results: dict[str, dict[str, Any]] = {}
	configs = model_configs or {}
	logger.info("Training started for %s model(s)", len(selected_models))

	for model_name in selected_models:
		started = time.perf_counter()
		captured_warnings: list[warnings.WarningMessage] = []
		warning_messages: list[str] = []
		try:
			pipeline = build_model_pipeline(
				model_name,
				problem_type,
				preprocessing_pipeline,
				configs.get(model_name),
				random_state,
			)
			with warnings.catch_warnings(record=True) as captured_warnings:
				warnings.simplefilter("always")
				pipeline.fit(X_train, y_train)
			warning_messages = [str(item.message) for item in captured_warnings]
			duration = round(time.perf_counter() - started, 4)
			parameters = pipeline.named_steps["model"].get_params(deep=False)
			trained_models[model_name] = {
				"pipeline": pipeline,
				"model_name": model_name,
				"problem_type": problem_type,
				"parameters": parameters,
				"status": "trained",
			}
			training_results[model_name] = {
				"status": "trained",
				"training_time_seconds": duration,
				"training_samples": len(X_train),
				"test_samples": len(X_test),
				"outlier_rows_removed": outlier_rows_removed,
				"features_before_preprocessing": X.shape[1],
				"warnings": warning_messages,
			}
			logger.info("Training completed for %s in %.4f seconds", model_name, duration)
		except Exception as exc:  # model-specific failures should not hide other results
			duration = round(time.perf_counter() - started, 4)
			warning_messages = [str(item.message) for item in captured_warnings]
			training_results[model_name] = {
				"status": "failed",
				"training_time_seconds": duration,
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
		"X_train": X_train,
		"X_test": X_test,
		"y_train": y_train,
		"y_test": y_test,
		"test_size": test_size,
		"random_state": random_state,
		"problem_type": problem_type,
		"outlier_rows_removed": outlier_rows_removed,
	}


train_selected_models = train_models
