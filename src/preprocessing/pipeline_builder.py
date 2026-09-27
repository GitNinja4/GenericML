"""Serializable preprocessing configuration and sklearn pipeline assembly."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.preprocessing.encoding import create_encoder
from src.preprocessing.feature_selection import create_feature_selector
from src.preprocessing.missing_values import create_categorical_imputer, create_numeric_imputer
from src.preprocessing.outliers import IQRTransformer
from src.preprocessing.scaling import create_scaler


@dataclass
class PreprocessingConfig:
	"""User-selected, serializable preprocessing choices."""

	numeric_missing_strategy: str = "None"
	numeric_constant_value: float = 0.0
	categorical_missing_strategy: str = "None"
	categorical_constant_value: str = "Missing"
	outlier_enabled: bool = False
	outlier_method: str = "IQR"
	outlier_action: str = "Clip"
	lower_multiplier: float = 1.5
	upper_multiplier: float = 1.5
	encoding_strategy: str = "One-Hot Encoding"
	scaling_strategy: str = "None"
	feature_selection_enabled: bool = False
	feature_selection_method: str = "SelectKBest"
	feature_selection_k: int = 10
	selected_features: list[str] = field(default_factory=list)
	excluded_features: list[str] = field(default_factory=list)
	target_column: str | None = None
	problem_type: str | None = None

	def to_dict(self) -> dict[str, Any]:
		"""Return a JSON-compatible configuration dictionary."""
		return asdict(self)

	@classmethod
	def from_dict(cls, values: dict[str, Any]) -> "PreprocessingConfig":
		"""Recreate a configuration from its serialized form."""
		return cls(**values)


def validate_preprocessing_config(
	config: PreprocessingConfig,
	available_features: list[str],
	target_column: str | None,
) -> None:
	"""Validate user choices before storing an active configuration."""
	if config.numeric_missing_strategy not in {"None", "Mean", "Median", "Constant"}:
		raise ValueError(f"Unsupported numerical missing-value strategy: {config.numeric_missing_strategy}")
	if config.categorical_missing_strategy not in {"None", "Most Frequent", "Constant"}:
		raise ValueError(f"Unsupported categorical missing-value strategy: {config.categorical_missing_strategy}")
	if config.encoding_strategy not in {"None", "One-Hot Encoding", "Ordinal Encoding"}:
		raise ValueError(f"Unsupported categorical encoding strategy: {config.encoding_strategy}")
	if config.scaling_strategy not in {"None", "StandardScaler", "MinMaxScaler", "RobustScaler"}:
		raise ValueError(f"Unsupported scaling strategy: {config.scaling_strategy}")
	if not config.selected_features:
		raise ValueError("Please select at least one feature.")
	if len(config.selected_features) != len(set(config.selected_features)):
		raise ValueError("Feature columns must be unique.")
	if target_column and target_column in config.selected_features:
		raise ValueError("The target column cannot be included in feature columns.")
	unknown = set(config.selected_features) - set(available_features)
	if unknown:
		raise ValueError(f"Unknown feature columns: {sorted(unknown)}")
	if config.feature_selection_enabled and not 1 <= config.feature_selection_k <= len(config.selected_features):
		raise ValueError("Feature-selection K must be between one and the number of selected features.")
	if config.outlier_enabled and config.outlier_method != "IQR":
		raise ValueError("Only IQR outlier handling is supported in Phase 4.")
	if config.outlier_enabled and config.outlier_action not in {"Clip", "Remove"}:
		raise ValueError("Outlier action must be Clip or Remove.")
	if config.outlier_enabled and (config.lower_multiplier < 0 or config.upper_multiplier < 0):
		raise ValueError("Outlier multipliers must be non-negative.")
	if config.problem_type not in {"classification", "regression"}:
		raise ValueError("A valid classification or regression problem type is required.")


def build_preprocessing_pipeline(
	config: PreprocessingConfig,
	numeric_columns: list[str],
	categorical_columns: list[str],
) -> Pipeline:
	"""Build an unfitted leakage-safe sklearn preprocessing pipeline.

	The caller must split data first and call ``pipeline.fit(X_train, y_train)``.
	This function never fits on the uploaded dataset.
	"""
	selected_numeric = [column for column in config.selected_features if column in numeric_columns]
	selected_categorical = [column for column in config.selected_features if column in categorical_columns]
	numeric_steps: list[tuple[str, Any]] = []
	numeric_imputer = create_numeric_imputer(config.numeric_missing_strategy, config.numeric_constant_value)
	if numeric_imputer is not None:
		numeric_steps.append(("imputer", numeric_imputer))
	if config.outlier_enabled:
		if config.outlier_enabled and config.outlier_action == "Clip":
			numeric_steps.append(("outliers", IQRTransformer("clip", config.lower_multiplier, config.upper_multiplier)))
	scaler = create_scaler(config.scaling_strategy)
	if scaler is not None:
		numeric_steps.append(("scaler", scaler))

	categorical_steps: list[tuple[str, Any]] = []
	categorical_imputer = create_categorical_imputer(config.categorical_missing_strategy, config.categorical_constant_value)
	if categorical_imputer is not None:
		categorical_steps.append(("imputer", categorical_imputer))
	encoder = create_encoder(config.encoding_strategy)
	if encoder is not None:
		categorical_steps.append(("encoder", encoder))

	transformers: list[tuple[str, Any, list[str]]] = []
	if selected_numeric:
		transformers.append(("numeric", Pipeline(numeric_steps) if numeric_steps else "passthrough", selected_numeric))
	if selected_categorical:
		transformers.append(("categorical", Pipeline(categorical_steps) if categorical_steps else "passthrough", selected_categorical))
	if not transformers:
		raise ValueError("At least one numerical or categorical feature is required.")

	steps: list[tuple[str, Any]] = [("preprocessor", ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False))]
	if config.feature_selection_enabled:
		steps.append(("feature_selection", create_feature_selector(config.problem_type or "", config.feature_selection_method, config.feature_selection_k)))
	return Pipeline(steps)


def get_processed_feature_names(pipeline: Pipeline) -> list[str]:
	"""Return transformed feature names after a pipeline has been fitted."""
	names = pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
	if "feature_selection" in pipeline.named_steps:
		selector = pipeline.named_steps["feature_selection"]
		names = [name for name, selected in zip(names, selector.get_support()) if selected]
	return names


def fit_and_transform_training_data(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
	"""Fit preprocessing on training data only and return transformed training data."""
	return pipeline.fit_transform(X_train, y_train)
