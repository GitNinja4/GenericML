"""Reusable helpers for constructing complete model pipelines."""

from __future__ import annotations

from typing import Any

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.pipeline import Pipeline as ImbalancedPipeline

from src.models.model_factory import create_model
from src.training.smote import TrainingDataImputer


def build_model_pipeline(
	model_name: str,
	problem_type: str,
	preprocessing_pipeline: Any,
	parameters: dict[str, Any] | None = None,
	random_state: int | None = None,
	sampling_method: str = "Disabled",
	categorical_features: list[int] | None = None,
	y_train: Any = None,
	preprocessing_config: dict[str, Any] | None = None,
) -> Pipeline:
	"""Create a fresh sampler, preprocessing, and model pipeline."""
	model = create_model(model_name, problem_type, parameters)
	if random_state is not None and "random_state" in model.get_params():
		model.set_params(random_state=random_state)
	steps: list[tuple[str, Any]] = []
	if sampling_method != "Disabled":
		if problem_type != "classification" or y_train is None:
			raise ValueError("SMOTE and SMOTENC are available only for classification training.")
		class_counts = y_train.value_counts()
		if class_counts.empty or int(class_counts.min()) < 2:
			raise ValueError("Oversampling requires at least two training samples in every class.")
		neighbors = min(5, int(class_counts.min()) - 1)
		if sampling_method == "SMOTE":
			if categorical_features:
				raise ValueError("SMOTE requires numerical features only. Choose SMOTENC for categorical features.")
			sampler = SMOTE(random_state=random_state, k_neighbors=neighbors)
		elif sampling_method == "SMOTENC":
			if not categorical_features:
				raise ValueError("SMOTENC requires at least one categorical feature. Choose SMOTE for numerical features.")
			sampler = SMOTENC(
				categorical_features=categorical_features,
				random_state=random_state,
				k_neighbors=neighbors,
			)
		else:
			raise ValueError(f"Unsupported oversampling method: {sampling_method}")
		config = preprocessing_config or {}
	if sampling_method == "Disabled":
		steps.append(("preprocessor", clone(preprocessing_pipeline)))
		steps.append(("model", model))
		return Pipeline(steps)
	preprocessing_steps = [(name, clone(transformer)) for name, transformer in preprocessing_pipeline.steps]
	if sampling_method == "SMOTE":
		steps.extend(preprocessing_steps)
		steps.extend([("sampler", sampler), ("model", model)])
		return ImbalancedPipeline(steps)
	config = preprocessing_config or {}
	steps.extend(
		[
			(
				"sampling_imputer",
				TrainingDataImputer(
					numeric_strategy=str(config.get("numeric_missing_strategy", "None")),
					numeric_constant=float(config.get("numeric_constant_value", 0.0)),
					categorical_strategy=str(config.get("categorical_missing_strategy", "None")),
					categorical_constant=str(config.get("categorical_constant_value", "Missing")),
				),
			),
			("sampler", sampler),
		]
	)
	steps.extend(preprocessing_steps)
	steps.append(("model", model))
	return ImbalancedPipeline(steps)
