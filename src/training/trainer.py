"""Reusable helpers for constructing complete model pipelines."""

from __future__ import annotations

from typing import Any

from sklearn.base import clone
from sklearn.pipeline import Pipeline

from src.models.model_factory import create_model


def build_model_pipeline(
	model_name: str,
	problem_type: str,
	preprocessing_pipeline: Any,
	parameters: dict[str, Any] | None = None,
	random_state: int | None = None,
) -> Pipeline:
	"""Create a fresh preprocessing-plus-model pipeline for one model."""
	model = create_model(model_name, problem_type, parameters)
	if random_state is not None and "random_state" in model.get_params():
		model.set_params(random_state=random_state)
	return Pipeline([("preprocessor", clone(preprocessing_pipeline)), ("model", model)])
