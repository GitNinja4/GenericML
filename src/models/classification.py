"""Classification estimator factories."""

from __future__ import annotations

from typing import Any

def create_classification_model(model_name: str, parameters: dict[str, Any] | None = None) -> Any:
	"""Create a registered classification model through the central registry."""
	from src.models.model_factory import create_model

	return create_model(model_name, "classification", parameters)
