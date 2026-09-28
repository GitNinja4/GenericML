"""Public model factory backed by the central registry."""

from __future__ import annotations

from typing import Any

from src.models.registry import get_model_metadata


def create_model(
	model_name: str,
	problem_type: str,
	parameters: dict[str, Any] | None = None,
) -> Any:
	"""Create a model after validating its registered problem type."""
	metadata = get_model_metadata(model_name)
	if metadata.problem_type not in {problem_type, "both"}:
		raise ValueError(f"Model '{model_name}' is not available for {problem_type}.")
	resolved_parameters = dict(metadata.default_parameters)
	resolved_parameters.update(parameters or {})
	constructor = metadata.constructors[problem_type]
	return constructor(**resolved_parameters)
