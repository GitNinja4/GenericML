"""Registry-derived hyperparameter search spaces."""

from __future__ import annotations

from typing import Any

from src.models.registry import get_model_metadata


def _numeric_values(parameter: Any, default: int | float | None) -> list[Any]:
	if parameter.control == "optional_int":
		return list(parameter.options)
	if parameter.control == "bool":
		return [True, False]
	if default is None:
		return [parameter.minimum] if parameter.minimum is not None else []
	if parameter.control == "int":
		values = {int(default), max(int(parameter.minimum or 1), int(default) // 2), int(default) * 2}
	else:
		step = float(parameter.step or 0.1)
		values = {float(default), max(float(parameter.minimum or 0), float(default) - step), float(default) + step}
	if parameter.maximum is not None:
		values = {value for value in values if value <= parameter.maximum}
	return sorted(values)


def build_search_space(model_name: str, problem_type: str) -> dict[str, list[Any]]:
	"""Build a bounded, registry-derived search space for a registered model."""
	metadata = get_model_metadata(model_name)
	if metadata.problem_type not in {problem_type, "both"}:
		raise ValueError(f"Model '{model_name}' is not available for {problem_type}.")
	space: dict[str, list[Any]] = {}
	for parameter in metadata.parameter_schema:
		default = metadata.default_parameters.get(parameter.name)
		if parameter.control == "select":
			values = list(parameter.options)
		else:
			values = _numeric_values(parameter, default)
		if values:
			space[parameter.name] = values
	return space
