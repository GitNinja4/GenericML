"""Schema generation and validation for single-row prediction input."""

from __future__ import annotations

from typing import Any

import pandas as pd


def build_prediction_schema(dataset: pd.DataFrame, target_column: str, selected_features: list[str] | None = None) -> list[dict[str, Any]]:
	"""Describe model input fields from the uploaded dataset schema."""
	features = selected_features or [column for column in dataset.columns if column != target_column]
	missing = [column for column in features if column not in dataset.columns]
	if missing:
		raise ValueError(f"Prediction features are missing from the dataset: {missing}")
	rows = []
	for column in features:
		series = dataset[column]
		kind = "numeric" if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series) else "boolean" if pd.api.types.is_bool_dtype(series) else "categorical"
		categories = [str(value) for value in series.dropna().unique().tolist()] if kind == "categorical" else []
		rows.append({"name": column, "dtype": str(series.dtype), "kind": kind, "categories": categories, "default": _default_value(series, kind)})
	return rows


def _default_value(series: pd.Series, kind: str) -> Any:
	if kind == "numeric":
		value = series.median()
		return float(value) if pd.notna(value) else 0.0
	if kind == "boolean":
		return bool(series.mode(dropna=True).iloc[0]) if not series.mode(dropna=True).empty else False
	mode = series.mode(dropna=True)
	return str(mode.iloc[0]) if not mode.empty else ""


def validate_prediction_frame(frame: pd.DataFrame, schema: list[dict[str, Any]]) -> pd.DataFrame:
	"""Validate and order one prediction row for the fitted pipeline."""
	expected = [item["name"] for item in schema]
	if list(frame.columns) != expected:
		raise ValueError("Prediction input columns do not match the trained pipeline schema.")
	if len(frame) != 1:
		raise ValueError("Prediction requires exactly one input row.")
	return frame.loc[:, expected]
