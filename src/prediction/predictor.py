"""Prediction orchestration for persisted pipelines."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.prediction.schema import validate_prediction_frame


def predict_with_pipeline(pipeline: Any, values: dict[str, Any], schema: list[dict[str, Any]], problem_type: str) -> dict[str, Any]:
	"""Run one schema-validated input row through the complete fitted pipeline."""
	if pipeline is None:
		raise ValueError("A final fitted pipeline is required before prediction.")
	frame = validate_prediction_frame(pd.DataFrame([values]), schema)
	prediction = pipeline.predict(frame)
	result: dict[str, Any] = {"prediction": prediction[0], "input": values}
	if problem_type == "classification" and hasattr(pipeline, "predict_proba"):
		probabilities = pipeline.predict_proba(frame)[0]
		result["probabilities"] = {str(label): float(value) for label, value in zip(pipeline.classes_, probabilities)}
	return result
