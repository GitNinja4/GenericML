"""Evaluation orchestration for the protected test partition."""

from __future__ import annotations

from typing import Any

from src.evaluation.classification_metrics import calculate_classification_metrics
from src.evaluation.regression_metrics import calculate_regression_metrics


def evaluate_pipeline(pipeline: Any, X_test: Any, y_test: Any, problem_type: str) -> dict[str, Any]:
	"""Predict and score a fitted pipeline on test data only."""
	if pipeline is None:
		raise ValueError("A fitted final pipeline is required for evaluation.")
	if X_test is None or y_test is None or len(X_test) == 0:
		raise ValueError("The protected test partition is unavailable.")
	if problem_type == "classification":
		predicted = pipeline.predict(X_test)
		return {"problem_type": problem_type, "sample_count": len(X_test), "predicted": predicted.tolist(), **calculate_classification_metrics(y_test, predicted)}
	if problem_type == "regression":
		predicted = pipeline.predict(X_test)
		return {"problem_type": problem_type, "sample_count": len(X_test), **calculate_regression_metrics(y_test, predicted)}
	raise ValueError(f"Unsupported problem type: {problem_type}")
