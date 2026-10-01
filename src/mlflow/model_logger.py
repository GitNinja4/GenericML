"""Model logging utilities for MLflow."""

from __future__ import annotations

from typing import Any

import mlflow.sklearn


def log_sklearn_pipeline(pipeline: Any, artifact_path: str = "model") -> None:
	"""Log the complete fitted preprocessing-plus-model pipeline."""
	mlflow.sklearn.log_model(pipeline, artifact_path=artifact_path)
