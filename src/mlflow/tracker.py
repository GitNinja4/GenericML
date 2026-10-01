"""MLflow experiment tracking utilities."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow

from src.mlflow.artifacts import log_json_artifact
from src.mlflow.model_logger import log_sklearn_pipeline


def log_model_run(
	experiment_name: str,
	model_name: str,
	pipeline: Any,
	parameters: dict[str, Any],
	metrics: dict[str, float],
	tags: dict[str, str] | None = None,
) -> dict[str, Any]:
	"""Log one complete experiment run and return the actual MLflow identifiers."""
	if not experiment_name.strip():
		raise ValueError("Experiment name cannot be empty.")
	mlflow.set_experiment(experiment_name.strip())
	with mlflow.start_run(run_name=model_name) as run:
		mlflow.set_tags({"model": model_name, **(tags or {})})
		clean_parameters = {key: str(value) for key, value in parameters.items() if value is not None}
		mlflow.log_params(clean_parameters)
		clean_metrics = {key: float(value) for key, value in metrics.items() if value is not None}
		mlflow.log_metrics(clean_metrics)
		log_sklearn_pipeline(pipeline)
		log_json_artifact(
			"run_summary.json",
			{
				"experiment_name": experiment_name.strip(),
				"model_name": model_name,
				"parameters": clean_parameters,
				"metrics": clean_metrics,
				"logged_at": datetime.now(timezone.utc).isoformat(),
			},
		)
		return {
			"run_id": run.info.run_id,
			"status": "FINISHED",
			"experiment_name": experiment_name.strip(),
			"model_name": model_name,
			"artifact_uri": run.info.artifact_uri,
			"timestamp": datetime.now(timezone.utc).isoformat(),
			"parameters": clean_parameters,
			"metrics": clean_metrics,
		}
