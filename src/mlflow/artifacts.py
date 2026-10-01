"""Small artifact helpers for MLflow runs."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import mlflow


def log_json_artifact(filename: str, payload: dict[str, Any]) -> None:
	"""Log a JSON payload as a temporary MLflow artifact."""
	with tempfile.TemporaryDirectory() as directory:
		path = Path(directory) / filename
		path.write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")
		mlflow.log_artifact(str(path))
