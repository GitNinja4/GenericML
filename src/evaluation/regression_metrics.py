"""Regression metrics calculated on an evaluation partition."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_regression_metrics(y_true: Any, y_predicted: Any) -> dict[str, Any]:
	"""Return error metrics and plot-ready values without fitting anything."""
	mse = float(mean_squared_error(y_true, y_predicted))
	return {
		"mae": float(mean_absolute_error(y_true, y_predicted)),
		"mse": mse,
		"rmse": float(np.sqrt(mse)),
		"r2": float(r2_score(y_true, y_predicted)),
		"actual": np.asarray(y_true).tolist(),
		"predicted": np.asarray(y_predicted).tolist(),
		"residuals": (np.asarray(y_true) - np.asarray(y_predicted)).tolist(),
	}
