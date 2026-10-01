"""Classification metrics calculated on an evaluation partition."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score


def calculate_classification_metrics(y_true: Any, y_predicted: Any) -> dict[str, Any]:
	"""Return aggregate and per-class metrics without fitting anything."""
	labels = list(np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_predicted)])))
	return {
		"accuracy": float(accuracy_score(y_true, y_predicted)),
		"precision": float(precision_score(y_true, y_predicted, average="weighted", zero_division=0)),
		"recall": float(recall_score(y_true, y_predicted, average="weighted", zero_division=0)),
		"f1": float(f1_score(y_true, y_predicted, average="weighted", zero_division=0)),
		"labels": [str(label) for label in labels],
		"confusion_matrix": confusion_matrix(y_true, y_predicted, labels=labels).tolist(),
		"classification_report": classification_report(y_true, y_predicted, labels=labels, output_dict=True, zero_division=0),
	}
