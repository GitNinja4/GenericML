"""Problem-type-aware feature selection factories."""

from __future__ import annotations

from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression


def create_feature_selector(problem_type: str, method: str, k: int) -> SelectKBest:
	"""Create a SelectKBest selector with a score function for the target type."""
	if method != "SelectKBest":
		raise ValueError(f"Unsupported feature-selection method: {method}")
	if k < 1:
		raise ValueError("Feature-selection K must be at least 1.")
	if problem_type == "classification":
		score_function = mutual_info_classif
	elif problem_type == "regression":
		score_function = mutual_info_regression
	else:
		raise ValueError(f"Unsupported problem type: {problem_type}")
	return SelectKBest(score_func=score_function, k=k)
