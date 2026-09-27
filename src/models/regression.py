"""Regression estimator factories."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import (
	AdaBoostRegressor,
	BaggingRegressor,
	ExtraTreesRegressor,
	GradientBoostingRegressor,
	HistGradientBoostingRegressor,
	RandomForestRegressor,
	StackingRegressor,
	VotingRegressor,
)
from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, LinearRegression, Ridge, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import RANSACRegressor, TheilSenRegressor
from sklearn.svm import LinearSVR, NuSVR, SVR
from sklearn.tree import DecisionTreeRegressor


def create_regression_model(model_name: str, parameters: dict[str, Any] | None = None) -> Any:
	"""Create a supported regression estimator from user parameters."""
	params = dict(parameters or {})
	factories = {
		"Linear Regression": LinearRegression,
		"Ridge": Ridge,
		"Lasso": Lasso,
		"ElasticNet": ElasticNet,
		"SGD Regressor": SGDRegressor,
		"Huber Regressor": HuberRegressor,
		"RANSAC Regressor": RANSACRegressor,
		"Theil-Sen Regressor": TheilSenRegressor,
		"Decision Tree": DecisionTreeRegressor,
		"Decision Tree Regressor": DecisionTreeRegressor,
		"Extra Trees Regressor": ExtraTreesRegressor,
		"Random Forest": RandomForestRegressor,
		"Random Forest Regressor": RandomForestRegressor,
		"Bagging Regressor": BaggingRegressor,
		"Voting Regressor": VotingRegressor,
		"Stacking Regressor": StackingRegressor,
		"Gradient Boosting Regressor": GradientBoostingRegressor,
		"HistGradientBoosting Regressor": HistGradientBoostingRegressor,
		"AdaBoost Regressor": AdaBoostRegressor,
		"KNN": KNeighborsRegressor,
		"KNN Regressor": KNeighborsRegressor,
		"SVR": SVR,
		"NuSVR": NuSVR,
		"LinearSVR": LinearSVR,
	}
	try:
		factory = factories[model_name]
	except KeyError as exc:
		raise ValueError(f"Unsupported regression model: {model_name}") from exc
	if model_name == "Voting Regressor":
		return VotingRegressor(
			estimators=[("linear", LinearRegression()), ("forest", RandomForestRegressor(n_estimators=50, random_state=42))],
			**params,
		)
	if model_name == "Stacking Regressor":
		return StackingRegressor(
			estimators=[("linear", LinearRegression()), ("forest", RandomForestRegressor(n_estimators=50, random_state=42))],
			final_estimator=Ridge(),
			**params,
		)
	return factory(**params)
