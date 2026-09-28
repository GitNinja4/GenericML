"""Single source of truth for registered model constructors and UI metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from sklearn.ensemble import (
	AdaBoostClassifier,
	AdaBoostRegressor,
	ExtraTreesClassifier,
	ExtraTreesRegressor,
	GradientBoostingClassifier,
	GradientBoostingRegressor,
	HistGradientBoostingClassifier,
	HistGradientBoostingRegressor,
	RandomForestClassifier,
	RandomForestRegressor,
)
from sklearn.linear_model import (
	ElasticNet,
	Lasso,
	LinearRegression,
	LogisticRegression,
	Ridge,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


@dataclass(frozen=True)
class ModelParameter:
	"""A registry-owned definition for one generic parameter control."""

	name: str
	label: str
	control: str
	minimum: int | float | None = None
	maximum: int | float | None = None
	step: int | float | None = None
	options: tuple[Any, ...] = ()
	format: str | None = None


@dataclass(frozen=True)
class ModelMetadata:
	"""Constructor, defaults, family, and parameter controls for one model."""

	name: str
	problem_type: str
	family: str
	description: str
	default_parameters: dict[str, Any]
	constructors: dict[str, Callable[..., Any]]
	parameter_schema: tuple[ModelParameter, ...] = ()


def _parameter(
	name: str,
	label: str,
	control: str,
	minimum: int | float | None = None,
	maximum: int | float | None = None,
	step: int | float | None = None,
	options: tuple[Any, ...] = (),
	format: str | None = None,
) -> ModelParameter:
	return ModelParameter(name, label, control, minimum, maximum, step, options, format)


def _classification(
	name: str,
	family: str,
	description: str,
	constructor: Callable[..., Any],
	defaults: dict[str, Any] | None = None,
	parameters: tuple[ModelParameter, ...] = (),
) -> ModelMetadata:
	return ModelMetadata(name, "classification", family, description, defaults or {}, {"classification": constructor}, parameters)


def _regression(
	name: str,
	family: str,
	description: str,
	constructor: Callable[..., Any],
	defaults: dict[str, Any] | None = None,
	parameters: tuple[ModelParameter, ...] = (),
) -> ModelMetadata:
	return ModelMetadata(name, "regression", family, description, defaults or {}, {"regression": constructor}, parameters)


def _both(
	name: str,
	family: str,
	description: str,
	classification_constructor: Callable[..., Any],
	regression_constructor: Callable[..., Any],
	defaults: dict[str, Any],
	parameters: tuple[ModelParameter, ...] = (),
) -> ModelMetadata:
	return ModelMetadata(
		name,
		"both",
		family,
		description,
		defaults,
		{"classification": classification_constructor, "regression": regression_constructor},
		parameters,
	)


_DEPTH = _parameter("max_depth", "Max depth", "optional_int", options=(None, 3, 5, 10, 20))
_MIN_SPLIT = _parameter("min_samples_split", "Minimum samples split", "int", 2, step=1)
_MIN_LEAF = _parameter("min_samples_leaf", "Minimum samples leaf", "int", 1, step=1)
_N_ESTIMATORS = _parameter("n_estimators", "Number of estimators", "int", 10, 1000, 10)
_C = _parameter("C", "C", "float", 0.0001, step=0.1)
_KERNEL = _parameter("kernel", "Kernel", "select", options=("rbf", "linear", "poly"))
_GAMMA = _parameter("gamma", "Gamma", "select", options=("scale", "auto"))
_ALPHA = _parameter("alpha", "Alpha", "float", 0.0, step=0.1)
_MAX_ITER = _parameter("max_iter", "Max iterations", "int", 100, step=100)
_LEARNING_RATE = _parameter("learning_rate", "Learning rate", "float", 0.001, 1.0, 0.05)
_MAX_LEAF_NODES = _parameter("max_leaf_nodes", "Maximum leaf nodes", "optional_int", options=(None, 15, 31, 63, 127))
_EPSILON = _parameter("epsilon", "Epsilon", "float", 0.0, step=0.05)


MODEL_REGISTRY: dict[str, ModelMetadata] = {
	"Logistic Regression": _classification("Logistic Regression", "Linear", "A linear classifier that estimates class probabilities.", LogisticRegression, {"C": 1.0, "max_iter": 1000}, (_C, _MAX_ITER)),
	"Naive Bayes": _classification("Naive Bayes", "Probabilistic", "A probabilistic classifier based on Bayes' theorem.", GaussianNB, {"var_smoothing": 1e-9}, (_parameter("var_smoothing", "Variance smoothing", "float", 1e-12, 1e-6, 1e-10, format="%.2e"),)),
	"K-Nearest Neighbors (KNN)": _classification("K-Nearest Neighbors (KNN)", "Distance-Based", "Classifies a sample from its nearest training examples.", KNeighborsClassifier, {"n_neighbors": 5, "weights": "uniform", "metric": "minkowski"}, (_parameter("n_neighbors", "Number of neighbors", "int", 1, step=1), _parameter("weights", "Weights", "select", options=("uniform", "distance")), _parameter("metric", "Metric", "select", options=("minkowski", "manhattan", "euclidean")))),
	"Support Vector Machine (SVM)": _classification("Support Vector Machine (SVM)", "Kernel-Based", "A maximum-margin classifier with configurable kernels.", SVC, {"C": 1.0, "kernel": "rbf", "gamma": "scale", "probability": True}, (_C, _KERNEL, _GAMMA, _parameter("probability", "Probability estimates", "bool"))),
	"Decision Tree": _classification("Decision Tree", "Tree-Based", "A tree classifier that learns decision rules from features.", DecisionTreeClassifier, {"max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1}, (_DEPTH, _MIN_SPLIT, _MIN_LEAF)),
	"Random Forest": _classification("Random Forest", "Ensemble", "An ensemble of randomized decision trees.", RandomForestClassifier, {"n_estimators": 100, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1}, (_N_ESTIMATORS, _DEPTH, _MIN_SPLIT, _MIN_LEAF)),
	"Extra Trees": _classification("Extra Trees", "Ensemble", "An ensemble of highly randomized decision trees.", ExtraTreesClassifier, {"n_estimators": 100, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1}, (_N_ESTIMATORS, _DEPTH, _MIN_SPLIT, _MIN_LEAF)),
	"AdaBoost": _classification("AdaBoost", "Boosting", "Boosts weak learners by focusing on difficult examples.", AdaBoostClassifier, {"n_estimators": 50, "learning_rate": 1.0}, (_N_ESTIMATORS, _LEARNING_RATE)),
	"Gradient Boosting": _classification("Gradient Boosting", "Boosting", "Builds an additive classifier from sequential decision trees.", GradientBoostingClassifier, {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3}, (_N_ESTIMATORS, _LEARNING_RATE, _DEPTH)),
	"HistGradientBoosting": _classification("HistGradientBoosting", "Boosting", "A histogram-based gradient boosting classifier.", HistGradientBoostingClassifier, {"max_iter": 100, "learning_rate": 0.1, "max_leaf_nodes": 31}, (_MAX_ITER, _LEARNING_RATE, _MAX_LEAF_NODES)),
	"Linear Regression": _regression("Linear Regression", "Linear", "A linear model that estimates a continuous target.", LinearRegression, {"fit_intercept": True}, (_parameter("fit_intercept", "Fit intercept", "bool"),)),
	"Ridge Regression": _regression("Ridge Regression", "Linear", "Linear regression with L2 regularization.", Ridge, {"alpha": 1.0}, (_ALPHA,)),
	"Lasso Regression": _regression("Lasso Regression", "Linear", "Linear regression with L1 regularization.", Lasso, {"alpha": 0.001, "max_iter": 5000}, (_parameter("alpha", "Alpha", "float", 0.0001, step=0.001, format="%.4f"), _MAX_ITER)),
	"Elastic Net": _regression("Elastic Net", "Linear", "Linear regression with combined L1 and L2 regularization.", ElasticNet, {"alpha": 1.0, "l1_ratio": 0.5}, (_ALPHA, _parameter("l1_ratio", "L1 ratio", "float", 0.0, 1.0, 0.05))),
	"K-Nearest Neighbors Regressor": _regression("K-Nearest Neighbors Regressor", "Distance-Based", "Predicts a continuous target from nearby examples.", KNeighborsRegressor, {"n_neighbors": 5, "weights": "uniform", "metric": "minkowski"}, (_parameter("n_neighbors", "Number of neighbors", "int", 1, step=1), _parameter("weights", "Weights", "select", options=("uniform", "distance")), _parameter("metric", "Metric", "select", options=("minkowski", "manhattan", "euclidean")))),
	"Support Vector Regressor (SVR)": _regression("Support Vector Regressor (SVR)", "Kernel-Based", "A support-vector model for continuous targets.", SVR, {"C": 1.0, "kernel": "rbf", "gamma": "scale", "epsilon": 0.1}, (_C, _KERNEL, _GAMMA, _EPSILON)),
	"Decision Tree Regressor": _regression("Decision Tree Regressor", "Tree-Based", "A tree model for continuous target decisions.", DecisionTreeRegressor, {"max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1}, (_DEPTH, _MIN_SPLIT, _MIN_LEAF)),
	"Random Forest Regressor": _regression("Random Forest Regressor", "Ensemble", "An ensemble of randomized trees for continuous targets.", RandomForestRegressor, {"n_estimators": 100, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1}, (_N_ESTIMATORS, _DEPTH, _MIN_SPLIT, _MIN_LEAF)),
	"Extra Trees Regressor": _regression("Extra Trees Regressor", "Ensemble", "An ensemble of highly randomized regression trees.", ExtraTreesRegressor, {"n_estimators": 100, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1}, (_N_ESTIMATORS, _DEPTH, _MIN_SPLIT, _MIN_LEAF)),
	"AdaBoost Regressor": _regression("AdaBoost Regressor", "Boosting", "Builds a strong regressor by focusing on difficult examples.", AdaBoostRegressor, {"n_estimators": 50, "learning_rate": 1.0}, (_N_ESTIMATORS, _LEARNING_RATE)),
	"Gradient Boosting Regressor": _regression("Gradient Boosting Regressor", "Boosting", "Builds an additive regressor from sequential trees.", GradientBoostingRegressor, {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3}, (_N_ESTIMATORS, _LEARNING_RATE, _DEPTH)),
	"HistGradientBoosting Regressor": _regression("HistGradientBoosting Regressor", "Boosting", "A histogram-based gradient boosting regressor.", HistGradientBoostingRegressor, {"max_iter": 100, "learning_rate": 0.1, "max_leaf_nodes": 31}, (_MAX_ITER, _LEARNING_RATE, _MAX_LEAF_NODES)),
}


def get_available_models(problem_type: str) -> list[str]:
	"""Return model names valid for classification or regression."""
	if problem_type not in {"classification", "regression"}:
		raise ValueError(f"Unsupported problem type: {problem_type}")
	return [metadata.name for metadata in MODEL_REGISTRY.values() if metadata.problem_type in {problem_type, "both"}]


def get_model_families(problem_type: str) -> list[str]:
	"""Return registry-derived family filters for the selected problem type."""
	models = [MODEL_REGISTRY[name] for name in get_available_models(problem_type)]
	return ["All Families", *dict.fromkeys(metadata.family for metadata in models)]


def get_models_by_family(problem_type: str, family: str = "All Families") -> list[str]:
	"""Return compatible model names filtered by a registry family."""
	names = get_available_models(problem_type)
	if family == "All Families":
		return names
	return [name for name in names if MODEL_REGISTRY[name].family == family]


def get_model_metadata(model_name: str) -> ModelMetadata:
	"""Return metadata for a registered model."""
	try:
		return MODEL_REGISTRY[model_name]
	except KeyError as exc:
		raise ValueError(f"Unknown model: {model_name}") from exc