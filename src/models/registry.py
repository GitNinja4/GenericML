"""Single source of truth for the dataset-aware model library."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ModelMetadata:
    """Educational metadata used by model cards, filters, and factories."""

    name: str
    problem_type: str
    family: str
    description: str
    default_parameters: dict[str, Any]
    factory: Callable[[str, dict[str, Any] | None], Any]


def _classification_factory(name: str, parameters: dict[str, Any] | None = None) -> Any:
    from src.models.classification import create_classification_model

    return create_classification_model(name, parameters)


def _regression_factory(name: str, parameters: dict[str, Any] | None = None) -> Any:
    from src.models.regression import create_regression_model

    return create_regression_model(name, parameters)


def _classification(name: str, family: str, description: str, parameters: dict[str, Any] | None = None) -> ModelMetadata:
    return ModelMetadata(name, "classification", family, description, parameters or {}, _classification_factory)


def _regression(name: str, family: str, description: str, parameters: dict[str, Any] | None = None) -> ModelMetadata:
    return ModelMetadata(name, "regression", family, description, parameters or {}, _regression_factory)


MODEL_REGISTRY: dict[str, ModelMetadata] = {
    "Logistic Regression": _classification("Logistic Regression", "Linear", "A linear classifier that estimates class probabilities.", {"C": 1.0, "max_iter": 1000}),
    "Ridge Classifier": _classification("Ridge Classifier", "Linear", "A linear classifier with L2 regularization."),
    "SGD Classifier": _classification("SGD Classifier", "Linear", "A linear model optimized with stochastic gradient descent."),
    "Perceptron": _classification("Perceptron", "Linear", "A simple linear classifier based on thresholded updates."),
    "Decision Tree": ModelMetadata("Decision Tree", "both", "Tree-Based", "A tree model that recursively splits data into decision rules.", {"max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1}, _classification_factory),
    "Extra Trees Classifier": _classification("Extra Trees Classifier", "Tree-Based", "An ensemble of highly randomized decision trees."),
    "Random Forest": ModelMetadata("Random Forest", "both", "Ensemble", "An ensemble of decision trees that combines many randomized trees.", {"n_estimators": 100, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1}, _classification_factory),
    "Bagging Classifier": _classification("Bagging Classifier", "Ensemble", "An ensemble that trains models on randomized samples."),
    "Voting Classifier": _classification("Voting Classifier", "Ensemble", "Combines predictions from complementary base classifiers."),
    "Stacking Classifier": _classification("Stacking Classifier", "Ensemble", "Learns a final classifier from several base model predictions."),
    "Gradient Boosting Classifier": _classification("Gradient Boosting Classifier", "Boosting", "Builds an additive classifier from sequential decision trees."),
    "HistGradientBoosting Classifier": _classification("HistGradientBoosting Classifier", "Boosting", "A histogram-based gradient boosting classifier."),
    "AdaBoost Classifier": _classification("AdaBoost Classifier", "Boosting", "Builds a strong classifier by focusing on difficult examples."),
    "KNN": ModelMetadata("KNN", "both", "Distance-Based", "A model that predicts from nearby training examples.", {"n_neighbors": 5, "weights": "uniform", "metric": "minkowski"}, _classification_factory),
    "SVM": _classification("SVM", "Kernel-Based", "A maximum-margin classifier with configurable kernels.", {"C": 1.0, "kernel": "rbf", "gamma": "scale", "probability": True}),
    "NuSVC": _classification("NuSVC", "Kernel-Based", "A support-vector classifier using a nu parameter."),
    "Gaussian Naive Bayes": _classification("Gaussian Naive Bayes", "Probabilistic", "A probabilistic classifier for continuous features."),
    "Multinomial Naive Bayes": _classification("Multinomial Naive Bayes", "Probabilistic", "A probabilistic classifier for count-like non-negative features."),
    "Bernoulli Naive Bayes": _classification("Bernoulli Naive Bayes", "Probabilistic", "A probabilistic classifier for binary-valued features."),
    "Complement Naive Bayes": _classification("Complement Naive Bayes", "Probabilistic", "A Naive Bayes variant designed for imbalanced text-like data."),
    "Linear Discriminant Analysis": _classification("Linear Discriminant Analysis", "Discriminant Analysis", "A generative linear discriminant model."),
    "Quadratic Discriminant Analysis": _classification("Quadratic Discriminant Analysis", "Discriminant Analysis", "A generative discriminant model with class-specific covariance."),
    "Linear Regression": _regression("Linear Regression", "Linear", "A linear model that estimates a continuous target."),
    "Ridge": _regression("Ridge", "Linear", "Linear regression with L2 regularization.", {"alpha": 1.0}),
    "Lasso": _regression("Lasso", "Linear", "Linear regression with L1 regularization.", {"alpha": 0.001, "max_iter": 5000}),
    "ElasticNet": _regression("ElasticNet", "Linear", "Linear regression with combined L1 and L2 regularization."),
    "SGD Regressor": _regression("SGD Regressor", "Linear", "A linear regressor optimized with stochastic gradient descent."),
    "Huber Regressor": _regression("Huber Regressor", "Robust Regression", "A robust linear model that reduces the effect of outliers."),
    "RANSAC Regressor": _regression("RANSAC Regressor", "Robust Regression", "A robust estimator that fits consensus subsets."),
    "Theil-Sen Regressor": _regression("Theil-Sen Regressor", "Robust Regression", "A robust regression estimator based on spatial medians."),
    "Decision Tree Regressor": _regression("Decision Tree Regressor", "Tree-Based", "A tree model for continuous target decisions."),
    "Extra Trees Regressor": _regression("Extra Trees Regressor", "Tree-Based", "An ensemble of highly randomized regression trees."),
    "Random Forest Regressor": _regression("Random Forest Regressor", "Ensemble", "An ensemble of randomized trees for continuous targets."),
    "Bagging Regressor": _regression("Bagging Regressor", "Ensemble", "An ensemble regressor trained on randomized samples."),
    "Voting Regressor": _regression("Voting Regressor", "Ensemble", "Averages predictions from several base regressors."),
    "Stacking Regressor": _regression("Stacking Regressor", "Ensemble", "Learns a final regressor from several base predictions."),
    "Gradient Boosting Regressor": _regression("Gradient Boosting Regressor", "Boosting", "Builds an additive regressor from sequential trees."),
    "HistGradientBoosting Regressor": _regression("HistGradientBoosting Regressor", "Boosting", "A histogram-based gradient boosting regressor."),
    "AdaBoost Regressor": _regression("AdaBoost Regressor", "Boosting", "Builds a strong regressor by focusing on difficult examples."),
    "KNN Regressor": _regression("KNN Regressor", "Distance-Based", "Predicts a continuous target from nearby examples."),
    "SVR": _regression("SVR", "Kernel-Based", "A support-vector model for continuous targets.", {"C": 1.0, "kernel": "rbf", "gamma": "scale", "epsilon": 0.1}),
    "NuSVR": _regression("NuSVR", "Kernel-Based", "A support-vector regressor using a nu parameter."),
    "LinearSVR": _regression("LinearSVR", "Kernel-Based", "A fast linear support-vector regressor."),
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
