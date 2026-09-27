"""Classification estimator factories."""

from __future__ import annotations

from typing import Any

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
	AdaBoostClassifier,
	BaggingClassifier,
	ExtraTreesClassifier,
	GradientBoostingClassifier,
	HistGradientBoostingClassifier,
	RandomForestClassifier,
	StackingClassifier,
	VotingClassifier,
)
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import NuSVC, SVC
from sklearn.tree import DecisionTreeClassifier


def create_classification_model(model_name: str, parameters: dict[str, Any] | None = None) -> Any:
	"""Create a supported classification estimator from user parameters."""
	params = dict(parameters or {})
	factories = {
		"Logistic Regression": LogisticRegression,
		"Ridge Classifier": RidgeClassifier,
		"SGD Classifier": SGDClassifier,
		"Perceptron": Perceptron,
		"Decision Tree": DecisionTreeClassifier,
		"Extra Trees Classifier": ExtraTreesClassifier,
		"Random Forest": RandomForestClassifier,
		"Bagging Classifier": BaggingClassifier,
		"Voting Classifier": VotingClassifier,
		"Stacking Classifier": StackingClassifier,
		"Gradient Boosting Classifier": GradientBoostingClassifier,
		"HistGradientBoosting Classifier": HistGradientBoostingClassifier,
		"AdaBoost Classifier": AdaBoostClassifier,
		"KNN": KNeighborsClassifier,
		"SVM": SVC,
		"NuSVC": NuSVC,
		"Gaussian Naive Bayes": GaussianNB,
		"Multinomial Naive Bayes": MultinomialNB,
		"Bernoulli Naive Bayes": BernoulliNB,
		"Complement Naive Bayes": ComplementNB,
		"Linear Discriminant Analysis": LinearDiscriminantAnalysis,
		"Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis,
	}
	try:
		factory = factories[model_name]
	except KeyError as exc:
		raise ValueError(f"Unsupported classification model: {model_name}") from exc
	if model_name == "Voting Classifier":
		return VotingClassifier(
			estimators=[("logistic", LogisticRegression(max_iter=1000)), ("forest", RandomForestClassifier(n_estimators=50, random_state=42))],
			voting="hard",
			**params,
		)
	if model_name == "Stacking Classifier":
		return StackingClassifier(
			estimators=[("logistic", LogisticRegression(max_iter=1000)), ("forest", RandomForestClassifier(n_estimators=50, random_state=42))],
			final_estimator=LogisticRegression(max_iter=1000),
			**params,
		)
	return factory(**params)
