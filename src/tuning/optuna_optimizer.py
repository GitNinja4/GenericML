"""Training-only hyperparameter tuning helpers."""

from __future__ import annotations

import time
from typing import Any

import optuna
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score


def _score_direction(scoring: str) -> str:
	return "maximize" if not scoring.startswith("neg_") else "maximize"


def tune_pipeline(
	estimator: Any,
	X_train: Any,
	y_train: Any,
	search_space: dict[str, list[Any]],
	method: str,
	cv: int,
	scoring: str,
	random_state: int,
	n_iter: int = 20,
	trials: int = 20,
) -> dict[str, Any]:
	"""Tune a complete preprocessing/model pipeline using training data only."""
	if method not in {"Grid Search", "Random Search", "Optuna"}:
		raise ValueError(f"Unsupported tuning method: {method}")
	if not search_space:
		raise ValueError("The selected model has no configurable parameters to tune.")
	if cv < 2:
		raise ValueError("Tuning requires at least two cross-validation folds.")
	started = time.perf_counter()
	parameter_space = {f"model__{name}": values for name, values in search_space.items()}
	if method == "Grid Search":
		search = GridSearchCV(clone(estimator), parameter_space, cv=cv, scoring=scoring, refit=True, n_jobs=-1)
		search.fit(X_train, y_train)
		best_estimator = search.best_estimator_
		best_params = {key.removeprefix("model__"): value for key, value in search.best_params_.items()}
		best_score = float(search.best_score_)
		candidate_count = len(search.cv_results_["params"])
	elif method == "Random Search":
		search = RandomizedSearchCV(
			clone(estimator),
			parameter_space,
			n_iter=min(n_iter, max(1, _candidate_count(search_space))),
			cv=cv,
			scoring=scoring,
			refit=True,
			random_state=random_state,
			n_jobs=-1,
		)
		search.fit(X_train, y_train)
		best_estimator = search.best_estimator_
		best_params = {key.removeprefix("model__"): value for key, value in search.best_params_.items()}
		best_score = float(search.best_score_)
		candidate_count = len(search.cv_results_["params"])
	else:
		try:
			from optuna.samplers import TPESampler
		except ImportError as exc:  # pragma: no cover
			raise ValueError("Optuna is not installed in the active environment.") from exc
		study = optuna.create_study(direction=_score_direction(scoring), sampler=TPESampler(seed=random_state))

		def objective(trial: optuna.Trial) -> float:
			parameters: dict[str, Any] = {}
			for name, values in search_space.items():
				parameters[name] = trial.suggest_categorical(name, values)
			candidate = clone(estimator).set_params(**{f"model__{name}": value for name, value in parameters.items()})
			return float(cross_val_score(candidate, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1).mean())

		study.optimize(objective, n_trials=trials, show_progress_bar=False)
		best_params = dict(study.best_params)
		best_estimator = clone(estimator).set_params(**{f"model__{name}": value for name, value in best_params.items()})
		best_estimator.fit(X_train, y_train)
		best_score = float(study.best_value)
		candidate_count = len(study.trials)
	return {
		"best_estimator": best_estimator,
		"best_params": best_params,
		"best_score": best_score,
		"candidate_count": candidate_count,
		"tuning_time_seconds": round(time.perf_counter() - started, 4),
		"method": method,
		"cv_folds": cv,
		"scoring": scoring,
	}


def _candidate_count(search_space: dict[str, list[Any]]) -> int:
	count = 1
	for values in search_space.values():
		count *= max(1, len(values))
	return count
