"""User-controlled hyperparameter tuning using training data and CV only."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.tuning.optuna_optimizer import tune_pipeline
from src.tuning.search_spaces import build_search_space
from ui.components import navigate, render_cta, render_empty_state, render_page_header


def _scoring(problem_type: str) -> tuple[str, str]:
	configured = st.session_state.get("validation_configuration") or {}
	metric = configured.get("scoring_metric") or ("accuracy" if problem_type == "classification" else "r2")
	if metric in {"mae", "mse", "rmse"}:
		return f"neg_{metric}", metric.upper()
	return metric, "F1 Score" if metric == "f1" else metric.title()


def _cv_limit(y_train: pd.Series, problem_type: str) -> int:
	if problem_type == "classification":
		return max(2, min(10, int(y_train.value_counts().min())))
	return max(2, min(10, len(y_train)))


def _render_search_space(search_space: dict[str, list[object]]) -> None:
	with st.container(border=True):
		st.markdown("#### Search space configuration")
		rows = [
			{"Parameter": name, "Values": ", ".join(str(value) for value in values)}
			for name, values in search_space.items()
		]
		st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
		st.caption("Search values are generated from the selected model's registry definition.")


def render_tuning_tab() -> None:
	"""Tune a trained model without exposing the protected test partition."""
	dataset = st.session_state.get("dataset")
	problem_type = st.session_state.get("problem_type") or st.session_state.get("detected_problem_type")
	trained_models = st.session_state.get("trained_models", {})
	render_page_header("Hyperparameter Tuning", "Optimize model parameters for better performance.", "07 · Optional")
	with st.container(border=True):
		st.success(
			"Tuning uses training data and cross-validation only. The protected test set remains locked until Evaluation.",
			icon=":material/lock:",
		)
	if dataset is None or problem_type not in {"classification", "regression"}:
		render_empty_state("Dataset not ready", "Complete dataset setup before tuning a model.", "database")
		return
	if not trained_models:
		render_empty_state("No trained models", "Train at least one model before starting optional tuning.", "model_training")
		render_cta("Go to Training →", "training", "tuning_to_training")
		return
	X_train = st.session_state.get("X_train")
	y_train = st.session_state.get("y_train")
	if X_train is None or y_train is None:
		st.error("The training partition is unavailable. Recreate the split before tuning.")
		return

	model_name = st.selectbox("Select model to tune", list(trained_models), key="tuning_model")
	model_record = trained_models[model_name]
	search_space = build_search_space(model_name, problem_type)
	if not search_space:
		st.info("This model has no registry-configured parameters to tune.")
		return
	_render_search_space(search_space)
	scoring, scoring_label = _scoring(problem_type)

	control_column, budget_column = st.columns([1.45, 0.9], vertical_alignment="top")
	with control_column:
		with st.container(border=True):
			st.markdown("#### Search configuration")
			method = st.selectbox("Tuning method", ["Grid Search", "Random Search", "Optuna"], key="tuning_method")
			max_cv = _cv_limit(y_train, problem_type)
			stored_cv = (st.session_state.get("validation_configuration") or {}).get("cv_folds")
			default_cv = min(max_cv, int(stored_cv)) if stored_cv else min(3, max_cv)
			cv_folds = st.number_input("Number of CV folds", min_value=2, max_value=max_cv, value=default_cv, step=1, key="tuning_cv_folds")
			st.caption(f"Scoring: {scoring_label} · test data is not inspected")
	with budget_column:
		with st.container(border=True):
			st.markdown("#### Run budget")
			if method == "Grid Search":
				st.metric("Parameter combinations", _candidate_count(search_space))
				budget = _candidate_count(search_space)
			else:
				label = "Number of iterations" if method == "Random Search" else "Number of trials"
				budget = st.number_input(label, min_value=1, max_value=200, value=20, step=1, key="tuning_budget")
			st.number_input("Random state", min_value=0, step=1, key="tuning_random_state")

	if st.button("Start Hyperparameter Tuning", key="start_tuning", type="primary", width="stretch", icon=":material/auto_awesome:"):
		try:
			with st.spinner("Running cross-validated tuning on training data..."):
				result = tune_pipeline(
					model_record["pipeline"],
					X_train,
					y_train,
					search_space,
					method,
					int(cv_folds),
					scoring,
					int(st.session_state["tuning_random_state"]),
					n_iter=int(budget) if method == "Random Search" else 20,
					trials=int(budget) if method == "Optuna" else 20,
				)
			st.session_state.setdefault("tuning_results", {})[model_name] = result
			st.success("Tuning completed on training data.")
		except Exception as exc:
			st.error(str(exc))

	result = st.session_state.get("tuning_results", {}).get(model_name)
	if result:
		results_column, summary_column = st.columns([1.6, 0.85], vertical_alignment="top")
		with results_column:
			with st.container(border=True):
				st.markdown("#### Tuning results")
				metric_columns = st.columns(4)
				metric_columns[0].metric("Best CV score", f"{result['best_score']:.4f}")
				metric_columns[1].metric("Candidates / trials", result["candidate_count"])
				metric_columns[2].metric("Tuning time", f"{result['tuning_time_seconds']:.2f} s")
				metric_columns[3].metric("Method", result["method"])
				st.dataframe(
					pd.DataFrame([{"Parameter": name, "Best value": str(value)} for name, value in result["best_params"].items()]),
					width="stretch",
					hide_index=True,
				)
		with summary_column:
			with st.container(border=True):
				st.markdown("#### Best pipeline")
				st.write(f"Model: **{model_name}**")
				st.write(f"Validation: **{cv_folds}-fold CV**")
				st.write(f"Score: **{scoring_label}**")
			if st.button("Use Best Parameters", key=f"use_best_{model_name}", type="primary", width="stretch"):
				st.session_state.setdefault("tuned_models", {})[model_name] = result
				st.session_state["trained_models"][model_name]["pipeline"] = result["best_estimator"]
				st.session_state["trained_models"][model_name]["parameters"].update(result["best_params"])
				st.session_state["model_configs"].setdefault(model_name, {}).update(result["best_params"])
				st.session_state["best_pipeline"] = result["best_estimator"]
				st.session_state["final_model_name"] = model_name
				st.success("Best parameters adopted. The tuned pipeline is ready for Evaluation.")

	back_column, _, next_column = st.columns([1, 1, 2])
	with back_column:
		if st.button("← Back to Training", key="tuning_to_training_back", width="stretch"):
			navigate("training")
	with next_column:
		render_cta("Continue to Evaluation →", "evaluation", "tuning_to_evaluation", disabled=st.session_state.get("best_pipeline") is None)


def _candidate_count(search_space: dict[str, list[object]]) -> int:
	count = 1
	for values in search_space.values():
		count *= max(1, len(values))
	return count
