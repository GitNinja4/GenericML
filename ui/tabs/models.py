"""User-controlled model selection and training UI."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import streamlit as st

from src.models.registry import get_available_models, get_model_families, get_model_metadata, get_models_by_family
from src.training.workflow import train_models
from src.utils.logging import get_logger
from ui.components import navigate, render_cta, render_empty_state, render_page_header, render_stat_cards

logger = get_logger(__name__)


def _optional_depth(label: str, key: str) -> int | None:
	value = st.selectbox(label, ["None", 3, 5, 10, 20], key=key)
	return None if value == "None" else int(value)


def _model_parameters(model_name: str, problem_type: str, expanded: bool = False) -> dict[str, Any]:
	"""Render a compact parameter editor for one selected model."""
	with st.expander(f"{model_name} parameters", expanded=expanded):
		st.caption(get_model_metadata(model_name).description)
		if model_name == "Logistic Regression":
			return {
				"C": st.number_input("C", min_value=0.0001, value=1.0, step=0.1, key=f"model_{model_name}_c"),
				"max_iter": st.number_input("Max iterations", min_value=100, value=1000, step=100, key=f"model_{model_name}_max_iter"),
			}
		if model_name in {"Decision Tree", "Decision Tree Regressor"}:
			return {
				"max_depth": _optional_depth("Max depth", f"model_{model_name}_tree_depth"),
				"min_samples_split": st.number_input("Minimum samples split", min_value=2, value=2, step=1, key=f"model_{model_name}_tree_split"),
				"min_samples_leaf": st.number_input("Minimum samples leaf", min_value=1, value=1, step=1, key=f"model_{model_name}_tree_leaf"),
			}
		if model_name in {"Random Forest", "Random Forest Regressor"}:
			return {
				"n_estimators": st.number_input("Number of trees", min_value=10, value=100, step=10, key=f"model_{model_name}_forest_estimators"),
				"max_depth": _optional_depth("Max depth", f"model_{model_name}_forest_depth"),
				"min_samples_split": st.number_input("Minimum samples split", min_value=2, value=2, step=1, key=f"model_{model_name}_forest_split"),
				"min_samples_leaf": st.number_input("Minimum samples leaf", min_value=1, value=1, step=1, key=f"model_{model_name}_forest_leaf"),
			}
		if model_name in {"KNN", "KNN Regressor"}:
			return {
				"n_neighbors": st.number_input("Number of neighbors", min_value=1, value=5, step=1, key=f"model_{model_name}_knn_neighbors"),
				"weights": st.selectbox("Weights", ["uniform", "distance"], key=f"model_{model_name}_knn_weights"),
				"metric": st.selectbox("Metric", ["minkowski", "manhattan", "euclidean"], key=f"model_{model_name}_knn_metric"),
			}
		if model_name in {"SVM", "SVR"}:
			parameters: dict[str, Any] = {
				"C": st.number_input("C", min_value=0.0001, value=1.0, step=0.1, key=f"model_{model_name}_c"),
				"kernel": st.selectbox("Kernel", ["rbf", "linear", "poly"], key=f"model_{model_name}_kernel"),
				"gamma": st.selectbox("Gamma", ["scale", "auto"], key=f"model_{model_name}_gamma"),
			}
			if model_name == "SVM":
				parameters["probability"] = True
			else:
				parameters["epsilon"] = st.number_input("Epsilon", min_value=0.0, value=0.1, step=0.05, key="model_svr_epsilon")
			return parameters
		if model_name == "Ridge":
			return {"alpha": st.number_input("Alpha", min_value=0.0, value=1.0, step=0.1, key="model_ridge_alpha")}
		if model_name == "Lasso":
			return {
				"alpha": st.number_input("Alpha", min_value=0.0001, value=0.001, step=0.001, format="%.4f", key="model_lasso_alpha"),
				"max_iter": st.number_input("Max iterations", min_value=100, value=5000, step=100, key="model_lasso_max_iter"),
			}
		metadata = get_model_metadata(model_name)
		parameters: dict[str, Any] = {}
		if "alpha" in metadata.default_parameters:
			parameters["alpha"] = st.number_input("Alpha", min_value=0.0, value=float(metadata.default_parameters["alpha"]), step=0.1, key=f"model_{model_name}_alpha")
		if "max_iter" in metadata.default_parameters:
			parameters["max_iter"] = st.number_input("Max iterations", min_value=100, value=int(metadata.default_parameters["max_iter"]), step=100, key=f"model_{model_name}_max_iter")
		if "epsilon" in metadata.default_parameters:
			parameters["epsilon"] = st.number_input("Epsilon", min_value=0.0, value=float(metadata.default_parameters["epsilon"]), step=0.05, key=f"model_{model_name}_epsilon")
		if "n_estimators" in metadata.default_parameters:
			parameters["n_estimators"] = st.number_input("Number of estimators", min_value=10, value=int(metadata.default_parameters["n_estimators"]), step=10, key=f"model_{model_name}_estimators")
		if "C" in metadata.default_parameters:
			parameters["C"] = st.number_input("C", min_value=0.0001, value=float(metadata.default_parameters["C"]), step=0.1, key=f"model_{model_name}_c")
		return parameters


def _clear_training_state() -> None:
	"""Clear fitted artifacts while preserving currently rendered widget values."""
	st.session_state["trained_models"] = {}
	st.session_state["training_results"] = {}
	st.session_state["training_configuration_signature"] = None
	st.session_state["X_train"] = None
	st.session_state["X_test"] = None
	st.session_state["y_train"] = None
	st.session_state["y_test"] = None


def _configuration_signature(
	selected_models: list[str],
	model_configs: dict[str, dict[str, Any]],
	test_size: float,
	random_state: int,
	preprocessing_config: dict[str, Any],
) -> str:
	return repr((selected_models, model_configs, test_size, random_state, preprocessing_config))


def _render_workflow_progress() -> None:
	steps = (
		("Dataset", "complete", "✓"),
		("EDA", "complete", "✓"),
		("Preprocessing", "complete", "✓"),
		("Models", "active", "4"),
		("Tuning", "upcoming", "5"),
		("Evaluation", "upcoming", "6"),
		("MLflow", "upcoming", "7"),
		("Prediction", "upcoming", "8"),
	)
	step_markup = "".join(
		f'<div class="model-progress-step {state}"><span class="model-progress-marker">{marker}</span>'
		f'<span class="model-progress-label">{label}</span></div>'
		for label, state, marker in steps
	)
	st.markdown(
		"""
		<style>
		.model-workflow { position: relative; display: grid; grid-template-columns: repeat(8, minmax(0, 1fr)); gap: 0.2rem; margin: 0 0 1rem; padding: 0.55rem 0.35rem; overflow-x: auto; border: 1px solid var(--ml-border); border-radius: 10px; background: var(--ml-panel); }
		.model-workflow::before { content: ""; position: absolute; top: 1.2rem; left: 6%; right: 6%; height: 1px; background: var(--ml-border); }
		.model-progress-step { position: relative; z-index: 1; display: flex; min-width: 3.7rem; flex-direction: column; align-items: center; gap: 0.25rem; color: var(--ml-muted); font-size: 0.65rem; }
		.model-progress-marker { display: grid; place-items: center; width: 1.35rem; height: 1.35rem; border: 1px solid var(--ml-border); border-radius: 50%; background: var(--ml-panel-soft); box-shadow: 0 0 0 3px var(--ml-panel); font-size: 0.68rem; font-weight: 700; }
		.model-progress-step.complete .model-progress-marker { border-color: transparent; background: #16a34a; color: #fff; }
		.model-progress-step.complete .model-progress-label { color: var(--ml-text); }
		.model-progress-step.active { color: var(--ml-primary); font-weight: 700; }
		.model-progress-step.active .model-progress-marker { border-color: var(--ml-primary); background: var(--ml-primary); color: #fff; box-shadow: 0 0 0 3px var(--ml-panel), 0 0 0 5px light-dark(#dbeafe, #1e3a5f); }
		.model-progress-label { white-space: nowrap; }
		</style>
		<div class="model-workflow">
		""" + step_markup + "</div>",
		unsafe_allow_html=True,
	)


def _toggle_model_selection(model_name: str) -> None:
	selected_models = list(st.session_state.get("selected_models", []))
	if model_name in selected_models:
		selected_models.remove(model_name)
	else:
		selected_models.append(model_name)
	st.session_state["selected_models"] = selected_models
	_clear_training_state()


def _render_dataset_information(dataset: pd.DataFrame, target_column: str, problem_type: str) -> None:
	with st.container(border=True):
		st.caption("Dataset information")
		st.write(st.session_state.get("dataset_name") or "Current dataset")
		st.caption(f"{len(dataset):,} rows · {dataset.shape[1]:,} columns")
		st.caption(f"Target: {target_column} ({problem_type.title()})")
		st.caption(f"Preprocessed features: {len(st.session_state.get('selected_features', []))}")


def _render_model_card(model_name: str, problem_type: str, selected: bool) -> None:
	metadata = get_model_metadata(model_name)
	with st.container(border=True):
		st.markdown(f"#### {metadata.name}")
		st.caption(metadata.family)
		st.write(metadata.description)
		st.button(
			"Selected ✓" if selected else "Select",
			key=f"model_toggle_{problem_type}_{model_name}",
			type="primary" if selected else "secondary",
			width="stretch",
			on_click=_toggle_model_selection,
			args=(model_name,),
		)
		with st.expander("Details"):
			st.caption(f"Compatible with {metadata.problem_type.replace('both', 'classification and regression')}.")
			st.write(metadata.description)
			if metadata.default_parameters:
				st.caption("Default parameters")
				st.json(metadata.default_parameters)
			else:
				st.caption("Uses the estimator's standard defaults.")


def _render_model_library(problem_type: str, selected_models: list[str]) -> list[str]:
	compatible_models = get_available_models(problem_type)
	family_options = get_model_families(problem_type)
	if st.session_state.get("model_family_filter") not in family_options:
		st.session_state["model_family_filter"] = "All Families"

	with st.container(border=True):
		st.markdown("#### Model library")
		search_column, family_column, sort_column = st.columns([1.45, 1, 0.85])
		with search_column:
			search = st.text_input("Search models", key="model_search", placeholder="Search models")
		with family_column:
			family = st.selectbox("Model family", family_options, key="model_family_filter")
		with sort_column:
			sort_by = st.selectbox("Sort by", ["Registry order", "Name A–Z", "Family"], key="model_sort")

		filtered_models = get_models_by_family(problem_type, family)
		search_term = search.strip().casefold()
		if search_term:
			filtered_models = [
				name
				for name in filtered_models
				if search_term in name.casefold()
				or search_term in get_model_metadata(name).description.casefold()
				or search_term in get_model_metadata(name).family.casefold()
			]
		if sort_by == "Name A–Z":
			filtered_models = sorted(filtered_models, key=str.casefold)
		elif sort_by == "Family":
			filtered_models = sorted(filtered_models, key=lambda name: (get_model_metadata(name).family, name.casefold()))

		st.caption(f"{len(filtered_models)} compatible models · {len(selected_models)} selected")
		st.multiselect(
			"Select one or more models",
			compatible_models,
			key="selected_models",
			placeholder="Choose compatible models",
		)
		card_columns = st.columns(3)
		for index, model_name in enumerate(filtered_models):
			with card_columns[index % len(card_columns)]:
				_render_model_card(model_name, problem_type, model_name in selected_models)
		if not filtered_models:
			st.info("No models match this search and family filter.")
	return list(st.session_state.get("selected_models", []))


def _render_selected_models(selected_models: list[str], model_configs: dict[str, dict[str, Any]]) -> None:
	with st.container(border=True):
		st.markdown(f"#### Selected models ({len(selected_models)})")
		if not selected_models:
			st.caption("Selected models will appear here for review.")
			return
		rows = []
		for model_name in selected_models:
			metadata = get_model_metadata(model_name)
			parameters = model_configs.get(model_name, {})
			rows.append(
				{
					"Model": metadata.name,
					"Type": metadata.problem_type.replace("both", "classification / regression").title(),
					"Family": metadata.family,
					"Parameters": ", ".join(f"{key}: {value}" for key, value in parameters.items()) or "Default",
				}
			)
		st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def render_models_tab() -> None:
	"""Render model selection and explicit training controls."""
	dataset = st.session_state.get("dataset")
	if dataset is None:
		render_page_header("Model selection & training", "Choose algorithms, configure their parameters, and train them.", "04 · Experiment")
		render_empty_state("No dataset available", "Upload a dataset before selecting models.", "model_training")
		return
	problem_type = st.session_state.get("problem_type") or st.session_state.get("detected_problem_type")
	target_column = st.session_state.get("target_column")
	preprocessing_pipeline = st.session_state.get("preprocessing_pipeline")
	selected_features = st.session_state.get("selected_features", [])
	if not problem_type or not target_column:
		st.warning("Select and validate a target on the Dataset tab first.")
		return
	if preprocessing_pipeline is None or not st.session_state.get("preprocessing_applied"):
		render_empty_state("Preprocessing is not applied", "Configure and apply preprocessing before selecting and training models.", "lock")
		return
	if not selected_features:
		st.error("No feature columns are available for training.")
		return
	compatible_models = get_available_models(problem_type)
	selected_before_run = st.session_state.get("selected_models", [])
	compatible_selection = [name for name in selected_before_run if name in compatible_models]
	if compatible_selection != selected_before_run:
		st.session_state["selected_models"] = compatible_selection
		_clear_training_state()

	_render_workflow_progress()
	page_heading, dataset_column = st.columns([2.8, 1], vertical_alignment="center")
	with page_heading:
		render_page_header("Model selection & training", "Choose algorithms, configure their parameters, and train them.", "04 · Experiment")
	with dataset_column:
		_render_dataset_information(dataset, target_column, problem_type)

	test_size = float(st.session_state.get("train_test_size", 0.2))
	test_samples = min(len(dataset), math.ceil(len(dataset) * test_size))
	train_samples = max(0, len(dataset) - test_samples)
	render_stat_cards(
		[
			("Problem type", problem_type.title()),
			("Target column", target_column),
			("Features", len(selected_features)),
			("Training samples", f"{train_samples:,}"),
			("Test samples", f"{test_samples:,}"),
		]
	)

	model_library_column, configuration_column = st.columns([1.8, 1], vertical_alignment="top")
	with model_library_column:
		selected_models = _render_model_library(problem_type, compatible_selection)
	with configuration_column:
		with st.container(border=True):
			st.markdown("#### Model configuration")
			if not selected_models:
				st.info("Select one or more models to configure their parameters.")
			model_configs = {
				model_name: _model_parameters(model_name, problem_type, expanded=index == 0)
				for index, model_name in enumerate(selected_models)
			}

	_render_selected_models(selected_models, model_configs)

	training_column, summary_column = st.columns([1.4, 1], vertical_alignment="top")
	with training_column:
		with st.container(border=True):
			st.markdown("#### Training configuration")
			split_col, seed_col = st.columns(2)
			test_size = split_col.selectbox("Test size", [0.1, 0.2, 0.25, 0.3], format_func=lambda value: f"{int(value * 100)}%", key="train_test_size")
			random_state = seed_col.number_input("Random state", min_value=0, step=1, key="train_random_state")
			if problem_type == "classification":
				st.caption("Classification uses stratified splitting when every class has enough samples.")
			else:
				st.caption("Regression uses a regular random train/test split.")
	with summary_column:
		with st.container(border=True):
			st.markdown("#### Training summary")
			st.write(f"Models: **{', '.join(selected_models) if selected_models else 'None selected'}**")
			st.write(f"Test size: **{int(test_size * 100)}%** · Random state: **{int(random_state)}**")
			st.write("Preprocessing: **Applied**")

	signature = _configuration_signature(selected_models, model_configs, test_size, int(random_state), st.session_state.get("preprocessing_config", {}))
	previous_signature = st.session_state.get("training_configuration_signature")
	if previous_signature and previous_signature != signature and st.session_state.get("trained_models"):
		_clear_training_state()
		st.info("Configuration changed. Previous trained models were cleared; train again.")

	back_column, spacer_column, train_column = st.columns([1, 1, 2], vertical_alignment="center")
	with back_column:
		if st.button("← Back to preprocessing", key="models_to_preprocessing", width="stretch"):
			navigate("preprocessing")
	with train_column:
		train_clicked = st.button("Train selected models", type="primary", key="train_selected_models", width="stretch", icon=":material/play_arrow:")

	if train_clicked:
		if not selected_models:
			st.error("Please select at least one model.")
			return
		X = dataset[selected_features].copy()
		y = dataset[target_column].copy()
		try:
			with st.status("Training selected models...", expanded=True) as status:
				st.write("Splitting dataset before fitting preprocessing...")
				result = train_models(
					X,
					y,
					preprocessing_pipeline,
					selected_models,
					problem_type,
					float(test_size),
					int(random_state),
					model_configs,
					preprocessing_config=st.session_state.get("preprocessing_config", {}),
				)
				st.write("Training complete.")
				status.update(label="Training complete", state="complete")
		except Exception as exc:
			logger.exception("Training workflow failed")
			st.error(str(exc))
			return
		st.session_state["model_configs"] = model_configs
		st.session_state["trained_models"] = result["trained_models"]
		st.session_state["training_results"] = result["training_results"]
		st.session_state["training_configuration_signature"] = signature
		st.session_state["X_train"] = result["X_train"]
		st.session_state["X_test"] = result["X_test"]
		st.session_state["y_train"] = result["y_train"]
		st.session_state["y_test"] = result["y_test"]

	training_results = st.session_state.get("training_results", {})
	if training_results:
		with st.container(border=True):
			st.markdown("#### Training results")
			if not st.session_state.get("trained_models"):
				st.error("None of the selected models trained successfully. Review the error details below.")
			rows = []
			for model_name, training_result in training_results.items():
				rows.append(
					{
						"Model": model_name,
						"Status": training_result["status"],
						"Training Time (s)": training_result.get("training_time_seconds"),
						"Training Samples": training_result.get("training_samples"),
						"Test Samples": training_result.get("test_samples"),
						"Error": training_result.get("error", ""),
						"Warnings": "; ".join(training_result.get("warnings", [])),
					}
				)
			st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
			st.caption("Phase 5 reports training status only. Model evaluation and ranking belong to Phase 6.")
			if st.session_state.get("trained_models"):
				render_cta("Evaluate models →", "evaluation", "models_to_evaluation")
