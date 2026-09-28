"""User-controlled model selection and training UI."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from src.models.registry import ModelMetadata, ModelParameter, get_available_models, get_model_families, get_model_metadata, get_models_by_family
from ui.components import navigate, render_cta, render_empty_state, render_page_header, render_stat_cards, render_workflow_progress


def _parameter_widget_key(model_name: str, parameter_name: str) -> str:
	model_slug = "_".join(model_name.casefold().split())
	return f"model_param_{model_slug}_{parameter_name}"


def _render_parameter_control(metadata: ModelMetadata, parameter: ModelParameter) -> Any:
	stored_config = st.session_state.get("model_configs", {}).get(metadata.name, {})
	key = _parameter_widget_key(metadata.name, parameter.name)
	if key not in st.session_state:
		st.session_state[key] = stored_config.get(parameter.name, metadata.default_parameters.get(parameter.name))
	kwargs: dict[str, Any] = {"key": key}
	if parameter.control in {"float", "int"}:
		if parameter.minimum is not None:
			kwargs["min_value"] = parameter.minimum
		if parameter.maximum is not None:
			kwargs["max_value"] = parameter.maximum
		if parameter.step is not None:
			kwargs["step"] = parameter.step
		if parameter.format is not None:
			kwargs["format"] = parameter.format
		return st.number_input(parameter.label, **kwargs)
	if parameter.control == "select":
		return st.selectbox(parameter.label, list(parameter.options), key=key)
	if parameter.control == "optional_int":
		return st.selectbox(
			parameter.label,
			list(parameter.options),
			key=key,
			format_func=lambda value: "None" if value is None else str(value),
		)
	if parameter.control == "bool":
		return st.checkbox(parameter.label, key=key)
	raise ValueError(f"Unsupported parameter control: {parameter.control}")


def _model_parameters(metadata: ModelMetadata, expanded: bool = False) -> dict[str, Any]:
	"""Render controls from the parameter schema registered for this model."""
	with st.expander(f"{metadata.name} parameters", expanded=expanded):
		st.caption(metadata.description)
		if not metadata.parameter_schema:
			st.caption("Uses registered defaults.")
			return {}
		return {
			parameter.name: _render_parameter_control(metadata, parameter)
			for parameter in metadata.parameter_schema
		}


def _clear_training_state() -> None:
	"""Clear fitted artifacts while preserving currently rendered widget values."""
	st.session_state["trained_models"] = {}
	st.session_state["training_results"] = {}
	st.session_state["training_configuration_signature"] = None


def _render_workflow_progress() -> None:
	render_workflow_progress("models")


def _toggle_model_selection(model_name: str) -> None:
	problem_type = st.session_state.get("model_selection_widget_type", "classification")
	selected_by_type = dict(st.session_state.get("selected_models_by_type", {}))
	selected_models = list(selected_by_type.get(problem_type, []))
	if model_name in selected_models:
		selected_models.remove(model_name)
	else:
		selected_models.append(model_name)
	selected_by_type[problem_type] = selected_models
	st.session_state["selected_models_by_type"] = selected_by_type
	if problem_type == st.session_state.get("problem_type"):
		st.session_state["selected_models"] = selected_models
	st.session_state["model_selection_widget"] = selected_models
	if problem_type == st.session_state.get("problem_type"):
		_clear_training_state()


def _sync_model_selection() -> None:
	problem_type = st.session_state.get("model_selection_widget_type", "classification")
	selected_models = list(st.session_state["model_selection_widget"])
	selected_by_type = dict(st.session_state.get("selected_models_by_type", {}))
	selected_by_type[problem_type] = selected_models
	st.session_state["selected_models_by_type"] = selected_by_type
	if problem_type == st.session_state.get("problem_type"):
		st.session_state["selected_models"] = selected_models
		_clear_training_state()


def _render_dataset_information(dataset: pd.DataFrame, target_column: str, problem_type: str) -> None:
	with st.container(border=True):
		st.caption("Dataset information")
		st.write(st.session_state.get("dataset_name") or "Current dataset")
		st.caption(f"{len(dataset):,} rows · {dataset.shape[1]:,} columns")
		st.caption(f"Target: {target_column} ({problem_type.title()})")
		st.caption(f"Selected features: {len(st.session_state.get('selected_features', []))}")


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


def _render_model_library(problem_type: str, selected_models: list[str]) -> tuple[list[str], str]:
	if st.session_state.get("model_type_filter") not in {"Classification", "Regression"}:
		st.session_state["model_type_filter"] = problem_type.title()
	model_type_by_label = {"Classification": "classification", "Regression": "regression"}
	with st.container(border=True):
		st.markdown("#### Model library")
		search_column, type_column, family_column, sort_column = st.columns([1.4, 0.9, 1, 0.85])
		with search_column:
			search = st.text_input("Search models", key="model_search", placeholder="Search models")
		with type_column:
			model_type_label = st.selectbox("Model type", list(model_type_by_label), key="model_type_filter")
		library_problem_type = model_type_by_label[model_type_label]
		compatible_models = get_available_models(library_problem_type)
		family_options = get_model_families(library_problem_type)
	if st.session_state.get("model_family_filter") not in family_options:
		st.session_state["model_family_filter"] = "All Families"
	with family_column:
		family = st.selectbox("Model family", family_options, key="model_family_filter")
	with sort_column:
		sort_by = st.selectbox("Sort by", ["Registry order", "Name A–Z", "Family"], key="model_sort")

	filtered_models = get_models_by_family(library_problem_type, family)
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

	selected_by_type = dict(st.session_state.get("selected_models_by_type", {}))
	selected_for_type = [name for name in selected_by_type.get(library_problem_type, []) if name in compatible_models]
	if (
		st.session_state.get("model_selection_widget_type") != library_problem_type
		or "model_selection_widget" not in st.session_state
	):
		st.session_state["model_selection_widget"] = selected_for_type
		st.session_state["model_selection_widget_type"] = library_problem_type
	st.caption(f"{len(filtered_models)} compatible models · {len(selected_for_type)} selected")
	st.multiselect(
		"Select one or more models",
		compatible_models,
		key="model_selection_widget",
		on_change=_sync_model_selection,
		placeholder="Choose compatible models",
	)
	models_by_family: dict[str, list[str]] = {}
	for model_name in filtered_models:
		models_by_family.setdefault(get_model_metadata(model_name).family, []).append(model_name)
	for family_name, family_models in models_by_family.items():
		st.markdown(f"#### {family_name}")
		card_columns = st.columns(3)
		for index, model_name in enumerate(family_models):
			with card_columns[index % len(card_columns)]:
				_render_model_card(model_name, library_problem_type, model_name in selected_for_type)
	if not filtered_models:
		st.info("No models match this search and family filter.")
	return list(st.session_state.get("model_selection_widget", [])), library_problem_type


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
		render_page_header("Select and configure models", "Choose models for the detected problem type and configure their parameters.", "05 · Models")
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
		render_empty_state("Preprocessing is not configured", "Complete preprocessing configuration before selecting models.", "lock")
		return
	if not selected_features:
		st.error("No feature columns are available for training.")
		return
	compatible_models = get_available_models(problem_type)
	selected_by_type = dict(st.session_state.get("selected_models_by_type", {}))
	selected_before_run = list(st.session_state.get("selected_models", []))
	if selected_before_run and not selected_by_type.get(problem_type):
		selected_by_type[problem_type] = selected_before_run
	selected_by_type[problem_type] = [name for name in selected_by_type.get(problem_type, []) if name in compatible_models]
	st.session_state["selected_models_by_type"] = selected_by_type
	st.session_state["selected_models"] = selected_by_type[problem_type]

	_render_workflow_progress()
	page_heading, dataset_column = st.columns([2.8, 1], vertical_alignment="center")
	with page_heading:
		render_page_header("Select and configure models", "Choose compatible algorithms and configure them from registered parameter controls.", "05 · Models")
	with dataset_column:
		_render_dataset_information(dataset, target_column, problem_type)

	train_samples = len(st.session_state["X_train"]) if st.session_state.get("split_completed") else 0
	test_samples = len(st.session_state["X_test"]) if st.session_state.get("split_completed") else 0
	render_stat_cards(
		[
			("Problem type", problem_type.title()),
			("Target column", target_column),
			("Features", len(selected_features)),
			("Training samples", f"{train_samples:,}" if train_samples else "Split pending"),
			("Test samples", f"{test_samples:,}" if test_samples else "Split pending"),
		]
	)

	model_library_column, configuration_column = st.columns([1.8, 1], vertical_alignment="top")
	with model_library_column:
		selected_models, library_problem_type = _render_model_library(problem_type, selected_by_type[problem_type])
	with configuration_column:
		with st.container(border=True):
			st.markdown("#### Model configuration")
			if not selected_models:
				st.info("Select one or more models to configure their parameters.")
			model_configs = dict(st.session_state.get("model_configs", {}))
			for index, model_name in enumerate(selected_models):
				model_configs[model_name] = _model_parameters(get_model_metadata(model_name), expanded=index == 0)
			st.session_state["model_configs"] = model_configs

	_render_selected_models(selected_models, model_configs)
	back_column, spacer_column, continue_column = st.columns([1, 1, 2], vertical_alignment="center")
	with back_column:
		if st.button("← Back to preprocessing", key="models_to_preprocessing", width="stretch"):
			navigate("preprocessing")
	with continue_column:
		if library_problem_type != problem_type:
			st.caption("Switch Model Type to match the selected dataset target before continuing.")
		render_cta(
			"Continue to Training →",
			"training",
			"models_to_training",
			disabled=not st.session_state.get("selected_models") or library_problem_type != problem_type,
		)
