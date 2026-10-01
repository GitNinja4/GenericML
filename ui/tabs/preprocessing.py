"""User-controlled preprocessing configuration UI."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.data.analyzer import detect_high_cardinality, detect_outliers, summarize_dataset
from src.preprocessing.pipeline_builder import (
	PreprocessingConfig,
	build_preprocessing_pipeline,
	validate_preprocessing_config,
)
from src.utils.logging import get_logger
from src.utils.session_state import clear_preprocessing_state
from ui.components import navigate, render_cta, render_empty_state, render_page_header

logger = get_logger(__name__)


def _feature_types(dataset: pd.DataFrame, target_column: str | None) -> tuple[list[str], list[str]]:
	"""Return numerical and categorical feature columns, excluding the target."""
	candidates = [column for column in dataset.columns if column != target_column]
	numeric = dataset[candidates].select_dtypes(include="number").columns.tolist()
	categorical = dataset[candidates].select_dtypes(include=["object", "category", "bool"]).columns.tolist()
	return numeric, categorical


def _render_dataset_information(dataset: pd.DataFrame, target_column: str | None, problem_type: str | None) -> None:
	summary = summarize_dataset(dataset)
	with st.container(border=True):
		st.caption("Dataset information")
		st.write(st.session_state.get("dataset_name") or "Uploaded dataset")
		st.caption(f"{summary['rows']:,} rows · {summary['columns']:,} columns")
		st.caption(f"Target: {target_column or 'Not selected'} ({problem_type or 'Not selected'})")


def _render_summary(dataset: pd.DataFrame, target_column: str | None, problem_type: str | None) -> None:
	summary = summarize_dataset(dataset)
	feature_columns = [column for column in dataset.columns if column != target_column]
	feature_dataset = dataset[feature_columns]
	with st.container(horizontal=True):
		for label, value in (
			("Total features", len(feature_columns)),
			("Numerical features", len(feature_dataset.select_dtypes(include="number").columns)),
			("Categorical features", len(feature_dataset.select_dtypes(include=["object", "category", "bool"]).columns)),
			("Target column", f"{target_column or 'Not selected'} ({problem_type or 'Not selected'})"),
		):
			st.metric(label, value, border=True)
	warnings: list[str] = []
	if summary["missing_values"]:
		warnings.append("Missing values detected.")
	outlier_summary = detect_outliers(dataset)
	if not outlier_summary.empty and (outlier_summary["Outlier Count"] > 0).any():
		warnings.append("Potential outliers detected.")
	if not detect_high_cardinality(dataset).empty:
		warnings.append("High-cardinality categorical features detected.")
	numeric = dataset.select_dtypes(include="number")
	if len(numeric.columns) > 1 and (numeric.std(numeric_only=True) > 0).any():
		warnings.append("Numerical features may have different scales.")
	if warnings:
		st.caption(" · ".join(warnings))


def _render_feature_selection(dataset: pd.DataFrame, target_column: str | None) -> list[str]:
	st.markdown("#### 1. Feature selection")
	feature_options = [column for column in dataset.columns if column != target_column]
	if not feature_options:
		st.error("No feature columns are available after excluding the target.")
		return []
	if not st.session_state.get("prep_features_initialized", False):
		st.session_state["prep_selected_features"] = feature_options.copy()
		st.session_state["prep_features_initialized"] = True
	left, right = st.columns(2)
	if left.button("Select all features", key="prep_select_all"):
		st.session_state["prep_selected_features"] = feature_options.copy()
	if right.button("Deselect all features", key="prep_deselect_all"):
		st.session_state["prep_selected_features"] = []
	selected = st.multiselect(
		"Include features",
		feature_options,
		key="prep_selected_features",
	)
	st.caption("The target column is kept separate and cannot be selected as a feature.")
	return selected


def _render_missing_controls(has_numeric: bool, has_categorical: bool) -> tuple[str, float, str, str]:
	st.markdown("#### 1. Missing value handling")
	numeric_strategy = st.selectbox(
		"Numerical strategy",
		["None", "Mean", "Median", "Constant", "Drop"],
		key="prep_numeric_missing",
		disabled=not has_numeric,
	)
	numeric_constant = 0.0
	if has_numeric and numeric_strategy == "Constant":
		numeric_constant = st.number_input("Numerical constant value", value=0.0, key="prep_numeric_constant")
	categorical_strategy = st.selectbox(
		"Categorical strategy",
		["None", "Most Frequent", "Constant", "Drop"],
		key="prep_categorical_missing",
		disabled=not has_categorical,
	)
	categorical_constant = "Missing"
	if has_categorical and categorical_strategy == "Constant":
		categorical_constant = st.text_input("Categorical constant value", value="Missing", key="prep_categorical_constant")
	return (
		numeric_strategy if has_numeric else "None",
		float(numeric_constant),
		categorical_strategy if has_categorical else "None",
		categorical_constant,
	)


def _render_outlier_controls(has_numeric: bool) -> tuple[bool, str, str, float, float]:
	st.markdown("#### 2. Outlier handling")
	enabled = st.checkbox("Enable outlier handling", value=False, key="prep_outliers_enabled", disabled=not has_numeric)
	if not has_numeric:
		st.info("Select numerical features to configure outlier handling.")
		return False, "IQR", "Clip", 1.5, 1.5
	if not enabled:
		st.info("No outlier treatment will be applied.")
		return False, "IQR", "Clip", 1.5, 1.5
	method = st.selectbox("Method", ["IQR"], key="prep_outlier_method")
	action = st.selectbox("Action", ["Clip", "Remove"], key="prep_outlier_action")
	lower = st.number_input("Lower IQR multiplier", min_value=0.0, value=1.5, step=0.1, key="prep_lower_multiplier")
	upper = st.number_input("Upper IQR multiplier", min_value=0.0, value=1.5, step=0.1, key="prep_upper_multiplier")
	if action == "Remove":
		st.info("Outlier rows are removed from the training split only; held-out test rows remain unchanged.")
	return enabled, method, action, float(lower), float(upper)


def _render_encoding_controls(has_categorical: bool) -> str:
	st.markdown("#### 3. Encoding")
	if not has_categorical:
		st.info("No categorical features are selected.")
	strategy = st.selectbox(
		"Encoding strategy",
		["One-Hot Encoding", "Ordinal Encoding", "None"],
		key="prep_encoding",
		disabled=not has_categorical,
	)
	if not has_categorical:
		return "None"
	if strategy == "Ordinal Encoding":
		st.info("Ordinal encoding introduces an artificial numerical ordering when no category order is supplied.")
	if strategy == "None" and has_categorical:
		st.warning("Many scikit-learn models require categorical columns to be encoded numerically.")
	return strategy


def _render_scaling_controls(has_numeric: bool) -> str:
	st.markdown("#### 4. Feature scaling")
	if not has_numeric:
		st.info("No numerical features are selected.")
	strategy = st.selectbox(
		"Scaling strategy",
		["None", "StandardScaler", "MinMaxScaler", "RobustScaler"],
		key="prep_scaling",
		disabled=not has_numeric,
	)
	st.caption("StandardScaler uses mean and standard deviation; MinMaxScaler maps a range; RobustScaler uses median and IQR.")
	return strategy if has_numeric else "None"


def _render_feature_selection_controls(feature_count: int) -> tuple[bool, str, int]:
	st.markdown("#### 5. Optional feature selection")
	enabled = st.checkbox("Enable feature selection", value=False, key="prep_feature_selection_enabled")
	method = st.selectbox("Method", ["SelectKBest"], key="prep_feature_selection_method", disabled=not enabled)
	max_features = max(1, feature_count)
	stored_k = int(st.session_state.get("prep_feature_selection_k", min(10, max_features)))
	if not 1 <= stored_k <= max_features:
		st.session_state["prep_feature_selection_k"] = min(max(stored_k, 1), max_features)
	k = st.number_input("Number of features", min_value=1, max_value=max_features, value=min(10, max_features), step=1, key="prep_feature_selection_k", disabled=not enabled)
	return enabled, method, int(k)


def _render_configuration(config: PreprocessingConfig) -> None:
	st.markdown(
		f"**Missing values:** Numerical → {config.numeric_missing_strategy}; "
		f"Categorical → {config.categorical_missing_strategy}"
	)
	st.markdown(
		f"**Outliers:** "
		f"{config.outlier_method} {config.outlier_action.lower()}ing"
		if config.outlier_enabled
		else "**Outliers:** Disabled"
	)
	st.markdown(f"**Encoding:** {config.encoding_strategy}")
	st.markdown(f"**Scaling:** {config.scaling_strategy}")
	feature_selection = (
		f"{config.feature_selection_method} — K={config.feature_selection_k}"
		if config.feature_selection_enabled
		else "Disabled"
	)
	st.markdown(f"**Feature selection:** {feature_selection}")
	st.caption("This configuration is serializable and will be fitted on training data only.")
	if config.encoding_strategy == "One-Hot Encoding":
		st.info("Unseen categories are ignored safely during transformation.")


def _pipeline_steps_markup(config: PreprocessingConfig, applied: bool) -> str:
	steps = [
		("Feature inclusion", f"{len(config.selected_features)} input features", bool(config.selected_features)),
		(
			"Missing values",
			f"{config.numeric_missing_strategy} · {config.categorical_missing_strategy}",
			config.numeric_missing_strategy != "None" or config.categorical_missing_strategy != "None",
		),
		(
			"Outlier handling",
			f"{config.outlier_method} · {config.outlier_action}" if config.outlier_enabled else "Disabled",
			config.outlier_enabled,
		),
		("Categorical encoding", config.encoding_strategy, config.encoding_strategy != "None"),
		("Feature scaling", config.scaling_strategy, config.scaling_strategy != "None"),
		(
			"Feature selection (optional)",
			f"{config.feature_selection_method} · {config.feature_selection_k} features" if config.feature_selection_enabled else "Disabled",
			config.feature_selection_enabled,
		),
		("Model-ready", "Applied · ready to fit" if applied else "Apply settings to prepare", applied),
	]
	return (
		"""
		<style>
		.prep-pipeline { display: grid; gap: 0.25rem; }
		.prep-step { position: relative; display: grid; grid-template-columns: 1.7rem minmax(0, 1fr); gap: 0.6rem; padding: 0.55rem 0.45rem; border-radius: 8px; }
		.prep-step:not(:last-child)::after { content: ""; position: absolute; left: 1.24rem; top: 2.05rem; bottom: -0.35rem; width: 1px; background: var(--ml-border); }
		.prep-marker { z-index: 1; display: grid; place-items: center; width: 1.55rem; height: 1.55rem; border: 1px solid var(--ml-border); border-radius: 50%; background: var(--ml-panel); color: var(--ml-muted); font-size: 0.72rem; font-weight: 700; }
		.prep-step.is-configured { background: var(--ml-panel-soft); }
		.prep-step.is-configured .prep-marker { border-color: transparent; background: var(--ml-primary); color: #fff; }
		.prep-step.is-ready .prep-marker { border-color: transparent; background: #16a34a; color: #fff; }
		.prep-step-title { color: var(--ml-text); font-size: 0.82rem; font-weight: 700; line-height: 1.25; }
		.prep-step-detail { margin-top: 0.12rem; color: var(--ml-muted); font-size: 0.7rem; line-height: 1.3; }
		</style>
		<div class="prep-pipeline">
		"""
		+ "".join(
			f'<div class="prep-step {"is-ready" if title == "Model-ready" and enabled else "is-configured" if enabled else ""}">'
			f'<span class="prep-marker">{"✓" if enabled else index}</span>'
			f'<div><div class="prep-step-title">{title}</div><div class="prep-step-detail">{detail}</div></div></div>'
			for index, (title, detail, enabled) in enumerate(steps, start=1)
		)
		+ "</div>"
	)


def _render_pipeline(config: PreprocessingConfig, applied: bool):
	with st.container(border=True):
		st.markdown("#### Preprocessing pipeline")
		steps_placeholder = st.empty()
		apply_placeholder = st.empty()
		apply_clicked = apply_placeholder.button(
			"Apply preprocessing configuration",
			type="primary",
			key="apply_preprocessing",
			width="stretch",
		)
		steps_placeholder.markdown(_pipeline_steps_markup(config, applied), unsafe_allow_html=True)
	return apply_clicked, steps_placeholder


def _render_preview(dataset: pd.DataFrame, config: PreprocessingConfig, applied: bool) -> None:
	with st.container(border=True):
		preview_heading, preview_summary = st.columns([2, 1], vertical_alignment="center")
		with preview_heading:
			st.markdown("#### Preprocessing preview")
			st.caption("A preview of selected input data and feature metadata.")
		with preview_summary:
			st.caption(f"{len(dataset):,} rows × {len(config.selected_features):,} selected input features")

		input_tab, feature_tab = st.tabs(["Selected input data", "Feature information"])
		with input_tab:
			if config.selected_features:
				st.dataframe(dataset[config.selected_features].head(10), width="stretch", hide_index=True)
			else:
				st.info("Select one or more features to preview the input data.")
		with feature_tab:
			feature_rows = [
				{
					"Feature": column,
					"Data type": str(dataset[column].dtype),
					"Missing values": int(dataset[column].isna().sum()),
					"Status": "Included" if column in config.selected_features else "Excluded",
				}
				for column in dataset.columns
				if column != config.target_column
			]
			st.dataframe(pd.DataFrame(feature_rows), width="stretch", hide_index=True)
		st.caption(
			"Configuration applied. The pipeline will fit on training data only."
			if applied
			else "Transformed feature names and values are produced after fitting on training data."
		)


def render_preprocessing_tab() -> None:
	"""Render user-controlled preprocessing settings without modifying raw data."""
	dataset = st.session_state.get("dataset")
	if dataset is None:
		render_page_header("Data preprocessing", "Configure and apply transformations to prepare your data for modeling.", "04 · Prepare")
		render_empty_state("No dataset available", "Upload a dataset and select a target before configuring preprocessing.", "tune")
		return
	target_column = st.session_state.get("target_column")
	problem_type = st.session_state.get("problem_type") or st.session_state.get("detected_problem_type")
	heading_column, information_column = st.columns([2.4, 1], vertical_alignment="center")
	with heading_column:
		render_page_header("Preprocessing Configuration", "Configure preprocessing based on your EDA findings.", "04 · Prepare")
	with information_column:
		_render_dataset_information(dataset, target_column, problem_type)
	_render_summary(dataset, target_column, problem_type)
	numeric_columns, categorical_columns = _feature_types(dataset, target_column)
	main_column, pipeline_column = st.columns([2.2, 1], vertical_alignment="top")
	with main_column:
		with st.container(border=True):
			selected_features = _render_feature_selection(dataset, target_column)
			excluded_features = [column for column in dataset.columns if column not in selected_features and column != target_column]
			st.caption(f"{len(selected_features)} included · {len(excluded_features)} excluded · Target is kept separate")

		missing_column, outlier_column = st.columns(2, vertical_alignment="top")
		with missing_column:
			with st.container(border=True):
				selected_numeric = [column for column in selected_features if column in numeric_columns]
				selected_categorical = [column for column in selected_features if column in categorical_columns]
				numeric_strategy, numeric_constant, categorical_strategy, categorical_constant = _render_missing_controls(
					bool(selected_numeric),
					bool(selected_categorical),
				)
				numeric_missing = int(dataset[selected_numeric].isna().sum().sum()) if selected_numeric else 0
				categorical_missing = int(dataset[selected_categorical].isna().sum().sum()) if selected_categorical else 0
				st.caption(f"{numeric_missing:,} numerical · {categorical_missing:,} categorical missing values")
		with outlier_column:
			with st.container(border=True):
				outlier_enabled, outlier_method, outlier_action, lower, upper = _render_outlier_controls(bool(selected_numeric))

		encoding_column, scaling_column = st.columns(2, vertical_alignment="top")
		with encoding_column:
			with st.container(border=True):
				encoding_strategy = _render_encoding_controls(bool(selected_categorical))
		with scaling_column:
			with st.container(border=True):
				scaling_strategy = _render_scaling_controls(bool(selected_numeric))

		with st.container(border=True):
			selection_enabled, selection_method, selection_k = _render_feature_selection_controls(len(selected_features))

	config = PreprocessingConfig(
		numeric_missing_strategy=numeric_strategy,
		numeric_constant_value=numeric_constant,
		categorical_missing_strategy=categorical_strategy,
		categorical_constant_value=categorical_constant,
		outlier_enabled=outlier_enabled,
		outlier_method=outlier_method,
		outlier_action=outlier_action,
		lower_multiplier=lower,
		upper_multiplier=upper,
		encoding_strategy=encoding_strategy,
		scaling_strategy=scaling_strategy,
		feature_selection_enabled=selection_enabled,
		feature_selection_method=selection_method,
		feature_selection_k=selection_k,
		selected_features=list(selected_features),
		excluded_features=[column for column in dataset.columns if column not in selected_features and column != target_column],
		target_column=target_column,
		problem_type=problem_type,
	)

	active_config = st.session_state.get("preprocessing_config", {})
	settings_changed = bool(st.session_state.get("preprocessing_applied")) and active_config != config.to_dict()
	if settings_changed:
		clear_preprocessing_state()
		st.warning("Preprocessing settings have changed. Click Apply preprocessing configuration to update the active pipeline.")

	with pipeline_column:
		with st.container(border=True):
			st.markdown("#### Target column")
			st.metric("Target", target_column or "Not selected", problem_type or "Problem type not selected")
		applied = bool(st.session_state.get("preprocessing_applied")) and not settings_changed
		apply_clicked, pipeline_steps = _render_pipeline(config, applied)

	if apply_clicked:
		try:
			validate_preprocessing_config(config, [column for column in dataset.columns if column != target_column], target_column)
			pipeline = build_preprocessing_pipeline(config, selected_numeric, selected_categorical)
		except ValueError as exc:
			st.error(str(exc))
			return
		st.session_state["preprocessing_config"] = config.to_dict()
		st.session_state["preprocessing_pipeline"] = pipeline
		st.session_state["selected_features"] = list(selected_features)
		st.session_state["excluded_features"] = config.excluded_features
		st.session_state["preprocessing_applied"] = True
		st.session_state["processed_feature_names"] = []
		logger.info("Preprocessing configuration applied for %s", st.session_state.get("dataset_name", "dataset"))
		st.success("Preprocessing configuration applied. The pipeline is ready to fit on training data.")

	applied = (
		bool(st.session_state.get("preprocessing_applied"))
		and st.session_state.get("preprocessing_config") == config.to_dict()
	)
	pipeline_steps.markdown(_pipeline_steps_markup(config, applied), unsafe_allow_html=True)

	if st.session_state.get("preprocessing_applied"):
		active_config = PreprocessingConfig.from_dict(st.session_state["preprocessing_config"])
		with pipeline_column:
			with st.expander("Active configuration"):
				_render_configuration(active_config)
		_render_preview(dataset, active_config, applied)
	else:
		_render_preview(dataset, config, False)
	back_column, spacer_column, models_column = st.columns([1, 1, 2], vertical_alignment="center")
	with back_column:
		if st.button("← Back to Train / Test Split", key="preprocessing_to_split", width="stretch"):
			navigate("train_test_split")
	with models_column:
		render_cta("Configure models →", "models", "preprocessing_to_models", disabled=not st.session_state.get("preprocessing_applied"))
