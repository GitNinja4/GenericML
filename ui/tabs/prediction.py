"""Schema-driven single-row prediction UI."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.prediction.predictor import predict_with_pipeline
from src.prediction.schema import build_prediction_schema
from ui.components import render_empty_state, render_page_header


def _render_input(schema: list[dict[str, object]]) -> dict[str, object]:
	values: dict[str, object] = {}
	with st.form("prediction_form", clear_on_submit=False):
		st.markdown("#### Input features")
		columns = st.columns(2)
		for index, item in enumerate(schema):
			name = str(item["name"])
			kind = item["kind"]
			with columns[index % 2]:
				if kind == "numeric":
					default = float(item["default"])
					values[name] = st.number_input(name, value=default, key=f"prediction_numeric_{name}")
				elif kind == "boolean":
					values[name] = st.checkbox(name, value=bool(item["default"]), key=f"prediction_boolean_{name}")
				elif item["categories"] and len(item["categories"]) <= 100:
					categories = list(item["categories"])
					values[name] = st.selectbox(name, categories, index=categories.index(item["default"]) if item["default"] in categories else 0, key=f"prediction_category_{name}")
				else:
					values[name] = st.text_input(name, value=str(item["default"]), key=f"prediction_text_{name}")
		submitted = st.form_submit_button("Predict", type="primary", width="stretch", icon=":material/online_prediction:")
	if submitted:
		return values
	return {}


def render_prediction_tab() -> None:
	"""Make one prediction using the saved complete inference pipeline."""
	render_page_header("Prediction", "Make predictions using the saved trained pipeline.", "10 · Predict")
	pipeline = st.session_state.get("best_pipeline")
	dataset = st.session_state.get("dataset")
	target_column = st.session_state.get("target_column")
	problem_type = st.session_state.get("problem_type") or st.session_state.get("detected_problem_type")
	if pipeline is None or dataset is None or not target_column:
		render_empty_state("Final pipeline required", "Complete Evaluation and select a final trained pipeline before making predictions.", "lock")
		return
	try:
		schema = build_prediction_schema(dataset, target_column, st.session_state.get("selected_features") or None)
	except ValueError as exc:
		st.error(str(exc))
		return
	st.session_state["prediction_schema"] = schema
	input_column, summary_column = st.columns([1.55, 0.85], vertical_alignment="top")
	with input_column:
		with st.container(border=True):
			st.info("Inputs follow the uploaded data schema and use the fitted training pipeline.", icon=":material/schema:")
		values = _render_input(schema)
	if values:
		try:
			result = predict_with_pipeline(pipeline, values, schema, problem_type)
			st.session_state["prediction_result"] = result
		except Exception as exc:
			st.error(str(exc))

	with summary_column:
		with st.container(border=True):
			st.markdown("#### Prediction context")
			st.write(f"Model: **{st.session_state.get('final_model_name') or 'Final pipeline'}**")
			st.write(f"Task: **{problem_type.title()}**")
			st.write(f"Target: **{target_column}**")
			st.write(f"Input features: **{len(schema)}**")
			st.caption(st.session_state.get("dataset_name") or "Current dataset")
		result = st.session_state.get("prediction_result")
		if result:
			with st.container(border=True):
				st.markdown("#### Prediction result")
				st.success(f"{result['prediction']}", icon=":material/check_circle:")
				if result.get("probabilities"):
					st.markdown("**Class probabilities**")
					st.dataframe(
						pd.DataFrame([{"Class": label, "Probability": value} for label, value in result["probabilities"].items()]),
						width="stretch",
						hide_index=True,
					)
