"""MLflow experiment tracking UI."""

from __future__ import annotations

import streamlit as st

from src.mlflow.tracker import log_model_run
from ui.components import render_cta, render_empty_state, render_page_header


def _run_parameters(record: dict[str, object], model_name: str) -> dict[str, object]:
	parameters: dict[str, object] = {f"model.{key}": value for key, value in record.get("parameters", {}).items()}
	parameters["model_name"] = model_name
	parameters["sampling_method"] = record.get("sampling_method", "Disabled")
	parameters["validation_strategy"] = record.get("validation_configuration", {}).get("validation_strategy", "None")
	parameters["preprocessing"] = record.get("preprocessing_config", {})
	return parameters


def _run_metrics(record: dict[str, object], evaluation: dict[str, object], tuning: dict[str, object]) -> dict[str, float]:
	metrics: dict[str, float] = {}
	for name, value in record.get("training_metrics", {}).items():
		metrics[f"train_{name}"] = float(value)
	for name, value in evaluation.items():
		if isinstance(value, (int, float)) and name not in {"sample_count"}:
			metrics[f"test_{name}"] = float(value)
	if tuning.get("best_score") is not None:
		metrics["tuning_best_cv_score"] = float(tuning["best_score"])
	return metrics


def render_mlflow_tab() -> None:
	"""Log the final fitted pipeline and actual session results to MLflow."""
	render_page_header("MLflow Tracking", "Track experiments and model versions.", "09 · Track")
	with st.container(border=True):
		st.info("Only actual session parameters, metrics, the final pipeline, and generated run summary are logged.", icon=":material/verified:")
	model_name = st.session_state.get("final_model_name")
	trained_models = st.session_state.get("trained_models", {})
	evaluation = st.session_state.get("evaluation_results", {})
	if not model_name or model_name not in trained_models or not evaluation:
		render_empty_state("Evaluation required", "Select a final model and complete test evaluation before logging an experiment.", "assessment")
		render_cta("Go to Evaluation →", "evaluation", "mlflow_to_evaluation")
		return
	model_record = trained_models[model_name]
	pipeline = st.session_state.get("best_pipeline") or model_record.get("pipeline")
	if pipeline is None:
		st.error("The final fitted pipeline is unavailable.")
		return

	configuration_column, model_column = st.columns([1.55, 0.85], vertical_alignment="top")
	with configuration_column:
		with st.container(border=True):
			st.subheader("Experiment configuration")
			experiment_name = st.text_input("Experiment name", value="GenericML_Experiment", key="mlflow_experiment_name")
			st.caption(f"Final model: **{model_name}** · Test metrics: **{len(evaluation)} values**")
			if st.button("Start MLflow Logging", key="start_mlflow_logging", type="primary", width="stretch", icon=":material/cloud_upload:"):
				try:
					tuning = st.session_state.get("tuning_results", {}).get(model_name, {})
					run_info = log_model_run(
						experiment_name,
						model_name,
						pipeline,
						_run_parameters(model_record, model_name),
						_run_metrics(model_record, evaluation, tuning),
						tags={"problem_type": str(st.session_state.get("problem_type")), "model_state": str(evaluation.get("model_state", "Original"))},
					)
					st.session_state["mlflow_run_information"] = run_info
					st.success("MLflow run logged successfully.")
				except Exception as exc:
					st.error(str(exc))
	with model_column:
		with st.container(border=True):
			st.markdown("#### Model to track")
			st.write(f"Model: **{model_name}**")
			st.write(f"Task: **{str(st.session_state.get('problem_type') or 'Unknown').title()}**")
			st.write(f"Test metrics: **{len(evaluation)}**")
			st.caption("The complete fitted pipeline and actual session metrics are logged.")

	run_info = st.session_state.get("mlflow_run_information", {})
	if run_info:
		results_column, summary_column = st.columns([1.55, 0.85], vertical_alignment="top")
		with results_column:
			run_tabs = st.tabs(["Run details", "Metrics", "Parameters", "Artifacts"])
			with run_tabs[0]:
				st.write(f"Run ID: **{run_info['run_id']}**")
				st.write(f"Run status: **{run_info['status']}**")
				st.write(f"Timestamp: **{run_info['timestamp']}**")
				st.write(f"Model: **{run_info['model_name']}**")
			with run_tabs[1]:
				st.dataframe([{"Metric": key, "Value": value} for key, value in run_info.get("metrics", {}).items()], width="stretch", hide_index=True)
			with run_tabs[2]:
				st.dataframe([{"Parameter": key, "Value": value} for key, value in run_info.get("parameters", {}).items()], width="stretch", hide_index=True)
			with run_tabs[3]:
				st.write(f"Artifact URI: **{run_info['artifact_uri']}**")
				st.caption("The logged model artifact is the complete fitted preprocessing-plus-model pipeline.")
		with summary_column:
			with st.container(border=True):
				st.markdown("#### Logged run")
				st.write(f"Status: **{run_info['status']}**")
				st.write(f"Model: **{run_info['model_name']}**")
				st.write(f"Metrics: **{len(run_info.get('metrics', {}))}**")
				render_cta("Continue to Prediction →", "prediction", "mlflow_to_prediction")
