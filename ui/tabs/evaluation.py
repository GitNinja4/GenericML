"""Evaluation of the selected final pipeline on the protected test partition."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.evaluation.evaluator import evaluate_pipeline
from ui.components import navigate, render_cta, render_empty_state, render_page_header


def _classification_results(result: dict[str, object]) -> None:
	metrics = st.columns(4)
	for column, label, key in zip(metrics, ("Accuracy", "Precision", "Recall", "F1 Score"), ("accuracy", "precision", "recall", "f1")):
		column.metric(label, f"{float(result[key]):.4f}")
	chart_column, report_column = st.columns([1.1, 1])
	with chart_column:
		st.markdown("#### Confusion matrix")
		figure = px.imshow(
			result["confusion_matrix"],
			text_auto=True,
			labels={"x": "Predicted label", "y": "True label", "color": "Count"},
			 x=result["labels"],
			y=result["labels"],
			color_continuous_scale="Blues",
		)
		figure.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=360)
		st.plotly_chart(figure, width="stretch", key="evaluation_confusion_matrix")
	with report_column:
		st.markdown("#### Classification report")
		report = result["classification_report"]
		rows = []
		for label, values in report.items():
			if isinstance(values, dict):
				rows.append({"Class": label, "Precision": values.get("precision", 0.0), "Recall": values.get("recall", 0.0), "F1": values.get("f1-score", 0.0), "Support": values.get("support", 0)})
		st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _regression_results(result: dict[str, object]) -> None:
	metrics = st.columns(4)
	for column, label, key in zip(metrics, ("MAE", "MSE", "RMSE", "R²"), ("mae", "mse", "rmse", "r2")):
		column.metric(label, f"{float(result[key]):.4f}")
	plot_column, residual_column = st.columns(2)
	with plot_column:
		st.markdown("#### Actual vs predicted")
		figure = px.scatter(x=result["actual"], y=result["predicted"], labels={"x": "Actual", "y": "Predicted"})
		figure.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=340)
		st.plotly_chart(figure, width="stretch", key="evaluation_actual_predicted")
	with residual_column:
		st.markdown("#### Residual plot")
		figure = px.scatter(x=result["predicted"], y=result["residuals"], labels={"x": "Predicted", "y": "Residual"})
		figure.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=340)
		st.plotly_chart(figure, width="stretch", key="evaluation_residuals")


def render_evaluation_tab() -> None:
	"""Evaluate only the explicitly selected final pipeline on X_test/y_test."""
	dataset = st.session_state.get("dataset")
	problem_type = st.session_state.get("problem_type") or st.session_state.get("detected_problem_type")
	trained_models = st.session_state.get("trained_models", {})
	render_page_header("Evaluation", "Evaluate the final model on the unseen test set.", "08 · Test Evaluation")
	with st.container(border=True):
		st.warning(
			"This is the first workflow stage that uses X_test and y_test. Test metrics are not sent back to training or tuning.",
			icon=":material/shield:",
		)
	if dataset is None or problem_type not in {"classification", "regression"}:
		render_empty_state("Dataset not ready", "Complete dataset setup before evaluation.", "database")
		return
	if not st.session_state.get("split_completed") or st.session_state.get("X_test") is None:
		render_empty_state("Protected test set unavailable", "Complete the Train / Test Split stage before evaluation.", "lock")
		return
	if not trained_models:
		render_empty_state("No trained models", "Train a model before evaluating it on the protected test set.", "model_training")
		render_cta("Go to Training →", "training", "evaluation_to_training")
		return

	model_names = list(trained_models)
	model_name = st.selectbox("Select final model", model_names, key="evaluation_model")
	model_state = "Tuned" if model_name in st.session_state.get("tuned_models", {}) else "Original"
	st.caption(f"Configuration: **{model_state}** · Model: **{model_name}**")
	if st.button("Use Trained Model", key="use_evaluation_model", type="secondary", icon=":material/check_circle:"):
		st.session_state["best_pipeline"] = trained_models[model_name]["pipeline"]
		st.session_state["final_model_name"] = model_name
		st.success(f"{model_name} selected as the final model. It is ready for test evaluation.")

	if st.session_state.get("final_model_name") == model_name and st.session_state.get("best_pipeline") is not None:
		if st.button("Evaluate on Test Set", key="evaluate_test_set", type="primary", width="stretch", icon=":material/assessment:"):
			try:
				result = evaluate_pipeline(
					st.session_state["best_pipeline"],
					st.session_state["X_test"],
					st.session_state["y_test"],
					problem_type,
				)
				result["model_name"] = model_name
				result["model_state"] = model_state
				st.session_state["evaluation_results"] = result
				st.success("Evaluation completed on the protected test set.")
			except Exception as exc:
				st.error(str(exc))

	result = st.session_state.get("evaluation_results") or {}
	if result and result.get("model_name") == model_name:
		results_column, summary_column = st.columns([1.7, 0.85], vertical_alignment="top")
		with results_column:
			with st.container(border=True):
				st.markdown("#### Test performance")
				st.caption(f"{result['sample_count']:,} protected test rows · {result['model_state']} model")
				if problem_type == "classification":
					_classification_results(result)
				else:
					_regression_results(result)
		with summary_column:
			with st.container(border=True):
				st.markdown("#### Final model summary")
				st.write(f"Model: **{model_name}**")
				st.write(f"Configuration: **{result['model_state']}**")
				st.write(f"Sampling: **{trained_models[model_name].get('sampling_method', 'Disabled')}**")
				st.write(f"Validation: **{trained_models[model_name].get('validation_configuration', {}).get('validation_strategy', 'None')}**")
				st.write(f"Test samples: **{result['sample_count']:,}**")
				st.caption("The protected test set is used only in this evaluation stage.")
			render_cta("Continue to MLflow →", "mlflow", "evaluation_to_mlflow")


