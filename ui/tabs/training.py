"""Training and cross-validation UI using only the stored training partition."""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from src.models.model_factory import create_model
from src.training.workflow import train_models
from src.utils.logging import get_logger
from ui.components import navigate, render_cta, render_empty_state, render_page_header

logger = get_logger(__name__)

CLASSIFICATION_SCORING = {
	"accuracy": "Accuracy",
	"precision": "Precision",
	"recall": "Recall",
	"f1": "F1 Score",
}
REGRESSION_SCORING = {
	"mae": "MAE",
	"mse": "MSE",
	"rmse": "RMSE",
	"r2": "R²",
}


def _clear_training_results() -> None:
	st.session_state["trained_models"] = {}
	st.session_state["training_results"] = {}
	st.session_state["training_configuration_signature"] = None
	st.session_state["cv_configuration"] = None
	st.session_state["validation_configuration"] = None


def _training_signature(
	selected_models: list[str],
	model_configs: dict[str, dict[str, object]],
	preprocessing_config: dict[str, object],
	cv_config: dict[str, object],
) -> str:
	return repr((selected_models, model_configs, preprocessing_config, cv_config))


def _model_card(model_name: str, result: dict[str, object], problem_type: str, scoring_metric: str) -> None:
	with st.container(border=True):
		heading, time_column = st.columns([3, 1], vertical_alignment="center")
		with heading:
			st.markdown(f"#### {model_name}")
			if result["status"] == "trained":
				st.success("Trained")
			else:
				st.error("Training failed")
		with time_column:
			st.metric("Training time", f"{result.get('training_time_seconds', 0):.2f} s")
		if result["status"] != "trained":
			st.error(str(result.get("error", "Training failed.")))
			return

		training_metrics = result["training_metrics"]
		metric_names = list(training_metrics)
		if problem_type == "classification":
			class_counts = [item["count"] for item in result.get("class_distribution", [])]
			if class_counts and max(class_counts) / min(class_counts) >= 1.5:
				metric_names = [name for name in ("precision", "recall", "f1", "accuracy") if name in training_metrics]
		st.markdown("**Training metrics**")
		st.dataframe(
			pd.DataFrame(
				[
					{"Metric": CLASSIFICATION_SCORING.get(name, REGRESSION_SCORING.get(name, name)), "Score": value}
					for name, value in ((name, training_metrics[name]) for name in metric_names)
				]
			),
			width="stretch",
			hide_index=True,
		)
		if problem_type == "classification":
			st.caption(f"Precision, recall, and F1 use {result.get('classification_average', 'weighted')} averaging.")
		if result.get("validation_metrics_available"):
			st.markdown(f"**Validation metrics · {result['cv_folds']}-Fold CV**")
			st.dataframe(
				pd.DataFrame(
					[
						{
							"Metric": CLASSIFICATION_SCORING.get(name, REGRESSION_SCORING.get(name, name)),
							"Mean ± Std": f"{values['mean']:.3f} ± {values['std']:.3f}",
						}
						for name in metric_names
						for values in (result["cv_metrics"][name],)
					]
				),
				width="stretch",
				hide_index=True,
			)
			selected_cv = result["cv_metrics"][scoring_metric]
			train_score = float(training_metrics[scoring_metric])
			cv_score = float(selected_cv["mean"])
			lower_is_better = scoring_metric in {"mae", "mse", "rmse"}
			gap = cv_score - train_score if lower_is_better else train_score - cv_score
			st.caption(
				f"{result['validation_method']} · {result['cv_folds']} folds · "
				f"{CLASSIFICATION_SCORING.get(scoring_metric, REGRESSION_SCORING.get(scoring_metric))}: "
				f"mean {cv_score:.4f}, std {float(selected_cv['std']):.4f} · train/validation gap {gap:.4f}"
			)
			diagnostic = result["diagnostic"]
			if diagnostic["status"] == "possible_overfitting":
				st.warning(f"⚠ Possible Overfitting\n\n{diagnostic['message']}")
			elif diagnostic["status"] == "possible_underfitting":
				st.info(f"ℹ Possible Underfitting\n\n{diagnostic['message']}")
			else:
				st.success(f"✓ {diagnostic['message']}")
		else:
			st.info(result["diagnostic"]["message"])
		st.caption(f"Sampling: {result.get('sampling_method', 'Disabled')}")

		with st.expander("Model details", expanded=False):
			parameter_rows = [{"Parameter": name, "Value": str(value)} for name, value in result["parameters"].items()]
			st.markdown("**Parameters used**")
			st.dataframe(pd.DataFrame(parameter_rows), width="stretch", hide_index=True)
			st.markdown("**Training configuration**")
			preprocessing_summary = result.get("preprocessing_summary") or ", ".join(
				f"{name.replace('_', ' ')}: {value}"
				for name, value in result.get("preprocessing_config", {}).items()
				if name not in {"selected_features", "excluded_features", "target_column"}
			) or "Configured preprocessing pipeline"
			st.write(f"Preprocessing: {preprocessing_summary}")
			st.write(f"Oversampling: {result.get('sampling_method', 'Disabled')}")
			st.write(f"Validation strategy: {result.get('validation_strategy', 'None')}")
			st.write(f"Final fit samples: {result.get('final_fit_samples', 0):,}")
			st.write(f"Training rows removed as missing: {result.get('missing_rows_removed', 0):,}")
			st.write(f"Training rows removed as outliers: {result.get('outlier_rows_removed', 0):,}")
			if result.get("validation_metrics_available"):
				st.markdown("**Individual fold scores**")
				st.json(result["cv_metrics"])
				st.json({"Validation configuration": result["validation_configuration"]})
			st.markdown("**Preprocessing configuration**")
			st.json(result.get("preprocessing_config", {}))
			st.markdown("**Training-target class distribution**")
			st.json(result.get("class_distribution"))


def render_training_tab() -> None:
	"""Train and validate models using X_train/y_train; never access holdout data."""
	dataset = st.session_state.get("dataset")
	problem_type = st.session_state.get("problem_type") or st.session_state.get("detected_problem_type")
	target_column = st.session_state.get("target_column")
	heading_column, protection_column = st.columns([2.3, 1], vertical_alignment="center")
	with heading_column:
		render_page_header("Training", "Train the selected models using training data only.", "06 · Training")
	with protection_column:
		with st.container(border=True):
			st.markdown("🔒 **TEST DATA PROTECTED**")
			st.caption("The test set has not been used for preprocessing, sampling, training, validation, or model selection. It will be used only during the Evaluation stage.")
	if dataset is None or not target_column or not problem_type:
		render_empty_state("Dataset not ready", "Upload a dataset and select a target before training.", "database")
		return
	if not st.session_state.get("split_completed"):
		render_empty_state("Train/test split required", "Create the training and protected holdout partitions first.", "shuffle")
		render_cta("Go to Train / Test Split →", "train_test_split", "training_to_split")
		return
	expected_split_source = {
		"dataset_fingerprint": st.session_state.get("dataset_fingerprint"),
		"target_column": target_column,
		"problem_type": problem_type,
	}
	if st.session_state.get("train_test_split_source") != expected_split_source:
		st.session_state["split_completed"] = False
		st.session_state["X_train"] = None
		st.session_state["y_train"] = None
		st.session_state["train_test_split_source"] = None
		_clear_training_results()
		st.warning("The dataset target or problem type changed. Recreate the train/test split before training.")
		render_cta("Go to Train / Test Split →", "train_test_split", "training_to_split_stale")
		return
	preprocessing_pipeline = st.session_state.get("preprocessing_pipeline")
	if preprocessing_pipeline is None or not st.session_state.get("preprocessing_applied"):
		render_empty_state("Preprocessing configuration required", "Configure preprocessing before training. No preprocessing is fitted here on the full dataset.", "tune")
		render_cta("Go to Preprocessing →", "preprocessing", "training_to_preprocessing")
		return
	selected_models = list(st.session_state.get("selected_models", []))
	model_configs = st.session_state.get("model_configs", {})
	selected_features = st.session_state.get("selected_features", [])
	if not selected_models:
		render_empty_state("No models selected", "Select at least one model before training.", "model_training")
		render_cta("Back to Models →", "models", "training_to_models")
		return
	if not selected_features:
		st.error("No feature columns are configured for training.")
		return
	X_train = st.session_state.get("X_train")
	y_train = st.session_state.get("y_train")
	if X_train is None or y_train is None:
		st.session_state["split_completed"] = False
		st.session_state["train_test_split_source"] = None
		_clear_training_results()
		st.warning("The training partition is incomplete. Recreate the train/test split before training.")
		render_cta("Go to Train / Test Split →", "train_test_split", "training_to_split_incomplete")
		return
	missing_features = [column for column in selected_features if column not in X_train.columns]
	if missing_features:
		st.error(f"Selected features are missing from the training partition: {missing_features}")
		return
	train_features = X_train.loc[:, selected_features].copy()
	if train_features.empty or y_train.empty:
		st.error("Training data is empty. Recreate the train/test split before training.")
		return
	if len(train_features) != len(y_train) or not train_features.index.equals(y_train.index):
		st.error("Training features and target are not row-aligned. Recreate the train/test split.")
		return
	if y_train.isna().any():
		st.error("Training target contains missing values. Correct the target before training.")
		return
	if problem_type == "classification" and y_train.nunique() < 2:
		st.error("Classification training requires at least two target classes.")
		return
	if problem_type == "regression" and not pd.api.types.is_numeric_dtype(y_train.dtype):
		st.error("Regression training requires a numeric target.")
		return
	metric_options = CLASSIFICATION_SCORING if problem_type == "classification" else REGRESSION_SCORING
	default_metric = "accuracy" if problem_type == "classification" else "r2"
	if st.session_state.get("training_scoring_metric") not in metric_options:
		st.session_state["training_scoring_metric"] = default_metric
	max_folds = min(10, len(y_train))
	if problem_type == "classification":
		max_folds = min(max_folds, int(y_train.value_counts().min()))
	validation_options = ["None"] + (["K-Fold Cross-Validation"] if max_folds >= 2 else [])
	if st.session_state.get("training_validation_strategy") not in validation_options:
		st.session_state["training_validation_strategy"] = "None"
	default_folds = max(2, min(5, max_folds))
	if max_folds >= 2 and not 2 <= int(st.session_state.get("training_cv_folds", default_folds)) <= max_folds:
		st.session_state["training_cv_folds"] = default_folds
	categorical_columns = train_features.select_dtypes(include=["object", "category", "string", "bool"]).columns.tolist()
	samplers = ["Disabled", "SMOTE", "SMOTENC"]
	if st.session_state.get("training_sampling_method") not in samplers:
		st.session_state["training_sampling_method"] = "Disabled"
	configuration_column, balance_column, overview_column = st.columns([1.15, 1, 0.9], vertical_alignment="top")
	with configuration_column:
		with st.container(border=True):
			st.markdown("#### Training configuration")
			validation_strategy = st.selectbox(
				"Validation strategy",
				validation_options,
				key="training_validation_strategy",
			)
			random_state = st.number_input("Random state", min_value=0, step=1, key="training_random_state")
			if validation_strategy == "K-Fold Cross-Validation":
				cv_folds = st.number_input("Number of folds", min_value=2, max_value=max_folds, step=1, key="training_cv_folds")
				shuffle = st.checkbox("Shuffle folds", key="training_shuffle_folds")
				use_default_scoring = st.checkbox("Use default scoring", key="training_use_default_scoring")
				if use_default_scoring:
					st.session_state["training_scoring_metric"] = default_metric
					st.caption(f"Scoring metric: **{metric_options[default_metric]}**")
				else:
					st.selectbox(
						"Scoring metric",
						list(metric_options),
						key="training_scoring_metric",
						format_func=lambda value: metric_options[value],
					)
				if problem_type == "classification":
					st.caption("Stratified folds preserve class proportions. Each fold fits its own preprocessing and sampling.")
				else:
					st.caption("Preprocessing is fitted independently inside each training fold.")
			else:
				cv_folds = 0
				shuffle = False
				st.info("Without cross-validation, training metrics alone cannot assess generalization.")
			if max_folds < 2:
				st.warning("K-Fold validation requires at least two rows per class/fold.")
	with balance_column:
		with st.container(border=True):
			st.markdown("#### Class imbalance handling")
			sampling_method = "Disabled"
			if problem_type == "classification":
				sampling_method = st.selectbox(
					"Oversampling",
					samplers,
					key="training_sampling_method",
				)
				class_counts = y_train.value_counts()
				imbalance_ratio = float(class_counts.max() / class_counts.min())
				if imbalance_ratio >= 1.5:
					st.warning(f"Training class ratio is {imbalance_ratio:.2f}. Oversampling is optional.")
				else:
					st.caption(f"Class ratio: {imbalance_ratio:.2f}; no substantial imbalance detected.")
				st.caption("SMOTE supports numeric features; choose SMOTENC when categorical features are present.")
			else:
				st.caption("Oversampling is available for classification only.")
	with overview_column:
		with st.container(border=True):
			st.markdown("#### Training data overview")
			st.metric("Training samples", f"{len(train_features):,}")
			st.metric("Test samples · protected", f"{len(st.session_state['X_test']):,}")
			st.metric("Features", len(selected_features))
			st.caption(f"Problem type: **{problem_type.title()}**")
			if problem_type == "classification":
				class_counts = y_train.value_counts()
				st.markdown("**Training class distribution**")
				st.dataframe(
					pd.DataFrame(
						{
							"Class": class_counts.index.astype(str),
							"Rows": class_counts.values,
							"Share": [f"{value / len(y_train):.1%}" for value in class_counts.values],
						}
					),
					width="stretch",
					hide_index=True,
				)
			st.caption("Training, validation, preprocessing, and sampling never use the protected test set.")

	cv_config = {
		"validation_strategy": validation_strategy,
		"validation_method": ("Stratified K-Fold" if problem_type == "classification" else "K-Fold") if validation_strategy != "None" else None,
		"cv_folds": int(cv_folds) if validation_strategy != "None" else None,
		"shuffle": bool(shuffle),
		"random_state": int(random_state),
		"scoring_metric": (default_metric if st.session_state.get("training_use_default_scoring", True) else st.session_state["training_scoring_metric"]) if validation_strategy != "None" else None,
		"sampling_method": sampling_method,
	}
	configuration_signature = _training_signature(
		selected_models,
		model_configs,
		st.session_state.get("preprocessing_config", {}),
		cv_config,
	)
	if st.session_state.get("training_configuration_signature") not in {None, configuration_signature}:
		_clear_training_results()

	back_column, _, run_column = st.columns([1, 1, 2], vertical_alignment="center")
	with back_column:
		if st.button("← Back to Models", key="training_to_models_back", width="stretch"):
			navigate("models")
	with run_column:
		train_clicked = st.button(
			"Train Selected Models",
			key="train_selected_models",
			type="primary",
			width="stretch",
			icon=":material/play_arrow:",
		)
	if train_clicked:
		sampling_error = None
		if sampling_method == "SMOTE" and categorical_columns:
			sampling_error = "SMOTE requires numerical-only features. Choose SMOTENC for this training data."
		elif sampling_method == "SMOTENC" and not categorical_columns:
			sampling_error = "SMOTENC requires categorical features. Choose SMOTE for this numerical-only training data."
		if sampling_error is None:
			for model_name in selected_models:
				try:
					create_model(model_name, problem_type, model_configs.get(model_name))
				except Exception as exc:
					sampling_error = f"Invalid parameters for {model_name}: {exc}"
					break
		if sampling_error:
			st.error(sampling_error)
		else:
			_clear_training_results()
			try:
				with st.status("Training selected models on training data...", expanded=True) as status:
					if validation_strategy == "None":
						st.write(f"Fitting each model on {len(train_features):,} training rows without cross-validation.")
					else:
						st.write(f"Running {cv_config['validation_method']} on {len(train_features):,} training rows.")
					st.write("The test partition is not passed to preprocessing, sampling, training, or validation.")
					result = train_models(
						train_features,
						y_train,
						preprocessing_pipeline,
						selected_models,
						problem_type,
						model_configs=model_configs,
						preprocessing_config=st.session_state.get("preprocessing_config", {}),
						cv_folds=cv_config["cv_folds"] or 2,
						random_state=cv_config["random_state"],
						shuffle=cv_config["shuffle"],
						scoring_metric=cv_config["scoring_metric"] or default_metric,
						validation_strategy=validation_strategy,
						sampling_method=sampling_method,
					)
					if not result["trained_models"]:
						status.update(label="Training failed", state="error")
					elif validation_strategy == "None":
						status.update(label="Training complete", state="complete")
					else:
						status.update(label="Training and validation complete", state="complete")
			except Exception as exc:
				logger.exception("Training workflow failed")
				st.error(str(exc))
			else:
				st.session_state["trained_models"] = result["trained_models"]
				st.session_state["training_results"] = result["training_results"]
				st.session_state["training_configuration_signature"] = configuration_signature
				st.session_state["cv_configuration"] = result["cv_configuration"]
				st.session_state["validation_configuration"] = result["validation_configuration"]
				st.session_state["training_class_distribution"] = result["class_distribution"]

	training_results = st.session_state.get("training_results", {})
	if not training_results:
		return
	succeeded = sum(result["status"] == "trained" for result in training_results.values())
	failed = len(training_results) - succeeded
	total_training_time = sum(float(result.get("training_time_seconds", 0)) for result in training_results.values())
	main_column, summary_column = st.columns([1.75, 0.9], vertical_alignment="top")
	with main_column:
		with st.container(border=True):
			st.markdown(f"#### Trained Models ({succeeded})")
			st.caption("Models are fitted on training data; validation uses CV folds from the training partition only.")
			if succeeded == 0:
				st.error("None of the selected models trained successfully. Review the per-model error details below.")
		for model_name, result in training_results.items():
			_model_card(model_name, result, problem_type, cv_config["scoring_metric"] or default_metric)
	with summary_column:
		with st.container(border=True):
			st.markdown("#### Training Summary")
			st.metric("Models attempted", len(training_results))
			st.metric("Successful", succeeded)
			st.metric("Failed", failed)
			st.metric("Total training time", f"{total_training_time:.2f} s")
			st.write("Problem type", f"**{problem_type.title()}**")
			st.write("Training dataset size", f"**{len(train_features):,} rows**")
			split_config = st.session_state.get("train_test_split_config", {})
			test_size = split_config.get("test_size")
			test_rows = (
				math.ceil(len(dataset) * test_size)
				if isinstance(test_size, float)
				else int(test_size)
				if isinstance(test_size, int)
				else None
			)
			st.write("Protected test dataset size", f"**{test_rows:,} rows**" if test_rows is not None else "Not inspected")
			st.write("Test set", "**NOT USED**")
			st.write("Validation strategy", f"**{validation_strategy}**")
			if validation_strategy != "None":
				st.write("Number of folds", f"**{cv_config['cv_folds']}**")
			st.write("Oversampling", f"**{sampling_method}**")
			st.write("Random state", f"**{cv_config['random_state']}**")
			if validation_strategy != "None":
				st.write("Scoring metric", f"**{metric_options[cv_config['scoring_metric']]}**")
	with st.container(border=True):
		st.markdown("#### Model comparison")
		if st.session_state.get("training_comparison_metric") not in metric_options:
			st.session_state["training_comparison_metric"] = cv_config["scoring_metric"] or default_metric
		comparison_metric = st.selectbox(
			"Compare by",
			list(metric_options),
			key="training_comparison_metric",
			format_func=lambda value: metric_options[value],
		)
		comparison_rows = []
		for model_name, result in training_results.items():
			if result["status"] != "trained":
				continue
			row = {"Model": model_name, "Training": result["training_metrics"][comparison_metric]}
			if result.get("validation_metrics_available"):
				fold_result = result["cv_metrics"][comparison_metric]
				row["CV mean"] = fold_result["mean"]
				row["CV std"] = fold_result["std"]
			comparison_rows.append(row)
		if comparison_rows:
			st.dataframe(pd.DataFrame(comparison_rows), width="stretch", hide_index=True)
			if validation_strategy == "None":
				st.caption("Comparison based on training metrics only. Without validation, underfitting and generalization cannot be reliably assessed.")
			else:
				st.caption("Comparison uses training and cross-validation metrics from the training partition only. The protected test set is not used.")
		with st.container(border=True):
			st.markdown("#### Next Steps")
			st.markdown("Review training metrics and validation results when available.")
			st.markdown("Use any train–validation gap as a diagnostic, not a final performance judgment.")
			st.markdown("The holdout will be used for the first time in Evaluation.")
			if st.session_state.get("trained_models"):
				render_cta("Continue to Tuning →", "tuning", "training_to_tuning")