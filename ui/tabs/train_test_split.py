"""User-controlled, leakage-safe raw train/test splitting UI."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.training.splitter import split_dataset
from ui.components import navigate, render_empty_state, render_page_header


def _invalidate_split() -> None:
	for key in ("X_train", "X_test", "y_train", "y_test"):
		st.session_state[key] = None
	st.session_state["split_completed"] = False
	st.session_state["train_test_split_source"] = None
	st.session_state["trained_models"] = {}
	st.session_state["training_results"] = {}
	st.session_state["training_configuration_signature"] = None


def _render_data_flow(dataset: pd.DataFrame, feature_count: int, target_column: str, problem_type: str) -> None:
	with st.container(border=True):
		st.markdown("#### Dataset information")
		st.write(st.session_state.get("dataset_name") or "Current dataset")
		st.caption(f"{len(dataset):,} rows · {dataset.shape[1]:,} columns")
		st.caption(f"Target: {target_column} · {problem_type.title()}")

	with st.container(border=True):
		st.markdown("#### Data flow · leakage-safe")
		steps = [
			("Raw dataset", f"{len(dataset):,} rows · unchanged"),
			("Separate X and y", f"X: {feature_count} raw features · y: 1 target"),
			("Train / test split", "Current step" if not st.session_state.get("split_completed") else "Split complete"),
		]
		for index, (title, detail) in enumerate(steps):
			with st.container(border=True):
				st.markdown(f"**{title}**")
				st.caption(detail)
			if index < len(steps) - 1:
				st.markdown("<div style='text-align:center;color:var(--ml-primary);'>↓</div>", unsafe_allow_html=True)
		st.markdown("<div style='text-align:center;color:var(--ml-primary);'>↙　　↘</div>", unsafe_allow_html=True)
		train_column, test_column = st.columns(2)
		with train_column:
			with st.container(border=True):
				st.markdown("**Training set**")
				st.caption(
					f"{len(st.session_state['X_train']):,} rows · X_train, y_train"
					if st.session_state.get("split_completed")
					else "X_train, y_train · awaiting split"
				)
		with test_column:
			with st.container(border=True):
				st.markdown("**Testing set**")
				st.caption(
					f"{len(st.session_state['X_test']):,} rows · X_test, y_test"
					if st.session_state.get("split_completed")
					else "X_test, y_test · awaiting split"
				)


def _render_split_results(problem_type: str) -> None:
	X_train = st.session_state["X_train"]
	X_test = st.session_state["X_test"]
	y_train = st.session_state["y_train"]
	y_test = st.session_state["y_test"]
	st.markdown("#### Split summary")
	st.success(
		"Train/test split completed successfully. The test set is now locked and will only be used during Evaluation.",
		icon=":material/verified_user:",
	)
	train_column, test_column = st.columns(2)
	with train_column:
		with st.container(border=True):
			st.markdown("**Training set**")
			st.metric("Rows", f"{len(X_train):,}", f"{len(X_train) / (len(X_train) + len(X_test)):.0%} of dataset")
			st.caption(f"X_train: {X_train.shape[1]} raw features · y_train: 1 target")
	with test_column:
		with st.container(border=True):
			st.markdown("**Test set · protected**")
			st.metric("Rows", f"{len(X_test):,}", f"{len(X_test) / (len(X_train) + len(X_test)):.0%} of dataset")
			st.caption("X_test and y_test remain untouched until Evaluation.")
	if problem_type == "classification":
		counts = pd.concat(
			[
				y_train.value_counts().rename("Training set"),
				y_test.value_counts().rename("Testing set"),
			],
			axis=1,
		).fillna(0).astype(int)
		with st.container(border=True):
			st.markdown("#### Class distribution")
			st.caption("Compare the target proportions in the training and protected test partitions.")
			st.bar_chart(counts)


def render_train_test_split_tab() -> None:
	"""Split untouched features and target before preprocessing configuration."""
	dataset = st.session_state.get("dataset")
	if dataset is None:
		render_page_header("Train / Test Split", "Create a reproducible holdout before preprocessing.", "03 · Split")
		render_empty_state("No dataset available", "Upload and validate a dataset before creating a split.", "database")
		return
	target_column = st.session_state.get("target_column")
	problem_type = st.session_state.get("problem_type") or st.session_state.get("detected_problem_type")
	if not target_column or target_column not in dataset.columns or problem_type not in {"classification", "regression"}:
		render_page_header("Train / Test Split", "Create a reproducible holdout before preprocessing.", "03 · Split")
		st.warning("Select and validate a target on the Dataset tab before splitting.")
		return
	feature_columns = [column for column in dataset.columns if column != target_column]
	if not feature_columns:
		render_page_header("Train / Test Split", "Create a reproducible holdout before preprocessing.", "03 · Split")
		st.error("The dataset must contain at least one feature column in addition to the target.")
		return
	page_column, information_column = st.columns([1.8, 1], vertical_alignment="top")
	with page_column:
		render_page_header(
			"Train / Test Split",
			"Split the dataset into training and test sets.",
			"03 · Train / Test Split",
		)
		st.info(
			"Leakage-safe workflow: preprocessing choices may be configured later, but transformers must be fitted only on X_train after this split.",
			icon=":material/shield_lock:",
		)
		with st.container(border=True):
			st.markdown("#### Split configuration")
			left_column, right_column = st.columns(2)
			with left_column:
				test_size_percent = st.slider(
					"Test size",
					min_value=10,
					max_value=50,
					step=5,
					key="train_test_size_percent",
					format="%d%%",
				)
				test_size = test_size_percent / 100
				st.session_state["train_test_size"] = test_size
			with right_column:
				random_state = st.number_input(
					"Random state",
					min_value=0,
					step=1,
					key="train_random_state",
				)
			if problem_type == "classification":
				stratified = st.checkbox(
					"Stratified split for classification",
					key="train_test_stratified",
				)
				st.caption("Preserves class proportions in both partitions when the class counts allow it.")
			else:
				stratified = False
				st.caption("Regression uses a reproducible random split without stratification.")
			st.caption(f"{test_size:.0%} of rows will be held out for testing.")
		current_config = {
			"test_size": float(test_size),
			"random_state": int(random_state),
			"stratified": bool(stratified),
		}
		stored_config = st.session_state.get("train_test_split_config", {})
		if st.session_state.get("split_completed") and stored_config != current_config:
			_invalidate_split()
			st.info("Split settings changed. Create a new split before continuing.")
		st.session_state["train_test_split_config"] = current_config
		if st.button(
			"Perform Train / Test Split",
			key="perform_train_test_split",
			type="primary",
			width="stretch",
			icon=":material/swap_horiz:",
		):
			X = dataset[feature_columns].copy()
			y = dataset[target_column].copy()
			try:
				partitions = split_dataset(
					X,
					y,
					problem_type,
					current_config["test_size"],
					current_config["random_state"],
					stratified=current_config["stratified"],
				)
			except ValueError as exc:
				_invalidate_split()
				st.error(str(exc))
			else:
				_invalidate_split()
				for key, partition in zip(("X_train", "X_test", "y_train", "y_test"), partitions, strict=True):
					st.session_state[key] = partition
				st.session_state["train_test_split_config"] = current_config
				st.session_state["train_test_split_source"] = {
					"dataset_fingerprint": st.session_state.get("dataset_fingerprint"),
					"target_column": target_column,
					"problem_type": problem_type,
				}
				st.session_state["split_completed"] = True
		with information_column:
			_render_data_flow(dataset, len(feature_columns), target_column, problem_type)
	if st.session_state.get("split_completed"):
		with st.container(border=True):
			_render_split_results(problem_type)
	back_column, spacer_column, continue_column = st.columns([1, 1, 2])
	with back_column:
		if st.button("← Back to EDA", key="split_to_eda", width="stretch"):
			navigate("eda")
	with continue_column:
		if st.button(
			"Continue to Preprocessing →",
			key="split_to_preprocessing",
			type="primary",
			width="stretch",
			disabled=not st.session_state.get("split_completed"),
		):
			navigate("preprocessing")