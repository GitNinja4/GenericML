"""Dataset tab UI."""

from __future__ import annotations

import hashlib

import pandas as pd
import streamlit as st

from src.data.analyzer import (
    detect_problem_type,
    get_column_metadata,
    get_column_types,
    summarize_dataset,
    summarize_target,
)
from src.data.loader import load_csv
from src.data.validator import validate_dataset, validate_target
from src.utils.logging import get_logger
from src.utils.session_state import clear_preprocessing_state, clear_split_state, reset_dataset_dependent_state
from ui.components import render_cta, render_page_header, render_stat_cards, render_status_badge

logger = get_logger(__name__)


def _reset_dataset_dependent_state() -> None:
    """Reset downstream state when a new dataset is uploaded."""
    reset_dataset_dependent_state()


def render_dataset_tab() -> None:
    """Render the dataset upload, validation, and target configuration UI."""
    render_page_header("Dataset", "Upload your dataset and select the target column.", "01 · Start here")

    upload_generation = st.session_state.get("dataset_upload_generation", 0)
    with st.container(border=True):
        st.subheader("Upload CSV file")
        st.caption("CSV files are supported for this workflow.")
        upload_column, file_column = st.columns([1.6, 1], vertical_alignment="center")
        with upload_column:
            uploaded_file = st.file_uploader(
                "Drag and drop a CSV file here, or click to browse",
                type=["csv"],
                key=f"dataset_uploader_{upload_generation}",
            )
        with file_column:
            file_name = st.session_state.get("dataset_name")
            file_size = st.session_state.get("dataset_file_size")
            if uploaded_file is not None:
                file_name = uploaded_file.name
                file_size = uploaded_file.size

            if file_name:
                st.markdown("**File loaded successfully**")
                st.caption(f"{file_name}")
                if file_size is not None:
                    size_label = (
                        f"{file_size / (1024 * 1024):.1f} MB"
                        if file_size >= 1024 * 1024
                        else f"{file_size / 1024:.1f} KB"
                    )
                    dataset_shape = st.session_state.get("dataset_shape", (0, 0))
                    st.caption(f"{size_label} · {dataset_shape[0]:,} rows × {dataset_shape[1]:,} columns")
                else:
                    st.caption("Ready to analyze")
            else:
                st.caption("No file selected")

    if uploaded_file is not None:
        file_fingerprint = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
        is_new_dataset = st.session_state.get("dataset_fingerprint") != file_fingerprint
        if is_new_dataset:
            _reset_dataset_dependent_state()
        st.session_state["dataset_name"] = uploaded_file.name
        st.session_state["dataset_file_size"] = uploaded_file.size
        st.session_state["dataset_fingerprint"] = file_fingerprint

        try:
            df = load_csv(uploaded_file)
            validation = validate_dataset(df)
            st.session_state["dataset"] = df
            st.session_state["dataset_valid"] = True
            st.session_state["dataset_shape"] = df.shape
            st.session_state["numeric_columns"] = [
                column for column, data_type in get_column_types(df).items() if data_type == "numeric"
            ]
            st.session_state["categorical_columns"] = [
                column for column, data_type in get_column_types(df).items() if data_type == "categorical"
            ]
            st.session_state["boolean_columns"] = [
                column for column, data_type in get_column_types(df).items() if data_type == "boolean"
            ]
            if is_new_dataset:
                st.success("Dataset loaded successfully.")
            all_missing_columns = validation["warnings"]["all_missing_columns"]
            if all_missing_columns:
                st.warning(f"Columns contain only missing values: {', '.join(all_missing_columns)}.")
        except ValueError as exc:
            st.session_state["dataset"] = None
            st.session_state["dataset_valid"] = False
            st.error(str(exc))
            logger.warning("Dataset upload failed: %s", exc)
            return

    dataset = st.session_state.get("dataset")
    if dataset is None:
        st.info("Please upload a CSV dataset to continue.")
        return

    summary = summarize_dataset(dataset)
    with st.container(border=True):
        st.subheader("Dataset information")
        render_stat_cards(
            [
                ("Rows", f"{summary['rows']:,}"),
                ("Columns", summary["columns"]),
                ("Numerical features", summary["numerical_columns"]),
                ("Categorical features", summary["categorical_columns"] + summary["boolean_columns"]),
            ]
        )
        st.caption(
            f"Missing values: {summary['missing_values']:,}  ·  "
            f"Duplicate rows: {summary['duplicate_rows']:,}"
        )

    with st.container(border=True):
        preview_title, preview_control = st.columns([3, 1], vertical_alignment="center")
        preview_title.subheader("Dataset preview")
        with preview_control:
            preview_rows = st.selectbox("Rows to preview", [10, 25, 50, 100], index=0)
        st.dataframe(dataset.head(preview_rows), width="stretch")

    with st.container(border=True):
        st.subheader("Target column")
        st.caption("Choose the column the model should learn to predict. The application will not assume a target for you.")
        target_column_col, problem_type_col = st.columns([1.15, 1.85], vertical_alignment="bottom")
        with target_column_col:
            target_options = dataset.columns.tolist()
            target_value = st.session_state.get("target_column")
            target_index = target_options.index(target_value) if target_value in target_options else None
            target_column = st.selectbox(
                "Target column",
                options=target_options,
                index=target_index,
                placeholder="Select a target column",
            )
        with problem_type_col:
            selected_problem_type = st.session_state.get("problem_type_selection", "Automatic")
            if target_column is not None:
                if "problem_type_selector" not in st.session_state:
                    st.session_state["problem_type_selector"] = selected_problem_type
                selected_problem_type = st.radio(
                    "Problem type",
                    ["Automatic", "Classification", "Regression"],
                    key="problem_type_selector",
                    horizontal=True,
                )
                st.session_state["problem_type_selection"] = selected_problem_type

    previous_target_column = st.session_state.get("target_column")
    previous_problem_type = st.session_state.get("problem_type")
    if target_column is None:
        st.session_state["target_column"] = None
        st.session_state["problem_type"] = None
        st.info("Select a target column to continue configuring the dataset.")
        render_status_badge("Target selection required", "warning")
        return

    target_changed = previous_target_column is not None and previous_target_column != target_column
    if target_changed:
        clear_preprocessing_state(reset_controls=True)
        clear_split_state()
    st.session_state["target_column"] = target_column

    target_series = dataset[target_column]
    try:
        detected_problem_type = detect_problem_type(target_series)
    except ValueError as exc:
        if not target_changed:
            clear_preprocessing_state(reset_controls=True)
            clear_split_state()
        st.session_state["problem_type"] = None
        st.session_state["detected_problem_type"] = None
        st.error(str(exc))
        return
    st.session_state["detected_problem_type"] = detected_problem_type
    st.session_state["target_n_classes"] = int(target_series.nunique(dropna=True))

    if selected_problem_type == "Automatic":
        effective_problem_type = detected_problem_type
    else:
        effective_problem_type = selected_problem_type.lower()

    if effective_problem_type == "classification" and st.session_state["target_n_classes"] < 2:
        clear_preprocessing_state(reset_controls=True)
        clear_split_state()
        st.error("The selected target contains only one class. A classification model requires at least two classes.")
        return

    problem_type_changed = previous_problem_type is not None and previous_problem_type != effective_problem_type
    if problem_type_changed and not target_changed:
        clear_preprocessing_state(reset_controls=True)
        clear_split_state()

    if target_changed or problem_type_changed:
        st.info("Target or problem type changed. Preprocessing and training state were cleared.")

    st.session_state["problem_type"] = effective_problem_type

    if selected_problem_type != "Automatic" and selected_problem_type.lower() != detected_problem_type:
        st.warning(
            "The selected problem type differs from the automatically detected type. This may be intentional."
        )

    try:
        target_report = validate_target(target_series, problem_type=effective_problem_type)
    except ValueError as exc:
        st.error(str(exc))
        return

    if target_report.get("warnings"):
        for warning in target_report["warnings"]:
            st.warning(warning)

    with st.container(border=True):
        st.subheader("Problem type")
        if effective_problem_type == "classification":
            class_counts = target_series.dropna().value_counts(sort=False).rename_axis("Class").reset_index(name="Count")
            if class_counts["Class"].map(type).nunique() > 1:
                class_counts["Class"] = class_counts["Class"].map(repr)
            class_labels = "Binary classification" if st.session_state["target_n_classes"] == 2 else "Multiclass classification"
            cols = st.columns(3)
            cols[0].metric("Target", target_column)
            cols[1].metric("Detected type", class_labels)
            cols[2].metric("Target classes", st.session_state["target_n_classes"])
            st.info(f"Target has {st.session_state['target_n_classes']} unique classes.")
            st.dataframe(class_counts, width="stretch", hide_index=True)
        else:
            target_summary = summarize_target(target_series, problem_type="regression")
            cols = st.columns(4)
            cols[0].metric("Target", target_column)
            cols[1].metric("Detected type", "Regression")
            cols[2].metric("Missing values", target_summary["missing_count"])
            cols[3].metric("Unique values", target_summary["n_unique"])
            st.info("The selected target is numeric and will be treated as a regression problem.")
            st.dataframe(
                pd.DataFrame(
                    {
                        "Metric": ["Mean", "Median", "Min", "Max"],
                        "Value": [
                            target_summary["mean"],
                            target_summary["median"],
                            target_summary["min"],
                            target_summary["max"],
                        ],
                    }
                ),
                width="stretch",
            )

    with st.expander("Column metadata", expanded=False):
        metadata = get_column_metadata(dataset)
        st.dataframe(metadata, width="stretch", hide_index=True)

    render_status_badge("Dataset configuration is ready", "success")
    render_cta("Proceed to EDA →", "eda", "dataset_to_eda")
