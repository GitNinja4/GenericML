"""Helpers for managing session_state fields."""

from __future__ import annotations

from config.settings import DEFAULT_RANDOM_STATE, DEFAULT_TEST_SIZE


def clear_training_state() -> None:
    """Clear artifacts that depend on the current dataset or preprocessing."""
    import streamlit as st

    for key, default in {
        "selected_models": [],
        "model_configs": {},
        "training_results": {},
        "training_configuration_signature": None,
        "X_train": None,
        "X_test": None,
        "y_train": None,
        "y_test": None,
        "trained_models": {},
        "evaluation_results": {},
        "best_pipeline": None,
        "prediction_schema": {},
        "mlflow_run_information": {},
    }.items():
        st.session_state[key] = default


def clear_preprocessing_state(reset_controls: bool = False) -> None:
    """Clear the active preprocessing configuration and its downstream artifacts."""
    import streamlit as st

    for key, default in {
        "preprocessing_config": {},
        "preprocessing_pipeline": None,
        "selected_features": [],
        "excluded_features": [],
        "preprocessing_applied": False,
        "processed_feature_names": [],
    }.items():
        st.session_state[key] = default
    if reset_controls:
        for key, default in {
            "prep_selected_features": [],
            "prep_features_initialized": False,
            "prep_numeric_missing": "None",
            "prep_numeric_constant": 0.0,
            "prep_categorical_missing": "None",
            "prep_categorical_constant": "Missing",
            "prep_outliers_enabled": False,
            "prep_outlier_method": "IQR",
            "prep_outlier_action": "Clip",
            "prep_lower_multiplier": 1.5,
            "prep_upper_multiplier": 1.5,
            "prep_encoding": "One-Hot Encoding",
            "prep_scaling": "None",
            "prep_feature_selection_enabled": False,
            "prep_feature_selection_method": "SelectKBest",
            "prep_feature_selection_k": 1,
        }.items():
            st.session_state[key] = default
    clear_training_state()


def reset_dataset_dependent_state() -> None:
    """Clear dataset identity and every downstream workflow artifact."""
    import streamlit as st

    for key, default in {
        "dataset": None,
        "dataset_name": None,
        "dataset_file_size": None,
        "dataset_fingerprint": None,
        "dataset_valid": False,
        "target_column": None,
        "problem_type": None,
        "detected_problem_type": None,
        "problem_type_selection": "Automatic",
        "target_n_classes": None,
        "numeric_columns": [],
        "categorical_columns": [],
        "boolean_columns": [],
        "dataset_shape": (0, 0),
        "train_test_size": DEFAULT_TEST_SIZE,
        "train_random_state": DEFAULT_RANDOM_STATE,
    }.items():
        st.session_state[key] = default
    st.session_state.pop("problem_type_selector", None)
    st.session_state["visited_pages"] = ["dataset"]
    clear_preprocessing_state(reset_controls=True)


def initialize_session_state() -> None:
    """Set up the baseline session state keys used by the app."""
    import streamlit as st

    defaults = {
        "current_page": "dataset",
        "visited_pages": ["dataset"],
        "dataset": None,
        "dataset_name": None,
        "dataset_file_size": None,
        "dataset_fingerprint": None,
        "dataset_valid": False,
        "target_column": None,
        "problem_type": None,
        "detected_problem_type": None,
        "problem_type_selection": "Automatic",
        "target_n_classes": None,
        "numeric_columns": [],
        "categorical_columns": [],
        "boolean_columns": [],
        "dataset_shape": (0, 0),
        "preprocessing_config": {},
        "preprocessing_pipeline": None,
        "selected_features": [],
        "excluded_features": [],
        "prep_selected_features": [],
        "prep_features_initialized": False,
        "preprocessing_applied": False,
        "processed_feature_names": [],
        "selected_models": [],
        "model_configs": {},
        "train_test_size": DEFAULT_TEST_SIZE,
        "train_random_state": DEFAULT_RANDOM_STATE,
        "training_results": {},
        "training_configuration_signature": None,
        "X_train": None,
        "X_test": None,
        "y_train": None,
        "y_test": None,
        "trained_models": {},
        "evaluation_results": {},
        "best_pipeline": None,
        "prediction_schema": {},
        "mlflow_run_information": {},
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
