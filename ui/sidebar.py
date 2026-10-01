"""Sidebar presentation components."""

from __future__ import annotations

import streamlit as st

from config.settings import APP_NAME, APP_SUBTITLE
from src.utils.session_state import reset_dataset_dependent_state
from ui.components import PAGE_ICONS, PAGE_LABELS, navigate


def render_sidebar() -> None:
    """Render the application identity and current workflow progress."""
    with st.sidebar:
        st.markdown(
            f'<div class="ml-sidebar-brand"><div class="ml-sidebar-mark">◆</div>'
            f'<div><div class="ml-sidebar-title">{APP_NAME} <span>Studio</span></div>'
            f'<div class="ml-sidebar-subtitle">{APP_SUBTITLE}</div></div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="ml-sidebar-section-label">Workflow</div>', unsafe_allow_html=True)

        dataset_ready = (
            bool(st.session_state.get("dataset_valid"))
            and bool(st.session_state.get("target_column"))
            and bool(st.session_state.get("problem_type"))
        )
        visited_pages = set(st.session_state.get("visited_pages", ["dataset"]))
        eda_ready = dataset_ready and "eda" in visited_pages
        preprocessing_ready = bool(st.session_state.get("preprocessing_applied"))
        models_trained = bool(st.session_state.get("trained_models"))
        completed_steps = {
            "Dataset": dataset_ready,
            "EDA": eda_ready,
            "Train / Test Split": bool(st.session_state.get("split_completed")),
            "Preprocessing": preprocessing_ready,
            "Models": bool(st.session_state.get("selected_models")),
            "Training": models_trained,
            "Tuning": bool(st.session_state.get("tuning_results")),
            "Evaluation": bool(st.session_state.get("evaluation_results")),
            "MLflow": bool(st.session_state.get("mlflow_run_information")),
            "Prediction": st.session_state.get("best_pipeline") is not None,
        }

        for index, (page, label) in enumerate(PAGE_LABELS.items(), start=1):
            active = page == st.session_state.get("current_page")
            marker = "▶" if active else ("✓" if completed_steps[label] else str(index))
            button_type = "primary" if active else "secondary"
            button_label = f"{marker}  {index}  {label}" if active or completed_steps[label] else f"{index}  {label}"
            if st.button(button_label, key=f"nav_{page}", type=button_type, width="stretch", icon=f":material/{PAGE_ICONS[page]}:"):
                navigate(page)

        dataset = st.session_state.get("dataset")
        dataset_name = st.session_state.get("dataset_name") or "No dataset loaded"
        with st.container(border=True):
            st.markdown("**Session info**")
            st.caption(dataset_name)
            if dataset is not None:
                st.caption(f"Target: {st.session_state.get('target_column') or 'Not selected'}")
                st.caption(f"Rows: {len(dataset):,} · Features: {max(0, dataset.shape[1] - 1):,}")
            else:
                st.caption("Upload a CSV to begin.")
        if st.button("Reset workflow", key="reset_workflow", width="stretch", icon=":material/restart_alt:"):
            reset_dataset_dependent_state()
            st.session_state["dataset_upload_generation"] = st.session_state.get("dataset_upload_generation", 0) + 1
            st.session_state["current_page"] = "dataset"
            st.rerun()
