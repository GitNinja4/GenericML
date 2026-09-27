"""Sidebar presentation components."""

from __future__ import annotations

import streamlit as st

from config.settings import APP_NAME, APP_SUBTITLE
from ui.components import PAGE_ICONS, PAGE_LABELS, navigate


def render_sidebar() -> None:
    """Render the application identity and current workflow progress."""
    with st.sidebar:
        st.markdown(f"## {APP_NAME}")
        st.caption(APP_SUBTITLE)
        st.divider()
        st.markdown('<div style="font-size: 0.75rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: #94a3b8; margin-bottom: 0.8rem;">Workflow Progress</div>', unsafe_allow_html=True)

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
            "Preprocessing": preprocessing_ready,
            "Models": models_trained,
            "Tuning": False,
            "Evaluation": False,
            "MLflow": False,
            "Prediction": st.session_state.get("best_pipeline") is not None,
        }

        for page, label in PAGE_LABELS.items():
            marker = "●" if page == st.session_state.get("current_page") else ("✓" if completed_steps[label] else "○")
            button_type = "primary" if page == st.session_state.get("current_page") else "secondary"
            if st.button(f"{marker}  {label}", key=f"nav_{page}", type=button_type, width="stretch", icon=f":material/{PAGE_ICONS[page]}:"):
                navigate(page)

        st.caption("Your work stays in this session. Change upstream settings to refresh downstream results.")
