"""Polished placeholders for future workflow phases."""

from __future__ import annotations

import streamlit as st

from ui.components import navigate, render_empty_state, render_page_header


PLACEHOLDER_CONTENT = {
    "tuning": ("Hyperparameter tuning", "Optimize model parameters with structured search.", "auto_awesome", ["Optuna integration", "Parameter search", "Optimization trials", "Best-parameter configuration"]),
    "evaluation": ("Model evaluation", "Analyze and compare trained model performance.", "assessment", ["Classification metrics", "Regression metrics", "Confusion matrix", "ROC curve", "Model comparison"]),
    "mlflow": ("MLflow tracking", "Track experiments and manage trained models.", "monitoring", ["Experiment tracking", "Model versioning", "Parameter and metric logging", "Model comparison"]),
    "prediction": ("Prediction", "Make predictions using trained models.", "lightbulb", ["Single prediction input", "Batch prediction", "Saved model reuse", "Export predictions"]),
}


def render_placeholder_page(page: str) -> None:
    """Render a future-phase page without introducing its backend functionality."""
    title, subtitle, icon, planned = PLACEHOLDER_CONTENT[page]
    render_page_header(title, subtitle, "Coming next")
    render_empty_state("This feature will be available in the next phase.", "The current workflow is ready for this stage, but its backend is intentionally not enabled yet.", icon)
    st.subheader("Planned capabilities")
    for item in planned:
        st.markdown(f"- {item}")
    if page == "tuning":
        if st.button("Continue to evaluation", key="placeholder_tuning_next", width="stretch"):
            navigate("evaluation")
    elif page == "evaluation":
        if st.button("Track experiment", key="placeholder_evaluation_next", width="stretch"):
            navigate("mlflow")
    elif page == "mlflow":
        if st.button("Make predictions", key="placeholder_mlflow_next", width="stretch"):
            navigate("prediction")