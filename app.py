import streamlit as st

from config.settings import APP_SUBTITLE, APP_TITLE
from src.utils.session_state import initialize_session_state
from ui.components import inject_theme, render_workflow_progress
from ui.sidebar import render_sidebar
from ui.tabs.dataset import render_dataset_tab
from ui.tabs.eda import render_eda_tab
from ui.tabs.evaluation import render_evaluation_tab
from ui.tabs.mlflow import render_mlflow_tab
from ui.tabs.prediction import render_prediction_tab
from ui.tabs.models import render_models_tab
from ui.tabs.preprocessing import render_preprocessing_tab
from ui.tabs.train_test_split import render_train_test_split_tab
from ui.tabs.training import render_training_tab
from ui.tabs.tuning import render_tuning_tab

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=":material/analytics:",
    layout="wide",
)

initialize_session_state()
inject_theme()
render_sidebar()
render_workflow_progress(st.session_state.get("current_page", "dataset"))

current_page = st.session_state.get("current_page", "dataset")
if current_page == "dataset":
    render_dataset_tab()
elif current_page == "eda":
    render_eda_tab()
elif current_page == "train_test_split":
    render_train_test_split_tab()
elif current_page == "preprocessing":
    render_preprocessing_tab()
elif current_page == "models":
    render_models_tab()
elif current_page == "training":
    render_training_tab()
elif current_page == "tuning":
    render_tuning_tab()
elif current_page == "evaluation":
    render_evaluation_tab()
elif current_page == "mlflow":
    render_mlflow_tab()
elif current_page == "prediction":
    render_prediction_tab()
else:
    st.session_state["current_page"] = "dataset"
    render_dataset_tab()
