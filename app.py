import streamlit as st

from config.settings import APP_SUBTITLE, APP_TITLE
from src.utils.session_state import initialize_session_state
from ui.components import inject_theme, render_top_header
from ui.sidebar import render_sidebar
from ui.tabs.dataset import render_dataset_tab
from ui.tabs.eda import render_eda_tab
from ui.tabs.models import render_models_tab
from ui.tabs.placeholders import render_placeholder_page
from ui.tabs.preprocessing import render_preprocessing_tab

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=":material/analytics:",
    layout="wide",
)

initialize_session_state()
inject_theme()
render_sidebar()
render_top_header()

current_page = st.session_state.get("current_page", "dataset")
if current_page == "dataset":
    render_dataset_tab()
elif current_page == "eda":
    render_eda_tab()
elif current_page == "preprocessing":
    render_preprocessing_tab()
elif current_page == "models":
    render_models_tab()
elif current_page in {"tuning", "evaluation", "mlflow", "prediction"}:
    render_placeholder_page(current_page)
else:
    st.session_state["current_page"] = "dataset"
    render_dataset_tab()
