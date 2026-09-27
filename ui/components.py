"""Reusable visual and navigation components for the Streamlit shell."""

from __future__ import annotations

import streamlit as st

from config.settings import APP_NAME, APP_SUBTITLE, APP_TAGLINE

PAGE_LABELS = {
	"dataset": "Dataset",
	"eda": "EDA",
	"preprocessing": "Preprocessing",
	"models": "Models",
	"tuning": "Tuning",
	"evaluation": "Evaluation",
	"mlflow": "MLflow",
	"prediction": "Prediction",
}

PAGE_ICONS = {
	"dataset": "database",
	"eda": "analytics",
	"preprocessing": "tune",
	"models": "model_training",
	"tuning": "auto_awesome",
	"evaluation": "assessment",
	"mlflow": "monitoring",
	"prediction": "lightbulb",
}


def inject_theme() -> None:
	"""Apply the centralized visual language used by all pages."""
	st.markdown(
		"""
		<style>
		:root {
			--ml-primary: #2563eb;
			--ml-primary-strong: #1d4ed8;
			--ml-primary-soft: #dbeafe;
			--ml-accent: #8b5cf6;
			--ml-navy: #0b1220;
			--ml-navy-soft: #111827;
			--ml-success: #16a34a;
			--ml-warning: #b45309;
			--ml-error: #b91c1c;
			--ml-bg: #f3f7fb;
			--ml-panel: #ffffff;
			--ml-panel-soft: #edf4ff;
			--ml-text: #0f172a;
			--ml-muted: #475569;
			--ml-border: #dfe7f3;
		}
		[data-testid="stApp"] {
			--ml-primary: light-dark(#2563eb, #2563eb);
			--ml-primary-strong: light-dark(#1d4ed8, #1d4ed8);
			--ml-primary-soft: light-dark(#dbeafe, #172e50);
			--ml-bg: light-dark(#edf3fb, #0b1220);
			--ml-panel: light-dark(#ffffff, #111827);
			--ml-panel-soft: light-dark(#edf4ff, #17263c);
			--ml-text: light-dark(#0f172a, #f1f5f9);
			--ml-muted: light-dark(#475569, #b6c2d1);
			--ml-border: light-dark(#dfe7f3, #334155);
			--ml-control: light-dark(#ffffff, #1f2937);
			background: var(--ml-bg);
			color: var(--ml-text);
		}
		html, body {
			background: transparent;
		}
		[data-testid="stAppViewContainer"] {
			background: var(--ml-bg);
			color: var(--ml-text);
		}
		.stApp {
			background: var(--ml-bg);
			color: var(--ml-text);
		}
		div[data-testid="stHeader"] {
			background: linear-gradient(180deg, #0b1220 0%, #111827 100%) !important;
			border-bottom: 1px solid rgba(148, 163, 184, 0.18) !important;
			box-shadow: none !important;
		}
		div[data-testid="stHeader"] * {
			color: #f8fafc !important;
		}
		div[data-testid="stToolbar"] {
			background: rgba(15, 23, 42, 0.8) !important;
			border: 1px solid rgba(148, 163, 184, 0.2) !important;
			border-radius: 10px !important;
			box-shadow: 0 2px 10px rgba(15, 23, 42, 0.2) !important;
		}
		[data-testid="stMainMenuButton"] {
			background: transparent !important;
			color: #f8fafc !important;
			border: 0 !important;
			box-shadow: none !important;
			border-radius: 8px !important;
			opacity: 1 !important;
			visibility: visible !important;
		}
		[data-testid="stMainMenuButton"] svg {
			color: #f8fafc !important;
		}
		[data-testid="stMainMenuButton"]:hover {
			background: rgba(148, 163, 184, 0.16) !important;
			color: #ffffff !important;
		}
		button[kind="header"],
		[data-testid="stToolbar"] button,
		[data-testid="stToolbar"] [role="button"],
		[data-testid="baseButton-header"],
		button[data-testid="stHeaderActionButton"] {
			background: rgba(30, 41, 59, 0.9) !important;
			color: #f8fafc !important;
			border: 1px solid rgba(148, 163, 184, 0.28) !important;
		}
		[data-testid="stToolbar"] button svg,
		[data-testid="stToolbar"] [role="button"] svg {
			color: #f8fafc !important;
			fill: #f8fafc !important;
		}
		button[kind="header"]:hover,
		[data-testid="stToolbar"] button:hover,
		[data-testid="stToolbar"] [role="button"]:hover,
		[data-testid="baseButton-header"]:hover,
		button[data-testid="stHeaderActionButton"]:hover {
			background: rgba(37, 99, 235, 0.9) !important;
			color: #ffffff !important;
		}
		.block-container {
			padding-top: 1.5rem;
			padding-bottom: 3rem;
			max-width: 1400px;
		}
		[data-testid="stSidebar"] {
			background: linear-gradient(180deg, var(--ml-navy) 0%, var(--ml-navy-soft) 100%);
			border-right: 1px solid rgba(148, 163, 184, 0.15);
		}
		[data-testid="stSidebar"] * {
			color: #e2e8f0;
		}
		[data-testid="stSidebar"] .stButton > button {
			border: 1px solid rgba(148, 163, 184, 0.18);
			background: rgba(15, 23, 42, 0.35);
			color: #e2e8f0;
			text-align: left;
			justify-content: flex-start;
			border-radius: 12px;
			min-height: 2.8rem;
			padding: 0.72rem 0.9rem;
			font-weight: 600;
		}
		[data-testid="stSidebar"] .stButton > button:hover,
		[data-testid="stSidebar"] .stButton > button[kind="primary"] {
			background: linear-gradient(135deg, var(--ml-primary) 0%, var(--ml-primary-strong) 100%);
			color: #ffffff;
			border-color: rgba(96, 165, 250, 0.8);
			box-shadow: 0 8px 20px rgba(37, 99, 235, 0.2);
		}
		[data-testid="stSidebar"] .stButton > button:focus,
		.stButton > button:focus-visible {
			box-shadow: 0 0 0 3px rgba(96, 165, 250, 0.25);
		}
		.ml-brand {
			margin-bottom: 1.2rem;
			padding: 0.35rem 0.15rem;
		}
		.ml-brand-title {
			font-size: clamp(2rem, 2.5vw, 2.8rem);
			font-weight: 800;
			letter-spacing: -0.06em;
			line-height: 1.05;
			color: #f8fafc;
		}
		.ml-brand-title span {
			background: linear-gradient(135deg, #c4b5fd 0%, #93c5fd 100%);
			-webkit-background-clip: text;
			background-clip: text;
			color: transparent;
		}
		.ml-brand-subtitle {
			margin-top: 0.55rem;
			color: #a5b4cf;
			font-size: 0.8rem;
			line-height: 1.45;
		}
		.ml-topbar {
			margin: 0 0 1.25rem;
			padding: 0.75rem 1.2rem 1rem 1.2rem;
			border: 1px solid var(--ml-border);
			border-radius: 14px;
			background: var(--ml-panel);
			box-shadow: 0 6px 18px rgba(15, 23, 42, 0.02);
		}
		.ml-topbar-title {
			color: var(--ml-text);
			font-size: clamp(2rem, 2.6vw, 3rem);
			font-weight: 800;
			line-height: 1.1;
		}
		.ml-topbar-subtitle {
			margin-top: 0.2rem;
			color: var(--ml-muted);
			font-size: 0.95rem;
		}
		.ml-page-header {
			margin: 0.35rem 0 1.4rem;
			padding: 0.2rem 0;
		}
		.ml-page-header h1 {
			margin: 0;
			color: var(--ml-text);
			font-size: clamp(1.9rem, 2.5vw, 2.6rem);
			letter-spacing: -0.045em;
			line-height: 1.1;
		}
		.ml-page-header p {
			margin: 0.5rem 0 0;
			color: var(--ml-muted);
			font-size: 1rem;
		}
		.ml-eyebrow {
			color: var(--ml-primary);
			font-size: 0.72rem;
			font-weight: 800;
			letter-spacing: 0.12em;
			text-transform: uppercase;
			margin-bottom: 0.5rem;
		}
		.ml-card {
			background: var(--ml-panel);
			border: 1px solid var(--ml-border);
			border-radius: 14px;
			padding: 1rem;
			box-shadow: 0 8px 20px rgba(15, 23, 42, 0.04);
		}
		.ml-card h3 {
			margin: 0 0 0.25rem;
			color: var(--ml-text);
			font-size: 1rem;
		}
		.ml-card p {
			margin: 0;
			color: var(--ml-muted);
			font-size: 0.84rem;
		}
		.ml-status {
			display: inline-flex;
			align-items: center;
			gap: 0.35rem;
			border-radius: 999px;
			padding: 0.34rem 0.7rem;
			font-size: 0.73rem;
			font-weight: 700;
			border: 1px solid transparent;
		}
		.ml-status.success { background: light-dark(#dcfce7, #123522); color: light-dark(#166534, #86efac); border-color: rgba(22, 101, 52, 0.25); }
		.ml-status.warning { background: light-dark(#fef3c7, #3d2b12); color: light-dark(#92400e, #fcd34d); border-color: rgba(146, 64, 14, 0.3); }
		.ml-status.neutral { background: light-dark(#e2e8f0, #263244); color: var(--ml-muted); border-color: var(--ml-border); }
		.ml-app-tagline {
			color: var(--ml-muted);
			font-size: 0.86rem;
			font-weight: 600;
			text-align: right;
			letter-spacing: 0.02em;
		}
		.ml-empty-state {
			display: flex;
			align-items: center;
			gap: 0.9rem;
			padding: 1.1rem 1.15rem;
			border-radius: 14px;
			background: var(--ml-panel-soft);
			border: 1px solid rgba(37, 99, 235, 0.12);
			color: var(--ml-text);
		}
		.ml-empty-icon {
			font-size: 1.4rem;
			display: inline-flex;
			align-items: center;
			justify-content: center;
			width: 2.5rem;
			height: 2.5rem;
			border-radius: 12px;
			background: var(--ml-primary-soft);
		}
		.ml-empty-title {
			font-size: 1.05rem;
			font-weight: 700;
			margin-bottom: 0.15rem;
		}
		.ml-empty-message {
			color: var(--ml-muted);
			font-size: 0.92rem;
		}
		div[data-testid="stMetric"] {
			background: var(--ml-panel);
			border: 1px solid var(--ml-border);
			border-radius: 12px;
			padding: 0.8rem 0.9rem;
			box-shadow: 0 6px 18px rgba(15, 23, 42, 0.04);
		}
		div[data-testid="stExpander"] {
			border: 1px solid var(--ml-border);
			border-radius: 12px;
			background: var(--ml-panel);
			box-shadow: 0 6px 18px rgba(15, 23, 42, 0.02);
		}
		div[data-testid="stDataFrame"] {
			border-radius: 12px;
			overflow: hidden;
		}
		button[kind="primary"],
		[data-testid="stBaseButton-primary"] {
			background: linear-gradient(135deg, var(--ml-primary) 0%, var(--ml-primary-strong) 100%) !important;
			border: 1px solid rgba(96, 165, 250, 0.8) !important;
			color: #ffffff !important;
			box-shadow: 0 8px 16px rgba(37, 99, 235, 0.18);
		}
		button[kind="primary"]:hover,
		[data-testid="stBaseButton-primary"]:hover {
			background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%) !important;
			color: #ffffff !important;
		}
		button[kind="secondary"],
		[data-testid="stBaseButton-secondary"] {
			background: var(--ml-control) !important;
			border: 1px solid var(--ml-border) !important;
			color: var(--ml-text) !important;
		}
		button[kind="secondary"]:hover,
		[data-testid="stBaseButton-secondary"]:hover {
			background: var(--ml-panel-soft) !important;
			color: var(--ml-text) !important;
			border-color: rgba(37, 99, 235, 0.25) !important;
		}
		div[data-testid="stFileUploader"] {
			background: rgba(15, 23, 42, 0.04);
			border: 1px solid var(--ml-border);
			border-radius: 14px;
			padding: 0.2rem 0.15rem;
		}
		div[data-testid="stFileUploader"] section {
			background: var(--ml-control);
			border: 1px solid rgba(148, 163, 184, 0.25);
			border-radius: 12px;
			padding: 0.9rem 1rem;
		}
		div[data-testid="stFileUploader"] button,
		div[data-testid="stFileUploader"] [data-testid="baseButton-secondary"] {
			background: var(--ml-control) !important;
			color: var(--ml-text) !important;
			border: 1px solid rgba(96, 165, 250, 0.4) !important;
			font-weight: 600;
		}
		div[data-testid="stFileUploader"] button:hover,
		div[data-testid="stFileUploader"] [data-testid="baseButton-secondary"]:hover {
			background: var(--ml-primary-strong) !important;
			color: #ffffff !important;
		}
		div[data-testid="stFileUploader"] label,
		div[data-testid="stFileUploader"] .stFileUploaderLabel,
		div[data-testid="stFileUploader"] .stFileUploaderDropzone > div,
		label,
		.stTextInput > label,
		.stSelectbox > label,
		.stNumberInput > label,
		.stTextArea > label,
		.stRadio > label,
		.stCheckbox > label {
			color: var(--ml-text) !important;
			font-weight: 600 !important;
		}
		.stTextInput input,
		.stSelectbox div[data-baseweb="select"] > div,
		.stNumberInput input,
		.stTextArea textarea,
		div[data-baseweb="select"] {
			background: var(--ml-control) !important;
			color: var(--ml-text) !important;
			border: 1px solid var(--ml-border) !important;
			border-radius: 10px !important;
		}
		.stTextInput input::placeholder,
		.stTextArea textarea::placeholder,
		.stSelectbox input::placeholder {
			color: var(--ml-muted) !important;
		}
		[data-testid="stSidebar"] .stButton > button {
			background: rgba(15, 23, 42, 0.35) !important;
			color: #e2e8f0 !important;
		}
		[data-testid="stSidebar"] .stButton > button:hover,
		[data-testid="stSidebar"] .stButton > button[kind="primary"] {
			background: linear-gradient(135deg, var(--ml-primary) 0%, var(--ml-primary-strong) 100%) !important;
			color: #ffffff !important;
		}
		@media (max-width: 1024px) {
			.block-container {
				padding-left: 1rem;
				padding-right: 1rem;
			}
			[data-testid="stSidebar"] {
				width: min(300px, 100vw);
			}
		}
		@media (max-width: 768px) {
			.block-container {
				padding-top: 0.75rem;
				padding-left: 0.75rem;
				padding-right: 0.75rem;
			}
			.ml-brand-title {
				font-size: 1.7rem;
			}
			.ml-page-header h1 {
				font-size: 2rem;
			}
			.ml-app-tagline {
				display: none;
			}
			.ml-empty-state {
				display: block;
				padding: 0.9rem 1rem;
			}
			.ml-empty-icon {
				margin-bottom: 0.6rem;
				width: 2.1rem;
				height: 2.1rem;
			}
			[data-testid="stSidebar"] .stButton > button {
				padding: 0.6rem 0.75rem;
			}
		}
		</style>
		""",
		unsafe_allow_html=True,
	)


def navigate(page: str) -> None:
	"""Set the active page and rerun the app."""
	if page not in PAGE_LABELS:
		raise ValueError(f"Unknown page: {page}")
	st.session_state["current_page"] = page
	visited_pages = list(st.session_state.get("visited_pages", ["dataset"]))
	if page not in visited_pages:
		visited_pages.append(page)
	st.session_state["visited_pages"] = visited_pages
	st.rerun()


def render_top_header() -> None:
	"""Render the branded top header from the reference layout."""
	st.markdown(
		'''<div class="ml-topbar"><div style="display:flex; justify-content:space-between; align-items:flex-end; gap:1rem; flex-wrap:wrap;">'''
		+ f'''<div><div class="ml-topbar-title">{APP_NAME}</div><div class="ml-topbar-subtitle">{APP_SUBTITLE}</div></div>'''
		+ f'''<div class="ml-app-tagline">{APP_TAGLINE}</div></div></div>''',
		unsafe_allow_html=True,
	)


def render_page_header(title: str, subtitle: str, eyebrow: str | None = None) -> None:
	"""Render a consistent page title block."""
	eyebrow_markup = f'<div class="ml-eyebrow">{eyebrow}</div>' if eyebrow else ""
	st.markdown(
		f'<div class="ml-page-header">{eyebrow_markup}<h1>{title}</h1><p>{subtitle}</p></div>',
		unsafe_allow_html=True,
	)


def render_stat_cards(items: list[tuple[str, object]]) -> None:
	"""Render compact dashboard metrics."""
	columns = st.columns(min(4, max(1, len(items))))
	for column, (label, value) in zip(columns, items):
		column.metric(label, value, border=True)


def render_empty_state(title: str, message: str, icon: str = "info") -> None:
	"""Render a calm empty state for unavailable workflow stages."""
	icon_map = {
		"info": "ℹ️",
		"upload_file": "📁",
		"tune": "⚙️",
		"lock": "🔒",
		"model_training": "🧠",
		"analytics": "📊",
		"monitoring": "📈",
		"lightbulb": "💡",
		"assessment": "📋",
		"auto_awesome": "✨",
		"database": "🗂️",
	}
	icon_symbol = icon_map.get(icon, "ℹ️")
	st.markdown(
		f'''
		<div class="ml-empty-state">
			<div class="ml-empty-icon">{icon_symbol}</div>
			<div class="ml-empty-copy">
				<div class="ml-empty-title">{title}</div>
				<div class="ml-empty-message">{message}</div>
			</div>
		</div>
		''',
		unsafe_allow_html=True,
	)


def render_cta(label: str, page: str, key: str, disabled: bool = False) -> None:
	"""Render a full-width workflow CTA."""
	if st.button(label, key=key, type="primary", disabled=disabled, width="stretch"):
		navigate(page)


def render_status_badge(label: str, state: str = "neutral") -> None:
	"""Render a compact status badge."""
	st.markdown(f'<span class="ml-status {state}">{label}</span>', unsafe_allow_html=True)
