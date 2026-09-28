"""Interactive exploratory data analysis tab."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data.analyzer import (
	calculate_skewness,
	detect_constant_columns,
	detect_high_cardinality,
	detect_high_correlations,
	detect_outliers,
	get_categorical_summary,
	get_correlation_matrix,
	get_dataset_quality,
	get_missing_value_summary,
	get_numerical_summary,
	get_duplicate_summary,
	summarize_classification_target,
	summarize_dataset,
	summarize_regression_target,
)
from src.visualization.eda_plots import (
	create_boxplot,
	create_categorical_target_chart,
	create_category_chart,
	create_correlation_heatmap,
	create_histogram,
	create_missing_values_chart,
	create_scatter_plot,
	create_target_distribution,
)
from ui.components import render_cta, render_empty_state, render_page_header, render_workflow_progress


def _render_eda_header(df: pd.DataFrame, target_column: str | None, problem_type: str | None) -> None:
	heading_column, dataset_column = st.columns([2.7, 1], vertical_alignment="center")
	with heading_column:
		render_page_header(
			"Exploratory data analysis",
			"Explore and understand your dataset through visualizations and statistics.",
			"02 · Understand",
		)
	with dataset_column:
		with st.container(border=True):
			st.caption("Dataset info")
			st.write(st.session_state.get("dataset_name") or "Current dataset")
			st.caption(f"{len(df):,} rows · {df.shape[1]:,} columns")
			if target_column and target_column in df.columns:
				problem_label = (problem_type or "unknown").title()
				st.caption(f"Target: {target_column} ({problem_label})")


def _render_plot(figure: go.Figure, key: str, height: int = 250) -> None:
	theme = getattr(st.context, "theme", None)
	is_dark = getattr(theme, "type", "light") == "dark"
	figure.update_layout(
		template="plotly_dark" if is_dark else "plotly_white",
		paper_bgcolor="rgba(0,0,0,0)",
		plot_bgcolor="rgba(0,0,0,0)",
		font={"color": "#f1f5f9" if is_dark else "#0f172a"},
		margin={"l": 16, "r": 16, "t": 42, "b": 20},
		height=height,
	)
	st.plotly_chart(figure, width="stretch", key=key)


def _render_metrics(df: pd.DataFrame) -> None:
	summary = summarize_dataset(df)
	with st.container(horizontal=True):
		for label, value in (
			("Rows", f"{summary['rows']:,}"),
			("Columns", summary["columns"]),
			("Numerical", summary["numerical_columns"]),
			("Categorical", summary["categorical_columns"] + summary["boolean_columns"]),
			("Missing values", f"{summary['missing_values']:,}"),
			("Duplicate rows", f"{summary['duplicate_rows']:,}"),
		):
			st.metric(label, value, border=True)


def _render_data_summary_table(df: pd.DataFrame) -> None:
	numeric_columns = df.select_dtypes(include="number").columns.tolist()
	categorical_columns = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
	total_columns = len(df.columns)
	type_summary = pd.DataFrame(
		[
			{"Type": "Numerical", "Count": len(numeric_columns), "Percentage": round(len(numeric_columns) / total_columns * 100, 1) if total_columns else 0},
			{"Type": "Categorical", "Count": len(categorical_columns), "Percentage": round(len(categorical_columns) / total_columns * 100, 1) if total_columns else 0},
			{"Type": "Total", "Count": total_columns, "Percentage": 100.0 if total_columns else 0},
		]
	)
	with st.container(border=True):
		st.markdown("#### Dataset summary")
		st.dataframe(type_summary, width="stretch", hide_index=True, height=165)


def _render_data_type_distribution(df: pd.DataFrame) -> None:
	numeric_columns = df.select_dtypes(include="number").columns.tolist()
	categorical_columns = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
	type_counts = pd.DataFrame(
		[
			{"Type": "Numerical", "Count": len(numeric_columns)},
			{"Type": "Categorical", "Count": len(categorical_columns)},
		]
	)
	with st.container(border=True):
		st.markdown("#### Data types distribution")
		if int(type_counts["Count"].sum()) == 0:
			st.info("No numerical or categorical features to chart.")
		else:
			figure = px.pie(
				type_counts[type_counts["Count"] > 0],
				values="Count",
				names="Type",
				hole=0.62,
				color_discrete_sequence=["#2563eb", "#f97316"],
			)
			figure.update_traces(textposition="inside", textinfo="percent")
			_render_plot(figure, "eda_overview_type_distribution", 205)


def _render_key_insights(df: pd.DataFrame, target_column: str | None, problem_type: str | None) -> None:
	with st.container(border=True):
		st.markdown("#### Key insights")
		_render_insights(df, target_column, problem_type)


def _render_feature_tables(df: pd.DataFrame) -> None:
	numerical_columns = df.select_dtypes(include="number").columns.tolist()
	categorical_columns = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
	numerical_column, categorical_column = st.columns(2)
	with numerical_column:
		with st.container(border=True):
			st.markdown("#### Numerical features")
			numerical = get_numerical_summary(df)
			if numerical.empty:
				st.caption("No numerical features in this dataset.")
			else:
				numerical.insert(1, "Data Type", numerical["Feature"].map(lambda column: str(df[column].dtype)))
				st.dataframe(
					numerical[["Feature", "Data Type", "Mean", "Std Dev", "Min", "Max"]].round(2),
					width="stretch",
					hide_index=True,
					height=260,
				)
	with categorical_column:
		with st.container(border=True):
			st.markdown("#### Categorical features")
			if not categorical_columns:
				st.caption("No categorical features in this dataset.")
				return
		rows = []
		for column in categorical_columns:
			counts = df[column].fillna("<Missing>").value_counts(dropna=False)
			rows.append(
				{
					"Feature": column,
					"Unique Values": int(df[column].nunique(dropna=True)),
					"Top Value": str(counts.index[0]) if not counts.empty else "",
					"Frequency": int(counts.iloc[0]) if not counts.empty else 0,
				}
			)
		st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, height=260)


def _render_overview_distributions(df: pd.DataFrame) -> None:
	numeric_columns = df.select_dtypes(include="number").columns.tolist()
	categorical_columns = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
	numeric_column, categorical_column, correlation_column = st.columns([1, 1, 1.15])
	with numeric_column:
		with st.container(border=True):
			st.markdown("#### Numerical feature distribution")
			if numeric_columns:
				selected = st.selectbox("Select feature", numeric_columns, key="eda_overview_numeric_feature")
				_render_plot(create_histogram(df, selected), "eda_overview_numeric_distribution", 250)
			else:
				st.caption("No numerical features available.")
	with categorical_column:
		with st.container(border=True):
			st.markdown("#### Categorical feature distribution")
			if categorical_columns:
				selected = st.selectbox("Select feature", categorical_columns, key="eda_overview_categorical_feature")
				category_summary = get_categorical_summary(df, selected)
				_render_plot(create_category_chart(category_summary, selected), "eda_overview_categorical_distribution", 250)
			else:
				st.caption("No categorical features available.")
	with correlation_column:
		with st.container(border=True):
			st.markdown("#### Correlation heatmap")
			correlation = get_correlation_matrix(df)
			if correlation.shape[1] >= 2:
				_render_plot(create_correlation_heatmap(correlation), "eda_overview_correlation", 285)
			else:
				st.caption("Correlation analysis requires at least two numerical columns.")


def _render_overview_checks(df: pd.DataFrame, target_column: str | None, problem_type: str | None) -> None:
	missing_column, outlier_column, target_column_panel = st.columns([1, 1, 1.15])
	with missing_column:
		with st.container(border=True):
			st.markdown("#### Missing values")
			missing = get_missing_value_summary(df)
			if int(missing["Missing Count"].sum()) == 0:
				st.success("No missing values found in the dataset.")
			else:
				st.dataframe(missing[missing["Missing Count"] > 0].head(6), width="stretch", hide_index=True, height=180)
	with outlier_column:
		with st.container(border=True):
			st.markdown("#### Outlier analysis")
			outliers = detect_outliers(df)
			if outliers.empty:
				st.caption("No numerical features available.")
			else:
				flagged = outliers[outliers["Outlier Count"] > 0]
				if flagged.empty:
					st.success("No potential outliers detected.")
				else:
					st.dataframe(flagged[["Feature", "Outlier Count", "Outlier %"]], width="stretch", hide_index=True, height=180)
	with target_column_panel:
		with st.container(border=True):
			st.markdown("#### Target analysis")
			if not target_column or target_column not in df.columns or not problem_type:
				st.caption("Choose a target on the Dataset tab to see its distribution.")
			else:
				st.caption(f"{target_column} · {problem_type.title()}")
				_render_plot(create_target_distribution(df[target_column], problem_type), "eda_overview_target_distribution", 205)


def _render_overview(df: pd.DataFrame, target_column: str | None, problem_type: str | None) -> None:
	_render_metrics(df)
	data_column, type_column, insights_column = st.columns([1, 0.9, 1.2])
	with data_column:
		_render_data_summary_table(df)
	with type_column:
		_render_data_type_distribution(df)
	with insights_column:
		_render_key_insights(df, target_column, problem_type)
	_render_feature_tables(df)
	_render_overview_distributions(df)
	_render_overview_checks(df, target_column, problem_type)


def _render_quality(df: pd.DataFrame) -> None:
	with st.container(border=True):
		st.markdown("#### Data quality by column")
		quality = get_dataset_quality(df)
		st.dataframe(quality, width="stretch", hide_index=True)
		high_missing = quality[quality["Missing %"] >= 20]["Column"].tolist()
		high_cardinality = quality[quality["High Cardinality"]]["Column"].tolist()
		constants = quality[quality["Constant"]]["Column"].tolist()
		if high_missing:
			st.warning(f"High missingness detected in: {', '.join(high_missing)}.")
		if high_cardinality:
			st.warning(f"High-cardinality columns detected in: {', '.join(high_cardinality)}.")
		if constants:
			st.warning(f"Constant columns detected in: {', '.join(constants)}.")
		if not high_missing and not high_cardinality and not constants:
			st.success("No high-missingness, high-cardinality, or constant columns detected.")


def _render_missing_values(df: pd.DataFrame) -> None:
	missing = get_missing_value_summary(df)
	with st.container(border=True):
		st.markdown("#### Missing values by column")
		if int(missing["Missing Count"].sum()) == 0:
			st.success("No missing values detected.")
		else:
			left, right = st.columns([1.1, 1])
			with left:
				st.dataframe(missing[missing["Missing Count"] > 0], width="stretch", hide_index=True)
			with right:
				_render_plot(create_missing_values_chart(missing), "eda_missing_values_chart", 290)

	duplicate_summary = get_duplicate_summary(df)
	quality_column, duplicate_column = st.columns([1.2, 1])
	with quality_column:
		_render_quality(df)
	with duplicate_column:
		with st.container(border=True):
			st.markdown("#### Duplicate rows")
			count_column, percentage_column = st.columns(2)
			count_column.metric("Duplicate rows", f"{duplicate_summary['count']:,}")
			percentage_column.metric("Duplicate percentage", f"{duplicate_summary['percentage']:.2f}%")
			if duplicate_summary["count"]:
				st.warning(f"{duplicate_summary['count']:,} duplicate rows detected.")
				st.dataframe(df[df.duplicated(keep=False)], width="stretch", hide_index=True)
			else:
				st.success("No duplicate rows detected.")


def _render_numerical_analysis(df: pd.DataFrame, target_column: str | None) -> None:
	numeric_columns = df.select_dtypes(include="number").columns.tolist()
	if not numeric_columns:
		st.info("No numerical columns available.")
		return
	with st.container(border=True):
		st.markdown("#### Numerical feature summary")
		st.dataframe(get_numerical_summary(df), width="stretch", hide_index=True)
		selected = st.selectbox("Select numerical feature", numeric_columns, key="eda_numeric_feature")
		left, right = st.columns(2)
		with left:
			_render_plot(create_histogram(df, selected), "eda_numeric_histogram", 320)
		with right:
			_render_plot(create_boxplot(df, selected), "eda_numeric_boxplot", 320)
		st.dataframe(calculate_skewness(df), width="stretch", hide_index=True)
		if target_column and target_column in numeric_columns and len(numeric_columns) > 1:
			feature_options = [column for column in numeric_columns if column != target_column]
			feature = st.selectbox("Select feature to compare with target", feature_options, key="eda_regression_feature")
			_render_plot(create_scatter_plot(df, feature, target_column), "eda_numeric_target_scatter", 320)


def _render_categorical_analysis(df: pd.DataFrame, target_column: str | None, problem_type: str | None) -> None:
	categorical_columns = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
	if not categorical_columns:
		st.info("No categorical columns available.")
		return
	with st.container(border=True):
		st.markdown("#### Categorical feature distribution")
		selected = st.selectbox("Select categorical feature", categorical_columns, key="eda_categorical_feature")
		top_n = st.number_input("Show top categories", min_value=5, max_value=50, value=10, step=5, key="eda_top_n")
		category_summary = get_categorical_summary(df, selected)
		left, right = st.columns([1, 1.25])
		with left:
			st.dataframe(category_summary, width="stretch", hide_index=True)
		with right:
			if len(category_summary) > top_n:
				st.caption(f"Chart displays the top {top_n} categories by count.")
			_render_plot(create_category_chart(category_summary, selected, int(top_n)), "eda_category_chart", 320)
		if problem_type == "classification" and target_column and target_column in df.columns and selected != target_column:
			_render_plot(create_categorical_target_chart(df, selected, target_column), "eda_categorical_target_chart", 320)


def _render_outliers(df: pd.DataFrame) -> None:
	outliers = detect_outliers(df)
	if outliers.empty:
		st.info("No numerical columns available.")
		return
	with st.container(border=True):
		st.markdown("#### Potential outliers")
		st.caption("Potential outliers are identified with the 1.5 × IQR rule. No values are removed or changed.")
		st.dataframe(outliers, width="stretch", hide_index=True)
		numeric_columns = df.select_dtypes(include="number").columns.tolist()
		selected = st.selectbox("Inspect feature distribution", numeric_columns, key="eda_outlier_feature")
		_render_plot(create_boxplot(df, selected), "eda_outlier_boxplot", 320)


def _render_correlations(df: pd.DataFrame) -> None:
	correlation = get_correlation_matrix(df)
	if correlation.shape[1] < 2:
		st.info("Correlation analysis requires at least two numerical columns.")
		return
	chart_column, findings_column = st.columns([1.5, 1])
	with chart_column:
		with st.container(border=True):
			st.markdown("#### Numerical correlation heatmap")
			_render_plot(create_correlation_heatmap(correlation), "eda_correlation_heatmap", 480)
	with findings_column:
		with st.container(border=True):
			st.markdown("#### Strong correlations")
			strong = detect_high_correlations(df)
			if strong.empty:
				st.success("No strong correlations detected at the 0.90 threshold.")
			else:
				st.warning("Strong numerical correlations detected.")
				st.dataframe(strong, width="stretch", hide_index=True)


def _render_target_analysis(df: pd.DataFrame, target_column: str | None, problem_type: str | None) -> None:
	if not target_column or target_column not in df.columns or not problem_type:
		st.info("Select and validate a target column on the Dataset tab first.")
		return
	target = df[target_column]
	with st.container(border=True):
		st.markdown(f"#### Target distribution · {target_column}")
		chart_column, summary_column = st.columns([1.25, 1])
		with chart_column:
			_render_plot(create_target_distribution(target, problem_type), "eda_target_distribution", 360)
		with summary_column:
			if problem_type == "classification":
				report = summarize_classification_target(target)
				st.caption(f"{report['n_classes']} classes · {report['missing_count']} missing values")
				st.dataframe(report["class_counts"], width="stretch", hide_index=True)
				if report["imbalanced"]:
					st.warning("Potential class imbalance detected. SMOTE can be considered during modeling.")
			else:
				report = summarize_regression_target(target)
				st.dataframe(pd.DataFrame([report]), width="stretch", hide_index=True)
				_render_plot(create_boxplot(df, target_column), "eda_regression_target_boxplot", 250)


def _render_insights(df: pd.DataFrame, target_column: str | None, problem_type: str | None) -> None:
	summary = summarize_dataset(df)
	insights = [f"Dataset has {summary['rows']:,} rows and {summary['columns']:,} columns."]
	missing_columns = get_missing_value_summary(df)
	missing_columns = missing_columns[missing_columns["Missing Count"] > 0]
	if missing_columns.empty:
		insights.append("No missing values detected.")
	else:
		insights.append(f"{len(missing_columns)} column(s) contain missing values.")
	constants = detect_constant_columns(df)
	if constants:
		insights.append(f"Constant columns detected: {', '.join(constants)}.")
	high_cardinality = detect_high_cardinality(df)
	if not high_cardinality.empty:
		insights.append(f"{len(high_cardinality)} high-cardinality categorical feature(s) detected.")
	strong = detect_high_correlations(df)
	if not strong.empty:
		insights.append(f"{len(strong)} strong numerical correlation(s) detected.")
	skewed = calculate_skewness(df)
	if not skewed.empty and skewed["Strong Skewness"].any():
		insights.append("Strong skewness detected in one or more numerical features.")
	if problem_type == "classification" and target_column in df.columns:
		report = summarize_classification_target(df[target_column])
		if report["imbalanced"]:
			insights.append("Potential class imbalance detected in the selected target.")
	for insight in insights:
		st.markdown(f"- {insight}")


def render_eda_tab() -> None:
	"""Render observational EDA for the currently loaded dataset."""
	render_workflow_progress("eda")
	dataset = st.session_state.get("dataset")
	if dataset is None:
		render_page_header(
			"Exploratory data analysis",
			"Explore and understand your dataset through visualizations and statistics.",
			"02 · Understand",
		)
		render_empty_state("No dataset available", "Upload a CSV dataset from the Dataset tab to begin EDA.", "upload_file")
		return
	target_column = st.session_state.get("target_column")
	problem_type = st.session_state.get("problem_type") or st.session_state.get("detected_problem_type")
	_render_eda_header(dataset, target_column, problem_type)
	tabs = st.tabs(["Overview", "Missing Values", "Distributions", "Correlations", "Outliers", "Target Analysis"])
	with tabs[0]:
		_render_overview(dataset, target_column, problem_type)
	with tabs[1]:
		_render_missing_values(dataset)
	with tabs[2]:
		numerical_column, categorical_column = st.columns(2)
		with numerical_column:
			_render_numerical_analysis(dataset, target_column)
		with categorical_column:
			_render_categorical_analysis(dataset, target_column, problem_type)
	with tabs[3]:
		_render_correlations(dataset)
	with tabs[4]:
		_render_outliers(dataset)
	with tabs[5]:
		_render_target_analysis(dataset, target_column, problem_type)
	render_cta("Continue to Train / Test Split →", "train_test_split", "eda_to_train_test_split")
