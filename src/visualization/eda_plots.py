"""Plotly figures used by the exploratory data analysis tab."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _empty_figure(message: str) -> go.Figure:
	figure = go.Figure()
	figure.add_annotation(text=message, x=0.5, y=0.5, showarrow=False)
	figure.update_xaxes(visible=False)
	figure.update_yaxes(visible=False)
	return figure


def create_missing_values_chart(summary: pd.DataFrame) -> go.Figure:
	"""Create a bar chart of columns containing missing values."""
	data = summary[summary["Missing Count"] > 0]
	if data.empty:
		return _empty_figure("No missing values detected")
	return px.bar(data, x="Column", y="Missing Count", text="Missing %", title="Missing values by column")


def create_histogram(df: pd.DataFrame, column: str) -> go.Figure:
	"""Create a numerical feature histogram."""
	if column not in df.columns or df[column].dropna().empty:
		return _empty_figure("No values available")
	return px.histogram(df, x=column, nbins=min(40, max(10, int(df[column].nunique()))), title=f"Distribution of {column}")


def create_boxplot(df: pd.DataFrame, column: str) -> go.Figure:
	"""Create a numerical feature box plot."""
	if column not in df.columns or df[column].dropna().empty:
		return _empty_figure("No values available")
	return px.box(df, y=column, points="outliers", title=f"Box plot of {column}")


def create_category_chart(summary: pd.DataFrame, column: str, top_n: int = 10) -> go.Figure:
	"""Create a bounded category frequency chart."""
	data = summary.head(max(1, top_n))
	if data.empty:
		return _empty_figure("No categories available")
	return px.bar(data, x="Category", y="Count", text="Percentage", title=f"Top categories in {column}")


def create_correlation_heatmap(correlation: pd.DataFrame) -> go.Figure:
	"""Create an interactive numerical correlation heatmap."""
	if correlation.shape[1] < 2:
		return _empty_figure("Correlation requires at least two numerical columns")
	return px.imshow(correlation, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r", title="Numerical correlation")


def create_target_distribution(target: pd.Series, problem_type: str) -> go.Figure:
	"""Create a class chart or numerical target histogram."""
	if problem_type == "classification":
		counts = target.dropna().value_counts().rename_axis("Class").reset_index(name="Count")
		if counts.empty:
			return _empty_figure("No target values available")
		return px.bar(counts, x="Class", y="Count", title=f"Class distribution: {target.name}")
	if target.dropna().empty:
		return _empty_figure("No target values available")
	return px.histogram(target, x=target.name, nbins=30, title=f"Target distribution: {target.name}")


def create_scatter_plot(df: pd.DataFrame, feature: str, target: str) -> go.Figure:
	"""Create a feature-versus-target scatter plot using complete rows only."""
	data = df[[feature, target]].dropna()
	if data.empty:
		return _empty_figure("No complete rows available")
	return px.scatter(data, x=feature, y=target, title=f"{feature} versus {target}")


def create_categorical_target_chart(df: pd.DataFrame, feature: str, target: str) -> go.Figure:
	"""Create normalized class percentages by categorical feature."""
	data = df[[feature, target]].dropna()
	if data.empty:
		return _empty_figure("No complete rows available")
	counts = data.groupby([feature, target], observed=False).size().rename("Count").reset_index()
	counts["Percentage"] = counts["Count"] / counts.groupby(feature, observed=False)["Count"].transform("sum") * 100
	return px.bar(counts, x=feature, y="Percentage", color=target, barmode="group", title=f"{feature} versus {target}")
