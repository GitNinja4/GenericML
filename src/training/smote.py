"""Training-only imputation used before oversampling raw feature data."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TrainingDataImputer(BaseEstimator, TransformerMixin):
	"""Apply configured imputations while preserving raw feature columns."""

	def __init__(
		self,
		numeric_strategy: str = "None",
		numeric_constant: float = 0.0,
		categorical_strategy: str = "None",
		categorical_constant: str = "Missing",
	) -> None:
		self.numeric_strategy = numeric_strategy
		self.numeric_constant = numeric_constant
		self.categorical_strategy = categorical_strategy
		self.categorical_constant = categorical_constant

	def fit(self, X: pd.DataFrame, y: Any = None) -> "TrainingDataImputer":
		self.feature_names_in_ = list(X.columns)
		self.statistics_: dict[str, Any] = {}
		for column in self.feature_names_in_:
			values = X[column].dropna()
			is_categorical = (
				pd.api.types.is_bool_dtype(X[column].dtype)
				or pd.api.types.is_object_dtype(X[column].dtype)
				or isinstance(X[column].dtype, pd.CategoricalDtype)
				or pd.api.types.is_string_dtype(X[column].dtype)
			)
			if not is_categorical and pd.api.types.is_numeric_dtype(X[column].dtype):
				if self.numeric_strategy == "Mean" and not values.empty:
					self.statistics_[column] = float(values.mean())
				elif self.numeric_strategy == "Median" and not values.empty:
					self.statistics_[column] = float(values.median())
				elif self.numeric_strategy == "Constant":
					self.statistics_[column] = self.numeric_constant
			else:
				if self.categorical_strategy == "Most Frequent" and not values.empty:
					self.statistics_[column] = values.mode().iloc[0]
				elif self.categorical_strategy == "Constant":
					self.statistics_[column] = self.categorical_constant
		missing_without_strategy = [
			column for column in self.feature_names_in_
			if X[column].isna().any() and column not in self.statistics_
		]
		if missing_without_strategy:
			raise ValueError(
				"Oversampling requires complete features. Configure numeric/categorical imputation for: "
				f"{missing_without_strategy}"
			)
		return self

	def transform(self, X: pd.DataFrame) -> pd.DataFrame:
		transformed = X.loc[:, self.feature_names_in_].copy()
		for column, value in self.statistics_.items():
			if transformed[column].isna().any():
				transformed[column] = transformed[column].astype(object).where(transformed[column].notna(), value)
		return transformed
