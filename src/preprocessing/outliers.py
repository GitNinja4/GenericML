"""Leakage-safe numerical outlier transformation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class IQRTransformer(BaseEstimator, TransformerMixin):
	"""Clip or filter numerical rows using IQR bounds learned during ``fit``.

	Bounds are learned only from the data passed to ``fit``. For ``remove``,
	``get_support_mask`` exposes the rows retained by the latest transform so a
	future training workflow can keep features and targets aligned.
	"""

	def __init__(self, action: str = "clip", lower_multiplier: float = 1.5, upper_multiplier: float = 1.5) -> None:
		self.action = action
		self.lower_multiplier = lower_multiplier
		self.upper_multiplier = upper_multiplier

	def fit(self, X: pd.DataFrame | np.ndarray, y: object = None) -> "IQRTransformer":
		if self.action not in {"clip", "remove"}:
			raise ValueError("Outlier action must be 'clip' or 'remove'.")
		if self.lower_multiplier < 0 or self.upper_multiplier < 0:
			raise ValueError("IQR multipliers must be non-negative.")
		frame = self._to_frame(X)
		self.feature_names_in_ = frame.columns.to_numpy()
		q1 = frame.quantile(0.25)
		q3 = frame.quantile(0.75)
		iqr = q3 - q1
		self.lower_bounds_ = q1 - self.lower_multiplier * iqr
		self.upper_bounds_ = q3 + self.upper_multiplier * iqr
		self.n_features_in_ = frame.shape[1]
		return self

	def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
		frame = self._to_frame(X)
		self._check_fitted(frame)
		violations = frame.lt(self.lower_bounds_, axis="columns") | frame.gt(self.upper_bounds_, axis="columns")
		self.support_mask_ = ~violations.any(axis=1).to_numpy()
		if self.action == "remove":
			transformed = frame.loc[self.support_mask_]
		else:
			transformed = frame.clip(self.lower_bounds_, self.upper_bounds_, axis="columns")
		return transformed if isinstance(X, pd.DataFrame) else transformed.to_numpy()

	def fit_transform(self, X: pd.DataFrame | np.ndarray, y: object = None, **fit_params: object) -> pd.DataFrame | np.ndarray:
		return self.fit(X, y).transform(X)

	def get_support_mask(self) -> np.ndarray:
		"""Return the row mask from the latest transformation."""
		if not hasattr(self, "support_mask_"):
			raise RuntimeError("IQRTransformer must transform data before a support mask is available.")
		return self.support_mask_

	@staticmethod
	def _to_frame(X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
		if isinstance(X, pd.DataFrame):
			return X.copy()
		array = np.asarray(X)
		return pd.DataFrame(array, columns=[f"feature_{index}" for index in range(array.shape[1])])

	def _check_fitted(self, frame: pd.DataFrame) -> None:
		if not hasattr(self, "lower_bounds_"):
			raise RuntimeError("IQRTransformer must be fitted before transform.")
		if frame.shape[1] != self.n_features_in_:
			raise ValueError("Input feature count does not match fitted outlier transformer.")
