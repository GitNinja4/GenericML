"""Create raw train/test partitions before any preprocessing is fitted.

Preprocessing configuration can be created before splitting, but all preprocessing
transformers must be fitted only on the training data after the train/test split.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(
	X: pd.DataFrame,
	y: pd.Series,
	problem_type: str,
	test_size: float = 0.2,
	random_state: int = 42,
	stratified: bool | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
	"""Split raw features and target without fitting or applying preprocessing."""
	if not 0 < test_size < 1:
		raise ValueError("Test size must be greater than 0 and less than 1.")
	if not isinstance(random_state, int):
		raise ValueError("Random state must be an integer.")
	if X.empty or len(y) == 0:
		raise ValueError("The dataset must contain at least one row and one feature.")
	if len(X) != len(y):
		raise ValueError("Feature and target row counts must match.")
	if not X.index.equals(y.index):
		raise ValueError("Feature and target indices must match.")
	if problem_type not in {"classification", "regression"}:
		raise ValueError("Problem type must be classification or regression.")
	missing_targets = int(y.isna().sum())
	if missing_targets:
		raise ValueError(f"The target contains {missing_targets} missing value(s). Remove rows with missing targets before training.")
	if problem_type == "classification" and y.nunique(dropna=True) < 2:
		raise ValueError("Classification requires at least two target classes.")
	if problem_type == "regression" and not pd.api.types.is_numeric_dtype(y.dtype):
		raise ValueError("The regression target must be numeric.")
	use_stratification = problem_type == "classification" and stratified is not False
	stratify = y if use_stratification else None
	try:
		return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)
	except ValueError as exc:
		if problem_type == "classification":
			raise ValueError(f"Stratified classification split failed: {exc}") from exc
		raise ValueError(f"Train/test split failed: {exc}") from exc