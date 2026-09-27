"""Missing-value transformer factories."""

from __future__ import annotations

from sklearn.impute import SimpleImputer


def create_numeric_imputer(strategy: str, fill_value: str | float | int = 0) -> SimpleImputer | None:
	"""Create the requested numerical imputer, or ``None`` for no imputation."""
	if strategy == "None":
		return None
	if strategy not in {"Mean", "Median", "Constant"}:
		raise ValueError(f"Unsupported numerical missing-value strategy: {strategy}")
	kwargs = {"strategy": strategy.lower()}
	if strategy == "Constant":
		kwargs["fill_value"] = fill_value
	return SimpleImputer(**kwargs)


def create_categorical_imputer(strategy: str, fill_value: str = "Missing") -> SimpleImputer | None:
	"""Create the requested categorical imputer, or ``None`` for no imputation."""
	if strategy == "None":
		return None
	if strategy not in {"Most Frequent", "Constant"}:
		raise ValueError(f"Unsupported categorical missing-value strategy: {strategy}")
	kwargs = {"strategy": "most_frequent" if strategy == "Most Frequent" else "constant"}
	if strategy == "Constant":
		kwargs["fill_value"] = fill_value
	return SimpleImputer(**kwargs)
