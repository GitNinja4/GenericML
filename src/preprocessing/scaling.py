"""Numerical feature scaling factories."""

from __future__ import annotations

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


def create_scaler(strategy: str) -> StandardScaler | MinMaxScaler | RobustScaler | None:
	"""Create the requested numerical scaler."""
	if strategy == "None":
		return None
	scalers = {
		"StandardScaler": StandardScaler,
		"MinMaxScaler": MinMaxScaler,
		"RobustScaler": RobustScaler,
	}
	if strategy not in scalers:
		raise ValueError(f"Unsupported scaling strategy: {strategy}")
	return scalers[strategy]()
