"""Categorical encoder factories."""

from __future__ import annotations

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def create_encoder(strategy: str) -> OneHotEncoder | OrdinalEncoder | None:
	"""Create a safe categorical encoder for the selected strategy."""
	if strategy == "None":
		return None
	if strategy == "One-Hot Encoding":
		try:
			return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
		except TypeError:  # pragma: no cover - compatibility with older sklearn
			return OneHotEncoder(handle_unknown="ignore", sparse=False)
	if strategy == "Ordinal Encoding":
		return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
	raise ValueError(f"Unsupported categorical encoding strategy: {strategy}")
