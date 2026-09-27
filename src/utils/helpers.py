"""Generic helper utilities used by the app."""

from __future__ import annotations

from typing import Any


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float or return a safe fallback."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
