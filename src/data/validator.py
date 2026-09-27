"""Dataset validation utilities."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.utils.exceptions import DatasetValidationError, MissingTargetError


def validate_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Validate a DataFrame before it is used in the ML workflow."""
    if df is None or not isinstance(df, pd.DataFrame):
        raise DatasetValidationError("The dataset is missing or invalid.")

    if df.empty or df.shape[0] == 0 or df.shape[1] == 0:
        raise DatasetValidationError("The uploaded dataset is empty.")

    duplicate_columns = df.columns[df.columns.duplicated()].tolist()
    if duplicate_columns:
        duplicate_columns = list(dict.fromkeys(duplicate_columns))
        raise DatasetValidationError(
            f"Duplicate column names detected: {duplicate_columns}. Please rename the columns and upload the dataset again."
        )

    all_missing_columns = [column for column in df.columns if df[column].isna().all()]
    return {
        "valid": True,
        "warnings": {
            "all_missing_columns": all_missing_columns,
        },
    }


def validate_target(target: pd.Series, problem_type: str | None = None) -> dict[str, Any]:
    """Validate a target column for the selected problem type."""
    if target is None or not isinstance(target, pd.Series):
        raise MissingTargetError("The selected target column is invalid.")

    clean_target = target.dropna()
    if clean_target.empty:
        raise MissingTargetError("The selected target column contains no usable values.")

    missing_count = int(target.isna().sum())
    warnings: list[str] = []

    if problem_type == "regression" and not pd.api.types.is_numeric_dtype(target.dtype):
        raise ValueError("Regression targets must be numeric. Select a numeric target or choose Classification.")

    if missing_count > 0:
        warnings.append(
            f"The selected target contains missing values. Missing target values: {missing_count}."
        )

    if problem_type == "classification" and clean_target.nunique() < 2:
        raise ValueError(
            "The selected target contains only one class. A classification model requires at least two classes."
        )

    return {
        "missing_count": missing_count,
        "n_unique": int(clean_target.nunique()),
        "warnings": warnings,
    }

