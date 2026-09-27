"""Dataset analysis utilities."""

from __future__ import annotations

import pandas as pd


def get_column_types(df: pd.DataFrame) -> dict[str, str]:
    """Classify each column as numeric, categorical, or boolean."""
    types: dict[str, str] = {}
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_bool_dtype(series):
            types[column] = "boolean"
        elif pd.api.types.is_numeric_dtype(series):
            types[column] = "numeric"
        else:
            types[column] = "categorical"
    return types


def detect_problem_type(target_series: pd.Series) -> str:
    """Infer a likely problem type for the selected target column."""
    if target_series is None or target_series.empty:
        raise ValueError("The selected target column is empty.")

    clean_target = target_series.dropna()
    if clean_target.empty:
        raise ValueError("The selected target column contains no usable values.")

    if pd.api.types.is_numeric_dtype(clean_target):
        unique_values = set(pd.unique(clean_target.to_numpy()))
        if unique_values.issubset({0, 1}):
            return "classification"
        unique_count = clean_target.nunique()
        values_are_integral = all(float(value).is_integer() for value in unique_values)
        if unique_count <= 10 and unique_count / len(clean_target) <= 0.5 and values_are_integral:
            return "classification"
        return "regression"

    if pd.api.types.is_bool_dtype(clean_target):
        return "classification"

    if clean_target.nunique() <= 10:
        return "classification"

    return "classification"


def summarize_dataset(df: pd.DataFrame) -> dict[str, object]:
    """Return dataset-level summary statistics for the dataset tab."""
    column_types = get_column_types(df)
    numeric_columns = [name for name, value in column_types.items() if value == "numeric"]
    categorical_columns = [name for name, value in column_types.items() if value == "categorical"]
    boolean_columns = [name for name, value in column_types.items() if value == "boolean"]

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "numerical_columns": len(numeric_columns),
        "categorical_columns": len(categorical_columns),
        "boolean_columns": len(boolean_columns),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "shape": df.shape,
    }


def get_column_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Build column metadata for the dataset table."""
    rows = []
    for column in df.columns:
        series = df[column]
        missing_count = int(series.isna().sum())
        missing_percent = (missing_count / len(df) * 100) if len(df) else 0.0
        rows.append(
            {
                "Column": column,
                "Data Type": str(series.dtype),
                "Missing Count": missing_count,
                "Missing %": round(missing_percent, 2),
                "Unique Count": int(series.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def summarize_target(target_series: pd.Series, problem_type: str) -> dict[str, object]:
    """Summarize a target series for display in the dataset tab."""
    clean_target = target_series.dropna()
    if problem_type == "classification":
        class_counts = clean_target.value_counts(dropna=False)
        return {
            "class_counts": class_counts,
            "missing_count": int(target_series.isna().sum()),
            "n_classes": int(clean_target.nunique()),
        }

    numeric_summary = clean_target.describe()
    return {
        "mean": float(numeric_summary.get("mean", 0.0)),
        "median": float(numeric_summary.get("50%", 0.0)),
        "min": float(numeric_summary.get("min", 0.0)),
        "max": float(numeric_summary.get("max", 0.0)),
        "missing_count": int(target_series.isna().sum()),
        "n_unique": int(clean_target.nunique()),
    }


def get_dataset_quality(df: pd.DataFrame, high_cardinality_threshold: float = 0.5) -> pd.DataFrame:
    """Return per-column quality metadata without changing ``df``."""
    rows = []
    row_count = len(df)
    for column in df.columns:
        series = df[column]
        unique_count = int(series.nunique(dropna=True))
        missing_count = int(series.isna().sum())
        high_cardinality = bool(row_count and unique_count / row_count >= high_cardinality_threshold)
        rows.append(
            {
                "Column": column,
                "Data Type": str(series.dtype),
                "Missing Count": missing_count,
                "Missing %": round(missing_count / row_count * 100, 2) if row_count else 0.0,
                "Unique Values": unique_count,
                "Constant": unique_count <= 1,
                "High Cardinality": high_cardinality,
            }
        )
    return pd.DataFrame(rows)


def get_missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing counts and percentages sorted from highest to lowest."""
    row_count = len(df)
    summary = pd.DataFrame(
        {
            "Column": df.columns,
            "Missing Count": df.isna().sum().astype(int).to_numpy(),
            "Missing %": (df.isna().sum() / row_count * 100 if row_count else 0).round(2).to_numpy(),
        }
    )
    return summary.sort_values(["Missing %", "Column"], ascending=[False, True]).reset_index(drop=True)


def get_duplicate_summary(df: pd.DataFrame) -> dict[str, float | int]:
    """Return duplicate row count and percentage."""
    duplicate_count = int(df.duplicated().sum())
    return {
        "count": duplicate_count,
        "percentage": round(duplicate_count / len(df) * 100, 2) if len(df) else 0.0,
    }


def get_numerical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for numerical columns."""
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return pd.DataFrame()
    summary = numeric.describe().T.rename(
        columns={"count": "Count", "mean": "Mean", "std": "Std Dev", "min": "Min", "25%": "25%", "50%": "Median", "75%": "75%", "max": "Max"}
    )
    return summary[["Count", "Mean", "Std Dev", "Min", "25%", "Median", "75%", "Max"]].reset_index(names="Feature")


def get_categorical_summary(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return counts and percentages for one categorical or boolean column."""
    if column not in df.columns:
        raise KeyError(f"Unknown column: {column}")
    values = df[column].astype("object").where(df[column].notna(), "<Missing>")
    counts = values.value_counts(dropna=False)
    result = counts.rename_axis("Category").reset_index(name="Count")
    result["Percentage"] = (result["Count"] / len(df) * 100 if len(df) else 0).round(2)
    return result


def detect_outliers(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Find potential numerical outliers using the 1.5 IQR rule."""
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    selected_columns = columns if columns is not None else numeric_columns
    rows = []
    for column in selected_columns:
        if column not in numeric_columns:
            continue
        values = df[column].dropna()
        if values.empty:
            rows.append({"Feature": column, "Q1": float("nan"), "Q3": float("nan"), "IQR": float("nan"), "Lower Bound": float("nan"), "Upper Bound": float("nan"), "Outlier Count": 0, "Outlier %": 0.0})
            continue
        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_count = int(((values < lower) | (values > upper)).sum())
        rows.append(
            {
                "Feature": column,
                "Q1": q1,
                "Q3": q3,
                "IQR": iqr,
                "Lower Bound": lower,
                "Upper Bound": upper,
                "Outlier Count": outlier_count,
                "Outlier %": round(outlier_count / len(values) * 100, 2),
            }
        )
    return pd.DataFrame(rows)


def get_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return Pearson correlations for numerical columns."""
    return df.select_dtypes(include="number").corr()


def detect_high_correlations(df: pd.DataFrame, threshold: float = 0.90) -> pd.DataFrame:
    """Return unique numerical feature pairs whose absolute correlation is high."""
    matrix = get_correlation_matrix(df)
    rows = []
    for index, first_column in enumerate(matrix.columns):
        for second_column in matrix.columns[index + 1 :]:
            correlation = matrix.loc[first_column, second_column]
            if pd.notna(correlation) and abs(correlation) >= threshold:
                rows.append({"Feature A": first_column, "Feature B": second_column, "Correlation": round(float(correlation), 4)})
    return pd.DataFrame(rows)


def detect_high_cardinality(df: pd.DataFrame, threshold: int = 50, proportion: float = 0.5) -> pd.DataFrame:
    """Return categorical columns with many distinct values."""
    rows = []
    for column in df.select_dtypes(include=["object", "category", "bool"]).columns:
        unique_count = int(df[column].nunique(dropna=True))
        if unique_count >= threshold or (len(df) and unique_count / len(df) >= proportion):
            rows.append({"Feature": column, "Unique Values": unique_count})
    return pd.DataFrame(rows)


def detect_constant_columns(df: pd.DataFrame) -> list[str]:
    """Return columns containing at most one non-missing value."""
    return [column for column in df.columns if df[column].nunique(dropna=True) <= 1]


def calculate_skewness(df: pd.DataFrame) -> pd.DataFrame:
    """Return skewness for each numerical feature."""
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return pd.DataFrame(columns=["Feature", "Skewness", "Strong Skewness"])
    result = numeric.skew().rename("Skewness").reset_index().rename(columns={"index": "Feature"})
    result["Strong Skewness"] = result["Skewness"].abs() > 1
    return result


def summarize_classification_target(target: pd.Series) -> dict[str, object]:
    """Return class counts, percentages, and missing values for a target."""
    counts = target.dropna().value_counts().rename_axis("Class").reset_index(name="Count")
    counts["Percentage"] = (counts["Count"] / len(target.dropna()) * 100 if target.dropna().size else 0).round(2)
    return {
        "column": target.name,
        "n_classes": int(target.nunique(dropna=True)),
        "missing_count": int(target.isna().sum()),
        "class_counts": counts,
        "imbalanced": bool(not counts.empty and counts["Percentage"].min() < 20),
    }


def summarize_regression_target(target: pd.Series) -> dict[str, float | int | str | None]:
    """Return descriptive statistics for a regression target."""
    values = target.dropna()
    return {
        "column": target.name,
        "count": int(values.size),
        "missing_count": int(target.isna().sum()),
        "mean": float(values.mean()) if not values.empty else float("nan"),
        "median": float(values.median()) if not values.empty else float("nan"),
        "std": float(values.std()) if not values.empty else float("nan"),
        "min": float(values.min()) if not values.empty else float("nan"),
        "max": float(values.max()) if not values.empty else float("nan"),
    }

