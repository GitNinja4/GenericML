"""Dataset loading utilities."""

from __future__ import annotations

import csv
from io import TextIOBase, TextIOWrapper
from os import PathLike
from typing import Any

import pandas as pd


def _csv_header(file: Any) -> list[str]:
    """Read the first CSV record without pandas' duplicate-name mangling."""
    if isinstance(file, (str, PathLike)):
        with open(file, encoding="utf-8-sig", errors="replace", newline="") as source:
            return next(csv.reader(source), [])

    if hasattr(file, "seek"):
        file.seek(0)
    if isinstance(file, TextIOBase):
        header = next(csv.reader(file), [])
    else:
        text_file = TextIOWrapper(file, encoding="utf-8-sig", errors="replace", newline="")
        try:
            header = next(csv.reader(text_file), [])
        finally:
            text_file.detach()
    if hasattr(file, "seek"):
        file.seek(0)
    return header


def load_csv(file: Any) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame with basic error handling."""
    if file is None:
        raise ValueError("No CSV file was uploaded.")

    try:
        header = _csv_header(file)
        seen: set[str] = set()
        duplicates: list[str] = []
        for name in header:
            if name in seen and name not in duplicates:
                duplicates.append(name)
            seen.add(name)
        if duplicates:
            raise ValueError(f"Duplicate column names detected: {duplicates}. Please rename the columns and upload the dataset again.")
        if hasattr(file, "seek"):
            file.seek(0)
        df = pd.read_csv(file, encoding_errors="replace")
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The uploaded CSV file is empty or has no header.") from exc
    except ValueError:
        raise
    except Exception as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Unable to read the CSV file: {exc}") from exc

    if df.empty:
        raise ValueError("The uploaded dataset is empty.")

    if df.shape[1] == 0:
        raise ValueError("The uploaded dataset contains no columns.")

    return df

