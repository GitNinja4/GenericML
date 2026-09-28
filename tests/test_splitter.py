import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split as sklearn_train_test_split

import src.training.splitter as splitter
from src.training.splitter import split_dataset


def test_split_dataset_returns_aligned_partitions_with_expected_sizes() -> None:
    X = pd.DataFrame({"value": range(20)}, index=range(100, 120))
    y = pd.Series([index % 2 for index in range(20)], index=X.index, name="target")

    X_train, X_test, y_train, y_test = split_dataset(X, y, "classification", test_size=0.25)

    assert len(X_train) == len(y_train) == 15
    assert len(X_test) == len(y_test) == 5
    assert X_train.index.equals(y_train.index)
    assert X_test.index.equals(y_test.index)
    assert set(X_train.index).isdisjoint(X_test.index)
    assert set(X_train.index) | set(X_test.index) == set(X.index)


def test_split_dataset_is_reproducible() -> None:
    X = pd.DataFrame({"value": range(40)})
    y = pd.Series([index % 2 for index in range(40)])

    first = split_dataset(X, y, "classification", test_size=0.25, random_state=7)
    second = split_dataset(X, y, "classification", test_size=0.25, random_state=7)

    for first_part, second_part in zip(first, second, strict=True):
        pd.testing.assert_frame_equal(first_part, second_part) if isinstance(first_part, pd.DataFrame) else pd.testing.assert_series_equal(first_part, second_part)


def test_classification_split_uses_stratification(monkeypatch: pytest.MonkeyPatch) -> None:
    X = pd.DataFrame({"value": range(40)})
    y = pd.Series(["a"] * 30 + ["b"] * 10)
    original_train_test_split = splitter.train_test_split
    received: dict[str, object] = {}

    def capture_split(*args: object, **kwargs: object) -> tuple[object, ...]:
        received.update(kwargs)
        return original_train_test_split(*args, **kwargs)

    monkeypatch.setattr(splitter, "train_test_split", capture_split)
    _, X_test, _, y_test = split_dataset(X, y, "classification", test_size=0.25, random_state=3)

    assert received["stratify"] is y
    assert len(X_test) == 10
    expected_proportions = y.value_counts(normalize=True)
    test_proportions = y_test.value_counts(normalize=True)
    for label, expected_proportion in expected_proportions.items():
        assert abs(test_proportions[label] - expected_proportion) <= 1 / len(y_test)


def test_regression_split_does_not_stratify(monkeypatch: pytest.MonkeyPatch) -> None:
    X = pd.DataFrame({"value": range(20)})
    y = pd.Series(np.linspace(0.0, 1.0, 20))
    original_train_test_split = splitter.train_test_split
    received: dict[str, object] = {}

    def capture_split(*args: object, **kwargs: object) -> tuple[object, ...]:
        received.update(kwargs)
        return original_train_test_split(*args, **kwargs)

    monkeypatch.setattr(splitter, "train_test_split", capture_split)
    split_dataset(X, y, "regression")

    assert received["stratify"] is None


def test_classification_can_disable_stratification(monkeypatch: pytest.MonkeyPatch) -> None:
    X = pd.DataFrame({"value": range(40)})
    y = pd.Series(["a"] * 30 + ["b"] * 10)
    original_train_test_split = splitter.train_test_split
    received: dict[str, object] = {}

    def capture_split(*args: object, **kwargs: object) -> tuple[object, ...]:
        received.update(kwargs)
        return original_train_test_split(*args, **kwargs)

    monkeypatch.setattr(splitter, "train_test_split", capture_split)
    split_dataset(X, y, "classification", stratified=False)

    assert received["stratify"] is None


def test_split_dataset_does_not_modify_inputs_or_preprocess_features() -> None:
    X = pd.DataFrame({"value": [1.0, np.nan, 3.0, 1000.0, 5.0, 6.0]})
    y = pd.Series([0, 1, 0, 1, 0, 1], name="target")
    original_X = X.copy(deep=True)
    original_y = y.copy(deep=True)
    expected = sklearn_train_test_split(X, y, test_size=0.33, random_state=2, stratify=y)

    actual = split_dataset(X, y, "classification", test_size=0.33, random_state=2)

    pd.testing.assert_frame_equal(X, original_X)
    pd.testing.assert_series_equal(y, original_y)
    for actual_part, expected_part in zip(actual, expected, strict=True):
        if isinstance(expected_part, pd.DataFrame):
            pd.testing.assert_frame_equal(actual_part, expected_part)
        else:
            pd.testing.assert_series_equal(actual_part, expected_part)