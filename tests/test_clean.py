# tests/test_add_zscore_outlier_flag.py
import pandas as pd
import pytest

from pgf.clean import add_zscore_outlier_flag, fix_col_names, null_percentage

def test_adds_columns_and_preserves_input():
    original = pd.DataFrame({"x": [1, 2, 3, 20, 4, 5]})
    out = add_zscore_outlier_flag(original, "x", z=2.0)

    # original is not modified
    assert list(original.columns) == ["x"]

    # new columns exist
    assert "x_z" in out.columns
    assert "x_is_outlier" in out.columns
    assert out.shape[0] == original.shape[0]

    # ensure expected rows are flagged
    assert out.at[3, "x_is_outlier"]
    assert not out.at[2, "x_is_outlier"]


def test_fix_col_names_normalizes_and_preserves_input():
    original = pd.DataFrame({" Amount ": [1], "User-Id": [2], "Total$": [3]})
    cleaned = fix_col_names(original)

    assert list(cleaned.columns) == ["amount", "user_id", "total_"]
    # ensure we did not mutate the caller's frame
    assert list(original.columns) == [" Amount ", "User-Id", "Total$"]


def test_null_percentage_returns_percent_of_missing_values():
    df = pd.DataFrame({"a": [1, None, 3], "b": [None, None, 5]})

    percentages = null_percentage(df)

    assert percentages["a"] == pytest.approx(100 * (1 / 3))
    assert percentages["b"] == pytest.approx(100 * (2 / 3))


def test_null_percentage_handles_empty_dataframe():
    df = pd.DataFrame(columns=["a", "b"])

    percentages = null_percentage(df)

    assert percentages == {"a": 0.0, "b": 0.0}


def test_null_percentage_counts_empty_strings_by_default():
    df = pd.DataFrame({"a": ["", "value", "  "]})

    percentages = null_percentage(df)

    assert percentages["a"] == pytest.approx(100 * (2 / 3))


def test_null_percentage_can_skip_empty_strings():
    df = pd.DataFrame({"a": ["", "value", "  "]})

    percentages = null_percentage(df, include_empty_strings=False)

    assert percentages["a"] == pytest.approx(0.0)
