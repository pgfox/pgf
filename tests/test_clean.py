# tests/test_add_zscore_outlier_flag.py
import pandas as pd
import numpy as np
import pytest

from pgf.clean import add_zscore_outlier_flag

def test_adds_columns_and_preserves_input():
    original = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
    out = add_zscore_outlier_flag(original, "x", z=2.0)

    # original is not modified
    assert list(original.columns) == ["x"]

    # new columns exist
    assert "x_z" in out.columns
    assert "x_is_outlier" in out.columns
    assert out.shape[0] == original.shape[0]