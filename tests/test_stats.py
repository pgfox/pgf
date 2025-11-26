import pandas as pd

from pgf.stats import iqr_bounds


def test_iqr_bounds_returns_expected_cutoffs():
    series = pd.Series([1, 2, 3, 4, 5, 6])

    lower, upper = iqr_bounds(series)

    assert lower == -1.5
    assert upper == 8.5
