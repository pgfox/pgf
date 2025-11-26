import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import pandas as pd
from matplotlib import pyplot as plt

from pgf.plot import histogram_bin_counts, qq_plot


def test_qq_plot_draws_points_and_reference_line():
    series = pd.Series([1, 2, 3, 4, 5])
    fig, ax = plt.subplots()
    try:
        returned = qq_plot(series, ax=ax)
        assert returned is ax
        # scatter adds a collection, reference line adds a line
        assert len(ax.collections) >= 1
        assert len(ax.lines) >= 1
    finally:
        plt.close(fig)


def test_qq_plot_rejects_empty_input():
    with pytest.raises(ValueError):
        qq_plot(pd.Series(dtype=float))


def test_histogram_bin_counts_matches_reference_values():
    series = pd.Series(range(1, 101))

    counts = histogram_bin_counts(series)

    assert counts == {
        "square_root": 10,
        "sturges": 8,
        "scott": 5,
        "freedman_diaconis": 5,
    }


def test_histogram_bin_counts_handles_degenerate_series_and_empty():
    single_value = pd.Series([5, 5, 5])
    assert histogram_bin_counts(single_value) == {
        "square_root": 2,
        "sturges": 3,
        "scott": 1,
        "freedman_diaconis": 1,
    }

    with pytest.raises(ValueError):
        histogram_bin_counts(pd.Series(dtype=float))
