import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import pandas as pd
from matplotlib import pyplot as plt

from pgf.plot import qq_plot


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
