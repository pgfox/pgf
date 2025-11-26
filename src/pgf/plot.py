"""Plotting helpers."""

from __future__ import annotations

from statistics import NormalDist
from typing import Optional

import numpy as np
import pandas as pd

__all__ = ["qq_plot"]


def qq_plot(
    series: pd.Series, ax: Optional["matplotlib.axes.Axes"] = None, *, marker: str = "o"
):
    """
    Draw a Q-Q plot comparing a sample distribution to the standard normal.

    Parameters
    ----------
    series:
        Input observations; NA values are ignored.
    ax:
        Optional matplotlib axes to draw on. If omitted a new figure and axes are
        created.
    marker:
        Marker style passed to ``Axes.scatter`` when plotting sample quantiles.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot, which allows further customization by callers.
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError("matplotlib is required to use qq_plot") from exc

    clean = series.dropna().astype(float)
    if clean.empty:
        raise ValueError("qq_plot requires at least one non-null observation")

    data_quantiles = np.sort(clean.to_numpy())
    probs = (np.arange(1, len(data_quantiles) + 1) - 0.5) / len(data_quantiles)
    normal = NormalDist()
    theoretical_quantiles = np.array([normal.inv_cdf(p) for p in probs])

    if ax is None:
        _, ax = plt.subplots()

    ax.scatter(theoretical_quantiles, data_quantiles, marker=marker, label="Data")

    min_bound = min(theoretical_quantiles.min(), data_quantiles.min())
    max_bound = max(theoretical_quantiles.max(), data_quantiles.max())
    ax.plot(
        [min_bound, max_bound],
        [min_bound, max_bound],
        color="red",
        linestyle="--",
        label="Ideal normal",
    )
    ax.set_title("Normal Q-Q Plot")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    ax.legend()
    return ax

