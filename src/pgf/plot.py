"""Plotting helpers."""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Optional

import numpy as np
import pandas as pd

__all__ = [
    "qq_plot",
    "histogram_plot",
    "histogram_bin_counts",
    "cdf_plot",
    "box_plot",
]


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


def histogram_plot(
    series: pd.Series,
    bins: int,
    ax: Optional["matplotlib.axes.Axes"] = None,
    *,
    density: bool = False,
) -> "matplotlib.axes.Axes":
    """Draw and return a histogram for the given series and bin count."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError("matplotlib is required to use histogram_plot") from exc

    clean = series.dropna().astype(float)
    if clean.empty:
        raise ValueError("histogram_plot requires at least one non-null observation")

    try:
        bin_count = int(bins)
    except (TypeError, ValueError) as exc:
        raise TypeError("bins must be an integer") from exc

    if bin_count < 1:
        raise ValueError("bins must be greater than zero")

    if ax is None:
        _, ax = plt.subplots()

    ax.hist(clean.to_numpy(), bins=bin_count, density=density, edgecolor="black")
    ax.set_title("Histogram")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density" if density else "Frequency")
    return ax


def cdf_plot(
    series: pd.Series,
    ax: Optional["matplotlib.axes.Axes"] = None,
    *,
    label: Optional[str] = "Empirical CDF",
) -> "matplotlib.axes.Axes":
    """
    Draw the empirical cumulative distribution function for `series`.

    Parameters
    ----------
    label:
        Optional legend label. Defaults to the Series name or "Empirical CDF".
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError("matplotlib is required to use cdf_plot") from exc

    clean = series.dropna().astype(float)
    if clean.empty:
        raise ValueError("cdf_plot requires at least one non-null observation")

    values = np.sort(clean.to_numpy())
    cumulative = np.arange(1, len(values) + 1) / len(values)
    label_text = label 
    x_label = series.name if series.name else "Value"


    if ax is None:
        _, ax = plt.subplots()

    ax.step(values, cumulative, where="post", label=label_text)
    ax.set_ylim(0, 1.05)
    ax.set_title(label_text)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Cumulative Probability")
    ax.legend()
    return ax


def histogram_bin_counts(series: pd.Series) -> dict[str, int]:
    """Return recommended histogram bin counts from four common rules."""
    clean = series.dropna().astype(float)
    n = len(clean)
    if n == 0:
        raise ValueError("histogram_bin_counts requires at least one value")

    def _positive(value: float) -> int:
        return max(1, int(math.ceil(value)))

    counts: dict[str, int] = {
        "square_root": _positive(math.sqrt(n)),
        "sturges": _positive(1 + math.log2(n)),
    }

    data_range = clean.max() - clean.min()
    if data_range == 0:
        counts["scott"] = counts["freedman_diaconis"] = 1
        return counts

    std = float(clean.std(ddof=1))
    scott_width = 3.5 * std / (n ** (1 / 3))
    counts["scott"] = _positive(data_range / scott_width) if scott_width > 0 else 1

    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = float(q3 - q1)
    fd_width = 2 * iqr / (n ** (1 / 3))
    counts["freedman_diaconis"] = (
        _positive(data_range / fd_width) if fd_width > 0 else 1
    )
    return counts


def box_plot(
    df: pd.DataFrame,
    value_columns: Optional[list[str]] = None,
    *,
    segment_on: Optional[str] = None,
    ax: Optional["matplotlib.axes.Axes"] = None,
    vertical: bool = False,
) -> "matplotlib.axes.Axes":
    """
    Draw a box plot for numeric columns, optionally segmented by another column.

    Parameters
    ----------
    df:
        Input frame. All numeric columns (or the ones provided via
        ``value_columns``) are plotted.
    value_columns:
        Optional subset of numeric columns to visualize. Defaults to every
        numeric column.
    segment_on:
        Optional column name used to split the frame into multiple boxes. When
        omitted the plot shows one box for each numeric column. When set, the
        data are grouped by this column and each group yields a box composed of
        the selected value columns.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError("matplotlib is required to use box_plot") from exc

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if value_columns is not None:
        missing = [col for col in value_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Columns not in DataFrame: {missing}")
        numeric_cols = [col for col in value_columns if col in numeric_cols]

    if segment_on:
        if segment_on not in df.columns:
            raise ValueError(f"Column '{segment_on}' not found in DataFrame")
        numeric_cols = [col for col in numeric_cols if col != segment_on]

    if not numeric_cols:
        raise ValueError("box_plot requires at least one numeric column")

    data: list[np.ndarray] = []
    labels: list[str] = []

    if segment_on:
        for label, group in df.groupby(segment_on):
            flattened = group[numeric_cols].to_numpy().ravel()
            flattened = flattened[~np.isnan(flattened)]
            if flattened.size == 0:
                continue
            data.append(flattened)
            labels.append(str(label))
    else:
        for col in numeric_cols:
            column_data = df[col].dropna().to_numpy()
            if column_data.size == 0:
                continue
            data.append(column_data)
            labels.append(col)

    if not data:
        raise ValueError("box_plot requires non-empty numeric data to plot")

    if ax is None:
        _, ax = plt.subplots()

    orientation = "vertical" if vertical else "horizontal"
    ax.boxplot(data, orientation=orientation)
    if vertical:
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)
    else:
        ax.set_yticks(range(1, len(labels) + 1))
        ax.set_yticklabels(labels)
    title = "Box Plot"
    if segment_on:
        title += f" by {segment_on}"
    ax.set_title(title)
    ax.set_ylabel("Value")
    return ax
