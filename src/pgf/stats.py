"""Statistical helper utilities."""

from __future__ import annotations

import pandas as pd

__all__: list[str] = ["iqr_bounds"]


def iqr_bounds(series: pd.Series, k: float = 1.5) -> tuple[float, float]:
    """Return lower and upper outlier cutoffs using the IQR method."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return lower, upper
