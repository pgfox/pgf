import pandas as pd

def add_zscore_outlier_flag(
    df: pd.DataFrame, col: str, z: float = 3.0
) -> pd.DataFrame:
    """Add zscore and outlier flag columns for a numeric column."""
    df = df.copy()
    mean = df[col].mean()
    std = df[col].std(ddof=0)
    df[f"{col}_z"] = (df[col] - mean) / std
    df[f"{col}_is_outlier"] = df[f"{col}_z"].abs() > z
    return df
