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

def fix_col_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with trimmed, snake-cased, lowercase column names."""
    new_df = df.copy()
    new_df.columns = (
        new_df.columns.str.strip().str.replace(r"\W+", "_", regex=True).str.lower()
    )
    return new_df


def null_percentage(
    df: pd.DataFrame, *, include_empty_strings: bool = True
) -> dict[str, float]:
    """Return a dictionary with the percentage of null values per column."""
    if df.empty:
        return {col: 0.0 for col in df.columns}

    total = len(df)
    percentages: dict[str, float] = {}
    for col in df.columns:
        series = df[col].copy()
        if include_empty_strings:
            str_mask = series.apply(lambda value: isinstance(value, str))
            if str_mask.any():
                stripped = series[str_mask].str.strip()
                series.loc[str_mask] = stripped
                series.loc[str_mask & stripped.eq("")] = pd.NA

        nulls = series.isna().sum()
        percentages[col] = (nulls / total) * 100
    return percentages
