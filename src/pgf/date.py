import pandas as pd

def add_date_time(
    df: pd.DataFrame, date_col: str, include_time: bool = True
) -> pd.DataFrame:
    """
    Add date and time related columns from a date string column.
    """
    
    df = df.copy()
    df['date'] = pd.to_datetime(df[date_col])

    df['year'] = df.date.dt.year
    df['month'] = df.date.dt.month
    df['day'] = df.date.dt.day
    df['quarter'] = df.date.dt.quarter
    df['dayofweek'] = df.date.dt.dayofweek  # Monday=0, Sunday=6
    df['is_month_start'] = df.date.dt.is_month_start
    df['is_month_end'] = df.date.dt.is_month_end
    if include_time:
        df['time'] = df.date.dt.time
        df['minutes'] = df.date.dt.minute
        df['hours'] = df.date.dt.hour


    return df
