# tests/test_add_zscore_outlier_flag.py
import pandas as pd
import numpy as np
import pytest

from pgf.date import add_date_time

def test_add_date_time():
    '''
    Test add_date_time with include_time=True
    ''' 
    data = {
    'id': [1, 2, 3, 4, 5],
    'date_str': [
        '2025-01-15',
        '2025-03-22',
        '2025-06-10',
        '2025-09-05',
        '2025-12-19'
        ],
    }

    df = pd.DataFrame(data)
    out = add_date_time(df, "date_str")

    # new columns exist
    assert "date" in out.columns
    assert "month" in out.columns
    assert "year" in out.columns
    assert "day" in out.columns
    assert "quarter" in out.columns
    assert 'dayofweek' in out.columns
    assert 'minutes' in out.columns
    assert 'minutes' in out.columns

    out_row = out.iloc[[0]]

    assert out_row['year'].values[0] == 2025 
    assert out_row['month'].values[0] == 1      
    assert out_row['day'].values[0] == 15     
    assert out_row['quarter'].values[0] == 1
    
    

def test_add_date_no_time():
    '''
    Test add_date_time with include_time=False
    '''
    data = {
    'id': [1, 2, 3, 4, 5],
    'date_str': [
        '2025-01-15',
        '2025-03-22',
        '2025-06-10',
        '2025-09-05',
        '2025-12-19'
        ],
    }

    df = pd.DataFrame(data)
    out = add_date_time(df, "date_str", include_time=False)

    # new columns exist
    assert "date" in out.columns
    assert "month" in out.columns
    assert "year" in out.columns
    assert "day" in out.columns
    assert "quarter" in out.columns
    assert 'dayofweek' in out.columns
    assert not 'minutes' in out.columns
    assert not 'minutes' in out.columns