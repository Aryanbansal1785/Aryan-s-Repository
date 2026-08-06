import pandas as pd
from src.pipeline import apply_standardize_date_iso, apply_normalize_province


def test_pipeline_date_and_province():
    df = pd.DataFrame({
        'txn_date': ['2023/12/01', '12-02-2023', '03 Dec 2023'],
        'province': ['CA', 'California', 'ca']
    })
    df2 = apply_standardize_date_iso(df.copy(), 'txn_date')
    assert 'txn_date_before' in df2.columns
    df3 = apply_normalize_province(df2, 'province')
    assert df3['province'].isin(['CA','ON','ca']).any()
