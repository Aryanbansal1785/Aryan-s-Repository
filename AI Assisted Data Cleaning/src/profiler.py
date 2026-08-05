"""
Profiler stub: examines a DataFrame and returns a lightweight profile suitable for
building LLM prompts. This is intentionally small — the LLM integration will live
here later and supports MOCK mode.
"""
import pandas as pd


def profile_dataframe(df: pd.DataFrame, max_rows=5):
    """Return a dict with column summaries and example rows."""
    profile = {}
    profile['row_count'] = len(df)
    profile['columns'] = {}
    for col in df.columns:
        col_ser = df[col]
        profile['columns'][col] = {
            'dtype': str(col_ser.dtype),
            'n_missing': int(col_ser.isna().sum()),
            'n_unique': int(col_ser.nunique(dropna=True)),
            'examples': col_ser.dropna().astype(str).head(max_rows).tolist()
        }
    profile['sample_rows'] = df.head(max_rows).to_dict(orient='records')
    return profile
