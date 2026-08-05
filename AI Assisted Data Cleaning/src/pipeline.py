"""
Pipeline stub: deterministic application of approved rules to a DataFrame.
Only simple actions implemented for demo: date standardization and province normalization.
"""
import pandas as pd
from dateutil import parser


def apply_standardize_date_iso(df: pd.DataFrame, column: str) -> pd.DataFrame:
    def to_iso(x):
        try:
            if pd.isna(x):
                return x
            return parser.parse(str(x)).date().isoformat()
        except Exception:
            return x
    df[column + '_before'] = df[column]
    df[column] = df[column].apply(to_iso)
    return df


def apply_normalize_province(df: pd.DataFrame, column: str) -> pd.DataFrame:
    mapping = {
        'ca': 'CA', 'california': 'CA', 'calif.': 'CA', 'ontario': 'ON', 'on': 'ON'
    }
    def norm(x):
        if pd.isna(x):
            return x
        k = str(x).strip().lower()
        return mapping.get(k, x)
    df[column + '_before'] = df[column]
    df[column] = df[column].apply(norm)
    return df


def apply_flag_missing(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df['_missing_' + column] = df[column].isna()
    return df
