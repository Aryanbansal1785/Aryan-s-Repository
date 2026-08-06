"""
Pipeline: deterministic application of approved rules to a DataFrame.

apply_standardize_date_iso and apply_flag_missing are fully generic — they work on
any column name. apply_normalize_province keeps the original hand-written mapping for
the demo's built-in province column (kept for backwards compatibility with existing
tests). apply_normalize_categorical is a generic replacement that works on ANY
categorical column: it groups values that are identical once you strip whitespace and
lowercase them, and rewrites each group to its most common original spelling.
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


def apply_normalize_categorical(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Generic categorical normalizer for any column on any dataset.

    Groups values that are the same after stripping whitespace and lowercasing,
    then maps every value in a group to that group's most frequent original spelling.
    E.g. ["CA", "ca", "California "] with "CA" appearing most often all become "CA".
    """
    df[column + '_before'] = df[column]

    def _key(x):
        if pd.isna(x):
            return None
        return str(x).strip().lower()

    non_null = df[column].dropna()
    if non_null.empty:
        return df

    canonical = {}
    for key, group in non_null.groupby(non_null.map(_key)):
        canonical[key] = group.value_counts().idxmax()

    def norm(x):
        if pd.isna(x):
            return x
        return canonical.get(_key(x), x)

    df[column] = df[column].apply(norm)
    return df


def apply_flag_missing(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df['_missing_' + column] = df[column].isna()
    return df
