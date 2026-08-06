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
