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


def apply_impute_missing(df: pd.DataFrame, column: str, strategy: str, custom_value=None) -> pd.DataFrame:
    """Fill (or drop) missing values in `column` using a human-chosen strategy.

    strategy is one of: 'mean', 'median', 'mode', 'custom', 'drop_rows'.
    Only a human decides which strategy to use — nothing here is auto-selected.
    """
    df = df.copy()

    if strategy == 'drop_rows':
        return df.dropna(subset=[column]).reset_index(drop=True)

    if strategy == 'mean':
        fill_value = df[column].mean()
    elif strategy == 'median':
        fill_value = df[column].median()
    elif strategy == 'mode':
        mode_vals = df[column].mode(dropna=True)
        fill_value = mode_vals.iloc[0] if not mode_vals.empty else None
    elif strategy == 'custom':
        fill_value = custom_value
        if pd.api.types.is_numeric_dtype(df[column]):
            try:
                fill_value = float(custom_value)
            except (TypeError, ValueError):
                pass
    else:
        raise ValueError(f"Unknown imputation strategy: {strategy}")

    df[column + '_before'] = df[column]
    df[column] = df[column].fillna(fill_value)
    return df
