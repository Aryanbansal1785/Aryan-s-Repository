import pandas as pd


def profile_dataframe(df: pd.DataFrame, max_rows=5):
    """Return a dict with column summaries and RULE-SPECIFIC row previews.

    Each potential issue (missing values, duplicate-casing values, odd date
    formats) gets its own preview built from the rows that ACTUALLY exhibit
    that issue -- computed over the full column, not a sample -- so the
    "preview affected rows" shown to a reviewer always matches what a rule
    would really touch, no matter how large the file is or where the issue
    happens to live in it.
    """
    profile = {}
    profile['row_count'] = len(df)
    profile['columns'] = {}
    for col in df.columns:
        col_ser = df[col]
        non_null = col_ser.dropna().astype(str)

        n_unique_raw = int(non_null.nunique())
        n_unique_normalized = int(non_null.str.strip().str.lower().nunique())

        fingerprints = non_null.str.replace(r'\d+', '#', regex=True)
        n_distinct_formats = int(fingerprints.nunique()) if not non_null.empty else 0

        sample_for_date_check = non_null.head(20).tolist()

        dup_examples = []
        dup_values = set()
        if n_unique_normalized < n_unique_raw:
            norm_to_raw = {}
            for v in non_null.unique():
                key = v.strip().lower()
                norm_to_raw.setdefault(key, set()).add(v)
            for vs in norm_to_raw.values():
                if len(vs) > 1:
                    dup_examples.append(sorted(vs))
                    dup_values.update(vs)
        dup_examples = dup_examples[:5]

        missing_mask = col_ser.isna()
        preview_missing = df[missing_mask].head(max_rows).to_dict(orient='records')

        preview_duplicates = []
        if dup_values:
            dup_mask = col_ser.astype(str).isin(dup_values) & ~missing_mask
            preview_duplicates = df[dup_mask].head(max_rows).to_dict(orient='records')

        preview_odd_format = []
        if n_distinct_formats > 1 and not fingerprints.empty:
            modal_fp = fingerprints.mode().iloc[0]
            odd_index = non_null[fingerprints != modal_fp].index
            preview_odd_format = df.loc[odd_index].head(max_rows).to_dict(orient='records')

        profile['columns'][col] = {
            'dtype': str(col_ser.dtype),
            'n_missing': int(col_ser.isna().sum()),
            'n_unique': n_unique_raw,
            'n_unique_normalized': n_unique_normalized,
            'n_distinct_formats': n_distinct_formats,
            'examples': non_null.head(max_rows).tolist(),
            'date_check_sample': sample_for_date_check,
            'duplicate_value_groups': dup_examples,
            'preview_missing': preview_missing,
            'preview_duplicates': preview_duplicates,
            'preview_odd_format': preview_odd_format,
        }
    profile['sample_rows'] = df.head(max_rows).to_dict(orient='records')
    return profile
