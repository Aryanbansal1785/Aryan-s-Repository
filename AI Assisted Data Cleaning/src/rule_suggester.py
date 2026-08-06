from typing import List, Dict
from dateutil import parser as dateparser

NUMERIC_DTYPE_PREFIXES = ("int", "float", "bool")


def _is_numeric_dtype(dtype_str: str) -> bool:
    return dtype_str.lower().startswith(NUMERIC_DTYPE_PREFIXES)


def _looks_like_date_column(stats: Dict) -> bool:
    sample = stats.get('date_check_sample', [])
    if len(sample) < 2:
        return False
    parsed = 0
    for val in sample:
        try:
            dateparser.parse(val, fuzzy=False)
            parsed += 1
        except Exception:
            continue
    most_parse = parsed >= max(1, len(sample) - 1)
    multiple_formats = stats.get('n_distinct_formats', 0) > 1
    return most_parse and multiple_formats


def _looks_like_inconsistent_categorical(stats: Dict) -> bool:
    return stats.get('n_unique_normalized', 0) < stats.get('n_unique', 0)


def suggest_rules_from_profile(profile: Dict) -> List[Dict]:
    rules = []
    cols = profile.get('columns', {})
    sample_rows = profile.get('sample_rows', [])
    counter = 0

    def next_id():
        nonlocal counter
        counter += 1
        return f"R{counter}"

    for col_name, stats in cols.items():
        dtype = stats.get('dtype', '')
        if _is_numeric_dtype(dtype):
            pass
        elif _looks_like_date_column(stats):
            rules.append({
                'id': next_id(),
                'title': f'Standardize "{col_name}" to ISO 8601',
                'description': f'Column "{col_name}" contains multiple date formats; convert to YYYY-MM-DD',
                'column': col_name,
                'action': 'standardize_date_iso',
                'example_preview': stats.get('preview_odd_format') or sample_rows,
            })
        elif _looks_like_inconsistent_categorical(stats):
            dup_groups = stats.get('duplicate_value_groups', [])
            examples_txt = '; '.join(str(g) for g in dup_groups[:3])
            rules.append({
                'id': next_id(),
                'title': f'Normalize "{col_name}" values',
                'description': (
                    f'Column "{col_name}" has {stats["n_unique"] - stats["n_unique_normalized"]} '
                    f'value(s) that are duplicates once casing/spacing is ignored. '
                    + (f'Examples: {examples_txt}' if examples_txt else '')
                ),
                'column': col_name,
                'action': 'normalize_categorical',
                'example_preview': stats.get('preview_duplicates') or sample_rows,
            })

        if stats.get('n_missing', 0) > 0:
            rules.append({
                'id': next_id(),
                'title': f'Handle missing values in "{col_name}"',
                'description': (
                    f'Column "{col_name}" has {stats["n_missing"]} missing value(s); '
                    'flag rows for human review (no value is auto-imputed)'
                ),
                'column': col_name,
                'action': 'flag_missing',
                'example_preview': stats.get('preview_missing') or sample_rows,
            })

    return rules
