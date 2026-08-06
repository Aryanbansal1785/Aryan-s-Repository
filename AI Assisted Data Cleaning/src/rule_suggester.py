"""
Rule suggester: analyzes a profile dict (from profiler.profile_dataframe) and proposes
data-cleaning rules.

This is a deterministic, rule-based "AI" (no external model call yet) — it looks at
column stats and guesses what needs cleaning. It works generically on ANY DataFrame,
not just the built-in demo columns, so it can drive suggestions for uploaded datasets too.

Detection heuristics:
- Date-like columns: most sampled values parse as dates, but more than one distinct
  format shows up -> suggest standardizing to ISO 8601.
- Inconsistent categorical columns: values collapse to fewer distinct entries once you
  strip whitespace and lowercase them (e.g. "CA", "ca", "California ") -> suggest
  normalizing to a canonical form.
- Any column with missing values -> suggest flagging them for human review.
"""
from typing import List, Dict
import re
from dateutil import parser as dateparser

NUMERIC_DTYPE_PREFIXES = ("int", "float", "bool")


def _is_numeric_dtype(dtype_str: str) -> bool:
    return dtype_str.lower().startswith(NUMERIC_DTYPE_PREFIXES)


def _looks_like_date_column(examples: List[str]) -> bool:
    if len(examples) < 2:
        return False
    parsed = 0
    formats_seen = set()
    for val in examples:
        try:
            dateparser.parse(val, fuzzy=False)
            parsed += 1
            # crude format fingerprint: replace digit runs so "2023/12/01" and
            # "2023/01/05" collapse to the same fingerprint, but "12-02-2023" doesn't.
            fmt = re.sub(r"\d+", "#", val)
            formats_seen.add(fmt)
        except Exception:
            continue
    most_parse = parsed >= max(1, len(examples) - 1)
    multiple_formats = len(formats_seen) > 1
    return most_parse and multiple_formats


def _looks_like_inconsistent_categorical(examples: List[str]) -> bool:
    if len(examples) < 2:
        return False
    raw_unique = set(examples)
    normalized_unique = set(e.strip().lower() for e in examples)
    # same underlying values but different casing/whitespace collapse to fewer entries
    return len(normalized_unique) < len(raw_unique)


def suggest_rules_from_profile(profile: Dict) -> List[Dict]:
    """Return a list of proposed rule dicts based on the profile. Deterministic mock 'AI'."""
    rules = []
    cols = profile.get('columns', {})
    sample_rows = profile.get('sample_rows', [])
    counter = 0

    def next_id():
        nonlocal counter
        counter += 1
        return f"R{counter}"

    for col_name, stats in cols.items():
        examples = stats.get('examples', [])
        dtype = stats.get('dtype', '')

        if not _is_numeric_dtype(dtype) and _looks_like_date_column(examples):
            rules.append({
                'id': next_id(),
                'title': f'Standardize "{col_name}" to ISO 8601',
                'description': f'Column "{col_name}" contains multiple date formats; convert to YYYY-MM-DD',
                'column': col_name,
                'action': 'standardize_date_iso',
                'example_preview': sample_rows,
            })
        elif not _is_numeric_dtype(dtype) and _looks_like_inconsistent_categorical(examples):
            rules.append({
                'id': next_id(),
                'title': f'Normalize "{col_name}" values',
                'description': (
                    f'Column "{col_name}" has inconsistent casing/spacing for what look '
                    'like the same values; standardize to one canonical spelling per value'
                ),
                'column': col_name,
                'action': 'normalize_categorical',
                'example_preview': sample_rows,
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
                'example_preview': sample_rows,
            })

    return rules
