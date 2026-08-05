"""
Rule suggester stub: builds a prompt for the LLM and parses simple structured replies.
Actual LLM calls will be in a later stage. For now this module provides a deterministic
mock suggestion set for demo and tests.
"""
from typing import List, Dict


def suggest_rules_from_profile(profile: Dict) -> List[Dict]:
    """Return a list of proposed rule dicts based on the profile. Mock implementation."""
    rules = []
    # Example: if txn_date has multiple formats, suggest standardizing
    cols = profile.get('columns', {})
    if 'txn_date' in cols and cols['txn_date']['n_unique'] > 1:
        rules.append({
            'id': 'R1',
            'title': 'Standardize txn_date to ISO 8601',
            'description': 'Column txn_date contains multiple formats; convert to YYYY-MM-DD',
            'column': 'txn_date',
            'action': 'standardize_date_iso',
            'example_preview': profile.get('sample_rows', [])
        })
    if 'province' in cols:
        rules.append({
            'id': 'R2',
            'title': 'Normalize province names',
            'description': 'Standardize province/county/state values to a canonical form',
            'column': 'province',
            'action': 'normalize_province',
            'example_preview': profile.get('sample_rows', [])
        })
    # Missing amounts
    if 'amount' in cols and cols['amount']['n_missing'] > 0:
        rules.append({
            'id': 'R3',
            'title': 'Handle missing amounts',
            'description': 'Flag or impute missing amount values (human review recommended)',
            'column': 'amount',
            'action': 'flag_missing',
            'example_preview': profile.get('sample_rows', [])
        })
    return rules
