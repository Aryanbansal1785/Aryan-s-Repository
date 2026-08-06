"""
Utility helpers used across modules.
"""
from rapidfuzz import fuzz


def fuzzy_similarity(a, b):
    return fuzz.token_sort_ratio(str(a), str(b))
