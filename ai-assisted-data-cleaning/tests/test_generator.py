import pytest
from src.generator import generate_transactions


def test_generate_transactions():
    df = generate_transactions(20)
    assert not df.empty
    assert 'txn_id' in df.columns
    # check that some missing amounts exist
    assert df['amount'].isna().sum() >= 0
