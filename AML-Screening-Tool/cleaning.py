"""
cleaning.py
Data validation & cleaning pipeline. Takes a raw, messy transaction export
(as produced by generator.py, or as a real bank export would look) and
produces a clean dataframe ready for the detection engine, plus a
data quality report describing exactly what was fixed or dropped.
"""

import re
import pandas as pd
from dateutil import parser as dateparser


AMOUNT_JUNK_RE = re.compile(r"[^0-9.\-]")


def _parse_amount(val):
    if pd.isna(val):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = AMOUNT_JUNK_RE.sub("", str(val))
    if s in ("", "-", "."):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_timestamp(val):
    if pd.isna(val) or str(val).strip() == "":
        return None
    try:
        return dateparser.parse(str(val))
    except (ValueError, OverflowError):
        return None


def clean_transactions(df: pd.DataFrame):
    """Returns (clean_df, report) where report is a dict of
    issue_type -> {count, detail} describing every fix/drop made."""
    report = {}
    df = df.copy()
    n_start = len(df)

    # --- 1. Exact duplicate transaction_id rows -> keep first occurrence ---
    dup_mask = df.duplicated(subset=["transaction_id"], keep="first")
    n_dupes = int(dup_mask.sum())
    if n_dupes:
        report["duplicate_transaction_id"] = {
            "count": n_dupes,
            "detail": "Exact duplicate transaction_id rows dropped (kept first occurrence).",
        }
    df = df[~dup_mask]

    # --- 2. Parse / normalize amount ---
    raw_amounts = df["amount"]
    parsed_amounts = raw_amounts.apply(_parse_amount)
    n_amount_reformatted = int((raw_amounts.astype(str) != parsed_amounts.astype(str)).sum())
    n_amount_unparseable = int(parsed_amounts.isna().sum())
    if n_amount_unparseable:
        report["amount_unparseable"] = {
            "count": n_amount_unparseable,
            "detail": "Amount field could not be parsed to a number; row dropped.",
        }
    df = df.assign(amount=parsed_amounts)
    df = df[df["amount"].notna()]

    # flag fat-finger outliers (>10x the account's own median amount) rather
    # than silently dropping them -- these get surfaced to the analyst, not
    # erased, since an outlier could itself be the fraud signal.
    acct_median = df.groupby("account_id")["amount"].transform("median")
    outlier_mask = (acct_median > 0) & (df["amount"] > acct_median * 8) & (df["amount"] > 5000)
    n_outliers = int(outlier_mask.sum())
    if n_outliers:
        report["amount_outlier_flagged"] = {
            "count": n_outliers,
            "detail": "Amount >8x the account's own median and >$5,000; kept but flagged for review, not auto-corrected.",
        }
    df["amount_outlier_flag"] = outlier_mask

    # --- 3. Parse / normalize timestamp (multiple inconsistent formats) ---
    raw_ts = df["timestamp"]
    parsed_ts = raw_ts.apply(_parse_timestamp)
    n_ts_bad = int(parsed_ts.isna().sum())
    if n_ts_bad:
        report["timestamp_unparseable"] = {
            "count": n_ts_bad,
            "detail": "Timestamp could not be parsed under any known format; row dropped.",
        }
    df = df.assign(timestamp=parsed_ts)
    df = df[df["timestamp"].notna()]
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    # --- 4. Normalize currency casing/whitespace ---
    n_currency_fixed = int((df["currency"].astype(str).str.strip().str.upper() != df["currency"].astype(str)).sum())
    if n_currency_fixed:
        report["currency_normalized"] = {
            "count": n_currency_fixed,
            "detail": "Currency code had inconsistent casing/whitespace (e.g. 'usd', 'Usd '); normalized to 'USD'.",
        }
    df["currency"] = df["currency"].astype(str).str.strip().str.upper()

    # --- 5. Trim whitespace on text fields ---
    text_cols = ["origin_country", "destination_country", "memo", "channel", "transaction_type"]
    n_whitespace = 0
    for c in text_cols:
        before = df[c].astype(str)
        after = before.str.strip()
        n_whitespace += int((before != after).sum())
        df[c] = after
    if n_whitespace:
        report["whitespace_trimmed"] = {
            "count": n_whitespace,
            "detail": "Leading/trailing whitespace stripped from text fields.",
        }

    # --- 6. Missing customer_id -> flagged, not dropped (transaction is
    #     still real and still needs to be screened) ---
    missing_cust = df["customer_id"].isna() | (df["customer_id"].astype(str).str.strip() == "")
    n_missing_cust = int(missing_cust.sum())
    if n_missing_cust:
        report["missing_customer_id"] = {
            "count": n_missing_cust,
            "detail": "customer_id missing; kept (transaction still screened) and flagged as a data-quality issue.",
        }
    df["customer_id"] = df["customer_id"].where(~missing_cust, None)
    df["missing_customer_flag"] = missing_cust

    n_end = len(df)
    report["_summary"] = {
        "rows_in": n_start,
        "rows_out": n_end,
        "rows_dropped": n_start - n_end,
    }

    return df.reset_index(drop=True), report


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "transactions_raw.csv"
    raw = pd.read_csv(path, dtype=str)
    clean, rpt = clean_transactions(raw)
    print(f"In: {rpt['_summary']['rows_in']}  Out: {rpt['_summary']['rows_out']}  Dropped: {rpt['_summary']['rows_dropped']}")
    for k, v in rpt.items():
        if k != "_summary":
            print(f"  {k}: {v['count']} -- {v['detail']}")
    clean.to_csv("transactions_clean.csv", index=False)
    print("Wrote transactions_clean.csv")
