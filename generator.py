"""
generator.py
Synthetic bank transaction generator for the AML Transaction Screening & Case
Management project.

Produces a realistic, DELIBERATELY MESSY transaction dataset with a known
number of injected AML/fraud patterns (ground truth), so detection accuracy
(precision/recall) can be measured honestly against something we control.

Ground truth columns (is_fraud, pattern) are written to a separate
answer-key file (ground_truth.csv) -- they are NOT included in the raw file
an "analyst" would work from, because a real analyst never sees the answer
key up front.

Usage:
    python generator.py --accounts 400 --days 45 --seed 42
"""

import argparse
import csv
import random
import uuid
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Config / reference data
# ---------------------------------------------------------------------------

LOW_RISK_COUNTRIES = ["US", "CA", "UK", "DE", "FR", "AU", "JP", "SG", "NL", "IE"]
# Placeholder codes standing in for FATF-monitored / high-risk jurisdictions.
# In a real deployment this list would be sourced from an actual FATF
# grey/black list feed rather than hardcoded.
HIGH_RISK_COUNTRIES = ["XA", "XB", "XC"]

TRANSACTION_TYPES = ["cash_deposit", "cash_withdrawal", "ach", "wire", "card", "transfer"]
CHANNELS = ["branch", "online", "mobile", "atm", "phone"]

FIRST_NAMES = ["James", "Maria", "Wei", "Amara", "Liam", "Sofia", "Raj", "Elena",
               "Noah", "Fatima", "Lucas", "Aisha", "Mateo", "Yuki", "Omar", "Ivy"]
LAST_NAMES = ["Smith", "Garcia", "Chen", "Okafor", "Muller", "Rossi", "Patel",
              "Kim", "Novak", "Haddad", "Silva", "Tanaka", "Nguyen", "Cohen"]


def rand_name(rng):
    return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"


def new_id(prefix, n):
    return f"{prefix}{n:07d}"


# ---------------------------------------------------------------------------
# Accounts
# ---------------------------------------------------------------------------

def generate_accounts(rng, n_accounts):
    """Create the account population. A small % are pre-designated 'bad
    actor' accounts that will have fraud patterns injected, and a small %
    are 'legit lookalikes' -- high-volume legitimate accounts (e.g. small
    businesses) whose normal behavior can still trip a naive rule, so the
    detection engine has real false positives to contend with."""
    accounts = []
    n_bad = max(2, round(n_accounts * 0.04))
    n_lookalike = max(2, round(n_accounts * 0.025))
    bad_idx = set(rng.sample(range(n_accounts), n_bad))
    remaining = [i for i in range(n_accounts) if i not in bad_idx]
    lookalike_idx = set(rng.sample(remaining, n_lookalike))

    for i in range(n_accounts):
        role = "bad_actor" if i in bad_idx else ("lookalike" if i in lookalike_idx else "normal")
        accounts.append({
            "account_id": new_id("ACC", i + 1),
            "customer_id": new_id("CUST", i + 1),
            "customer_name": rand_name(rng),
            "home_country": rng.choice(LOW_RISK_COUNTRIES),
            "role": role,
            # normal daily transaction volume/amount profile per account
            "avg_amount": rng.uniform(40, 900) if role != "lookalike" else rng.uniform(2000, 9000),
            "daily_tx_rate": rng.uniform(0.1, 1.2) if role != "lookalike" else rng.uniform(1.5, 3.0),
        })
    return accounts


# ---------------------------------------------------------------------------
# Baseline "normal" activity
# ---------------------------------------------------------------------------

def generate_normal_transactions(rng, accounts, start_date, n_days, counter):
    rows = []
    for acct in accounts:
        n_tx = max(0, int(rng.gauss(acct["daily_tx_rate"] * n_days, 3)))
        for _ in range(n_tx):
            day_offset = rng.uniform(0, n_days)
            ts = start_date + timedelta(days=day_offset, hours=rng.uniform(0, 24))
            ttype = rng.choice(TRANSACTION_TYPES)
            amount = max(5, rng.gauss(acct["avg_amount"], acct["avg_amount"] * 0.35))
            counter[0] += 1
            rows.append({
                "transaction_id": new_id("TXN", counter[0]),
                "account_id": acct["account_id"],
                "customer_id": acct["customer_id"],
                "timestamp": ts,
                "amount": round(amount, 2),
                "currency": "USD",
                "transaction_type": ttype,
                "origin_country": acct["home_country"],
                "destination_country": acct["home_country"] if ttype != "wire" else rng.choice(LOW_RISK_COUNTRIES),
                "channel": rng.choice(CHANNELS),
                "memo": rng.choice(["", "payment", "transfer", "invoice pmt", "misc", "n/a"]),
                "is_fraud": False,
                "pattern": "none",
            })
    return rows


# ---------------------------------------------------------------------------
# Injected AML / fraud patterns (ground truth = True)
# ---------------------------------------------------------------------------

def inject_structuring(rng, accounts, start_date, n_days, counter):
    """Multiple cash deposits just under $10k reporting threshold, same
    account, within a 24-48h window, summing well above $10k."""
    rows = []
    bad_accounts = [a for a in accounts if a["role"] == "bad_actor"]
    for acct in rng.sample(bad_accounts, k=max(1, len(bad_accounts) // 2)):
        day_offset = rng.uniform(0, n_days - 2)
        base_ts = start_date + timedelta(days=day_offset)
        n_deposits = rng.randint(3, 5)
        for i in range(n_deposits):
            ts = base_ts + timedelta(hours=rng.uniform(0, 30))
            amount = rng.uniform(8200, 9850)
            counter[0] += 1
            rows.append({
                "transaction_id": new_id("TXN", counter[0]),
                "account_id": acct["account_id"],
                "customer_id": acct["customer_id"],
                "timestamp": ts,
                "amount": round(amount, 2),
                "currency": "USD",
                "transaction_type": "cash_deposit",
                "origin_country": acct["home_country"],
                "destination_country": acct["home_country"],
                "channel": rng.choice(["branch", "atm"]),
                "memo": "",
                "is_fraud": True,
                "pattern": "structuring",
            })
    return rows


def inject_velocity(rng, accounts, start_date, n_days, counter):
    """Unusually high number of transactions from one account in a short
    (~1 hour) window -- e.g. account takeover / mule draining activity."""
    rows = []
    bad_accounts = [a for a in accounts if a["role"] == "bad_actor"]
    for acct in rng.sample(bad_accounts, k=max(1, len(bad_accounts) // 2)):
        day_offset = rng.uniform(0, n_days - 1)
        base_ts = start_date + timedelta(days=day_offset)
        n_tx = rng.randint(7, 12)
        for i in range(n_tx):
            ts = base_ts + timedelta(minutes=rng.uniform(0, 55))
            amount = rng.uniform(100, 1500)
            counter[0] += 1
            rows.append({
                "transaction_id": new_id("TXN", counter[0]),
                "account_id": acct["account_id"],
                "customer_id": acct["customer_id"],
                "timestamp": ts,
                "amount": round(amount, 2),
                "currency": "USD",
                "transaction_type": rng.choice(["card", "transfer", "ach"]),
                "origin_country": acct["home_country"],
                "destination_country": acct["home_country"],
                "channel": rng.choice(["online", "mobile"]),
                "memo": "",
                "is_fraud": True,
                "pattern": "velocity",
            })
    return rows


def inject_high_risk_geo(rng, accounts, start_date, n_days, counter):
    """Wires to/from placeholder high-risk jurisdictions."""
    rows = []
    bad_accounts = [a for a in accounts if a["role"] == "bad_actor"]
    for acct in rng.sample(bad_accounts, k=max(1, len(bad_accounts) // 2)):
        n_tx = rng.randint(1, 3)
        for _ in range(n_tx):
            day_offset = rng.uniform(0, n_days)
            ts = start_date + timedelta(days=day_offset, hours=rng.uniform(0, 24))
            amount = rng.uniform(3000, 15000)
            counter[0] += 1
            rows.append({
                "transaction_id": new_id("TXN", counter[0]),
                "account_id": acct["account_id"],
                "customer_id": acct["customer_id"],
                "timestamp": ts,
                "amount": round(amount, 2),
                "currency": "USD",
                "transaction_type": "wire",
                "origin_country": acct["home_country"],
                "destination_country": rng.choice(HIGH_RISK_COUNTRIES),
                "channel": "online",
                "memo": "",
                "is_fraud": True,
                "pattern": "high_risk_geo",
            })
    return rows


def inject_layering(rng, accounts, start_date, n_days, counter):
    """Large deposit quickly followed by a near-equal outbound transfer --
    classic layering (money moved through the account, not kept in it)."""
    rows = []
    bad_accounts = [a for a in accounts if a["role"] == "bad_actor"]
    for acct in rng.sample(bad_accounts, k=max(1, len(bad_accounts) // 2)):
        day_offset = rng.uniform(0, n_days - 1)
        ts_in = start_date + timedelta(days=day_offset, hours=rng.uniform(0, 20))
        amount = rng.uniform(5000, 25000)
        ts_out = ts_in + timedelta(minutes=rng.uniform(15, 110))
        counter[0] += 1
        rows.append({
            "transaction_id": new_id("TXN", counter[0]), "account_id": acct["account_id"],
            "customer_id": acct["customer_id"], "timestamp": ts_in, "amount": round(amount, 2),
            "currency": "USD", "transaction_type": "wire", "origin_country": rng.choice(LOW_RISK_COUNTRIES),
            "destination_country": acct["home_country"], "channel": "online", "memo": "",
            "is_fraud": True, "pattern": "layering",
        })
        counter[0] += 1
        rows.append({
            "transaction_id": new_id("TXN", counter[0]), "account_id": acct["account_id"],
            "customer_id": acct["customer_id"], "timestamp": ts_out, "amount": round(amount * rng.uniform(0.96, 1.0), 2),
            "currency": "USD", "transaction_type": "transfer", "origin_country": acct["home_country"],
            "destination_country": rng.choice(LOW_RISK_COUNTRIES), "channel": "online", "memo": "",
            "is_fraud": True, "pattern": "layering",
        })
    return rows


def inject_lookalikes(rng, accounts, start_date, n_days, counter):
    """Legitimate high-volume accounts (small businesses) that do
    recurring round-dollar transfers -- NOT fraud, but designed to trip a
    naive round-dollar rule so precision/recall are meaningful rather than
    trivially perfect."""
    rows = []
    lookalikes = [a for a in accounts if a["role"] == "lookalike"]
    for acct in lookalikes:
        n_tx = rng.randint(4, 10)
        for _ in range(n_tx):
            day_offset = rng.uniform(0, n_days)
            ts = start_date + timedelta(days=day_offset, hours=rng.uniform(8, 18))
            amount = rng.choice([1000, 2000, 5000, 10000, 15000])
            counter[0] += 1
            rows.append({
                "transaction_id": new_id("TXN", counter[0]),
                "account_id": acct["account_id"],
                "customer_id": acct["customer_id"],
                "timestamp": ts,
                "amount": float(amount),
                "currency": "USD",
                "transaction_type": "wire",
                "origin_country": acct["home_country"],
                "destination_country": acct["home_country"],
                "channel": "online",
                "memo": "payroll",
                "is_fraud": False,
                "pattern": "lookalike_legit",
            })
    return rows


# ---------------------------------------------------------------------------
# Messiness: real bank exports are never this clean
# ---------------------------------------------------------------------------

def apply_messiness(rng, rows):
    rows = [dict(r) for r in rows]  # copy

    # 1. Duplicate ~1.5% of rows verbatim (same transaction_id) - simulates
    #    a batch file being re-ingested.
    n_dupe = int(len(rows) * 0.015)
    for r in rng.sample(rows, k=n_dupe):
        rows.append(dict(r))

    # 2. Null out customer_id on ~3% of rows
    for r in rng.sample(rows, k=int(len(rows) * 0.03)):
        r["customer_id"] = ""

    # 3. Inconsistent currency casing on ~5%
    for r in rng.sample(rows, k=int(len(rows) * 0.05)):
        r["currency"] = rng.choice(["usd", "Usd", "USD "])

    # 4. Whitespace padding on text fields ~4%
    for r in rng.sample(rows, k=int(len(rows) * 0.04)):
        r["memo"] = f"  {r['memo']}  "
        r["origin_country"] = f" {r['origin_country']}"

    # 5. Amount formatted as string with $ and commas on ~6%
    for r in rng.sample(rows, k=int(len(rows) * 0.06)):
        r["amount"] = f"${r['amount']:,.2f}"

    # 6. A handful of outlier fat-finger amounts (extra zero) ~0.5%
    for r in rng.sample(rows, k=max(1, int(len(rows) * 0.005))):
        try:
            r["amount"] = round(float(r["amount"]) * 10, 2)
        except (ValueError, TypeError):
            pass

    return rows


def format_timestamp_messy(rng, ts):
    """Return the timestamp as one of several inconsistent string formats."""
    fmt = rng.choice([
        "%Y-%m-%dT%H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%d-%m-%Y %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
    ])
    return ts.strftime(fmt)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accounts", type=int, default=400)
    ap.add_argument("--days", type=int, default=45)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="transactions_raw.csv")
    ap.add_argument("--ground-truth-out", type=str, default="ground_truth.csv")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    start_date = datetime(2026, 1, 1)
    counter = [0]

    accounts = generate_accounts(rng, args.accounts)

    rows = []
    rows += generate_normal_transactions(rng, accounts, start_date, args.days, counter)
    rows += inject_structuring(rng, accounts, start_date, args.days, counter)
    rows += inject_velocity(rng, accounts, start_date, args.days, counter)
    rows += inject_high_risk_geo(rng, accounts, start_date, args.days, counter)
    rows += inject_layering(rng, accounts, start_date, args.days, counter)
    rows += inject_lookalikes(rng, accounts, start_date, args.days, counter)

    rng.shuffle(rows)
    rows = apply_messiness(rng, rows)
    rng.shuffle(rows)

    # ground truth kept separate, keyed by transaction_id
    ground_truth_fields = ["transaction_id", "is_fraud", "pattern"]
    raw_fields = ["transaction_id", "account_id", "customer_id", "timestamp", "amount",
                  "currency", "transaction_type", "origin_country", "destination_country",
                  "channel", "memo"]

    with open(args.out, "w", newline="") as f_raw, open(args.ground_truth_out, "w", newline="") as f_gt:
        w_raw = csv.DictWriter(f_raw, fieldnames=raw_fields)
        w_gt = csv.DictWriter(f_gt, fieldnames=ground_truth_fields)
        w_raw.writeheader()
        w_gt.writeheader()
        for r in rows:
            raw_row = {k: r[k] for k in raw_fields if k != "timestamp"}
            raw_row["timestamp"] = format_timestamp_messy(rng, r["timestamp"])
            w_raw.writerow(raw_row)
            w_gt.writerow({k: r[k] for k in ground_truth_fields})

    n_fraud = sum(1 for r in rows if r["is_fraud"])
    print(f"Wrote {len(rows)} transactions ({n_fraud} true fraud/AML, "
          f"{len(rows) - n_fraud} legitimate) to {args.out}")
    print(f"Ground truth written to {args.ground_truth_out}")


if __name__ == "__main__":
    main()
