"""
Five rules, each a distinct AML typology:
  1. structuring     -- SUM()/COUNT() OVER, multiple deposits under the
                         reporting threshold summing above it in one day
  2. velocity         -- COUNT() OVER ... RANGE BETWEEN (trailing 1h window)
  3. round_dollar     -- large round-number wires/transfers (layering signal)
  4. high_risk_geo    -- counterparty in a high risk jurisdiction
  5. layering         -- LEAD() OVER, large deposit followed quickly by a
                         near equal outbound transfer
"""

import sqlite3
import db as dbmod

HIGH_RISK_COUNTRIES = ("XA", "XB", "XC")


def rule_structuring(conn):
    sql = """
    WITH deposit_only AS (
        SELECT transaction_id, account_id, ts, amount
        FROM transactions
        WHERE transaction_type = 'cash_deposit' AND amount < 10000
    ),
    daily AS (
        SELECT transaction_id, account_id, date(ts) AS tx_date, amount,
               SUM(amount) OVER (PARTITION BY account_id, date(ts)) AS daily_total,
               COUNT(*)    OVER (PARTITION BY account_id, date(ts)) AS daily_count
        FROM deposit_only
    )
    SELECT transaction_id, account_id, tx_date, daily_total, daily_count
    FROM daily
    WHERE daily_total >= 10000 AND daily_count >= 2
    """
    flags = []
    for row in conn.execute(sql):
        detail = (f"{row['daily_count']} cash deposits on {row['tx_date']} for account "
                  f"{row['account_id']} totaling ${row['daily_total']:,.2f} "
                  f"(each individually under the $10,000 reporting threshold)")
        flags.append((row["transaction_id"], "structuring", detail, 3))
    return flags


def rule_velocity(conn):
    sql = """
    WITH timed AS (
        SELECT transaction_id, account_id, ts, julianday(ts) AS jd
        FROM transactions
    ),
    windowed AS (
        SELECT transaction_id, account_id, ts,
               COUNT(*) OVER (
                   PARTITION BY account_id ORDER BY jd
                   RANGE BETWEEN (1.0/24.0) PRECEDING AND CURRENT ROW
               ) AS tx_count_1h
        FROM timed
    )
    SELECT transaction_id, account_id, ts, tx_count_1h
    FROM windowed
    WHERE tx_count_1h >= 6
    """
    flags = []
    for row in conn.execute(sql):
        detail = f"{row['tx_count_1h']} transactions from account {row['account_id']} within a trailing 1-hour window"
        flags.append((row["transaction_id"], "velocity", detail, 2))
    return flags


def rule_round_dollar(conn):
    sql = """
    SELECT transaction_id, account_id, amount, transaction_type
    FROM transactions
    WHERE transaction_type IN ('wire', 'transfer')
      AND amount >= 5000
      AND CAST(amount AS INTEGER) = amount
      AND CAST(amount AS INTEGER) % 1000 = 0
    """
    flags = []
    for row in conn.execute(sql):
        detail = f"Round-dollar {row['transaction_type']} of ${row['amount']:,.2f} (common layering signature)"
        flags.append((row["transaction_id"], "round_dollar", detail, 1))
    return flags


def rule_high_risk_geo(conn):
    sql = f"""
    SELECT transaction_id, account_id, origin_country, destination_country
    FROM transactions
    WHERE origin_country IN {HIGH_RISK_COUNTRIES}
       OR destination_country IN {HIGH_RISK_COUNTRIES}
    """
    flags = []
    for row in conn.execute(sql):
        detail = f"Counterparty jurisdiction {row['origin_country']} -> {row['destination_country']} includes a high-risk code"
        flags.append((row["transaction_id"], "high_risk_geo", detail, 3))
    return flags


def rule_layering(conn):
    sql = """
    WITH ordered AS (
        SELECT transaction_id, account_id, ts, amount, transaction_type,
               julianday(ts) AS jd,
               LEAD(transaction_id)      OVER (PARTITION BY account_id ORDER BY ts) AS next_txn_id,
               LEAD(transaction_type)    OVER (PARTITION BY account_id ORDER BY ts) AS next_type,
               LEAD(amount)              OVER (PARTITION BY account_id ORDER BY ts) AS next_amount,
               LEAD(julianday(ts))       OVER (PARTITION BY account_id ORDER BY ts) AS next_jd
        FROM transactions
        WHERE transaction_type IN ('wire', 'cash_deposit', 'transfer', 'ach', 'cash_withdrawal')
    )
    SELECT transaction_id, account_id, ts, amount, next_txn_id, next_amount, next_jd, jd
    FROM ordered
    WHERE transaction_type IN ('wire', 'cash_deposit')
      AND next_type IN ('transfer', 'wire', 'cash_withdrawal')
      AND (next_jd - jd) BETWEEN 0 AND (3.0/24.0)
      AND amount > 0
      AND ABS(next_amount - amount) / amount < 0.08
      AND amount >= 3000
    """
    flags = []
    for row in conn.execute(sql):
        detail = (f"${row['amount']:,.2f} in, matched by ${row['next_amount']:,.2f} out "
                  f"within {(row['next_jd'] - row['jd']) * 24:.1f}h (account {row['account_id']})")
        flags.append((row["transaction_id"], "layering", detail, 3))
        flags.append((row["next_txn_id"], "layering", detail, 3))
    return flags


RULES = [rule_structuring, rule_velocity, rule_round_dollar, rule_high_risk_geo, rule_layering]


def run_all_rules(db_path=dbmod.DB_PATH):
    dbmod.clear_flags(db_path)
    all_flags = []
    with dbmod.get_conn(db_path) as conn:
        for rule_fn in RULES:
            all_flags.extend(rule_fn(conn))
    dbmod.insert_flags(all_flags, db_path)
    return all_flags


def evaluate_against_ground_truth(db_path=dbmod.DB_PATH):
    """Returns precision/recall/f1 of (any-rule-flagged) vs ground truth,
    plus a per-rule breakdown. Ground truth only exists because this is a
    synthetic demo -- a real deployment would substitute confirmed SAR
    outcomes over time instead."""
    with dbmod.get_conn(db_path) as conn:
        flagged = {r["transaction_id"] for r in conn.execute("SELECT DISTINCT transaction_id FROM flags")}
        truth_rows = conn.execute("SELECT transaction_id, is_fraud FROM ground_truth").fetchall()
        truth = {r["transaction_id"]: bool(r["is_fraud"]) for r in truth_rows}

        tp = sum(1 for tid in flagged if truth.get(tid, False))
        fp = sum(1 for tid in flagged if not truth.get(tid, False))
        fn = sum(1 for tid, is_f in truth.items() if is_f and tid not in flagged)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        rule_breakdown = conn.execute("""
            SELECT rule_name, COUNT(DISTINCT transaction_id) AS n_flagged
            FROM flags GROUP BY rule_name ORDER BY n_flagged DESC
        """).fetchall()

    return {
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "rule_breakdown": [dict(r) for r in rule_breakdown],
    }


if __name__ == "__main__":
    flags = run_all_rules()
    print(f"Inserted {len(flags)} flags across {len(RULES)} rules")
    metrics = evaluate_against_ground_truth()
    print(f"Precision: {metrics['precision']:.3f}  Recall: {metrics['recall']:.3f}  F1: {metrics['f1']:.3f}")
    print(f"TP={metrics['true_positives']} FP={metrics['false_positives']} FN={metrics['false_negatives']}")
    for r in metrics["rule_breakdown"]:
        print(f"  {r['rule_name']}: {r['n_flagged']} flagged")
