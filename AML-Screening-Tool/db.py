"""
db.py
SQLite persistence layer. Plain sqlite3 (stdlib) rather than an ORM, on
purpose: the whole point of this project is to demonstrate hand-written SQL
(CTEs, window functions) for the detection engine, and an ORM would hide
exactly the skill this project is meant to showcase.
"""

import sqlite3
from contextlib import contextmanager

DB_PATH = "aml_case_tool.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL,
    customer_id TEXT,
    ts TEXT NOT NULL,               -- ISO 'YYYY-MM-DD HH:MM:SS'
    amount REAL NOT NULL,
    currency TEXT,
    transaction_type TEXT,
    origin_country TEXT,
    destination_country TEXT,
    channel TEXT,
    memo TEXT,
    missing_customer_flag INTEGER DEFAULT 0,
    amount_outlier_flag INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_tx_account_ts ON transactions(account_id, ts);

CREATE TABLE IF NOT EXISTS data_quality_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT,
    issue_type TEXT,
    count INTEGER,
    detail TEXT,
    logged_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS flags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id TEXT NOT NULL,
    rule_name TEXT NOT NULL,
    rule_detail TEXT,
    risk_score INTEGER DEFAULT 1,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(transaction_id) REFERENCES transactions(transaction_id)
);

CREATE TABLE IF NOT EXISTS review_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id TEXT NOT NULL,
    decision TEXT NOT NULL,          -- 'cleared' | 'escalated' | 'confirmed_sar'
    reviewer TEXT,
    notes TEXT,
    decided_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(transaction_id) REFERENCES transactions(transaction_id)
);

-- Evaluation-only table. In a real deployment this would not exist; here it
-- holds the synthetic ground truth so the dashboard can report honest
-- precision/recall against known-injected patterns.
CREATE TABLE IF NOT EXISTS ground_truth (
    transaction_id TEXT PRIMARY KEY,
    is_fraud INTEGER,
    pattern TEXT
);
"""


@contextmanager
def get_conn(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path=DB_PATH):
    with get_conn(db_path) as conn:
        conn.executescript(SCHEMA)


def reset_db(db_path=DB_PATH):
    """Wipe transactional data but keep schema -- used when regenerating a
    fresh synthetic batch from the app."""
    with get_conn(db_path) as conn:
        conn.executescript("""
            DELETE FROM review_decisions;
            DELETE FROM flags;
            DELETE FROM data_quality_log;
            DELETE FROM ground_truth;
            DELETE FROM transactions;
        """)


def load_clean_transactions(df, db_path=DB_PATH):
    """df must have columns matching the transactions table (minus PK
    constraints handling -- upsert via INSERT OR REPLACE)."""
    cols = ["transaction_id", "account_id", "customer_id", "timestamp", "amount",
            "currency", "transaction_type", "origin_country", "destination_country",
            "channel", "memo", "missing_customer_flag", "amount_outlier_flag"]
    records = []
    for _, row in df.iterrows():
        records.append((
            row["transaction_id"], row["account_id"], row.get("customer_id"),
            row["timestamp"], float(row["amount"]), row.get("currency"),
            row.get("transaction_type"), row.get("origin_country"),
            row.get("destination_country"), row.get("channel"), row.get("memo"),
            int(bool(row.get("missing_customer_flag", False))),
            int(bool(row.get("amount_outlier_flag", False))),
        ))
    with get_conn(db_path) as conn:
        conn.executemany(f"""
            INSERT OR REPLACE INTO transactions
            (transaction_id, account_id, customer_id, ts, amount, currency,
             transaction_type, origin_country, destination_country, channel, memo,
             missing_customer_flag, amount_outlier_flag)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, records)


def load_data_quality_log(report, batch_id="batch-1", db_path=DB_PATH):
    rows = [(batch_id, k, v["count"], v["detail"]) for k, v in report.items() if k != "_summary"]
    with get_conn(db_path) as conn:
        conn.executemany(
            "INSERT INTO data_quality_log (batch_id, issue_type, count, detail) VALUES (?,?,?,?)",
            rows,
        )


def load_ground_truth(df, db_path=DB_PATH):
    records = [(row["transaction_id"], int(bool(row["is_fraud"])), row["pattern"]) for _, row in df.iterrows()]
    with get_conn(db_path) as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO ground_truth (transaction_id, is_fraud, pattern) VALUES (?,?,?)",
            records,
        )


def clear_flags(db_path=DB_PATH):
    with get_conn(db_path) as conn:
        conn.execute("DELETE FROM flags")


def insert_flags(flag_rows, db_path=DB_PATH):
    """flag_rows: list of (transaction_id, rule_name, rule_detail, risk_score)"""
    with get_conn(db_path) as conn:
        conn.executemany(
            "INSERT INTO flags (transaction_id, rule_name, rule_detail, risk_score) VALUES (?,?,?,?)",
            flag_rows,
        )


def record_decision(transaction_id, decision, reviewer, notes, db_path=DB_PATH):
    with get_conn(db_path) as conn:
        conn.execute(
            "INSERT INTO review_decisions (transaction_id, decision, reviewer, notes) VALUES (?,?,?,?)",
            (transaction_id, decision, reviewer, notes),
        )
