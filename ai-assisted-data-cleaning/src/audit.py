"""
Audit helpers: record application of rules to the local SQLite audit_log table.
"""
import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path.cwd() / "ai_assisted_data_cleaning.db"


def log_rule_application(rule_id, rule_version, rows_affected, batch_id, path: Path = DB_PATH):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute('INSERT INTO audit_log (rule_id, rule_version, applied_at, rows_affected, batch_id) VALUES (?,?,?,?,?)',
                (rule_id, rule_version, datetime.utcnow().isoformat(), rows_affected, batch_id))
    conn.commit(); conn.close()
