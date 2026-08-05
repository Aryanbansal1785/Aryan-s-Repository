"""
rules_store: simple SQLite-backed store for approved rules and versions.
"""
import sqlite3
from pathlib import Path
import json
from datetime import datetime

DB_PATH = Path.cwd() / "ai_assisted_data_cleaning.db"


def init_db(path: Path = DB_PATH):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute('''
    CREATE TABLE IF NOT EXISTS approved_rules (
        id TEXT PRIMARY KEY,
        version INTEGER,
        rule_json TEXT,
        approved_by TEXT,
        approved_at TEXT
    )
    ''')
    cur.execute('''
    CREATE TABLE IF NOT EXISTS audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        rule_id TEXT,
        rule_version INTEGER,
        applied_at TEXT,
        rows_affected INTEGER,
        batch_id TEXT
    )
    ''')
    conn.commit(); conn.close()


def save_approved_rule(rule_id: str, rule_dict: dict, approved_by: str, path: Path = DB_PATH):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    # find next version
    cur.execute('SELECT MAX(version) FROM approved_rules WHERE id = ?', (rule_id,))
    row = cur.fetchone()
    next_version = 1 if row is None or row[0] is None else row[0] + 1
    cur.execute('INSERT OR REPLACE INTO approved_rules (id, version, rule_json, approved_by, approved_at) VALUES (?,?,?,?,?)',
                (rule_id, next_version, json.dumps(rule_dict), approved_by, datetime.utcnow().isoformat()))
    conn.commit(); conn.close()
    return next_version


def list_approved_rules(path: Path = DB_PATH):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute('SELECT id, version, rule_json, approved_by, approved_at FROM approved_rules')
    rows = cur.fetchall(); conn.close()
    result = []
    for r in rows:
        result.append({'id': r[0], 'version': r[1], 'rule': json.loads(r[2]), 'approved_by': r[3], 'approved_at': r[4]})
    return result
