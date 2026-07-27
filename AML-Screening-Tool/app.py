"""
app.py
AML Transaction Screening & Case Management -- Streamlit app.

Three pages:
  1. Data pipeline  - generate a fresh synthetic batch (or upload one in the
                       same schema), see the data quality report, run the
                       detection engine.
  2. Review queue    - work flagged transactions like an AML analyst would:
                       see the evidence, clear or escalate, leave notes.
  3. Dashboard        - detection volume, precision/recall vs ground truth,
                       rule performance, review throughput.
"""

import os
import subprocess
import sys
from datetime import datetime

import pandas as pd
import streamlit as st

import db
import cleaning
import detection

os.chdir(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="AML Screening & Case Management", layout="wide")
db.init_db()

# Helpers

def run_pipeline(n_accounts, n_days, seed):
    subprocess.run(
        [sys.executable, "generator.py", "--accounts", str(n_accounts),
         "--days", str(n_days), "--seed", str(seed)],
        check=True,
    )
    db.reset_db()
    raw = pd.read_csv("transactions_raw.csv", dtype=str)
    clean, report = cleaning.clean_transactions(raw)
    db.load_clean_transactions(clean)
    db.load_data_quality_log(report, batch_id=f"seed-{seed}")
    gt = pd.read_csv("ground_truth.csv")
    db.load_ground_truth(gt)
    detection.run_all_rules()
    return report


def get_queue_df():
    with db.get_conn() as conn:
        df = pd.read_sql_query("""
            SELECT t.transaction_id, t.account_id, t.customer_id, t.ts, t.amount,
                   t.transaction_type, t.origin_country, t.destination_country,
                   t.channel, t.missing_customer_flag, t.amount_outlier_flag,
                   GROUP_CONCAT(DISTINCT f.rule_name) AS rules_triggered,
                   MAX(f.risk_score) AS max_risk_score,
                   COUNT(DISTINCT f.rule_name) AS n_rules
            FROM transactions t
            JOIN flags f ON f.transaction_id = t.transaction_id
            LEFT JOIN review_decisions rd ON rd.transaction_id = t.transaction_id
            WHERE rd.transaction_id IS NULL
            GROUP BY t.transaction_id
            ORDER BY max_risk_score DESC, n_rules DESC, t.ts DESC
        """, conn)
    return df


def get_transaction_detail(transaction_id):
    with db.get_conn() as conn:
        tx = pd.read_sql_query(
            "SELECT * FROM transactions WHERE transaction_id = ?", conn, params=(transaction_id,)
        ).iloc[0]
        flags_df = pd.read_sql_query(
            "SELECT rule_name, rule_detail, risk_score FROM flags WHERE transaction_id = ?",
            conn, params=(transaction_id,),
        )
        account_history = pd.read_sql_query("""
            SELECT transaction_id, ts, amount, transaction_type, origin_country, destination_country
            FROM transactions WHERE account_id = ? ORDER BY ts DESC LIMIT 15
        """, conn, params=(tx["account_id"],))
    return tx, flags_df, account_history

# Sidebar navigation

st.sidebar.title("AML Screening Tool")
page = st.sidebar.radio("Go to", ["Data Pipeline", "Review Queue", "Dashboard"])

with db.get_conn() as _conn:
    _n_tx = _conn.execute("SELECT COUNT(*) c FROM transactions").fetchone()["c"]
st.sidebar.caption(f"{_n_tx:,} transactions currently loaded")


# Page 1: Data Pipeline

if page == "Data Pipeline":
    st.title("Data Pipeline")
    st.write(
        "Generate a fresh synthetic transaction batch (deliberately messy, "
        "with known AML/fraud patterns injected), run it through cleaning, "
        "load it into the database, and score it with the rule-based "
        "detection engine."
    )

    col1, col2, col3 = st.columns(3)
    n_accounts = col1.number_input("Accounts", min_value=50, max_value=2000, value=400, step=50)
    n_days = col2.number_input("Days of activity", min_value=7, max_value=180, value=45, step=7)
    seed = col3.number_input("Random seed", min_value=0, max_value=99999, value=42, step=1)

    if st.button("Generate new batch and run detection", type="primary"):
        with st.spinner("Generating data, cleaning, and scoring..."):
            report = run_pipeline(n_accounts, n_days, seed)
        st.success("Pipeline complete.")
        st.session_state["last_report"] = report

    if "last_report" in st.session_state:
        st.subheader("Data quality report")
        rpt = st.session_state["last_report"]
        summary = rpt.get("_summary", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows in", summary.get("rows_in", "-"))
        c2.metric("Rows out", summary.get("rows_out", "-"))
        c3.metric("Rows dropped", summary.get("rows_dropped", "-"))

        issues = [{"issue": k, **v} for k, v in rpt.items() if k != "_summary"]
        if issues:
            st.dataframe(pd.DataFrame(issues), width="stretch", hide_index=True)

    st.divider()
    st.subheader("Or upload your own file")
    st.caption(
        "Must match the raw schema: transaction_id, account_id, customer_id, "
        "timestamp, amount, currency, transaction_type, origin_country, "
        "destination_country, channel, memo. No ground truth needed -- "
        "precision/recall just won't be available on the dashboard."
    )
    uploaded = st.file_uploader("Upload transactions CSV", type=["csv"])
    if uploaded is not None and st.button("Load uploaded file"):
        raw = pd.read_csv(uploaded, dtype=str)
        clean, report = cleaning.clean_transactions(raw)
        db.reset_db()
        db.load_clean_transactions(clean)
        db.load_data_quality_log(report, batch_id="uploaded")
        detection.run_all_rules()
        st.session_state["last_report"] = report
        st.success(f"Loaded {len(clean)} cleaned transactions and ran detection.")
        st.rerun()


# Page 2: Review Queue

elif page == "Review Queue":
    st.title("Review Queue")
    try:
        queue_df = get_queue_df()
        if queue_df.empty:
            st.info("No open flagged transactions. Generate a batch from the Data Pipeline page.")
        else:
            st.write(f"**{len(queue_df)}** flagged transactions awaiting review.")
            display_cols = ["transaction_id", "account_id", "ts", "amount", "transaction_type",
                             "rules_triggered", "max_risk_score", "n_rules"]
            st.dataframe(queue_df[display_cols], width="stretch", hide_index=True, height=300)
            st.divider()
            selected_id = st.selectbox("Select a transaction to review", queue_df["transaction_id"].tolist())
            if selected_id:
                tx, flags_df, history = get_transaction_detail(selected_id)
                colA, colB = st.columns([1, 1])
                with colA:
                    st.subheader("Transaction")
                    st.write(f"**ID:** {tx['transaction_id']}")
                    st.write(f"**Account:** {tx['account_id']}  |  **Customer:** {tx['customer_id'] or '⚠️ missing'}")
                    st.write(f"**Time:** {tx['ts']}")
                    st.write(f"**Amount:** ${tx['amount']:,.2f} {tx['currency']}")
                    st.write(f"**Type / channel:** {tx['transaction_type']} via {tx['channel']}")
                    st.write(f"**Route:** {tx['origin_country']} -> {tx['destination_country']}")
                    if tx["amount_outlier_flag"]:
                        st.warning("Data-quality flag: amount is a statistical outlier for this account.")
                    if tx["missing_customer_flag"]:
                        st.warning("Data-quality flag: customer_id was missing on ingest.")
                with colB:
                    st.subheader("Why it was flagged")
                    for _, f in flags_df.iterrows():
                        st.write(f"**{f['rule_name']}** (risk {f['risk_score']}): {f['rule_detail']}")
                st.subheader("Recent activity on this account")
                st.dataframe(history, width="stretch", hide_index=True)
                st.subheader("Decision")
                reviewer = st.text_input("Reviewer name", value="Aryan Bansal")
                notes = st.text_area("Notes")
                d1, d2, d3 = st.columns(3)
                if d1.button("Clear (false positive)"):
                    db.record_decision(selected_id, "cleared", reviewer, notes)
                    st.rerun()
                if d2.button("Escalate for further review"):
                    db.record_decision(selected_id, "escalated", reviewer, notes)
                    st.rerun()
                if d3.button("Confirm SAR-worthy", type="primary"):
                    db.record_decision(selected_id, "confirmed_sar", reviewer, notes)
                    st.rerun()
    except Exception as e:
        st.exception(e)

# Page 3: Dashboard

elif page == "Dashboard":
    st.title("Dashboard")

    with db.get_conn() as conn:
        has_gt = conn.execute("SELECT COUNT(*) c FROM ground_truth").fetchone()["c"] > 0

    if has_gt:
        metrics = detection.evaluate_against_ground_truth()
        st.caption(
            "Precision/recall below are measured against the synthetic ground truth "
            "(known-injected patterns) -- an evaluation aid only available because this "
            "is a demo. In production these would come from confirmed SAR outcomes over time."
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precision", f"{metrics['precision']:.1%}")
        c2.metric("Recall", f"{metrics['recall']:.1%}")
        c3.metric("F1", f"{metrics['f1']:.2f}")
        c4.metric("True positives", metrics["true_positives"])

        st.subheader("Flags by rule")
        rb = pd.DataFrame(metrics["rule_breakdown"]).set_index("rule_name")
        st.bar_chart(rb)
    else:
        st.info("No ground truth loaded (custom upload) -- precision/recall unavailable.")

    with db.get_conn() as conn:
        vol = pd.read_sql_query("""
            SELECT date(t.ts) AS day, COUNT(DISTINCT f.transaction_id) AS flagged
            FROM flags f JOIN transactions t ON t.transaction_id = f.transaction_id
            GROUP BY date(t.ts) ORDER BY day
        """, conn)
        decisions = pd.read_sql_query("""
            SELECT decision, COUNT(*) as n FROM review_decisions GROUP BY decision
        """, conn)
        queue_size = conn.execute("""
            SELECT COUNT(DISTINCT f.transaction_id) c FROM flags f
            LEFT JOIN review_decisions rd ON rd.transaction_id = f.transaction_id
            WHERE rd.transaction_id IS NULL
        """).fetchone()["c"]

    st.subheader("Flagged transaction volume over time")
    if not vol.empty:
        st.line_chart(vol.set_index("day"))
    else:
        st.caption("No flags yet.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Review outcomes")
        if not decisions.empty:
            st.bar_chart(decisions.set_index("decision"))
        else:
            st.caption("No decisions recorded yet.")
    with col2:
        st.subheader("Open queue size")
        st.metric("Awaiting review", queue_size)
