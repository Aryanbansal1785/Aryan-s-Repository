
import streamlit as st
import pandas as pd
import os
from src.generator import load_sample_csv, generate_transactions
from src.profiler import profile_dataframe
from src.rule_suggester import suggest_rules_from_profile
from src.rules_store import init_db, save_approved_rule, list_approved_rules
from src.pipeline import apply_standardize_date_iso, apply_normalize_province, apply_flag_missing
from src.audit import log_rule_application
from pathlib import Path
import json

st.set_page_config(page_title="AI Assisted Data Cleaning", layout="wide")

DB_PATH = Path(os.getenv('DATABASE_PATH', 'ai_assisted_data_cleaning.db'))
init_db(DB_PATH)

st.title('AI Assisted Data Cleaning (Demo)')

col1, col2 = st.columns([1,2])
with col1:
    st.header('Data')
    sample = load_sample_csv()
    st.write('Sample rows')
    st.dataframe(sample.head(10))
    if st.button('Regenerate synthetic (50 rows)'):
        df = generate_transactions(50)
        st.write(df.head())
    st.markdown('---')
    st.header('Approved rules')
    rules = list_approved_rules(DB_PATH)
    st.write(rules)

with col2:
    st.header('AI Proposals (mock)')
    profile = profile_dataframe(sample)
    proposals = suggest_rules_from_profile(profile)
    for p in proposals:
        st.subheader(p['title'])
        st.write(p['description'])
        st.write('Preview:')
        st.json(p['example_preview'])
        cols = st.columns([1,1,1])
        if cols[0].button(f"Approve {p['id']}"):
            # store approved rule minimal
            approved_by = os.getenv('REVIEWER_NAME', 'DemoReviewer')
            ver = save_approved_rule(p['id'], p, approved_by, DB_PATH)
            st.success(f"Approved {p['id']} as version {ver}")
        if cols[1].button(f"Apply {p['id']} (demo)"):
            # apply deterministic demo actions
            df = sample.copy()
            if p['action'] == 'standardize_date_iso':
                df = apply_standardize_date_iso(df, p['column'])
            if p['action'] == 'normalize_province':
                df = apply_normalize_province(df, p['column'])
            if p['action'] == 'flag_missing':
                df = apply_flag_missing(df, p['column'])
            st.write(df.head())
        if cols[2].button(f"Reject {p['id']}"):
            st.info(f"Rejected {p['id']}")

st.sidebar.header('Run pipeline (approved rules)')
if st.sidebar.button('Run full pipeline on sample (demo)'):
    df = sample.copy()
    approved = list_approved_rules(DB_PATH)
    batch_id = 'batch-demo-1'
    rows_total = len(df)
    for r in approved:
        rid = r['id']; rule = r['rule']; ver = r['version']
        if rule.get('action') == 'standardize_date_iso':
            df = apply_standardize_date_iso(df, rule['column'])
            log_rule_application(rid, ver, rows_total, batch_id, DB_PATH)
        if rule.get('action') == 'normalize_province':
            df = apply_normalize_province(df, rule['column'])
            log_rule_application(rid, ver, rows_total, batch_id, DB_PATH)
    st.success('Pipeline run complete (demo)')
    st.write(df.head())

st.markdown('---')
st.write('Audit log stored in sqlite at: ', str(DB_PATH))
