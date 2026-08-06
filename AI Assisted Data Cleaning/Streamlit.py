import streamlit as st
import pandas as pd
import os
from pathlib import Path

from src.generator import load_sample_csv, generate_transactions
from src.profiler import profile_dataframe
from src.rule_suggester import suggest_rules_from_profile
from src.rules_store import init_db, save_approved_rule, list_approved_rules
from src.pipeline import (
    apply_standardize_date_iso,
    apply_normalize_province,
    apply_normalize_categorical,
    apply_flag_missing,
)
from src.audit import log_rule_application

st.set_page_config(page_title="AI Assisted Data Cleaning", layout="wide")

DB_PATH = Path(os.getenv('DATABASE_PATH', 'ai_assisted_data_cleaning.db'))
init_db(DB_PATH)

# Maps a rule's "action" string to the function that actually performs it.
ACTIONS = {
    'standardize_date_iso': apply_standardize_date_iso,
    'normalize_province': apply_normalize_province,
    'normalize_categorical': apply_normalize_categorical,
    'flag_missing': apply_flag_missing,
}

# Session state

if 'df' not in st.session_state:
    st.session_state.df = load_sample_csv()
if 'source_label' not in st.session_state:
    st.session_state.source_label = 'Sample data'
if 'rule_status' not in st.session_state:
    st.session_state.rule_status = {}  # rule_id -> "approved" | "rejected"

st.title('AI Assisted Data Cleaning (Demo)')
st.caption(
    'Upload any CSV — or use the built-in sample — and review AI-proposed cleaning '
    'rules before anything is applied. Nothing changes your data without a click.'
)


# Sidebar: data source + pipeline controls

st.sidebar.header('1. Data source')
source_choice = st.sidebar.radio('Choose a dataset', ['Use sample data', 'Upload my own CSV'])

if source_choice == 'Upload my own CSV':
    uploaded = st.sidebar.file_uploader('Upload a CSV file', type=['csv'])
    if uploaded is not None:
        try:
            st.session_state.df = pd.read_csv(uploaded)
            st.session_state.source_label = uploaded.name
            st.session_state.rule_status = {}
            st.sidebar.success(f'Loaded "{uploaded.name}" — {len(st.session_state.df)} rows')
        except Exception as e:
            st.sidebar.error(f'Could not read that file: {e}')
else:
    if st.session_state.source_label != 'Sample data':
        if st.sidebar.button('Switch back to sample data'):
            st.session_state.df = load_sample_csv()
            st.session_state.source_label = 'Sample data'
            st.session_state.rule_status = {}
            st.rerun()

df = st.session_state.df

st.sidebar.markdown('---')
st.sidebar.header('2. Run pipeline')
run_pipeline = st.sidebar.button('Run full pipeline (approved rules)')

st.sidebar.markdown('---')
st.sidebar.header('Approved rules')
approved_rules = list_approved_rules(DB_PATH)
if approved_rules:
    for r in approved_rules:
        st.sidebar.write(f"✅ {r['rule'].get('title', r['id'])} (v{r['version']})")
else:
    st.sidebar.caption('None yet — approve a proposal to see it here.')


# Top metrics

profile = profile_dataframe(df, max_rows=15)
m1, m2, m3 = st.columns(3)
m1.metric('Rows', profile['row_count'])
m2.metric('Columns', len(profile['columns']))
total_missing = sum(c['n_missing'] for c in profile['columns'].values())
m3.metric('Missing values', total_missing)

tab_data, tab_proposals = st.tabs(['Data', 'AI Proposals'])

# Data tab

with tab_data:
    st.subheader(f'Preview — {st.session_state.source_label}')
    st.dataframe(df.head(20), use_container_width=True)
    if source_choice == 'Use sample data' and st.button('Regenerate synthetic (50 rows)'):
        st.session_state.df = generate_transactions(50)
        st.session_state.rule_status = {}
        st.rerun()

# AI Proposals tab

with tab_proposals:
    proposals = suggest_rules_from_profile(profile)
    if not proposals:
        st.info('No issues detected in this dataset.')

    for p in proposals:
        status = st.session_state.rule_status.get(p['id'])
        with st.container(border=True):
            header_col, badge_col = st.columns([4, 1])
            header_col.subheader(p['title'])
            if status == 'approved':
                badge_col.success('Approved')
            elif status == 'rejected':
                badge_col.error('Rejected')

            st.write(p['description'])
            with st.expander('Preview affected rows'):
                st.json(p['example_preview'])

            c1, c2, c3 = st.columns(3)
            if c1.button('Approve', key=f"approve_{p['id']}"):
                approved_by = os.getenv('REVIEWER_NAME', 'DemoReviewer')
                ver = save_approved_rule(p['id'], p, approved_by, DB_PATH)
                st.session_state.rule_status[p['id']] = 'approved'
                st.success(f"Approved {p['id']} as version {ver}")
                st.rerun()
            if c2.button('Apply (preview)', key=f"apply_{p['id']}"):
                action_fn = ACTIONS.get(p['action'])
                if action_fn:
                    preview_df = action_fn(df.copy(), p['column'])
                    st.write(preview_df.head(10))
                else:
                    st.warning('No handler implemented for this action yet.')
            if c3.button('Reject', key=f"reject_{p['id']}"):
                st.session_state.rule_status[p['id']] = 'rejected'
                st.info(f"Rejected {p['id']}")
                st.rerun()

# Full pipeline run

if run_pipeline:
    approved = list_approved_rules(DB_PATH)
    cleaned = df.copy()
    batch_id = 'batch-demo-1'
    rows_total = len(cleaned)
    applied_count = 0
    for r in approved:
        rid, rule, ver = r['id'], r['rule'], r['version']
        action_fn = ACTIONS.get(rule.get('action'))
        if action_fn:
            cleaned = action_fn(cleaned, rule['column'])
            log_rule_application(rid, ver, rows_total, batch_id, DB_PATH)
            applied_count += 1

    st.markdown('---')
    st.success(f'Pipeline run complete — {applied_count} approved rule(s) applied.')
    st.dataframe(cleaned.head(20), use_container_width=True)
    st.download_button(
        'Download cleaned CSV',
        cleaned.to_csv(index=False).encode('utf-8'),
        file_name='cleaned_data.csv',
        mime='text/csv',
    )

st.markdown('---')
st.caption(f'Audit log stored in SQLite at: {DB_PATH}')
