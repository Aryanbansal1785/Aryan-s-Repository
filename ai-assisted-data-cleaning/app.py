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
    apply_impute_missing,
)
from src.audit import log_rule_application

st.set_page_config(page_title="AI Assisted Data Cleaning", layout="wide")

DB_PATH = Path(os.getenv('DATABASE_PATH', 'ai_assisted_data_cleaning.db'))
init_db(DB_PATH)

STRATEGY_LABELS = {
    'mean': 'Fill with mean',
    'median': 'Fill with median',
    'mode': 'Fill with most common value (mode)',
    'custom': 'Fill with a custom value',
    'drop_rows': 'Drop rows with missing value',
}


def apply_rule(df: pd.DataFrame, rule: dict) -> pd.DataFrame:
    action = rule.get('action')
    column = rule.get('column')
    if action == 'standardize_date_iso':
        return apply_standardize_date_iso(df, column)
    if action == 'normalize_province':
        return apply_normalize_province(df, column)
    if action == 'normalize_categorical':
        return apply_normalize_categorical(df, column)
    if action == 'flag_missing':
        return apply_flag_missing(df, column)
    if action == 'impute_missing':
        return apply_impute_missing(df, column, rule.get('strategy'), rule.get('custom_value'))
    return None

if 'df' not in st.session_state:
    st.session_state.df = load_sample_csv()
if 'source_label' not in st.session_state:
    st.session_state.source_label = 'Sample data'
if 'rule_status' not in st.session_state:
    st.session_state.rule_status = {}

st.title('AI Assisted Data Cleaning (Demo)')
st.caption(
    'Upload any CSV — or use the built-in sample — and review AI-proposed cleaning '
    'rules before anything is applied. Nothing changes your data without a click.'
)

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

profile = profile_dataframe(df, max_rows=15)
m1, m2, m3 = st.columns(3)
m1.metric('Rows', profile['row_count'])
m2.metric('Columns', len(profile['columns']))
total_missing = sum(c['n_missing'] for c in profile['columns'].values())
m3.metric('Missing values', total_missing)

tab_data, tab_proposals = st.tabs(['Data', 'AI Proposals'])

with tab_data:
    st.subheader(f'Preview — {st.session_state.source_label}')
    st.dataframe(df.head(20), use_container_width=True)
    if source_choice == 'Use sample data' and st.button('Regenerate synthetic (50 rows)'):
        st.session_state.df = generate_transactions(50)
        st.session_state.rule_status = {}
        st.rerun()

with tab_proposals:
    proposals = suggest_rules_from_profile(profile)
    if not proposals:
        st.info('No issues detected in this dataset.')

    for p in proposals:
        status = st.session_state.rule_status.get(p['id'])
        with st.container():
            header_col, badge_col = st.columns([4, 1])
            header_col.subheader(p['title'])
            if status == 'approved':
                badge_col.success('Approved')
            elif status == 'rejected':
                badge_col.error('Rejected')

            st.write(p['description'])
            with st.expander('Preview affected rows'):
                st.dataframe(pd.DataFrame(p['example_preview']), use_container_width=True)

            c1, c2, c3 = st.columns(3)
            if c1.button('Approve', key=f"approve_{p['id']}"):
                approved_by = os.getenv('REVIEWER_NAME', 'DemoReviewer')
                ver = save_approved_rule(p['id'], p, approved_by, DB_PATH)
                st.session_state.rule_status[p['id']] = 'approved'
                st.success(f"Approved {p['id']} as version {ver}")
                st.rerun()
            if c2.button('Apply (preview)', key=f"apply_{p['id']}"):
                preview_df = apply_rule(df.copy(), p)
                if preview_df is not None:
                    before_col = p['column'] + '_before'

                    if p['action'] == 'flag_missing':
                        flag_col = '_missing_' + p['column']
                        changed = preview_df[preview_df[flag_col]]
                        label = 'row(s) flagged as missing'
                    elif before_col in preview_df.columns:
                        changed_mask = (
                            preview_df[p['column']].astype(str)
                            != preview_df[before_col].astype(str)
                        )
                        changed = preview_df[changed_mask]
                        label = 'row(s) actually changed by this rule'
                    else:
                        changed = preview_df
                        label = 'row(s) — no before/after comparison available'

                    total_changed = len(changed)
                    if total_changed == 0:
                        st.info('This rule found no rows to change in the current data.')
                    else:
                        st.caption(f'Showing {min(10, total_changed)} of {total_changed} {label}:')
                        st.dataframe(changed.head(10), use_container_width=True)
                else:
                    st.warning('No handler implemented for this action yet.')
            if c3.button('Reject', key=f"reject_{p['id']}"):
                st.session_state.rule_status[p['id']] = 'rejected'
                st.info(f"Rejected {p['id']}")
                st.rerun()

            if p['action'] == 'flag_missing':
                st.markdown('**What should happen to these missing values?**')
                col_dtype = profile['columns'][p['column']]['dtype']
                is_numeric = col_dtype.lower().startswith(('int', 'float'))

                options = ['Leave flagged only (no fill)']
                if is_numeric:
                    options += [STRATEGY_LABELS['mean'], STRATEGY_LABELS['median']]
                options += [
                    STRATEGY_LABELS['mode'],
                    STRATEGY_LABELS['custom'],
                    STRATEGY_LABELS['drop_rows'],
                ]
                label_to_strategy = {v: k for k, v in STRATEGY_LABELS.items()}

                strategy_choice = st.selectbox(
                    'Strategy', options, key=f"strategy_{p['id']}", label_visibility='collapsed'
                )

                custom_value = None
                if strategy_choice == STRATEGY_LABELS['custom']:
                    custom_value = st.text_input(
                        'Value to fill with', key=f"customval_{p['id']}"
                    )

                if strategy_choice != 'Leave flagged only (no fill)':
                    strategy_key = label_to_strategy[strategy_choice]
                    ic1, ic2 = st.columns(2)

                    if ic1.button('Preview this fix', key=f"preview_impute_{p['id']}"):
                        impute_rule = {'column': p['column'], 'action': 'impute_missing',
                                        'strategy': strategy_key, 'custom_value': custom_value}
                        result = apply_rule(df.copy(), impute_rule)
                        if strategy_key == 'drop_rows':
                            st.caption(
                                f"Would drop {len(df) - len(result)} row(s) missing "
                                f"'{p['column']}' (out of {len(df)} total)."
                            )
                            st.dataframe(result.head(10), use_container_width=True)
                        else:
                            before_col = p['column'] + '_before'
                            filled_mask = result[before_col].isna()
                            filled = result[filled_mask]
                            st.caption(f"Would fill {len(filled)} row(s):")
                            st.dataframe(filled.head(10), use_container_width=True)

                    if ic2.button('Approve this fix as a rule', key=f"approve_impute_{p['id']}"):
                        impute_rule = {
                            'id': f"{p['id']}-impute",
                            'title': f"{p['title']} — {strategy_choice}",
                            'description': (
                                f"Human-selected strategy for missing values in "
                                f"\"{p['column']}\": {strategy_choice}."
                            ),
                            'column': p['column'],
                            'action': 'impute_missing',
                            'strategy': strategy_key,
                            'custom_value': custom_value,
                        }
                        approved_by = os.getenv('REVIEWER_NAME', 'DemoReviewer')
                        ver = save_approved_rule(impute_rule['id'], impute_rule, approved_by, DB_PATH)

                        result = apply_rule(df.copy(), impute_rule)
                        if strategy_key == 'drop_rows':
                            preview_table = result.head(10)
                            note = f"Would drop {len(df) - len(result)} row(s) missing '{p['column']}'."
                        else:
                            before_col = p['column'] + '_before'
                            preview_table = result[result[before_col].isna()].head(10)
                            note = f"Filled {len(result[result[before_col].isna()])} row(s)."
                        st.session_state[f"approved_result_{p['id']}"] = {
                            'version': ver,
                            'strategy_choice': strategy_choice,
                            'note': note,
                            'table': preview_table,
                        }
                        st.rerun()

                approved_result = st.session_state.get(f"approved_result_{p['id']}")
                if approved_result:
                    st.success(
                        f"Approved as version {approved_result['version']}: "
                        f"{approved_result['strategy_choice']}. {approved_result['note']}"
                    )
                    st.dataframe(approved_result['table'], use_container_width=True)

if run_pipeline:
    approved = list_approved_rules(DB_PATH)
    cleaned = df.copy()
    batch_id = 'batch-demo-1'
    rows_total = len(cleaned)
    applied_count = 0
    for r in approved:
        rid, rule, ver = r['id'], r['rule'], r['version']
        result = apply_rule(cleaned, rule)
        if result is not None:
            cleaned = result
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
