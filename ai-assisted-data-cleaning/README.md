# AI Assisted Data Cleaning and Standardization Pipeline

A lightweight, demoable implementation (Python + pandas + Streamlit + SQLite) of an "AI proposes, human approves, rules versioned" workflow for cleaning messy tabular data.

Instead of silently transforming a dataset, the app profiles whatever CSV you give it, proposes specific cleaning rules for the issues it finds, and requires a human reviewer to explicitly Approve, Preview, or Reject each one before anything is applied. Every approved rule is versioned and logged to an audit trail.

# What it does

Upload any CSV, or use the built-in synthetic sample data generator
Detects issues automatically, computed over the full dataset (not just a sample), so it works reliably on large files:
Inconsistent date formats within a column → proposes standardizing to ISO 8601
Near duplicate categorical values (casing/whitespace, e.g. "CA" vs "ca" vs "California ") → proposes normalizing to one canonical spelling
Missing values in any column → flags them, and lets the reviewer choose how to handle them (fill with mean, median, mode, a custom value, or drop the affected rows)
Nothing is auto-applied. Every proposal has to be explicitly approved by a human before it's used, and "Apply (preview)" always shows the actual rows a rule would change — not just the first few rows of the file
Versioned, auditable rules. Approved rules are stored in SQLite with a version number and reviewer name; every pipeline run logs which rule, which version, and how many rows were touched
Run the full pipeline on all approved rules at once and download the cleaned CSV
Tested at scale

Validated against a real ~13,700-row project billing dataset (not just the synthetic demo data), correctly surfacing genuine issues — casing-inconsistent customer/project names and 95 missing project records — that a naive "check the first few rows" approach would have missed entirely.

# How to run (local dev)

bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
Project structure
app.py                     # Streamlit UI
src/
  generator.py              # synthetic sample-data generator
  profiler.py                # full-column dataset profiling (missing values, format/casing diversity)
  rule_suggester.py         # rule detection logic — deterministic today, designed to be swapped for a live LLM call later
  pipeline.py                # the actual data transformations (date standardization, categorical normalization, imputation, flagging)
  rules_store.py             # SQLite-backed storage for approved rules and versions
  audit.py                   # audit log of pipeline runs
sample_data/
  transactions_sample.csv    # small built-in demo dataset
tests/
  test_generator.py
  test_pipeline.py


# Current limitations / roadmap
Rule suggestions are currently deterministic (statistical profiling + heuristics), not backed by a live LLM call. The rule_suggester.py module is isolated specifically so this can be swapped in later without touching the rest of the app.
Categorical normalization groups values by casing/whitespace only it won't merge semantically equivalent but differently worded values (e.g. "USA" vs "United States").
Date-format detection relies on dateutil's default parsing, which can occasionally misread ambiguous day/month order (e.g. 01-06-2021).
