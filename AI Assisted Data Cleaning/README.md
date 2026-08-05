# AI Assisted Data Cleaning and Standardization Pipeline

This repository folder contains a lightweight, demoable implementation (pandas + SQLite) of the AI Assisted Data Cleaning and Standardization Pipeline.

Goals
- Demonstrate an "AI proposes, human approves, rules versioned" flow.
- Provide a Streamlit UI for human review of AI-proposed rules.
- Persist approved rules and an audit trail in SQLite.

Status
- Scaffold pushed. Next: implement the synthetic transactions generator and unit tests.

How to run (local dev)
1. python -m venv .venv
2. source .venv/bin/activate  # or .venv\Scripts\activate on Windows
3. pip install -r requirements.txt
4. copy .env.example to .env if you want to provide an OpenAI API key. If you do not provide a key the app runs in MOCK mode.
5. streamlit run streamlit_app.py

Notes
- Do not add secrets in code. Use environment variables or GitHub Secrets for CI.
