# AML Transaction Screening & Case Management

A live, working tool that simulates what an AML analyst actually does:
transactions come in, get screened against rule-based detection logic, land
in a review queue, and a human works through them making decisions —
clear, escalate, or confirm as SAR-worthy. A dashboard reports detection
performance and review throughput.

This extends the detection logic from the standalone *Fraud Detection & AML
Monitoring* project (Python/SQL, 6.3M transactions, 99.79% rule-based
detection accuracy) into an operational system with a UI, a database, and
an analyst workflow, instead of a one-time notebook analysis.

## Why the data is synthetic

The app runs on a synthetic transaction generator rather than a static
dataset, for two reasons:

1. **A live case-management tool needs a stream of new data**, not a single
   frozen file — a static dataset doesn't behave like transactions
   "arriving."
2. **Controlling the ground truth lets detection accuracy be measured
   honestly.** Every fraud/AML pattern is injected on purpose and tracked,
   so precision/recall numbers are real, not guessed. Public datasets
   (e.g. Kaggle's anonymized/PCA-transformed fraud sets) don't allow this —
   you can't point to "this is a structuring pattern" when the columns
   have been stripped of meaning.

The generated data is also **deliberately messy** — missing fields,
duplicate transaction IDs, inconsistent date formats and currency casing,
fat-finger amount typos — because real bank exports are never clean, and a
cleaning/validation step is part of the pipeline for that reason (see
`cleaning.py`).

## Architecture

```
generator.py   -> raw, messy synthetic transactions + a separate ground-truth
                   answer key (never shown to the "analyst")
cleaning.py    -> validates & cleans raw data, producing a data-quality report
db.py          -> SQLite schema + persistence (plain sqlite3, no ORM — the
                   detection engine's hand-written SQL is the point)
detection.py   -> 5 rule-based AML typologies, written as SQL CTEs and
                   window functions
app.py         -> Streamlit UI: Data Pipeline / Review Queue / Dashboard
```

Data flow: `generator.py` → `cleaning.py` → SQLite (`db.py`) →
`detection.py` scores every transaction and writes flags → the Streamlit
app reads/writes the same database as an analyst works the queue.

## The 5 detection rules

| Rule | Technique | Typology |
|---|---|---|
| `structuring` | `SUM()`/`COUNT() OVER` partitioned by account+day | Multiple cash deposits just under the $10k reporting threshold, summing above it in one day |
| `velocity` | `COUNT() OVER (... RANGE BETWEEN)` on a trailing 1-hour window | Unusually high transaction frequency (e.g. account takeover / mule draining) |
| `round_dollar` | Simple filter | Large round-number wires/transfers — a common layering signature |
| `high_risk_geo` | Simple filter | Counterparty in a high-risk jurisdiction (placeholder codes `XA`/`XB`/`XC` — swap for a real FATF list in production) |
| `layering` | `LEAD() OVER` partitioned by account, ordered by time | Large deposit followed quickly by a near-equal outbound transfer |

On the default generated batch, the engine scores **~66% precision / ~63%
recall** against ground truth — not a fake 99%, a genuinely defensible
number with explainable gaps (e.g. `velocity`'s trailing-window design
means the first few transactions in a burst haven't accumulated 6 events
yet — a real precision/recall tradeoff worth discussing in an interview
rather than hiding).

## Running locally

```bash
cd aml-screening-tool
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
streamlit run app.py
```

Then in the app: **Data Pipeline** page → "Generate new batch and run
detection" → go to **Review Queue** to work flagged transactions → check
**Dashboard** for metrics.

You can also run the pipeline standalone from the command line:

```bash
python generator.py --accounts 400 --days 45 --seed 42
python -c "
import pandas as pd, db, cleaning, detection
db.init_db(); db.reset_db()
raw = pd.read_csv('transactions_raw.csv', dtype=str)
clean, report = cleaning.clean_transactions(raw)
db.load_clean_transactions(clean)
db.load_data_quality_log(report)
db.load_ground_truth(pd.read_csv('ground_truth.csv'))
detection.run_all_rules()
print(detection.evaluate_against_ground_truth())
"
```

## Deploying for free (Streamlit Community Cloud)

1. Push this folder to a public (or private) GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, click "New app."
3. Point it at the repo, branch, and `app.py` as the entry point.
4. Deploy. You'll get a public URL like
   `https://<your-app>.streamlit.app` — put that on your resume next to
   the GitHub link.

Note: Streamlit Community Cloud's filesystem resets on redeploy/sleep, so
the SQLite database won't persist indefinitely between cold starts — that's
fine for a portfolio demo (the "Generate new batch" button rebuilds
everything in seconds), but worth knowing if you want persistence, in
which case swap `db.py`'s `sqlite3` connection for a hosted Postgres
instance (e.g. free tier on Render or Supabase) — the SQL in
`detection.py` is portable to Postgres with minimal changes (window
function syntax is nearly identical).

## Talking points for interviews

- **Why rule-based, not ML:** rule-based detection is explainable and
  auditable, which matters in AML/compliance where you need to justify
  every flag to a regulator. The tradeoff is discussed openly via the
  precision/recall numbers rather than papered over.
- **Data quality as a first-class step:** the cleaning report is not
  cosmetic — dropped/flagged row counts are logged per batch, which is
  what "reproducible, accurate reporting" (from the original resume
  project) looks like as a system instead of a one-off writeup.
- **SQL technique:** every rule is a CTE, most use window functions
  (`SUM() OVER`, `COUNT() OVER ... RANGE BETWEEN`, `LEAD() OVER`) — the
  exact skill listed on the resume, now doing real work against a live
  database rather than a static analysis.
- **Known limitations, stated honestly:** structuring is grouped by
  calendar day rather than a rolling 24h window (a deposit at 11pm and one
  at 1am the next day won't be linked) — a good discussion point on rule
  design tradeoffs and how you'd iterate on it.
