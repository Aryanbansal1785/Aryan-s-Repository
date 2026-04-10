# Fraud Detection & AML Transaction Monitoring
**Tools:** Snowflake · SQL · Python · Power BI

---

## What This Project Does

Analyzes 284,807 credit card transactions and 6.3M financial transactions to detect fraud and money laundering — then builds better detection rules from scratch using SQL.

---

## The Key Finding

The existing fraud detection system caught **16 out of 8,213 fraud cases** — a 99.8% miss rate.

Three SQL rules I built caught **8,146 out of 8,213** — a **500x improvement**.

| | Original System | This Project |
|---|---|---|
| Fraud caught | 16 | 8,146 |
| Detection rate | 0.19% | **99.79%** |

---

## What I Found

- 100% of fraud happens in just 2 transaction types: **TRANSFER** and **CASH_OUT**
- CASH_OUT fraud was completely invisible — 0 out of 4,116 cases caught by the existing system
- Top 10% amount transactions that are 2x above average flag CASH_OUT fraud with **99.86% precision**

---

## How It Was Built

**Python** — cleaned and merged 4 datasets, preserved all fraud cases during sampling

**Snowflake** — loaded data via internal staging, same approach used in production bank pipelines

**SQL** — 4 queries using CTEs, window functions (`NTILE`, `AVG OVER`, `LAG`), and `CASE WHEN` rules

---

## Why It Matters

AML transaction monitoring, KPI development, and Snowflake are listed requirements in data analyst roles at RBC, TD, and CIBC. This project demonstrates all three on real fraud datasets.

---

## How to Run

1. Sign up for a free Snowflake trial at snowflake.com
2. Run `sql/setup.sql` to create the database and tables
3. Upload CSVs from `/data` via Snowflake's Load Data feature
4. Run `sql/fraud_detection_queries.sql`

---

**Aryan Bansal** · [LinkedIn](https://www.linkedin.com/in/aryan-bansal17/) · [Kaggle](https://www.kaggle.com/work)
