# Fraud Detection & AML Transaction Monitoring

**Tools:** Snowflake · SQL · Python · PySpark · Apache Kafka · PostgreSQL · Redis · Docker · Power BI

## What This Project Does

Analyzes 284,807 credit card transactions and 6.3M financial transactions to detect fraud and money laundering — then builds better detection rules from scratch using SQL, and operationalizes those rules into a real-time streaming pipeline.

## The Key Finding

The existing fraud detection system caught **16 out of 8,213 fraud cases** — a 99.8% miss rate.

Three SQL rules I built caught **8,146 out of 8,213** — a **500x improvement**.

| | Original System | This Project |
|---|---|---|
| Fraud caught | 16 | 8,146 |
| Detection rate | 0.19% | **99.79%** |

## What I Found

- 100% of fraud happens in just 2 transaction types: **TRANSFER** and **CASH_OUT**
- CASH_OUT fraud was completely invisible — 0 out of 4,116 cases caught by the existing system
- Top 10% amount transactions that are 2x above average flag CASH_OUT fraud with **99.86% precision**
- The existing system never checked whether the transaction amount matched the actual balance change — Rule 3 exploits exactly this gap

## The 3 Detection Rules

### Rule 1 — Account Emptied
```sql
CASE WHEN balance_orig_after = 0
     AND balance_orig_before > 0
     THEN 1 ELSE 0 END AS rule_account_emptied
```
Flags transactions where the origin account is completely drained in a single TRANSFER or CASH_OUT. Real-world pattern: compromised accounts and money mules emptying funds before the bank can freeze them.

### Rule 2 — Large Amount
```sql
CASE WHEN amount > 200000
     THEN 1 ELSE 0 END AS rule_large_amount
```
Flags transactions above $200,000. Fraud median = $442k vs. clean median = $169k. 68% of all fraud cases exceed this threshold.

### Rule 3 — Balance Mismatch
```sql
CASE WHEN ABS((balance_orig_before - balance_orig_after) - amount) > 1
     THEN 1 ELSE 0 END AS rule_balance_mismatch
```
If you send $500, your balance should drop by exactly $500. Any difference greater than $1 means the records don't add up — the original system never checked this.

Combined:
```sql
SUM(CASE WHEN rule_account_emptied = 1
         OR rule_large_amount = 1
         OR rule_balance_mismatch = 1
         THEN 1 ELSE 0 END) AS flagged_by_any_rule
```

## How It Was Built

**Phase 1 — Analysis (Snowflake + SQL)**

- **Python** — cleaned and merged 4 datasets, preserved all fraud cases during sampling
- **Snowflake** — loaded data via internal staging, same approach used in production bank pipelines
- **SQL** — 4 queries using CTEs, window functions (`NTILE`, `AVG OVER`, `LAG`), and `CASE WHEN` rules
- **Power BI** — dashboard comparing detection rates, alert volumes, and average fraud amounts

**Phase 2 — Real-Time Pipeline (Kafka + PySpark + Docker)**

- **Apache Kafka** — transaction feed split into 3 partitions (one per detection rule) so all rules consume in parallel
- **PySpark Structured Streaming 3.5** — ports the 3 SQL rules into micro-batch streaming jobs processing 10 tx/sec
- **PostgreSQL** — immutable audit log of every flagged transaction with `UNIQUE(transaction_id, rule_triggered)` for idempotent writes
- **Redis** — live alert counters per rule for real-time dashboard stats
- **Docker Compose** — all 6 services (Kafka, Zookeeper, PostgreSQL, Redis, Spark, Kafka UI) run with a single command

## Architecture

```
[Transaction Simulator] → [Apache Kafka] → [PySpark Structured Streaming]
      Python                3 partitions        3 detection rules
                                                       ↓
                                              [PostgreSQL]
                                              Flagged transactions
                                              audit log
```

## Project Structure

```
fraud-detection-aml/
├── docker-compose.yml          # Full infrastructure
├── ingestion/
│   └── producer.py             # PaySim transaction simulator
├── detection/
│   └── fraud_detector.py       # PySpark Structured Streaming
├── storage/
│   └── init.sql                # PostgreSQL schema
└── fraud_detection.sql         # Original Snowflake SQL analysis
```

## Quick Start

```bash
# 1. Start infrastructure
docker-compose up -d

# 2. Create Kafka topics
docker exec paysim_kafka kafka-topics --bootstrap-server localhost:9092 \
  --create --topic paysim_transactions --partitions 3 --replication-factor 1

# 3. Start transaction simulator
pip install kafka-python
python3 ingestion/producer.py

# 4. Start detection engine
pip install pyspark==3.5.3 psycopg2-binary redis
spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.3,org.postgresql:postgresql:42.7.3 \
  detection/fraud_detector.py

# 5. Verify alerts in PostgreSQL
docker exec paysim_postgres psql -U fraud_user -d fraud_db -c \
  "SELECT rule_triggered, COUNT(*) as alerts FROM flagged_transactions GROUP BY rule_triggered;"
```

## Data Sources

| Dataset | Source | Link |
|---|---|---|
| Credit Card Fraud Detection | Kaggle — ULB Machine Learning Group | [kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| PaySim Financial Transactions | Kaggle — Edgar Lopez-Rojas | [kaggle.com/datasets/ealaxi/paysim1](https://www.kaggle.com/datasets/ealaxi/paysim1) |

**Aryan Bansal** · [LinkedIn](https://www.linkedin.com/in/aryan-bansal17/) · [Kaggle](https://www.kaggle.com/work)
