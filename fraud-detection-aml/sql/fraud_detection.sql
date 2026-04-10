
CREATE DATABASE FRAUD_DETECTION;

USE DATABASE FRAUD_DETECTION;

CREATE SCHEMA FRAUD_ANALYSIS;

USE SCHEMA FRAUD_ANALYSIS;

--  virtual warehouse (compute engine)
CREATE WAREHOUSE FRAUD_WH
    WAREHOUSE_SIZE = 'X-SMALL'
    AUTO_SUSPEND = 60
    AUTO_RESUME = TRUE;

USE WAREHOUSE FRAUD_WH;

-- Credit card fraud table
CREATE OR REPLACE TABLE FRAUD_DETECTION.FRAUD_ANALYSIS.CREDIT_CARD_TRANSACTIONS (
    time_seconds FLOAT,
    V1 FLOAT, V2 FLOAT, V3 FLOAT, V4 FLOAT, V5 FLOAT,
    V6 FLOAT, V7 FLOAT, V8 FLOAT, V9 FLOAT, V10 FLOAT,
    V11 FLOAT, V12 FLOAT, V13 FLOAT, V14 FLOAT, V15 FLOAT,
    V16 FLOAT, V17 FLOAT, V18 FLOAT, V19 FLOAT, V20 FLOAT,
    V21 FLOAT, V22 FLOAT, V23 FLOAT, V24 FLOAT, V25 FLOAT,
    V26 FLOAT, V27 FLOAT, V28 FLOAT,
    amount FLOAT,
    is_fraud INT
);

-- PaySim AML table
CREATE OR REPLACE TABLE FRAUD_DETECTION.FRAUD_ANALYSIS.PAYSIM_TRANSACTIONS (
    step_hour INT,
    transaction_type VARCHAR(20),
    amount FLOAT,
    account_origin VARCHAR(20),
    balance_orig_before FLOAT,
    balance_orig_after FLOAT,
    account_dest VARCHAR(20),
    balance_dest_before FLOAT,
    balance_dest_after FLOAT,
    is_fraud INT,
    is_flagged_by_system INT
);

-- stage to upload files from computer
CREATE OR REPLACE STAGE FRAUD_DETECTION.FRAUD_ANALYSIS.FRAUD_STAGE
    FILE_FORMAT = (TYPE = 'CSV' FIELD_OPTIONALLY_ENCLOSED_BY = '"' SKIP_HEADER = 1);


SELECT 'CREDIT_CARD' AS dataset, COUNT(*) AS total_rows, SUM(is_fraud) AS fraud_cases
FROM FRAUD_DETECTION.FRAUD_ANALYSIS.CREDIT_CARD_TRANSACTIONS
UNION ALL
SELECT 'PAYSIM' AS dataset, COUNT(*) AS total_rows, SUM(is_fraud) AS fraud_cases
FROM FRAUD_DETECTION.FRAUD_ANALYSIS.PAYSIM_TRANSACTIONS;

-- Query 1: Fraud rate by transaction type (PaySim)
SELECT 
    transaction_type,
    COUNT(*) AS total_transactions,
    SUM(is_fraud) AS fraud_cases,
    ROUND(SUM(is_fraud) * 100.0 / COUNT(*), 2) AS fraud_rate_pct,
    ROUND(AVG(amount), 2) AS avg_transaction_amount
FROM FRAUD_DETECTION.FRAUD_ANALYSIS.PAYSIM_TRANSACTIONS
GROUP BY transaction_type
ORDER BY fraud_rate_pct DESC;

-- Query 2: The system only caught 16 cases checking which ones it missed
SELECT 
    transaction_type,
    COUNT(*) AS total_fraud,
    SUM(is_flagged_by_system) AS caught_by_system,
    SUM(is_fraud) - SUM(is_flagged_by_system) as missed_by_system,
    ROUND((SUM(is_fraud) - SUM(is_flagged_by_system)) * 100.0 / SUM(is_fraud), 2) AS miss_rate_pct
FROM FRAUD_DETECTION.FRAUD_ANALYSIS.PAYSIM_TRANSACTIONS
where is_fraud=1
GROUP BY transaction_type
ORDER BY missed_by_system DESC;

-- Query 3: improved fraud detection rules
WITH transaction_features AS (
    SELECT
        account_origin,
        transaction_type,
        amount,
        balance_orig_before,
        balance_orig_after,
        is_fraud,
        -- Rule 1: Account emptied completely
        CASE WHEN balance_orig_after = 0 
             AND balance_orig_before > 0 
             THEN 1 ELSE 0 END AS rule_account_emptied,
        -- Rule 2: Large transaction over $200k
        CASE WHEN amount > 200000 
             THEN 1 ELSE 0 END AS rule_large_amount,
        -- Rule 3: Amount doesn't match balance change
        CASE WHEN ABS((balance_orig_before - balance_orig_after) - amount) > 1
             THEN 1 ELSE 0 END AS rule_balance_mismatch
    FROM FRAUD_DETECTION.FRAUD_ANALYSIS.PAYSIM_TRANSACTIONS
    WHERE transaction_type IN ('TRANSFER', 'CASH_OUT')
)
SELECT
    SUM(is_fraud) AS actual_fraud,
    SUM(rule_account_emptied) AS flagged_by_rule1,
    SUM(rule_large_amount) AS flagged_by_rule2,
    SUM(rule_balance_mismatch) AS flagged_by_rule3,
    SUM(CASE WHEN rule_account_emptied = 1 
             OR rule_large_amount = 1 
             OR rule_balance_mismatch = 1 
             THEN 1 ELSE 0 END) AS flagged_by_any_rule,
    ROUND(SUM(CASE WHEN rule_account_emptied = 1 
                   OR rule_large_amount = 1 
                   OR rule_balance_mismatch = 1 
                   THEN is_fraud ELSE 0 END) * 100.0 / SUM(is_fraud), 2) AS our_detection_rate
FROM transaction_features;

-- Query 4: High amount fraud detection with window functions
WITH account_stats AS (
    SELECT
        account_origin,
        transaction_type,
        amount,
        is_fraud,
        balance_orig_before,
        balance_orig_after,
        -- What percentile is this transaction amount?
        NTILE(100) OVER (
            PARTITION BY transaction_type
            ORDER BY amount
        ) AS amount_percentile,
        -- How does this compare to average for this type?
        ROUND(amount / AVG(amount) OVER (
            PARTITION BY transaction_type
        ), 2) AS amount_vs_avg
    FROM FRAUD_DETECTION.FRAUD_ANALYSIS.PAYSIM_TRANSACTIONS
    WHERE transaction_type IN ('TRANSFER', 'CASH_OUT')
)
SELECT
    transaction_type,
    COUNT(*) AS flagged,
    SUM(is_fraud) AS fraud_caught,
    ROUND(SUM(is_fraud) * 100.0 / COUNT(*), 2) AS precision_pct
FROM account_stats
WHERE amount_percentile >= 90
  AND amount_vs_avg > 2
GROUP BY transaction_type
ORDER BY precision_pct DESC;


SELECT transaction_type,
    COUNT(*) AS total_transactions,
    SUM(is_fraud) AS fraud_cases,
    ROUND(SUM(is_fraud) * 100.0 / COUNT(*), 2) AS fraud_rate_pct
FROM FRAUD_DETECTION.FRAUD_ANALYSIS.PAYSIM_TRANSACTIONS
GROUP BY transaction_type
ORDER BY fraud_rate_pct DESC;

SELECT 'Original System' AS method, 16 AS fraud_caught, 0.19 AS detection_rate
UNION ALL
SELECT 'Our SQL Rules', 8146, 99.79;

SELECT transaction_type, amount, is_fraud
FROM FRAUD_DETECTION.FRAUD_ANALYSIS.PAYSIM_TRANSACTIONS
WHERE transaction_type IN ('TRANSFER','CASH_OUT')
LIMIT 5000;