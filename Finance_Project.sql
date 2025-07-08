CREATE DATABASE bank_project;
USE bank_project;

CREATE TABLE bank_marketing (
  age INT,
  job VARCHAR(50),
  marital VARCHAR(20),
  education VARCHAR(50),
  default_val VARCHAR(10),
  housing VARCHAR(10),
  loan VARCHAR(10),
  contact VARCHAR(20),
  month VARCHAR(10),
  day_of_week VARCHAR(10),
  duration INT,
  campaign INT,
  pdays INT,
  previous INT,
  poutcome VARCHAR(20),
  emp_var_rate FLOAT,
  cons_price_idx FLOAT,
  cons_conf_idx FLOAT,
  euribor3m FLOAT,
  nr_employed FLOAT,
  y VARCHAR(5)
);

DROP TABLE IF EXISTS credit_cards;

CREATE TABLE credit_cards_alt (
  `CUST_ID` VARCHAR(20),
  `BALANCE` FLOAT,
  `BALANCE_FREQUENCY` FLOAT,
  `PURCHASES` FLOAT,
  `ONEOFF_PURCHASES` FLOAT,
  `INSTALLMENTS_PURCHASES` FLOAT,
  `CASH_ADVANCE` FLOAT,
  `PURCHASES_FREQUENCY` FLOAT,
  `ONEOFF_PURCHASES_FREQUENCY` FLOAT,
  `PURCHASES_INSTALLMENTS_FREQUENCY` FLOAT,
  `CASH_ADVANCE_FREQUENCY` FLOAT,
  `CASH_ADVANCE_TRX` INT,
  `PURCHASES_TRX` INT,
  `CREDIT_LIMIT` FLOAT,
  `PAYMENTS` FLOAT,
  `MINIMUM_PAYMENTS` FLOAT,
  `PRC_FULL_PAYMENT` FLOAT,
  `TENURE` INT
);

Select count(*) AS rows_bank FROM bank_marketing;
Select count(*) AS rows_cc FROM credit_cards_alt;

SHOW COLUMNS FROM bank_marketing;

/* 1  Customer‑behaviour snapshots (Bank Marketing)

/* 1‑A: Overall campaign success rate */
SELECT 
    COUNT(*)                                        AS total_customers,
    SUM(y = 'yes')                                  AS responders,
    ROUND(SUM(y = 'yes') * 100.0 / COUNT(*), 2)     AS response_pct
FROM bank_marketing;

/* 1‑B: Response rate by job segment (top 10) */
SELECT 
    job,
    COUNT(*)                                        AS total,
    SUM(y = 'yes')                                  AS responders,
    ROUND(SUM(y = 'yes') * 100.0 / COUNT(*), 2)     AS response_pct
FROM bank_marketing
GROUP BY job
ORDER BY response_pct DESC
LIMIT 10;

/* 2. Creating an enriched cc view */

/* 2‑A: helper view — one row per credit‑card customer */

create or replace view credit_cards_enriched as 
select
	cust_id,
    balance,
    purchases,
    cash_advance,
    credit_limit,
    tenure,
    /* utilisation: how much of the limit is used on avg */
    ROUND(balance / NULLIF(credit_limit, 0), 3)  as util_ratio,
    /* very rough profit proxy: 18 % APR on balance + 1.5 % interchange */
    ROUND(balance * 0.8 + purchases * 0.015, 2) as annual_profit,
    /* full‑payment behaviour */
    prc_full_payment
From credit_cards_alt;

/* 3. Product Profibility Insights (CC Table) */

/* 3‑A: Average profitability and utilisation by tenure */
Select
    tenure,
    Count(*) as customers,
    Round(avg(annual_profit),2) as avg_profit,
    Round(AVG(util_ratio)*100,2) as avg_util_pct
From credit_cards_enriched
Group by tenure
Order by tenure;

/* 3‑B: High‑ vs low‑profit segments (top and bottom 5 %) */
with ranked as (
  Select 
      cust_id, 
      annual_profit,
      NTILE(20) OVER (Order by annual_profit) as vintile_5p
	From credit_cards_enriched
)
Select 
	 case when vintile_5p = 20 Then 'Top 5 %'
		  when vintile_5p = 1 Then 'Bottom 5%'
	 end as segment, 
     Count(*)  as customers,
	 Round(avg(annual_profit),2) as avg_profit,
     Round(avg(util_ratio)*100,2) as avg_util_pct
From ranked
where vintile_5p in(1,20)
Group by segment;

/* 4. Revenue and Spend Trends */

/* 4‑A: Spend vs payments scatter (quick view of repayment behaviour) */
SELECT
    purchases             AS total_spend,
    payments              AS total_payments,
    (payments > purchases) AS overpay_flag  -- 1 if they pay more than they spend
FROM credit_cards_alt
LIMIT 200;   -- take a sample if the plot tool you use can’t handle 8 950 pts

 
/* 5. Churn Indicators /*

/* 5‑A: Dormant vs active customers */
SELECT
    SUM(pdays = 999)                       AS dormant_customers,
    SUM(pdays <> 999)                      AS active_customers,
    ROUND(SUM(pdays = 999)*100.0/COUNT(*),2) AS dormant_pct
FROM bank_marketing;

/* 5‑B: Response rate among dormant customers */
SELECT
    CASE WHEN pdays = 999 THEN 'Dormant' ELSE 'Active' END AS segment,
    COUNT(*)                                  AS customers,
    SUM(y = 'yes')                            AS responders,
    ROUND(SUM(y = 'yes')*100.0/COUNT(*),2)    AS response_pct
FROM bank_marketing
GROUP BY segment;

/* 6. Cross‑referencing: who’s profitable and responsive? */

/* 6‑A: Build small dim tables for age and utilisation bands */
CREATE OR REPLACE VIEW bank_age_band AS
SELECT
    CASE 
      WHEN age < 30 THEN '<30'
      WHEN age BETWEEN 30 AND 44 THEN '30‑44'
      WHEN age BETWEEN 45 AND 59 THEN '45‑59'
      ELSE '60+'
    END        AS age_band,
    COUNT(*)   AS customers,
    ROUND(SUM(y = 'yes')*100.0/COUNT(*),2) AS response_rate
FROM bank_marketing
GROUP BY age_band;

CREATE OR REPLACE VIEW cc_util_band AS
SELECT
    CASE 
      WHEN util_ratio < 0.25 THEN '<25 %'
      WHEN util_ratio < 0.50 THEN '25‑49 %'
      WHEN util_ratio < 0.75 THEN '50‑74 %'
      ELSE '75 %+'
    END        AS util_band,
    COUNT(*)   AS customers,
    ROUND(AVG(annual_profit),2) AS avg_profit
FROM credit_cards_enriched
GROUP BY util_band;

/* 6‑B: Simple report (no join) */
SELECT * FROM bank_age_band;
SELECT * FROM cc_util_band;




