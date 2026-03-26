# 💼 Wealth Management Lakehouse
### Medallion Architecture | Databricks | PySpark | Delta Lake

## 📌 Project Overview
A production-style data lakehouse pipeline built on Databricks that 
analyzes 10,000 retail banking customers across wealth tiers, age 
segments, and geographies to surface actionable insights for wealth 
management strategy.

Built using **medallion architecture** (Bronze → Silver → Gold), 
mirroring how real financial institutions like CIBC, RBC, and TD 
structure their enterprise data platforms.

---

## 🏗️ Architecture
```
Raw CSV (10,000 Bank Customers, 39 columns)
                    │
                    ▼
     ┌──────────────────────────────┐
     │        DATABRICKS            │
     │                              │
     │  🥉 BRONZE                   │
     │  Raw ingestion, untouched    │
     │         ↓                    │
     │  🥈 SILVER                   │
     │  Cleaned + 7 engineered      │
     │  features added              │
     │         ↓                    │
     │  🥇 GOLD                     │
     │  3 business aggregation      │
     │  tables                      │
     │         ↓                    │
     │  📊 SQL ANALYSIS             │
     │  7 insight queries           │
     └──────────────────────────────┘
```

---

## 🛠️ Tech Stack
| Tool | Purpose |
|------|---------|
| Databricks (Serverless) | Cloud compute platform |
| Apache Spark 3.4 | Distributed data processing |
| PySpark | Python API for Spark |
| Delta Lake | ACID-compliant storage format |
| Spark SQL | Business insight queries |

---

## 📂 Project Structure
| Notebook | What it does |
|----------|-------------|
| `01_bronze_ingestion` | Loads raw CSV into Spark, validates schema |
| `02_silver_transformation` | Cleans data, engineers 7 new features |
| `03_gold_aggregations` | Builds 3 business aggregation tables |
| `04_sql_analysis` | 7 SQL queries producing business insights |

---

## 🔧 Silver Layer — Feature Engineering
7 new features engineered from raw data:

| Feature | Description |
|---------|-------------|
| `WealthTier` | Segments customers: Mass Market → Ultra High Net Worth |
| `AgeGroup` | Age buckets: Under 30 / 30s / 40s / 50s / 60+ |
| `CreditBand` | Credit score bands: Poor / Fair / Good / Excellent |
| `InvestmentCount` | Number of investment products held (0–8) |
| `SalaryToDebtRatio` | Salary divided by debt — proxy for financial health |

---

## 🥇 Gold Layer — Business Tables
Three aggregation tables built for business consumption:

**`gold_wealth_tiers`** — Portfolio performance, churn rate, and 
investment behaviour by wealth segment

**`gold_age_groups`** — Returns, diversification, and risk profile 
by age group

**`gold_countries`** — Geographic breakdown across France, 
Spain, and Germany

---

## 📊 Key Findings
- Customers holding **5+ investment products** churn at half 
  the rate of those with 1–2 products
- The **40s age group** shows the highest average portfolio 
  returns across all segments  
- **High Net Worth** customers hold 5+ investment types on 
  average vs 2 for Mass Market customers
- **Credit score** is positively correlated with portfolio 
  diversification — Excellent band customers hold 2x more 
  investment products than Poor band

---

## 🚀 How to Replicate
1. Download dataset: [Wealth Management Customer Data](https://www.kaggle.com/datasets/rgupt44/wealth-management-customer-data)
2. Upload CSV to Databricks Volume
3. Run notebooks in order: `01` → `02` → `03` → `04`
4. All Delta tables created automatically

---

## 💡 Skills Demonstrated
- Medallion architecture design (Bronze/Silver/Gold)
- PySpark data cleaning and transformation
- Feature engineering for financial analytics
- Delta Lake table creation and management  
- Spark SQL aggregations and business logic
- Databricks Serverless compute
- Financial domain knowledge — wealth tiers, portfolio 
  metrics, churn analysis, debt-to-income ratios

---

## 👤 Author
**Aryan Bansal**  
Master of Professional Studies in Analytics — Northeastern University  
[LinkedIn](https://www.linkedin.com/in/aryan-bansal17/) | 
[Kaggle](https://www.kaggle.com/aryan1555) | 
[Medium](https://medium.com/@aryanbansal1712)
