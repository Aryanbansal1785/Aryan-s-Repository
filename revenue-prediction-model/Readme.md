# Revenue Prediction Model

A machine learning application that forecasts next month client revenue from historical billing data, built with **XGBoost** and served through an interactive **Streamlit** dashboard.

## Overview

This project analyzes historical project billing records (hours worked, billing rates, project counts, etc.) for a consulting firm's clients and predicts each client's revenue for the upcoming month. It was built to help finance and account management teams get an early read on revenue trends per customer.

## How It Works

1. **Data Ingestion & Cleaning** — Loads raw billing records from Excel, strips inconsistent formatting, parses dates, and drops incomplete rows.
2. **Feature Engineering** — Aggregates raw line-item data into monthly features per customer:
   - Total and average monthly revenue
   - Revenue volatility (standard deviation)
   - Billable hours and average billing rate
   - Number of distinct active projects
   - Calendar features (month, quarter)
   - **Lag features** — revenue from the previous 1, 2, and 3 months, so the model can learn trend and momentum per customer
3. **Model Training** — Trains an `XGBRegressor` with hyperparameter tuning via `GridSearchCV`, using `TimeSeriesSplit` cross-validation to respect the chronological order of the data (no lookahead leakage).
4. **Evaluation** — Holds out the most recent two months as a test set and reports **RMSE** and **R²**.
5. **Prediction** — For any given customer, takes their most recent monthly feature snapshot and predicts revenue for the following month.
6. **Interactive UI** — A Streamlit front end lets a user type in a customer name and instantly see the predicted revenue for next month.

## Screenshots

<img width="2940" height="946" alt="Screenshot 2026-07-01 at 5 29 26 PM" src="https://github.com/user-attachments/assets/5e2534cb-a4d4-4d55-8663-8115df827cf5" /> 

<img width="2940" height="946" alt="Screenshot 2026-07-01 at 5 25 21 PM" src="https://github.com/user-attachments/assets/5361904c-51b6-4146-9d7b-600881296990" /> 

## Tech Stack

- **Python**
- **pandas / numpy** — data wrangling and feature engineering
- **XGBoost** — gradient boosted regression model
- **scikit-learn** — hyperparameter tuning (`GridSearchCV`) and time-series cross-validation
- **Streamlit** — interactive web UI

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install pandas numpy scikit-learn xgboost streamlit openpyxl
```

### Data

The model expects an Excel file with (at minimum) the following columns:

- `Worked Date`
- `Customer Name`
- `Project`
- `Extended Price`
- `Billable Hours`
- `Hourly Billing Rate`

> **Note:** Update the `file_path` variable in the script to point to your local data file before running. Raw client data is not included in this repository for confidentiality reasons.

### Running the App

```bash
streamlit run Revenue_Prediction_Model.py
```

This will open the dashboard in your browser at `http://localhost:8501`. Enter a customer name to see their predicted revenue for the next month.

## Model Performance

Evaluated on a chronological hold-out set (the most recent two months of data):

| Metric | Value |
|---|---|
| RMSE | $1,953.54 |
| R² (accuracy) | 0.9862 (98.62%) |

The high R² indicates the model explains the large majority of variance in monthly customer revenue, driven primarily by the lag features capturing each customer's recent billing trend.

## Future Improvements

- Cache data loading and model training with `st.cache_resource` to avoid retraining on every UI interaction
- Add confidence intervals around predictions rather than a single point estimate
- Add a historical vs. predicted revenue trend chart per customer
- Support batch predictions for all customers at once
- Move the hardcoded file path to a config file or file-upload widget so the app is portable across machines
- Case insensitive / fuzzy customer name matching
