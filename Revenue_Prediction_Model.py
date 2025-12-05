import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from xgboost import XGBRegressor


file_path = "/Users/aryanbansal/Documents/Steeve and Associates/XN-Project_Data.xlsx"
df = pd.read_excel(file_path)
preview = df.head()

info = df.info()
stats = df.describe(include='all')

(preview, stats)

# DATA CLEANING 

df.columns = df.columns.str.strip()
df['Project'] = df['Project'].fillna('Unknown')
df['Worked Date'] = pd.to_datetime(df['Worked Date'], dayfirst=True, errors='coerce')
df = df.dropna(subset=['Worked Date', 'Customer Name', 'Extended Price'])

#Feature Engineering
df['YearMonth'] = df['Worked Date'].dt.to_period('M')
df['Month'] = df['Worked Date'].dt.month
df['Quarter'] = df['Worked Date'].dt.quarter


# Aggregate monthly features
monthly_features = df.groupby(['Customer Name', 'YearMonth']).agg(
    revenue_sum=('Extended Price', 'sum'),
    revenue_mean=('Extended Price', 'mean'),
    revenue_std=('Extended Price', 'std'),
    hours_sum=('Billable Hours', 'sum'),
    billingrate_mean=('Hourly Billing Rate', 'mean'),
    project_count=('Project', 'nunique'),
    month=('Month', 'first'),
    quarter=('Quarter', 'first')
).reset_index()


# Fill NaN std with 0
monthly_features['revenue_std'] = monthly_features['revenue_std'].fillna(0)

# lag features (previous 3 months' revenue)
def add_lags(group, n_lags=3):
    for lag in range(1, n_lags+1):
        group[f'revenue_lag_{lag}'] = group['revenue_sum'].shift(lag)
    return group

monthly_features = monthly_features.groupby('Customer Name').apply(add_lags).dropna().reset_index(drop=True)

#Preparing features and target
feature_cols = [
    'revenue_lag_1', 'revenue_lag_2', 'revenue_lag_3', 'revenue_mean', 'revenue_std', 'hours_sum', 'billingrate_mean', 'project_count', 'month', 'quarter']

X = monthly_features[feature_cols]
y = monthly_features['revenue_sum']

#Chronological train-test
monthly_features['YearMonth'] = monthly_features['YearMonth'].astype(str)
unique_months = sorted(monthly_features['YearMonth'].unique())
split_month = unique_months[-2] #last 2 months for testing

train_idx = monthly_features['YearMonth'] < split_month
test_idx = monthly_features['YearMonth'] >= split_month

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# 6. Train XGBoost model with hyperparameter tuning
param_grid = {
    'n_estimators': [50,100],
    'max_depth': [3,5],
    'learning_rate':[0.05,0.1]
}
tscv = TimeSeriesSplit(n_splits=3)
xgb = XGBRegressor(random_state=42)
grid_search = GridSearchCV(xgb, param_grid, cv=tscv, scoring='neg_root_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_


#Evaluation
preds = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print(f"Improved Test RMSE: {rmse:.2f}")

#PREDICT
def predict_next_month_revenue(customer_name):
    cust_data = monthly_features[monthly_features['Customer Name']== customer_name]
    if cust_data.empty:
        return None, None
    
    last_entry = cust_data.sort_values('YearMonth').iloc[-1]
    features = last_entry[feature_cols].values.reshape(1, -1)
    pred = best_model.predict(features)[0]

    last_month = pd.Period(last_entry['YearMonth'], freq='M')
    predicted_month = last_month + 1
    return pred, predicted_month

#Accuracy in %
from sklearn.metrics import r2_score
preds = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test,preds))
r2 = r2_score(y_test,preds)

print(f"Improved Test RMSE: {rmse:.2f}")
print(f"Model R² (accuracy): {r2:.4f} ({r2*100:.2f}%)")
    
# 10. Example usage
customer_to_search = "Macroservice"
prediction, month = predict_next_month_revenue(customer_to_search)
if prediction is not None:
    print(f"Predicted revenue for '{customer_to_search}' in {month.strftime('%B %Y')}: ${prediction:,.2f}")
else:
    print(f"No data available for customer '{customer_to_search}'.")


# --- Streamlit UI ---

import streamlit as st

st.title("Steeve and Associates Revenue Prediction")

customer_input = st.text_input("Enter Customer Name:")

if customer_input:
    prediction, month = predict_next_month_revenue(customer_input)
    if prediction is None:
        st.error(f"No data available for customer '{customer_input}'. Please check the name and try again.")
    else:
        st.success(f"Predicted revenue for '{customer_input}' in {month.strftime('%B %Y')}: ${prediction:,.2f}")