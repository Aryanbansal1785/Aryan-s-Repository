"""
Synthetic transactions generator with controlled, documented data quality issues.
This generator creates a pandas DataFrame with deliberate errors: missing values,
multiple date formats, inconsistent country/province spellings, and near-duplicates.
"""
import pandas as pd
from pathlib import Path
import random
from datetime import datetime

SAMPLE_PATH = Path(__file__).resolve().parents[1] / "sample_data" / "transactions_sample.csv"


def load_sample_csv():
    return pd.read_csv(SAMPLE_PATH)

def generate_transactions(n=100, random_seed=42):
    random.seed(random_seed)
    ids = [f"T{str(i+1).zfill(4)}" for i in range(n)]
    accounts = [f"A{1000 + (i % 200)}" for i in range(n)]
    amounts = [round(random.uniform(1, 1000), 2) for _ in range(n)]

    # Introduce some missing amounts
    for i in range(0, n, 25):
        amounts[i] = None

    currencies = ["USD" for _ in range(n)]

    # Mix date formats
    base_date = datetime(2023, 12, 1)
    dates = []
    for i in range(n):
        d = base_date
        if i % 3 == 0:
            dates.append(d.strftime("%Y/%m/%d"))
        elif i % 3 == 1:
            dates.append(d.strftime("%m-%d-%Y"))
        else:
            dates.append(d.strftime("%d %b %Y"))

    merchants = [random.choice(["Acme Store", "Example Mart", "Corner Shop", "BestBuy"]) for _ in range(n)]

    # Country/province inconsistencies
    countries = [random.choice(["USA", "United States", "United States of America", "US"]) for _ in range(n)]
    provinces = [random.choice(["CA", "California", "ca", "ON", "Ontario"]) for _ in range(n)]

    df = pd.DataFrame({
        "txn_id": ids,
        "account_id": accounts,
        "amount": amounts,
        "currency": currencies,
        "txn_date": dates,
        "merchant": merchants,
        "country": countries,
        "province": provinces,
    })

    # Introduce near-duplicates by copying some rows and slightly mutating
    for i in range(0, min(5, n)):
        row = df.iloc[i].copy()
        row['txn_id'] = f"DUP-{row['txn_id']}"
        # slight merchant variation
        row['merchant'] = row['merchant'].lower() if random.random() < 0.5 else row['merchant']
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    return df

if __name__ == '__main__':
    df = generate_transactions(50)
    print(df.head())
