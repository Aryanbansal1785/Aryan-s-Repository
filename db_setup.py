import pandas as pd
from sqlalchemy import create_engine
import os

def load_file_to_db(filepath: str):
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == ".csv":
        df = pd.read_csv(filepath, encoding="latin1")
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    
    
    df.columns = df.columns.str.replace(" ", "_").str.lower()
    
    
    os.makedirs("db", exist_ok=True)
    engine = create_engine("sqlite:///db/data.db")
    df.to_sql("uploaded_data", engine, if_exists="replace", index=False)
    
    print("Loaded:", len(df), "rows")
    print("Columns:", list(df.columns))
    
    return df.columns.tolist(), len(df)



