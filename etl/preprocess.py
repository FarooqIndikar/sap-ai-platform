import pandas as pd
import numpy as np

def load_data(path):
    df = pd.read_csv(path)
    return df

def clean_data(df):
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["due_days"] = pd.to_numeric(df["due_days"], errors="coerce")
    return df

def feature_engineering(df):
    df["high_value"] = np.where(df["amount"] > 100000, 1, 0)
    df["risk_score"] = df["amount"] * 0.6 + df["due_days"] * 0.4
    df["target"] = np.where(df["status"]=="unpaid", 1, 0)
    return df

if __name__ == "__main__":
    df = load_data("data/sap_data.csv")
    df = clean_data(df)
    df = feature_engineering(df)
    print(df.head())