import pandas as pd
import sys

try:
    df = pd.read_parquet("./initial_expedition/train-00000-of-00011.parquet")
    print("Columns:", df.columns.tolist())
    print("\nFirst row keys:", df.iloc[0].keys())
    print("\nSample 'hobbies' related columns:")
    for col in df.columns:
        if 'hobbies' in col or 'interest' in col:
            print(f"--- {col} ---")
            print(df[col].iloc[0])
            print("Type:", type(df[col].iloc[0]))
except Exception as e:
    print(e)
