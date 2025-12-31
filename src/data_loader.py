import pandas as pd

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Time" in df.columns:
        df["Time_hours"] = df["Time"] / 3600
    return df
