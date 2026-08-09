"""Reusable preprocessing helpers for the churn project."""

from pathlib import Path
import pandas as pd


DEFAULT_DATA_PATH = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")


def load_data(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the raw Telco churn CSV and normalize numeric fields."""
    df = pd.read_csv(path)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy with invalid TotalCharges rows removed."""
    cleaned = df.copy()
    if "TotalCharges" in cleaned.columns:
        cleaned = cleaned.dropna(subset=["TotalCharges"])
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    return cleaned
