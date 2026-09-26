"""
data_preprocessing.py
----------------------
Cleaning and feature-preparation functions shared by the notebook,
the training script, and the prediction script — so cleaning logic
lives in exactly one place.
"""

import pandas as pd
import numpy as np

CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
TARGET_COL = "Churn"
ID_COL = "customerID"


def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fix the real-world messiness injected into the raw data."""
    df = df.copy()

    # Drop exact duplicate rows
    df = df.drop_duplicates()

    # Normalize categorical text: strip whitespace, fix casing
    for col in CATEGORICAL_COLS + [TARGET_COL]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    df["gender"] = df["gender"].str.capitalize()

    # TotalCharges should be numeric; coerce and fill missing with tenure * MonthlyCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    fill_mask = df["TotalCharges"].isna()
    df.loc[fill_mask, "TotalCharges"] = (
        df.loc[fill_mask, "tenure"] * df.loc[fill_mask, "MonthlyCharges"]
    )

    # Drop rows missing the target (shouldn't happen, but defensive)
    df = df.dropna(subset=[TARGET_COL])

    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a few analyst-friendly engineered features."""
    df = df.copy()

    df["AvgMonthlySpend"] = np.where(
        df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"]
    )

    def tenure_bucket(t):
        if t <= 12:
            return "0-1 yr"
        elif t <= 24:
            return "1-2 yr"
        elif t <= 48:
            return "2-4 yr"
        else:
            return "4+ yr"

    df["TenureGroup"] = df["tenure"].apply(tenure_bucket)

    df["NumServices"] = (
        (df["OnlineSecurity"] == "Yes").astype(int)
        + (df["OnlineBackup"] == "Yes").astype(int)
        + (df["DeviceProtection"] == "Yes").astype(int)
        + (df["TechSupport"] == "Yes").astype(int)
        + (df["StreamingTV"] == "Yes").astype(int)
        + (df["StreamingMovies"] == "Yes").astype(int)
    )

    return df


def prepare_model_frame(df: pd.DataFrame):
    """
    Returns (X, y) ready for a scikit-learn pipeline:
    - y is binary (1 = Churn, 0 = No churn)
    - X keeps raw categorical/numeric columns; one-hot encoding happens
      inside the modeling pipeline (see train_model.py) to avoid leakage
      between train/test splits.
    """
    df = df.copy()
    y = (df[TARGET_COL] == "Yes").astype(int)
    feature_cols = CATEGORICAL_COLS + NUMERIC_COLS + ["AvgMonthlySpend", "TenureGroup", "NumServices"]
    feature_cols = [c for c in feature_cols if c in df.columns]
    X = df[feature_cols]
    return X, y


def full_pipeline(raw_path: str):
    """Convenience: raw CSV -> cleaned, feature-engineered, model-ready X/y."""
    df = load_raw(raw_path)
    df = clean_data(df)
    df = engineer_features(df)
    X, y = prepare_model_frame(df)
    return df, X, y
