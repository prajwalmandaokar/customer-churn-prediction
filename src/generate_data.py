"""
generate_data.py
----------------
Generates a realistic, synthetic telecom customer-churn dataset and saves it to
data/raw/customer_churn.csv

This mimics the structure of the well-known "Telco Customer Churn" style
datasets used across the data-analytics community, but every row here is
synthetically generated (with intentional real-world messiness) so the
project is fully self-contained and license-free.

Run directly:
    python src/generate_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42
N_CUSTOMERS = 5000

OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "customer_churn.csv"


def generate_dataset(n=N_CUSTOMERS, seed=RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    customer_id = [f"CUST-{10000 + i}" for i in range(n)]
    gender = rng.choice(["Male", "Female"], size=n)
    senior_citizen = rng.choice([0, 1], size=n, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], size=n, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], size=n, p=[0.30, 0.70])

    tenure_months = rng.integers(0, 73, size=n)  # 0-72 months

    phone_service = rng.choice(["Yes", "No"], size=n, p=[0.90, 0.10])
    multiple_lines = np.where(
        phone_service == "No",
        "No phone service",
        rng.choice(["Yes", "No"], size=n),
    )

    internet_service = rng.choice(
        ["DSL", "Fiber optic", "No"], size=n, p=[0.34, 0.44, 0.22]
    )

    def dependent_internet_feature(p_yes=0.5):
        vals = np.where(
            internet_service == "No",
            "No internet service",
            rng.choice(["Yes", "No"], size=n, p=[p_yes, 1 - p_yes]),
        )
        return vals

    online_security = dependent_internet_feature(0.35)
    online_backup = dependent_internet_feature(0.40)
    device_protection = dependent_internet_feature(0.40)
    tech_support = dependent_internet_feature(0.35)
    streaming_tv = dependent_internet_feature(0.45)
    streaming_movies = dependent_internet_feature(0.45)

    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.24, 0.21]
    )
    paperless_billing = rng.choice(["Yes", "No"], size=n, p=[0.59, 0.41])
    payment_method = rng.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        size=n,
    )

    # --- Monthly charges depend loosely on services selected (more services = higher bill)
    base = rng.normal(35, 8, size=n)
    internet_add = np.select(
        [internet_service == "DSL", internet_service == "Fiber optic", internet_service == "No"],
        [20, 45, 0],
    )
    extra_services = sum(
        np.where(col == "Yes", rng.uniform(3, 8, size=n), 0)
        for col in [
            online_security, online_backup, device_protection,
            tech_support, streaming_tv, streaming_movies,
        ]
    )
    monthly_charges = np.clip(base + internet_add + extra_services, 18, 145).round(2)

    # total charges roughly tenure * monthly (with noise), some early-tenure NaNs (real-world messiness)
    total_charges = (monthly_charges * tenure_months * rng.uniform(0.92, 1.05, size=n)).round(2)
    total_charges = np.where(tenure_months == 0, 0.0, total_charges)

    # --- Churn probability model (this is what makes the ML meaningful/learnable) ---
    logit = (
        -1.6
        + 1.9 * (contract == "Month-to-month")
        + 0.35 * (contract == "One year")
        - 0.05 * tenure_months
        + 0.012 * monthly_charges
        + 0.9 * (internet_service == "Fiber optic")
        - 0.7 * (online_security == "Yes")
        - 0.6 * (tech_support == "Yes")
        + 0.5 * (paperless_billing == "Yes")
        + 0.6 * (payment_method == "Electronic check")
        + 0.4 * (senior_citizen == 1)
        - 0.3 * (partner == "Yes")
        - 0.3 * (dependents == "Yes")
    )
    prob_churn = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, prob_churn)
    churn_label = np.where(churn == 1, "Yes", "No")

    df = pd.DataFrame(
        {
            "customerID": customer_id,
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure_months,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Churn": churn_label,
        }
    )

    # --- Inject realistic data-quality issues (great for portfolio storytelling) ---
    # 1) A handful of TotalCharges blanks (mirrors the real Telco dataset's known quirk)
    blank_idx = rng.choice(n, size=11, replace=False)
    df.loc[blank_idx, "TotalCharges"] = np.nan

    # 2) A few duplicate rows
    dup_rows = df.sample(15, random_state=seed)
    df = pd.concat([df, dup_rows], ignore_index=True)

    # 3) Inconsistent casing / whitespace in a categorical column (common real-world mess)
    messy_idx = rng.choice(df.index, size=40, replace=False)
    df.loc[messy_idx, "gender"] = df.loc[messy_idx, "gender"].str.upper() + "  "

    # 4) Shuffle rows
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return df


if __name__ == "__main__":
    df = generate_dataset()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Generated {len(df):,} rows -> {OUT_PATH}")
    print(f"Churn rate: {(df['Churn'] == 'Yes').mean():.2%}")
