
from pathlib import Path
import pandas as pd

# Load dataset
data_path = Path(__file__).parent / "insurance_claims.csv"
df = pd.read_csv(data_path)

# Dataset overview
print("\n--- DATASET OVERVIEW ---")
print(f"Number of claims: {len(df):,}")
print(f"Number of columns: {df.shape[1]}")

# Data quality
print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- DUPLICATE CLAIM IDs ---")
print(df["claim_id"].duplicated().sum())

# Target distribution
print("\n--- FRAUD DISTRIBUTION ---")
print(df["is_fraud"].value_counts())

print(f"\nFraud rate: {df['is_fraud'].mean():.2%}")

# Compare fraudulent and non-fraudulent claims
features = [
    "claim_amount",
    "policy_age_days",
    "previous_claims",
    "days_to_report",
    "police_report",
    "witness_count",
    "claim_to_value_ratio"
]

print("\n--- FRAUD VS NON-FRAUD COMPARISON ---")
print(df.groupby("is_fraud")[features].mean().T.round(3))

# Fraud rate by claim type
print("\n--- FRAUD RATE BY CLAIM TYPE ---")
print(
    df.groupby("claim_type")["is_fraud"]
    .mean()
    .mul(100)
    .round(2)
    .sort_values(ascending=False)
)

# Correlation with fraud
print("\n--- CORRELATION WITH FRAUD ---")
print(
    df[features + ["is_fraud"]]
    .corr()["is_fraud"]
    .sort_values(ascending=False)
    .round(3)
)
