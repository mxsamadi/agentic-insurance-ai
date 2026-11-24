
from pathlib import Path
import numpy as np
import pandas as pd

# Reproducibility
rng = np.random.default_rng(42)

n = 20000

# Generate synthetic insurance claims
df = pd.DataFrame({
    "claim_id": [f"CLM-{i:06d}" for i in range(1, n + 1)],
    "policy_id": rng.integers(10000, 25000, n),
    "policy_age_days": rng.integers(1, 3650, n),
    "annual_premium": rng.uniform(600, 3500, n).round(2),
    "claim_amount": rng.lognormal(8.5, 0.9, n).round(2),
    "vehicle_value": rng.uniform(5000, 80000, n).round(2),
    "claim_type": rng.choice(
        ["Collision", "Theft", "Fire", "Other"],
        size=n,
        p=[0.65, 0.15, 0.05, 0.15]
    ),
    "previous_claims": rng.poisson(0.8, n),
    "days_to_report": rng.exponential(3, n).astype(int),
    "police_report": rng.binomial(1, 0.75, n),
    "witness_count": rng.poisson(0.7, n)
})

# Generate repair estimates
df["repair_estimate"] = (
    df["claim_amount"] * rng.uniform(0.4, 1.2, n)
).round(2)

# Feature engineering
df["claim_to_value_ratio"] = (
    df["claim_amount"] / df["vehicle_value"]
).round(4)

# Simulate fraud probability using risk indicators
log_odds = (
    -3.8
    + 1.5 * (df["policy_age_days"] < 90)
    + 1.3 * (df["previous_claims"] >= 3)
    + 1.4 * (df["claim_to_value_ratio"] > 0.8)
    + 0.8 * (df["police_report"] == 0)
    + 0.9 * (df["days_to_report"] > 10)
    + 0.5 * (df["witness_count"] == 0)
    + 0.7 * (
        df["claim_amount"] > 1.8 * df["repair_estimate"]
    )
)

# Convert log-odds into probabilities
fraud_probability = 1 / (1 + np.exp(-log_odds))

# Generate probabilistic fraud labels
df["is_fraud"] = rng.binomial(1, fraud_probability)

# Save the dataset
output_path = Path(__file__).parent / "insurance_claims.csv"
df.to_csv(output_path, index=False)

# Display dataset summary
print("Dataset generated successfully!")
print(f"Total claims: {len(df):,}")
print(f"Fraudulent claims: {df['is_fraud'].sum():,}")
print(f"Fraud rate: {df['is_fraud'].mean():.2%}")
print(f"Saved to: {output_path}")

print("\nFirst five records:")
print(df.head())
