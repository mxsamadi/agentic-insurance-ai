
from pathlib import Path
from functools import lru_cache
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def load_claims():
    """Load the insurance claims dataset."""
    return pd.read_csv(ROOT / "data/insurance_claims.csv")


def get_claim(claim_id: str):
    """Retrieve a claim using its unique identifier."""

    df = load_claims()

    claim = df[df["claim_id"] == claim_id]

    if claim.empty:
        raise ValueError(f"Claim {claim_id} not found.")

    return claim.copy()


def prepare_features(claim):
    """Prepare features for machine learning inference."""

    features = claim.copy()

    features["claim_to_repair_ratio"] = (
        features["claim_amount"] /
        features["repair_estimate"]
    )

    features = features.drop(
        columns=["claim_id", "policy_id", "is_fraud"],
        errors="ignore"
    )

    return features
