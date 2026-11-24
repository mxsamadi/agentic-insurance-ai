
from pathlib import Path
from functools import lru_cache

import joblib
from langchain_core.tools import tool

from tools.data_utils import get_claim, prepare_features

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def load_risk_model():
    """Load the trained XGBoost pipeline."""
    return joblib.load(
        ROOT / "models/risk_model.joblib"
    )


@tool
def assess_fraud_risk(claim_id: str) -> dict:
    """Assess the fraud risk of an insurance claim
    using a trained XGBoost model.

    Args:
        claim_id: Unique claim ID, such as CLM-000001.
    """

    try:
        claim = get_claim(claim_id)
    except ValueError as error:
        return {"error": str(error)}

    features = prepare_features(claim)

    model = load_risk_model()

    risk_score = float(
        model.predict_proba(features)[0, 1]
    )

    if risk_score >= 0.7:
        risk_level = "High"
    elif risk_score >= 0.4:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        "claim_id": claim_id,
        "risk_score": round(risk_score, 4),
        "risk_level": risk_level,
        "model": "XGBoost",
        "claim_amount": float(
            claim["claim_amount"].iloc[0]
        ),
        "disclaimer": (
            "Uncalibrated model score based on synthetic "
            "data. Not proof of fraud."
        )
    }
