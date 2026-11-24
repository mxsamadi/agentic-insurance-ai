
from pathlib import Path
from functools import lru_cache

import joblib
from langchain_core.tools import tool

from tools.data_utils import get_claim, prepare_features

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def load_anomaly_model():
    """Load the trained Isolation Forest pipeline."""
    return joblib.load(
        ROOT / "models/anomaly_model.joblib"
    )


@tool
def detect_claim_anomaly(claim_id: str) -> dict:
    """Detect unusual characteristics in an insurance
    claim using a trained Isolation Forest model.

    Args:
        claim_id: Unique claim ID, such as CLM-000001.
    """

    try:
        claim = get_claim(claim_id)
    except ValueError as error:
        return {"error": str(error)}

    features = prepare_features(claim)

    model = load_anomaly_model()

    anomaly_score = float(
        -model.score_samples(features)[0]
    )

    prediction = int(model.predict(features)[0])

    is_anomaly = prediction == -1

    return {
        "claim_id": claim_id,
        "anomaly_score": round(anomaly_score, 4),
        "is_anomaly": is_anomaly,
        "anomaly_status": (
            "Unusual" if is_anomaly else "Normal"
        ),
        "model": "Isolation Forest",
        "disclaimer": (
            "An unusual claim is not necessarily "
            "fraudulent."
        )
    }
