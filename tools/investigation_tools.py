
from langchain_core.tools import tool
from tools.data_utils import get_claim


@tool
def investigate_claim(claim_id: str) -> dict:
    """Retrieve claim details and identify potential
    risk indicators for an insurance investigation.

    Args:
        claim_id: Unique claim ID, such as CLM-000001.
    """

    try:
        claim = get_claim(claim_id).iloc[0]
    except ValueError as error:
        return {"error": str(error)}

    risk_indicators = []

    if claim["policy_age_days"] < 90:
        risk_indicators.append("Recently purchased policy")

    if claim["previous_claims"] >= 3:
        risk_indicators.append("Multiple previous claims")

    if claim["claim_to_value_ratio"] > 0.8:
        risk_indicators.append("High claim-to-value ratio")

    if claim["days_to_report"] > 10:
        risk_indicators.append("Delayed claim reporting")

    if claim["police_report"] == 0:
        risk_indicators.append("No police report")

    if claim["witness_count"] == 0:
        risk_indicators.append("No witnesses")

    if claim["claim_amount"] > 1.8 * claim["repair_estimate"]:
        risk_indicators.append("Claim and repair cost discrepancy")

    return {
        "claim_id": claim_id,
        "claim_type": str(claim["claim_type"]),
        "claim_amount": float(claim["claim_amount"]),
        "vehicle_value": float(claim["vehicle_value"]),
        "repair_estimate": float(claim["repair_estimate"]),
        "policy_age_days": int(claim["policy_age_days"]),
        "previous_claims": int(claim["previous_claims"]),
        "days_to_report": int(claim["days_to_report"]),
        "police_report": bool(claim["police_report"]),
        "witness_count": int(claim["witness_count"]),
        "claim_to_value_ratio": float(
            claim["claim_to_value_ratio"]
        ),
        "risk_indicators": risk_indicators,
        "number_of_indicators": len(risk_indicators),
        "disclaimer": (
            "Risk indicators require human review "
            "and do not establish fraud."
        )
    }
