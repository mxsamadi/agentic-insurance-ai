from guardrails.citation_validator import validate_citations

from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from workflows.state import InvestigationState


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)


class ValidationResult(BaseModel):

    passed: bool

    issues: list[str] = Field(
        description="Unsupported claims or validation issues"
    )

    feedback: str

    human_review_required: bool


validator = llm.with_structured_output(ValidationResult)


def validation_agent(state: InvestigationState) -> dict:
    """Validate investigation results and agent analyses."""

    attempts = state.get("validation_attempts", 0) + 1

    required = state.get("required_agents", [])
    completed = state.get("completed_agents", [])

    issues = list(state.get("errors", []))

    # Check that required agents completed their work
    for agent in required:
        if agent not in completed:
            issues.append(f"Missing agent result: {agent}")

    # Map agents to their expected results
    result_keys = {
        "risk_agent": "risk_result",
        "anomaly_agent": "anomaly_result",
        "investigation_agent": "investigation_result"
    }

    # Validate required tool outputs
    for agent in required:
        result = state.get(result_keys[agent])

        if not result or "error" in result:
            issues.append(f"Invalid result from {agent}")

        elif result.get("claim_id") != state["claim_id"]:
            issues.append(f"Claim ID mismatch in {agent}")

    if not required:
        issues.append("No investigation plan available")

    # Check the risk score range
    risk = state.get("risk_result")

    if risk and "error" not in risk:
        score = risk.get("risk_score")

        if not isinstance(score, (int, float)):
            issues.append("Invalid risk score")

        elif not 0 <= score <= 1:
            issues.append("Risk score outside valid range")

    
    # Deterministic RAG citation validation
    if "investigation_agent" in required:

        investigation = (
            state.get("investigation_result") or {}
        )

        guidelines = investigation.get(
            "retrieved_guidelines", []
        )

        analysis = state.get(
            "investigation_analysis", ""
        )

        citation_result = validate_citations(
            analysis,
            guidelines
        )

        if not citation_result["passed"]:

            issues.append(
                "Deterministic citation validation failed."
            )

            if citation_result["invalid_citations"]:

                issues.append(
                    "Unauthorized citations: "
                    + ", ".join(
                        citation_result["invalid_citations"]
                    )
                )

            if citation_result["missing_citations"]:

                issues.append(
                    "Required source citations are missing."
                )

            if not guidelines:

                issues.append(
                    "No retrieved guidelines available."
                )


    # Deterministic RAG citation validation
    if "investigation_agent" in required:

        investigation = (
            state.get("investigation_result") or {}
        )

        guidelines = investigation.get(
            "retrieved_guidelines", []
        )

        analysis = state.get(
            "investigation_analysis", ""
        )

        citation_result = validate_citations(
            analysis,
            guidelines
        )

        if not citation_result["passed"]:

            issues.append(
                "Deterministic citation validation failed."
            )

            if citation_result["invalid_citations"]:

                issues.append(
                    "Unauthorized citations: "
                    + ", ".join(
                        citation_result["invalid_citations"]
                    )
                )

            if citation_result["missing_citations"]:

                issues.append(
                    "Required source citations are missing."
                )

            if not guidelines:

                issues.append(
                    "No retrieved guidelines available."
                )


    # Stop if deterministic validation fails
    if issues:
        return {
            "validation_passed": False,
            "validation_feedback": "; ".join(issues),
            "validation_attempts": attempts,
            "human_review_required": True,
            "execution_log": [
                "Deterministic validation failed"
            ]
        }

    # LLM-based validation
    prompt = f"""
    You are an independent AI model validation agent.

    User request:
    {state["question"]}

    Claim ID:
    {state["claim_id"]}

    XGBoost result:
    {state.get("risk_result")}

    Risk analysis:
    {state.get("risk_analysis")}

    Isolation Forest result:
    {state.get("anomaly_result")}

    Anomaly analysis:
    {state.get("anomaly_analysis")}

    Claim investigation result:
    {state.get("investigation_result")}

    Investigation analysis:
    {state.get("investigation_analysis")}

    Evaluate whether:
    1. Analyses are supported by tool results.
    2. Numerical results are reported accurately.
    3. No claim details or evidence were fabricated.
    4. Anomalies are distinguished from fraud.
    5. Risk scores are not presented as calibrated
       fraud probabilities.
    6. No automatic claim denial is recommended.
    7. Synthetic data limitations are acknowledged.
    8. The requested task has been addressed.

    Missing results are acceptable when the corresponding
    agent was not required for the user's request.

    Fail validation if material issues are identified.
    Recommend human review when appropriate.
    """

    review = validator.invoke(prompt)

    # Apply additional deterministic review rules
    anomaly = state.get("anomaly_result") or {}

    human_review = (
        review.human_review_required
        or not review.passed
        or bool(review.issues)
        or (risk or {}).get("risk_level") in ["Medium", "High"]
        or anomaly.get("is_anomaly", False)
    )

    passed = review.passed and not review.issues

    return {
        "validation_passed": passed,
        "validation_feedback": (
            review.feedback + " " + "; ".join(review.issues)
        ).strip(),
        "validation_attempts": attempts,
        "human_review_required": human_review,
        "execution_log": [
            f"Validation attempt {attempts}: "
            f"{'Passed' if passed else 'Failed'}"
        ]
    }


if __name__ == "__main__":

    state = {
        "claim_id": "CLM-000001",
        "question": "Assess the fraud risk of this claim.",
        "required_agents": ["risk_agent"],
        "completed_agents": ["risk_agent"],
        "risk_result": {
            "claim_id": "CLM-000001",
            "risk_score": 0.3138,
            "risk_level": "Low",
            "model": "XGBoost"
        },
        "risk_analysis": (
            "The claim has a low risk score of 0.3138. "
            "This is not a calibrated fraud probability. "
            "The assessment uses synthetic data and "
            "does not establish fraud."
        )
    }

    result = validation_agent(state)

    print("\n--- VALIDATION RESULT ---")

    for key, value in result.items():
        print(f"{key}: {value}")
