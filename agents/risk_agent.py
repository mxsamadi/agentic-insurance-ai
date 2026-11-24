
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from workflows.state import InvestigationState
from tools.risk_tools import assess_fraud_risk


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)


def risk_agent(state: InvestigationState) -> dict:
    """Assess and interpret insurance claim risk."""

    claim_id = state["claim_id"]

    # Execute the XGBoost risk assessment tool
    result = assess_fraud_risk.invoke(
        {"claim_id": claim_id}
    )

    # Handle tool errors
    if "error" in result:
        return {
            "risk_result": result,
            "errors": [result["error"]],
            "execution_log": ["Risk Agent failed"]
        }

    # Interpret the model results using the LLM
    prompt = f"""
    You are a specialized insurance risk assessment agent.

    Analyze the following XGBoost model results:

    {result}

    Provide:
    1. A summary of the risk assessment.
    2. An interpretation of the risk score and level.
    3. A recommendation for further investigation.

    Rules:
    - Do not invent information.
    - The risk score is not a calibrated fraud probability.
    - Do not claim that fraud has occurred.
    - Do not recommend automatic claim denial.
    - Clearly mention that the data is synthetic.
    """

    response = llm.invoke(prompt)

    # Update the shared LangGraph state
    return {
        "risk_result": result,
        "risk_analysis": response.content,
        "completed_agents": ["risk_agent"],
        "execution_log": [
            f"Risk Agent completed assessment for {claim_id}"
        ]
    }


if __name__ == "__main__":

    state = {
        "claim_id": "CLM-000001",
        "question": "Assess the fraud risk of this claim."
    }

    result = risk_agent(state)

    print("\n--- RISK MODEL RESULT ---")
    print(result["risk_result"])

    print("\n--- AI RISK ANALYSIS ---")
    print(result.get("risk_analysis", "Analysis unavailable"))

    print("\n--- EXECUTION LOG ---")
    print(result["execution_log"])
