
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from workflows.state import InvestigationState
from tools.anomaly_tools import detect_claim_anomaly


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)


def anomaly_agent(state: InvestigationState) -> dict:
    """Detect and interpret unusual claim characteristics."""

    claim_id = state["claim_id"]

    # Execute the Isolation Forest tool
    result = detect_claim_anomaly.invoke(
        {"claim_id": claim_id}
    )

    # Handle tool errors
    if "error" in result:
        return {
            "anomaly_result": result,
            "errors": [result["error"]],
            "execution_log": ["Anomaly Agent failed"]
        }

    # Interpret the anomaly detection results
    prompt = f"""
    You are a specialized insurance anomaly detection agent.

    Analyze the following Isolation Forest results:

    {result}

    Provide:
    1. A summary of the anomaly detection results.
    2. An interpretation of the anomaly score and status.
    3. A recommendation for further investigation.

    Rules:
    - Do not invent information.
    - An anomaly does not necessarily indicate fraud.
    - The anomaly score is not a fraud probability.
    - Do not invent feature-level explanations.
    - Do not recommend automatic claim denial.
    - Clearly mention that the data is synthetic.
    """

    response = llm.invoke(prompt)

    # Update the shared LangGraph state
    return {
        "anomaly_result": result,
        "anomaly_analysis": response.content,
        "completed_agents": ["anomaly_agent"],
        "execution_log": [
            f"Anomaly Agent completed assessment for {claim_id}"
        ]
    }


if __name__ == "__main__":

    state = {
        "claim_id": "CLM-000001",
        "question": "Identify unusual claim characteristics."
    }

    result = anomaly_agent(state)

    print("\n--- ANOMALY MODEL RESULT ---")
    print(result["anomaly_result"])

    print("\n--- AI ANOMALY ANALYSIS ---")
    print(result.get("anomaly_analysis", "Analysis unavailable"))

    print("\n--- EXECUTION LOG ---")
    print(result["execution_log"])
