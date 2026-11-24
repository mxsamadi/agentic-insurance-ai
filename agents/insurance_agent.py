
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from tools.risk_tools import assess_fraud_risk
from tools.anomaly_tools import detect_claim_anomaly
from tools.investigation_tools import investigate_claim

# Load environment variables
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# Initialize the LLM
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)

# Register available tools
tools = [
    assess_fraud_risk,
    detect_claim_anomaly,
    investigate_claim
]

# Define agent instructions
SYSTEM_PROMPT = """
You are an insurance fraud investigation assistant.

Your responsibilities:
- Assess insurance claim risk.
- Identify unusual claim characteristics.
- Investigate potential risk indicators.
- Provide evidence-based recommendations.

Select the appropriate tools based on the user's request.

For a comprehensive investigation, use all three tools.

Rules:
- Never invent claim information or model scores.
- Use tool results as the source of factual information.
- Report tool errors honestly.
- Do not interpret uncalibrated scores as fraud probabilities.
- Distinguish anomalies from fraud.
- Never declare that a policyholder committed fraud.
- Never recommend automatic claim denial.
- Recommend human review when appropriate.
- Clearly state that the data is synthetic.
"""

# Create the AI agent
insurance_agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT
)


def run_investigation(question: str) -> str:
    """Execute an insurance investigation."""

    result = insurance_agent.invoke(
        {
            "messages": [
                {"role": "user", "content": question}
            ]
        },
        config={"recursion_limit": 20}
    )

    # Display tool execution history
    print("\n--- AGENT EXECUTION TRACE ---")

    for message in result["messages"]:
        if getattr(message, "tool_calls", None):
            for call in message.tool_calls:
                print(f"Tool selected: {call['name']}")
                print(f"Arguments: {call['args']}")

        elif message.type == "tool":
            print(f"Tool completed: {message.name}")

    return result["messages"][-1].content


if __name__ == "__main__":

    question = (
        "Conduct a comprehensive fraud investigation "
        "for claim CLM-000001."
    )

    response = run_investigation(question)

    print("\n--- INVESTIGATION REPORT ---")
    print(response)
