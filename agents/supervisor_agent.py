
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Literal

from workflows.state import InvestigationState


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)


class SupervisorPlan(BaseModel):

    required_agents: list[
        Literal[
            "risk_agent",
            "anomaly_agent",
            "investigation_agent"
        ]
    ] = Field(
        description="Specialized agents required for the task"
    )

    reasoning: str = Field(
        description="Brief justification for agent selection"
    )


planner = llm.with_structured_output(SupervisorPlan)


def supervisor_agent(state: InvestigationState) -> dict:
    """Plan and coordinate the investigation."""

    question = state["question"]
    claim_id = state["claim_id"]

    completed = state.get("completed_agents", [])
    required = state.get("required_agents", [])

    # Stop delegation if an error has occurred
    if state.get("errors"):
        return {
            "next_agent": "validation_agent",
            "execution_log": [
                "Supervisor detected an error"
            ]
        }

    # Create an investigation plan if one does not exist
    if not required:

        prompt = f"""
        You are the supervisor of an insurance
        fraud investigation system.

        User request:
        {question}

        Claim ID:
        {claim_id}

        Available agents:

        risk_agent:
        Calculates XGBoost fraud risk scores.

        anomaly_agent:
        Detects unusual claims using Isolation Forest.

        investigation_agent:
        Examines claim details and risk indicators.

        Select the agents necessary to answer the request.

        Rules:
        - For risk score requests, select risk_agent.
        - For anomaly requests, select anomaly_agent.
        - For claim details or risk indicators,
          select investigation_agent.
        - For comprehensive fraud investigations,
          select all three agents.
        - Select at least one agent.
        - Do not select duplicate agents.
        """

        plan = planner.invoke(prompt)

        # Enforce a consistent execution order
        agent_order = [
            "risk_agent",
            "anomaly_agent",
            "investigation_agent"
        ]

        required = [
            agent for agent in agent_order
            if agent in plan.required_agents
        ]

        # Fallback if the LLM returns an empty plan
        if not required:
            required = agent_order.copy()

        planning_log = [
            f"Supervisor plan: {required}",
            f"Planning justification: {plan.reasoning}"
        ]

    else:
        planning_log = []

    # Identify agents that have not completed their work
    remaining = [
        agent for agent in required
        if agent not in completed
    ]

    if remaining:

        next_agent = remaining[0]

        return {
            "required_agents": required,
            "next_agent": next_agent,
            "execution_log": planning_log + [
                f"Supervisor delegated to {next_agent}"
            ]
        }

    # All required agents have completed their work
    return {
        "required_agents": required,
        "next_agent": "validation_agent",
        "execution_log": planning_log + [
            "All required agents completed. "
            "Routing to validation."
        ]
    }


if __name__ == "__main__":

    state = {
        "claim_id": "CLM-000001",
        "question": (
            "Conduct a comprehensive fraud investigation "
            "for claim CLM-000001."
        ),
        "completed_agents": []
    }

    result = supervisor_agent(state)

    print("\n--- SUPERVISOR PLAN ---")
    print(result["required_agents"])

    print("\n--- NEXT AGENT ---")
    print(result["next_agent"])

    print("\n--- EXECUTION LOG ---")
    print(result["execution_log"])
