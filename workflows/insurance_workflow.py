
from langgraph.graph import StateGraph, START, END
import re

from workflows.state import InvestigationState

from agents.supervisor_agent import supervisor_agent
from agents.risk_agent import risk_agent
from agents.anomaly_agent import anomaly_agent
from agents.investigation_agent import investigation_agent
from agents.validation_agent import validation_agent


def safe_node(agent, name):
    """Handle agent execution failures."""

    def execute(state):
        try:
            return agent(state)
        except Exception as error:
            return {
                "errors": [
                    f"{name} failed: {type(error).__name__}"
                ],
                "execution_log": [
                    f"{name} execution failed"
                ],
                "validation_passed": False,
                "human_review_required": True
            }

    return execute


def route_supervisor(state: InvestigationState) -> str:
    """Determine which agent executes next."""

    if state.get("errors"):
        return "validation_agent"

    return state["next_agent"]


def generate_report(state: InvestigationState) -> dict:
    """Generate the final report from validated results."""

    claim_id = state["claim_id"]

    if not state.get("validation_passed", False):

        report = (
            f"INVESTIGATION REPORT: {claim_id}\n\n"
            "Status: Validation failed.\n"
            "Human review required.\n\n"
            f"Validation feedback: "
            f"{state.get('validation_feedback', 'Unavailable')}\n\n"
            f"Errors: {state.get('errors', [])}\n\n"
            "No automated claim decision has been made."
        )

        return {
            "final_report": report,
            "human_review_required": True,
            "execution_log": [
                "Unvalidated results withheld"
            ]
        }

    sections = [
        f"INVESTIGATION REPORT: {claim_id}",
        "Status: Validation passed"
    ]

    analyses = {
        "risk_analysis": "RISK ASSESSMENT",
        "anomaly_analysis": "ANOMALY DETECTION",
        "investigation_analysis": "CLAIM INVESTIGATION"
    }

    for key, heading in analyses.items():

        if state.get(key):
            sections.append(
                f"{heading}\n{state[key]}"
            )

    sections.append(
        "VALIDATION FEEDBACK\n"
        + state.get("validation_feedback", "")
    )

    sections.append(
        "HUMAN REVIEW FLAG: "
        + (
            "Required"
            if state.get("human_review_required")
            else "Not triggered"
        )
    )

    sections.append(
        "DISCLAIMER\n"
        "This report uses synthetic insurance data. "
        "Risk scores and anomalies are not proof of fraud. "
        "No automated claim decision has been made."
    )

    return {
        "final_report": "\n\n".join(sections),
        "execution_log": ["Final report generated"]
    }


# Initialize LangGraph
workflow = StateGraph(InvestigationState)

# Register agents as graph nodes
workflow.add_node(
    "supervisor_agent",
    safe_node(supervisor_agent, "Supervisor Agent")
)

workflow.add_node(
    "risk_agent",
    safe_node(risk_agent, "Risk Agent")
)

workflow.add_node(
    "anomaly_agent",
    safe_node(anomaly_agent, "Anomaly Agent")
)

workflow.add_node(
    "investigation_agent",
    safe_node(investigation_agent, "Investigation Agent")
)

workflow.add_node(
    "validation_agent",
    safe_node(validation_agent, "Validation Agent")
)

workflow.add_node("generate_report", generate_report)

# Define the starting point
workflow.add_edge(START, "supervisor_agent")

# Define dynamic routing
workflow.add_conditional_edges(
    "supervisor_agent",
    route_supervisor,
    {
        "risk_agent": "risk_agent",
        "anomaly_agent": "anomaly_agent",
        "investigation_agent": "investigation_agent",
        "validation_agent": "validation_agent"
    }
)

# Return control to the supervisor
workflow.add_edge("risk_agent", "supervisor_agent")
workflow.add_edge("anomaly_agent", "supervisor_agent")
workflow.add_edge(
    "investigation_agent",
    "supervisor_agent"
)

# Generate the final report after validation
workflow.add_edge(
    "validation_agent",
    "generate_report"
)

workflow.add_edge("generate_report", END)

# Compile the graph
insurance_workflow = workflow.compile()


def run_workflow(question: str) -> dict:
    """Execute the complete multi-agent workflow."""

    claim_ids = set(
        re.findall(
            r"\bCLM-\d{6}\b",
            question.upper()
        )
    )

    if len(claim_ids) != 1:
        raise ValueError(
            "Please provide exactly one valid claim ID."
        )

    claim_id = next(iter(claim_ids))

    initial_state = {
        "question": question,
        "claim_id": claim_id,
        "completed_agents": [],
        "required_agents": [],
        "execution_log": [],
        "errors": [],
        "validation_attempts": 0,
        "validation_passed": False,
        "human_review_required": False
    }

    return insurance_workflow.invoke(
        initial_state,
        config={"recursion_limit": 25}
    )


if __name__ == "__main__":

    question = (
        "Conduct a comprehensive fraud investigation "
        "for claim CLM-000001."
    )

    result = run_workflow(question)

    print("\n--- MULTI-AGENT EXECUTION TRACE ---")

    for entry in result["execution_log"]:
        print(entry)

    print("\n--- FINAL INVESTIGATION REPORT ---")
    print(result["final_report"])
