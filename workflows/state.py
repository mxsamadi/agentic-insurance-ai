
from typing import TypedDict, Annotated, Optional
import operator


class InvestigationState(TypedDict, total=False):

    # User input
    question: str
    claim_id: str

    # Supervisor decisions
    next_agent: str
    required_agents: list[str]

    # Specialized agent results
    risk_result: Optional[dict]
    anomaly_result: Optional[dict]
    investigation_result: Optional[dict]

    # Agent-generated analysis
    risk_analysis: str
    anomaly_analysis: str
    investigation_analysis: str

    # Execution history
    completed_agents: Annotated[list[str], operator.add]
    execution_log: Annotated[list[str], operator.add]

    # Validation
    validation_passed: bool
    validation_feedback: str
    validation_attempts: int

    # Final output
    final_report: str
    human_review_required: bool
    errors: Annotated[list[str], operator.add]
