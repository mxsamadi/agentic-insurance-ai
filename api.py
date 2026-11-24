
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import time
import re

from workflows.insurance_workflow import run_workflow


app = FastAPI(
    title="Agentic AI Insurance Risk & Fraud Detection",
    description=(
        "Multi-agent insurance investigation platform "
        "powered by LangGraph, XGBoost, Isolation Forest, "
        "RAG, and AI guardrails."
    ),
    version="1.0.0"
)


class InvestigationRequest(BaseModel):

    question: str = Field(
        min_length=10,
        max_length=2000
    )


class InvestigationResponse(BaseModel):

    claim_id: str
    status: str

    risk_score: Optional[float] = None
    risk_level: Optional[str] = None

    anomaly_score: Optional[float] = None
    is_anomaly: Optional[bool] = None

    selected_agents: list[str]
    completed_agents: list[str]

    validation_passed: bool
    human_review_required: bool

    sources: list[dict]

    final_report: str
    execution_log: list[str]

    latency_seconds: float


@app.get("/")
def home():

    return {
        "platform": "Agentic AI Insurance Platform",
        "version": "1.0.0",
        "documentation": "/docs",
        "data_type": "Synthetic insurance claims"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "service": "Insurance Investigation API"
    }


@app.post(
    "/investigate",
    response_model=InvestigationResponse
)
def investigate(request: InvestigationRequest):

    claim_ids = set(
        re.findall(
            r"\bCLM-\d{6}\b",
            request.question.upper()
        )
    )

    if len(claim_ids) != 1:

        raise HTTPException(
            status_code=400,
            detail="Provide exactly one valid claim ID."
        )

    start = time.perf_counter()

    try:

        result = run_workflow(request.question)

        latency = time.perf_counter() - start

        validated = result.get(
            "validation_passed", False
        )

        # Do not expose unvalidated analysis or scores
        risk = (
            result.get("risk_result") or {}
            if validated else {}
        )

        anomaly = (
            result.get("anomaly_result") or {}
            if validated else {}
        )

        investigation = (
            result.get("investigation_result") or {}
            if validated else {}
        )

        sources = [
            {
                "citation": item["citation"],
                "title": item["title"],
                "similarity_score": item["similarity_score"]
            }
            for item in investigation.get(
                "retrieved_guidelines", []
            )
        ]

        return InvestigationResponse(

            claim_id=result["claim_id"],

            status=(
                "completed"
                if validated
                else "withheld"
            ),

            risk_score=risk.get("risk_score"),
            risk_level=risk.get("risk_level"),

            anomaly_score=anomaly.get("anomaly_score"),
            is_anomaly=anomaly.get("is_anomaly"),

            selected_agents=result.get(
                "required_agents", []
            ),

            completed_agents=result.get(
                "completed_agents", []
            ),

            validation_passed=validated,

            human_review_required=result.get(
                "human_review_required", True
            ),

            sources=sources,

            final_report=(
                result["final_report"]
                if validated
                else "Validation failed. Human review required. "
                     "No automated claim decision has been made."
            ),

            execution_log=(
                result.get("execution_log", [])
                if validated
                else ["Investigation withheld by validation"]
            ),

            latency_seconds=round(latency, 2)
        )

    except Exception:

        raise HTTPException(
            status_code=500,
            detail="Investigation service failed."
        )
