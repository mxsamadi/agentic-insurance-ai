
from fastapi.testclient import TestClient
from unittest.mock import patch

from api import app


client = TestClient(app)


def test_home():

    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["version"] == "1.0.0"


def test_health():

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_missing_claim_id():

    response = client.post(
        "/investigate",
        json={"question": "Investigate an insurance claim."}
    )

    assert response.status_code == 400


def test_multiple_claim_ids():

    response = client.post(
        "/investigate",
        json={
            "question": (
                "Investigate CLM-000001 and CLM-000002."
            )
        }
    )

    assert response.status_code == 400


def test_empty_question():

    response = client.post(
        "/investigate",
        json={"question": ""}
    )

    assert response.status_code == 422


def test_successful_investigation():

    mock_result = {
        "claim_id": "CLM-000001",
        "risk_result": {
            "risk_score": 0.3138,
            "risk_level": "Low"
        },
        "required_agents": ["risk_agent"],
        "completed_agents": ["risk_agent"],
        "validation_passed": True,
        "human_review_required": False,
        "final_report": "Synthetic claim risk assessment.",
        "execution_log": ["Risk Agent completed"]
    }

    with patch(
        "api.run_workflow",
        return_value=mock_result
    ):

        response = client.post(
            "/investigate",
            json={
                "question": (
                    "Assess fraud risk for CLM-000001."
                )
            }
        )

    assert response.status_code == 200

    data = response.json()

    assert data["claim_id"] == "CLM-000001"
    assert data["status"] == "completed"
    assert data["validation_passed"] is True
    assert data["risk_score"] == 0.3138


def test_failed_validation():

    mock_result = {
        "claim_id": "CLM-000001",
        "risk_result": {
            "risk_score": 0.95,
            "risk_level": "High"
        },
        "required_agents": ["risk_agent"],
        "completed_agents": ["risk_agent"],
        "validation_passed": False,
        "human_review_required": True,
        "final_report": "Unvalidated internal analysis.",
        "execution_log": ["Validation failed"]
    }

    with patch(
        "api.run_workflow",
        return_value=mock_result
    ):

        response = client.post(
            "/investigate",
            json={
                "question": (
                    "Assess fraud risk for CLM-000001."
                )
            }
        )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "withheld"
    assert data["validation_passed"] is False
    assert data["human_review_required"] is True

    # Verify that unvalidated results are not exposed
    assert data["risk_score"] is None
    assert data["risk_level"] is None
    assert data["sources"] == []
    assert "Unvalidated internal analysis" not in data["final_report"]


def test_internal_server_error():

    with patch(
        "api.run_workflow",
        side_effect=RuntimeError("Simulated failure")
    ):

        response = client.post(
            "/investigate",
            json={
                "question": (
                    "Assess fraud risk for CLM-000001."
                )
            }
        )

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "Investigation service failed."
    )
