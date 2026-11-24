
import time
import pandas as pd

from workflows.insurance_workflow import run_workflow


test_cases = [
    {
        "name": "Risk assessment",
        "question": "What is the fraud risk score of CLM-000001?",
        "expected_agents": ["risk_agent"],
        "expected_validation": True
    },
    {
        "name": "Anomaly detection",
        "question": "Is claim CLM-000002 unusual?",
        "expected_agents": ["anomaly_agent"],
        "expected_validation": True
    },
    {
        "name": "Claim investigation",
        "question": "Identify risk indicators for CLM-000003.",
        "expected_agents": ["investigation_agent"],
        "expected_validation": True
    },
    {
        "name": "Combined assessment",
        "question": "Assess fraud risk and detect anomalies for CLM-000004.",
        "expected_agents": [
            "risk_agent",
            "anomaly_agent"
        ],
        "expected_validation": True
    },
    {
        "name": "Comprehensive investigation",
        "question": "Conduct a comprehensive fraud investigation for CLM-000005.",
        "expected_agents": [
            "risk_agent",
            "anomaly_agent",
            "investigation_agent"
        ],
        "expected_validation": True
    },
    {
        "name": "Nonexistent claim",
        "question": "Assess the fraud risk score of CLM-999999.",
        "expected_agents": ["risk_agent"],
        "expected_validation": False
    }
]


def evaluate():

    results = []

    for test in test_cases:

        print(f"\nRunning: {test['name']}")

        start = time.perf_counter()

        try:
            result = run_workflow(test["question"])

            latency = time.perf_counter() - start

            selected = result.get("required_agents", [])
            completed = result.get("completed_agents", [])

            routing_correct = (
                set(selected) == set(test["expected_agents"])
            )

            validation_correct = (
                result.get("validation_passed", False)
                == test["expected_validation"]
            )

            error_handling_correct = (
                not result.get("errors")
                and set(completed) == set(test["expected_agents"])
                if test["expected_validation"]
                else bool(result.get("errors"))
                and result.get("human_review_required", False)
            )

            report_generated = bool(
                result.get("final_report")
            )

            passed = (
                routing_correct
                and validation_correct
                and error_handling_correct
                and report_generated
            )

            results.append({
                "test": test["name"],
                "routing_correct": routing_correct,
                "validation_correct": validation_correct,
                "error_handling_correct": error_handling_correct,
                "report_generated": report_generated,
                "latency_seconds": round(latency, 2),
                "passed": passed
            })

            print("Selected agents:", selected)
            print("Completed agents:", completed)
            print("Validation:", result.get("validation_passed"))
            print("Latency:", round(latency, 2), "seconds")
            print("Test passed:", passed)

        except Exception as error:

            results.append({
                "test": test["name"],
                "routing_correct": False,
                "validation_correct": False,
                "error_handling_correct": False,
                "report_generated": False,
                "latency_seconds": round(
                    time.perf_counter() - start, 2
                ),
                "passed": False
            })

            print("Execution error:", error)

    df = pd.DataFrame(results)

    print("\n--- EVALUATION RESULTS ---")
    print(df.to_string(index=False))

    print("\n--- PERFORMANCE METRICS ---")

    print(
        f"Routing accuracy: "
        f"{df['routing_correct'].mean():.2%}"
    )

    print(
        f"Validation accuracy: "
        f"{df['validation_correct'].mean():.2%}"
    )

    print(
        f"Overall test pass rate: "
        f"{df['passed'].mean():.2%}"
    )

    print(
        f"Average latency: "
        f"{df['latency_seconds'].mean():.2f} seconds"
    )

    print(
        f"P95 latency: "
        f"{df['latency_seconds'].quantile(0.95):.2f} seconds"
    )

    df.to_csv(
        "evaluation/workflow_results.csv",
        index=False
    )

    print("\nEvaluation results saved successfully.")


if __name__ == "__main__":
    evaluate()
