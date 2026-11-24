
from agents.insurance_agent import run_investigation

questions = [
    "What is the fraud risk score of claim CLM-000001?",
    "Is claim CLM-000002 an unusual claim?",
    "What are the risk indicators for claim CLM-000003?",
    "Conduct a comprehensive fraud investigation for claim CLM-000004.",
    "Investigate claim CLM-999999."
]

if __name__ == "__main__":

    for i, question in enumerate(questions, start=1):

        print("\n" + "=" * 70)
        print(f"TEST {i}")
        print(f"Question: {question}")
        print("=" * 70)

        try:
            response = run_investigation(question)
            print("\n--- AGENT RESPONSE ---")
            print(response)

        except Exception as error:
            print(f"Execution error: {error}")
