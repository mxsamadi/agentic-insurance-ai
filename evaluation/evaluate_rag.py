
import time
import pandas as pd

from tools.retrieval_tools import retrieve_guidelines


test_cases = [
    {
        "query": "Recently purchased insurance policy",
        "relevant_sections": [2]
    },
    {
        "query": "Multiple previous insurance claims",
        "relevant_sections": [3]
    },
    {
        "query": "Claim amount exceeds vehicle value",
        "relevant_sections": [4]
    },
    {
        "query": "Delayed reporting of an accident",
        "relevant_sections": [5]
    },
    {
        "query": "Missing police report and witnesses",
        "relevant_sections": [6]
    },
    {
        "query": "Discrepancy between claim and repair estimate",
        "relevant_sections": [7]
    },
    {
        "query": "Statistically unusual insurance claims",
        "relevant_sections": [8]
    },
    {
        "query": "Human review and automatic claim denial",
        "relevant_sections": [9]
    },
    {
        "query": "Uncalibrated predictive fraud risk scores",
        "relevant_sections": [1]
    },
    {
        "query": "AI hallucination and fabricated evidence",
        "relevant_sections": [10]
    },
    {
        "query": "New policy with multiple previous claims",
        "relevant_sections": [2, 3]
    },
    {
        "query": "Delayed reporting and missing police report",
        "relevant_sections": [5, 6]
    }
]


def evaluate_rag(top_k=3):

    records = []

    for test in test_cases:

        query = test["query"]
        relevant = set(test["relevant_sections"])

        start = time.perf_counter()

        result = retrieve_guidelines.invoke({
            "query": query,
            "top_k": top_k
        })

        latency = time.perf_counter() - start

        if "error" in result:
            print(f"Retrieval failed: {result['error']}")

        retrieved = [
            int(item["citation"].split("Section ")[1])
            for item in result.get("guidelines", [])
        ]

        # Count relevant retrieved sections
        hits = len(set(retrieved) & relevant)

        precision = hits / top_k
        recall = hits / len(relevant)

        # Reciprocal rank
        reciprocal_rank = 0.0

        for rank, section in enumerate(retrieved, start=1):

            if section in relevant:
                reciprocal_rank = 1 / rank
                break

        records.append({
            "query": query,
            "expected_sections": sorted(relevant),
            "retrieved_sections": retrieved,
            "precision": precision,
            "recall": recall,
            "reciprocal_rank": reciprocal_rank,
            "hit": int(hits > 0),
            "latency_seconds": latency,
            "error": result.get("error", "")
        })

        print(f"\nQuery: {query}")
        print("Expected:", sorted(relevant))
        print("Retrieved:", retrieved)
        print(f"Precision@{top_k}: {precision:.2%}")
        print(f"Recall@{top_k}: {recall:.2%}")
        print(f"Reciprocal Rank: {reciprocal_rank:.4f}")

    df = pd.DataFrame(records)

    print("\n--- RAG EVALUATION SUMMARY ---")

    print(f"Total queries: {len(df)}")
    print(f"Precision@{top_k}: {df['precision'].mean():.2%}")
    print(f"Recall@{top_k}: {df['recall'].mean():.2%}")
    print(f"MRR@{top_k}: {df['reciprocal_rank'].mean():.4f}")
    print(f"Hit Rate@{top_k}: {df['hit'].mean():.2%}")

    print(
        f"Average latency: "
        f"{df['latency_seconds'].mean():.4f} seconds"
    )

    print(
        f"P95 latency: "
        f"{df['latency_seconds'].quantile(0.95):.4f} seconds"
    )

    print(f"Retrieval errors: {(df['error'] != '').sum()}")

    df.to_csv(
        f"evaluation/rag_results_k{top_k}.csv",
        index=False
    )


if __name__ == "__main__":
    evaluate_rag(top_k=3)
