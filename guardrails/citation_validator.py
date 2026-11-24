
import re


def validate_citations(
    analysis: str,
    retrieved_guidelines: list
) -> dict:
    """Validate citations against retrieved source metadata."""

    pattern = r"\bGUIDE-\d+\s*,?\s*Section\s+(\d+)\b"

    def extract_citations(text):

        matches = re.finditer(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        citations = set()

        for match in matches:

            document_id = re.search(
                r"GUIDE-\d+",
                match.group(0),
                flags=re.IGNORECASE
            ).group(0).upper()

            section = int(match.group(1))

            citations.add(
                f"{document_id}, Section {section}"
            )

        return citations

    # Identify authorized citations
    allowed_citations = set()

    for guideline in retrieved_guidelines:

        allowed_citations.update(
            extract_citations(
                guideline.get("citation", "")
            )
        )

    # Identify citations used in the generated analysis
    generated_citations = extract_citations(analysis)

    valid_citations = (
        generated_citations & allowed_citations
    )

    invalid_citations = (
        generated_citations - allowed_citations
    )

    missing_citations = (
        bool(allowed_citations)
        and not generated_citations
    )

    citation_accuracy = (
        len(valid_citations) / len(generated_citations)
        if generated_citations
        else None
    )

    passed = (
        bool(allowed_citations)
        and bool(generated_citations)
        and not invalid_citations
    )

    return {
        "passed": passed,
        "allowed_citations": sorted(allowed_citations),
        "generated_citations": sorted(generated_citations),
        "valid_citations": sorted(valid_citations),
        "invalid_citations": sorted(invalid_citations),
        "missing_citations": missing_citations,
        "citation_accuracy": citation_accuracy
    }


if __name__ == "__main__":

    guidelines = [
        {"citation": "GUIDE-001, Section 2"},
        {"citation": "GUIDE-001, Section 3"},
        {"citation": "GUIDE-001, Section 9"}
    ]

    test_cases = [
        {
            "name": "Valid citations",
            "analysis": (
                "Verify policy dates "
                "[GUIDE-001, Section 2] "
                "and review previous claims "
                "[GUIDE-001, Section 3]."
            ),
            "expected": True
        },
        {
            "name": "Fabricated citation",
            "analysis": (
                "Review the policy "
                "[GUIDE-001, Section 99]."
            ),
            "expected": False
        },
        {
            "name": "Missing citations",
            "analysis": "Additional investigation is recommended.",
            "expected": False
        },
        {
            "name": "Mixed valid and invalid citations",
            "analysis": (
                "Review policy dates "
                "[GUIDE-001, Section 2] "
                "and additional evidence "
                "[GUIDE-001, Section 50]."
            ),
            "expected": False
        },
        {
            "name": "Unretrieved existing section",
            "analysis": (
                "Review repair estimates "
                "[GUIDE-001, Section 7]."
            ),
            "expected": False
        }
    ]

    for test in test_cases:

        result = validate_citations(
            test["analysis"],
            guidelines
        )

        print(f"\n--- {test['name']} ---")

        for key, value in result.items():
            print(f"{key}: {value}")

        assert result["passed"] == test["expected"]

    print("\nAll citation validation tests passed.")
