"""
CARA - Greenfield Medical Centre AI Receptionist
relevance_checker.py - Checks if top retrieved chunk meets
relevance threshold before injecting into Claude prompt
"""

# ─────────────────────────────────────────
# THRESHOLD
# ─────────────────────────────────────────
RELEVANCE_THRESHOLD = 0.3


def is_relevant(top_score: float) -> bool:
    """
    Checks if the top retrieved chunk's cosine similarity
    score meets the relevance threshold.

    Args:
        top_score: cosine similarity score of top chunk (0.0 to 1.0)

    Returns:
        True if relevant enough to inject into prompt
        False if not relevant — CARA should ask patient to clarify
    """
    return top_score >= RELEVANCE_THRESHOLD


def check_relevance(chunks: list[dict]) -> tuple[bool, float]:
    """
    Takes list of retrieved chunks with their scores.
    Returns whether chunks are relevant and the top score.

    Args:
        chunks: list of dicts with keys:
                "text" — chunk text
                "score" — cosine similarity score
                "metadata" — chunk metadata

    Returns:
        (is_relevant: bool, top_score: float)
    """
    if not chunks:
        print("  [RelevanceChecker] No chunks returned — not relevant")
        return False, 0.0

    top_score = chunks[0]["score"]
    relevant = is_relevant(top_score)

    if relevant:
        print(f"  [RelevanceChecker] Top score: {top_score:.3f} >= {RELEVANCE_THRESHOLD} — RELEVANT ✅")
    else:
        print(f"  [RelevanceChecker] Top score: {top_score:.3f} < {RELEVANCE_THRESHOLD} — NOT RELEVANT ❌ — will ask clarification")

    return relevant, top_score


# ─────────────────────────────────────────
# TEST — run directly to verify
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("Testing RelevanceChecker...\n")

    # Simulate chunks with different scores
    test_cases = [
        {
            "label": "High relevance — clear appointment query",
            "chunks": [
                {"text": "Dr Patel available Monday 9am", "score": 0.89, "metadata": {}},
                {"text": "Dr Collins available Monday 10am", "score": 0.81, "metadata": {}},
                {"text": "Dr Bennett available Monday 11am", "score": 0.76, "metadata": {}},
            ]
        },
        {
            "label": "Medium relevance — borderline query",
            "chunks": [
                {"text": "Opening hours Monday to Friday 8am", "score": 0.42, "metadata": {}},
                {"text": "Saturday hours 8am to 12pm", "score": 0.38, "metadata": {}},
            ]
        },
        {
            "label": "Low relevance — unclear query needs clarification",
            "chunks": [
                {"text": "Some vaguely related chunk", "score": 0.28, "metadata": {}},
                {"text": "Another loosely related chunk", "score": 0.21, "metadata": {}},
            ]
        },
        {
            "label": "No chunks returned",
            "chunks": []
        }
    ]

    for case in test_cases:
        print(f"Case: {case['label']}")
        relevant, score = check_relevance(case["chunks"])
        print(f"Result: {'PROCEED with chunks' if relevant else 'ASK CLARIFICATION'}")
        print()