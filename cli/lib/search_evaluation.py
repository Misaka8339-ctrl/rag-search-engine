import json

from .query_enhancement import _enhance_with_llm


def evaluate_results(query: str, results: list[dict]) -> list[int]:
    if not results:
        return []

    formatted_results = [
        f"{i}. {result['title']} - {result.get('document', '')}"
        for i, result in enumerate(results, start=1)
    ]

    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(formatted_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers other than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents.
Return exactly one score for each result.
Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""

    response = _enhance_with_llm(prompt)
    scores = json.loads(response)

    if not isinstance(scores, list):
        raise ValueError("Evaluation must return a JSON list.")

    if len(scores) != len(results):
        raise ValueError(
            "Evaluation must return exactly one score per result."
        )

    if any(
        type(score) is not int or score not in (0, 1, 2, 3)
        for score in scores
    ):
        raise ValueError("Evaluation scores must be integers from 0 to 3.")

    return scores