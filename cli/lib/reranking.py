import math
import time
import json

from openai import APIError

from .query_enhancement import _enhance_with_llm
from sentence_transformers import CrossEncoder


def rerank_individual(query: str, documents: list[dict]) -> list[dict]:
    results = []
    last_call_finished = None

    for doc in documents:
        prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("document", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Output ONLY the number in your response, no other text or explanation.

Score:"""

        for attempt in range(3):
            # 普通调用和重试之间，都至少间隔 3 秒
            if last_call_finished is not None:
                remaining = 3.0 - (time.monotonic() - last_call_finished)
                if remaining > 0:
                    time.sleep(remaining)

            try:
                text = _enhance_with_llm(prompt)
                score = float(text)

                if not math.isfinite(score) or not 0 <= score <= 10:
                    raise ValueError("Re-rank score must be between 0 and 10.")

            except (APIError, ValueError, RuntimeError) as exc:
                if attempt == 2:
                    raise RuntimeError(
                        f"Failed to re-rank movie {doc['id']} after 3 attempts."
                    ) from exc
            else:
                # 创建新字典，保留 RRF 分数和两路排名
                results.append({
                    **doc,
                    "rerank_score": score,
                })
                break
            finally:
                last_call_finished = time.monotonic()

    # 同分时保留原来的 RRF 顺序
    return sorted(
        results,
        key=lambda result: result["rerank_score"],
        reverse=True,
    )

def rerank_batch(query: str, documents: list[dict]) -> list[dict]:
    if not documents:
        return []

    doc_list_str = json.dumps(
        [
            {
                "id": doc["id"],
                "title": doc["title"],
                "document": doc["document"],
            }
            for doc in documents
        ],
        ensure_ascii=False,
        indent=2,
    )

    prompt = f"""Rank the movies listed below by relevance to the following search query.

Query: "{query}"

Movies:
{doc_list_str}

Return the movie IDs in order of relevance, best match first.
Include every provided movie ID exactly once. Do not invent IDs.

Your response must be a raw JSON array of integers.
Do not wrap the JSON in Markdown. Do not use a ```json code block.
Do not include any explanatory text.

For example:
[75, 12, 34, 2, 1]

Ranking:"""

    response = _enhance_with_llm(prompt)
    ranked_ids = json.loads(response)

    # 验证返回格式及 ID，避免遗漏、重复或添加不存在的电影
    if not isinstance(ranked_ids, list):
        raise ValueError("Batch re-ranking must return a JSON array.")

    if any(type(doc_id) is not int for doc_id in ranked_ids):
        raise ValueError("Batch re-ranking IDs must be integers.")

    expected_ids = {doc["id"] for doc in documents}

    if (
        len(ranked_ids) != len(documents)
        or set(ranked_ids) != expected_ids
    ):
        raise ValueError(
            "Batch re-ranking must return every candidate ID exactly once."
        )

    rank_map = {
        doc_id: rank
        for rank, doc_id in enumerate(ranked_ids, start=1)
    }

    results = [
        {
            **doc,
            "rerank_rank": rank_map[doc["id"]],
        }
        for doc in documents
    ]

    return sorted(
        results,
        key=lambda result: result["rerank_rank"],
    )

def rerank_cross_encoder(
    query: str,
    documents: list[dict],
) -> list[dict]:
    if not documents:
        return []

    pairs = [
        [
            query,
            f"{doc.get('title', '')} - {doc.get('document', '')}",
        ]
        for doc in documents
    ]

    # 每次查询只创建一个模型实例
    cross_encoder = CrossEncoder(
        "cross-encoder/ms-marco-TinyBERT-L2-v2"
    )

    scores = cross_encoder.predict(pairs)

    results = [
        {
            **doc,
            "cross_encoder_score": float(score),
        }
        for doc, score in zip(documents, scores)
    ]

    return sorted(
        results,
        key=lambda result: result["cross_encoder_score"],
        reverse=True,
    )