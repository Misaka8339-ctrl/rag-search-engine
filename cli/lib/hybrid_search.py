from pathlib import Path

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents

        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()

        if not Path(self.idx.index_path).exists():
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        matches = self.idx.bm25_search(query, limit)

        results = []

        for doc_id, score in matches:
            doc = self.idx.docmap[doc_id]

            results.append(
                {
                    "id": doc["id"],
                    "title": doc["title"],
                    "document": doc["description"][:100],
                    "score": float(score),
                }
            )

        return results

    def weighted_search(
        self,
        query: str,
        alpha: float = 0.5,
        limit: int = 5,
    ) -> list[dict]:
        if not 0 <= alpha <= 1:
            raise ValueError("Alpha must be between 0 and 1.")

        if limit <= 0:
            return []

        candidate_limit = limit * 500

        bm25_results = self._bm25_search(query, candidate_limit)
        semantic_results = self.semantic_search.search_chunks(
            query,
            candidate_limit,
        )

        # 两路分数分别归一化
        bm25_scores = normalize_scores(
            [result["score"] for result in bm25_results]
        )
        semantic_scores = normalize_scores(
            [result["score"] for result in semantic_results]
        )

        combined = {}

        for results, scores, score_key in (
            (bm25_results, bm25_scores, "bm25_score"),
            (semantic_results, semantic_scores, "semantic_score"),
        ):
            for result, score in zip(results, scores):
                doc_id = result["id"]

                if doc_id not in combined:
                    combined[doc_id] = {
                        "id": doc_id,
                        "title": result["title"],
                        "document": result["document"],
                        "bm25_score": 0.0,
                        "semantic_score": 0.0,
                    }

                combined[doc_id][score_key] = score

        for result in combined.values():
            result["hybrid_score"] = hybrid_score(
                result["bm25_score"],
                result["semantic_score"],
                alpha,
            )

        return sorted(
            combined.values(),
            key=lambda result: result["hybrid_score"],
            reverse=True,
        )


    def rrf_search(
        self,
        query: str,
        k: int = 60,
        limit: int = 10,
    ) -> list[dict]:
        if k < 0:
            raise ValueError("K must be non-negative.")

        if limit <= 0:
            return []

        candidate_limit = limit * 500

        bm25_results = self._bm25_search(query, candidate_limit)
        semantic_results = self.semantic_search.search_chunks(
            query,
            candidate_limit,
        )

        combined = {}

        for results, rank_key in (
            (bm25_results, "bm25_rank"),
            (semantic_results, "semantic_rank"),
        ):
            for rank, result in enumerate(results, start=1):
                doc_id = result["id"]

                if doc_id not in combined:
                    combined[doc_id] = {
                        "id": doc_id,
                        "title": result["title"],
                        "document": result["document"],
                        "bm25_rank": None,
                        "semantic_rank": None,
                        "rrf_score": 0.0,
                    }

                combined[doc_id][rank_key] = rank
                combined[doc_id]["rrf_score"] += rrf_score(rank, k)

        return sorted(
            combined.values(),
            key=lambda result: result["rrf_score"],
            reverse=True,
        )



def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        return [1.0] * len(scores)

    return [
        (score - min_score) / (max_score - min_score)
        for score in scores
    ]

def hybrid_score(
    bm25_score: float,
    semantic_score: float,
    alpha: float = 0.5,
) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score

def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)