import math
import pickle
from collections import Counter

from .config import BM25_B, BM25_K1, CACHE_DIR
from .data import load_movies
from .text_processing import tokenize_text, tokenize_term


class InvertedIndex:
    def __init__(self) -> None:
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter[str]] = {}
        self.doc_lengths: dict[int, int] = {}
        self.doc_lengths_path = CACHE_DIR / "doc_lengths.pkl"
        self.index_path = CACHE_DIR / "index.pkl"

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)

        self.doc_lengths[doc_id] = len(tokens)
        self.term_frequencies[doc_id] = Counter()

        for token in tokens:
            if token not in self.index:
                self.index[token] = set()

            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1

    def get_documents(self, term: str) -> list[int]:
        return sorted(self.index.get(term, set()))

    def get_tf(self, doc_id: int, term: str) -> int:
        frequencies = self.term_frequencies.get(doc_id)

        if frequencies is None:
            return 0

        return frequencies[term]

    def build(self) -> None:
        self.index.clear()
        self.docmap.clear()
        self.term_frequencies.clear()
        self.doc_lengths.clear()

        for movie in load_movies():
            doc_id = movie["id"]
            text = f"{movie['title']} {movie['description']}"

            self.__add_document(doc_id, text)
            self.docmap[doc_id] = movie

    def save(self) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        with open(CACHE_DIR / "index.pkl", "wb") as f:
            pickle.dump(self.index, f)

        with open(CACHE_DIR / "docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)

        with open(CACHE_DIR / "term_frequencies.pkl", "wb") as f:
            pickle.dump(self.term_frequencies, f)

        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)

    def load(self) -> None:
        with open(CACHE_DIR / "index.pkl", "rb") as f:
            index = pickle.load(f)

        with open(CACHE_DIR / "docmap.pkl", "rb") as f:
            docmap = pickle.load(f)

        with open(CACHE_DIR / "term_frequencies.pkl", "rb") as f:
            term_frequencies = pickle.load(f)

        with open(self.doc_lengths_path, "rb") as f:
            doc_lengths = pickle.load(f)

        with open(self.index_path, "rb") as f:
            index = pickle.load(f)

        self.index = index
        self.docmap = docmap
        self.term_frequencies = term_frequencies
        self.doc_lengths = doc_lengths

    def get_idf(self, term: str) -> float:
        total_docs = len(self.docmap)
        matching_docs = len(self.index.get(term, set()))

        return math.log((total_docs + 1) / (matching_docs + 1))

    def get_tfidf(self, doc_id: int, term: str) -> float:
        return self.get_tf(doc_id, term) * self.get_idf(term)

    def get_bm25_idf(self, term: str) -> float:
        total_docs = len(self.docmap)
        document_frequency = len(self.index.get(term, set()))

        return math.log(
            (total_docs - document_frequency + 0.5)
            / (document_frequency + 0.5)
            + 1
        )

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0

        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def get_bm25_tf(
        self,
        doc_id: int,
        term: str,
        k1: float = BM25_K1,
        b: float = BM25_B,
    ) -> float:
        tf = self.get_tf(doc_id, term)

        if tf == 0:
            return 0.0

        avg_doc_length = self.__get_avg_doc_length()

        if avg_doc_length == 0:
            return 0.0

        doc_length = self.doc_lengths[doc_id]
        length_norm = 1 - b + b * (doc_length / avg_doc_length)

        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def bm25(self, doc_id: int, term: str) -> float:
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)

        return bm25_tf * bm25_idf

    def bm25_search(
        self, query: str, limit: int = 5
    ) -> list[tuple[int, float]]:
        if limit <= 0:
            return []

        query_tokens = tokenize_text(query)
        scores: dict[int, float] = {}

        for doc_id in self.docmap:
            total_score = 0.0

            for token in query_tokens:
                total_score += self.bm25(doc_id, token)

            scores[doc_id] = total_score

        ranked_documents = sorted(
            scores.items(),
            key=lambda item: (-item[1], item[0]),
        )

        return ranked_documents[:limit]


def search_movies(query: str) -> list[dict]:
    inverted_index = InvertedIndex()
    inverted_index.load()

    query_tokens = tokenize_text(query)

    results = []
    seen_ids = set()

    for token in query_tokens:
        doc_ids = inverted_index.get_documents(token)

        for doc_id in doc_ids:
            if doc_id in seen_ids:
                continue

            seen_ids.add(doc_id)
            results.append(inverted_index.docmap[doc_id])

            if len(results) >= 5:
                return results

    return results

def build_command() -> None:
    inverted_index = InvertedIndex()
    inverted_index.build()
    inverted_index.save()
    print("Inverted index built and saved.")

def tf_command(doc_id: int, term: str) -> None:
    token = tokenize_term(term)

    inverted_index = InvertedIndex()
    inverted_index.load()

    print(inverted_index.get_tf(doc_id, token))

def bm25_idf_command(term: str) -> float:
    inverted_index = InvertedIndex()
    inverted_index.load()

    token = tokenize_term(term)

    return inverted_index.get_bm25_idf(token)

def bm25_tf_command(
    doc_id: int,
    term: str,
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> float:
    inverted_index = InvertedIndex()
    inverted_index.load()

    token = tokenize_term(term)

    return inverted_index.get_bm25_tf(doc_id, token, k1, b)