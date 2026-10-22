import json

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import CACHE_DIR
from .data import load_movies

import re

class SemanticSearch:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text: str):
        if not text.strip():
            raise ValueError("Text must not be empty or whitespace-only.")

        embeddings = self.model.encode([text])

        return embeddings[0]

    def build_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {
            doc["id"]: doc for doc in documents
        }

        movie_texts = [
            f"{doc['title']}: {doc['description']}"
            for doc in documents
        ]

        self.embeddings = self.model.encode(
            movie_texts,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(CACHE_DIR / "movie_embeddings.npy", self.embeddings)

        return self.embeddings

    def load_or_create_embeddings(
        self, documents: list[dict]
    ) -> np.ndarray:
        self.documents = documents
        self.document_map = {
            doc["id"]: doc for doc in documents
        }

        embeddings_path = CACHE_DIR / "movie_embeddings.npy"

        if embeddings_path.exists():
            self.embeddings = np.load(
                embeddings_path,
                allow_pickle=False,
            )

            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.build_embeddings(documents)

    def search(self, query: str, limit: int = 5) -> list[dict]:
        if self.embeddings is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)

        scored_documents = []

        for i, document_embedding in enumerate(self.embeddings):
            similarity = cosine_similarity(query_embedding, document_embedding)
            document = self.documents[i]
            scored_documents.append((similarity, document))

        scored_documents.sort(key=lambda item: item[0], reverse=True)

        return [
            {
                "score": score,
                "title": document["title"],
                "description": document["description"],
            }
            for score, document in scored_documents[:max(0, limit)]
        ]


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(
        self,
        documents: list[dict],
    ) -> np.ndarray:
        self.documents = documents
        self.document_map = {
            document["id"]: document
            for document in documents
        }

        all_chunks = []
        chunk_metadata = []

        for movie_idx, document in enumerate(self.documents):
            description = document["description"]

            if not description.strip():
                continue

            # 与 semantic_chunk 命令复用同一个函数
            chunks = semantic_chunk_text(
                description,
                max_chunk_size=4,
                overlap=1,
            )

            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append(
                    {
                        "movie_idx": movie_idx,
                        "chunk_idx": chunk_idx,
                        "total_chunks": len(chunks),
                    }
                )

        self.chunk_embeddings = self.model.encode(
            all_chunks,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        self.chunk_metadata = chunk_metadata

        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        np.save(
            CACHE_DIR / "chunk_embeddings.npy",
            self.chunk_embeddings,
        )

        with open(
            CACHE_DIR / "chunk_metadata.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                {
                    "chunks": chunk_metadata,
                    "total_chunks": len(all_chunks),
                },
                f,
                indent=2,
            )

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(
        self,
        documents: list[dict],
    ) -> np.ndarray:
        self.documents = documents
        self.document_map = {
            document["id"]: document
            for document in documents
        }

        embeddings_path = CACHE_DIR / "chunk_embeddings.npy"
        metadata_path = CACHE_DIR / "chunk_metadata.json"

        if embeddings_path.exists() and metadata_path.exists():
            self.chunk_embeddings = np.load(
                embeddings_path,
                allow_pickle=False,
            )

            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            self.chunk_metadata = metadata["chunks"]
            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10) -> list[dict]:
        if self.chunk_embeddings is None or self.chunk_metadata is None:
            raise ValueError(
                "No chunk embeddings loaded. "
                "Call `load_or_create_chunk_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)

        chunk_scores = []

        for i, chunk_embedding in enumerate(self.chunk_embeddings):
            metadata = self.chunk_metadata[i]
            score = cosine_similarity(chunk_embedding, query_embedding)

            chunk_scores.append(
                {
                    "chunk_idx": metadata["chunk_idx"],
                    "movie_idx": metadata["movie_idx"],
                    "score": float(score),
                }
            )

        # 每部电影只保留得分最高的块
        movie_scores = {}

        for chunk_score in chunk_scores:
            movie_idx = chunk_score["movie_idx"]

            if (
                movie_idx not in movie_scores
                or chunk_score["score"] > movie_scores[movie_idx]["score"]
            ):
                movie_scores[movie_idx] = chunk_score

        sorted_scores = sorted(
            movie_scores.values(),
            key=lambda item: item["score"],
            reverse=True,
        )

        results = []

        for movie_score in sorted_scores[:max(0, limit)]:
            doc = self.documents[movie_score["movie_idx"]]

            results.append(
                {
                    "id": doc["id"],
                    "title": doc["title"],
                    "document": doc["description"][:100],
                    "score": round(movie_score["score"], 3),
                }
            )

        return results



def verify_model() -> None:
    search = SemanticSearch()

    print(f"Model loaded: {search.model}")
    print(f"Max sequence length: {search.model.max_seq_length}")

def embed_text(text: str) -> None:
    search = SemanticSearch()
    embedding = search.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings() -> None:
    search = SemanticSearch()
    documents = load_movies()

    embeddings = search.load_or_create_embeddings(documents)

    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors "
        f"in {embeddings.shape[1]} dimensions"
    )

def embed_query_text(query: str) -> None:
    search = SemanticSearch()
    embedding = search.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 0,) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than 0.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "Overlap must be non-negative and smaller than chunk size."
        )
    
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))

        # 已覆盖最后一个单词，不再生成只有重复内容的尾块
        if end >= len(words):
            break

        start = end - overlap

    return chunks

def semantic_chunk_text(
    text: str,
    max_chunk_size: int = 4,
    overlap: int = 0,
) -> list[str]:
    if max_chunk_size <= 0:
        raise ValueError("Max chunk size must be greater than 0.")

    if overlap < 0 or overlap >= max_chunk_size:
        raise ValueError(
            "Overlap must be non-negative and smaller than max chunk size."
        )

    text = text.strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)

     # 没有句末标点的单段文本，整体保留为一个句子
    if len(sentences) == 1 and not text.endswith((".", "!", "?")):
        sentences = [text]

    # 清理每个句子，并过滤空内容
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text.strip())
        if sentence.strip()
    ]

    chunks = []
    start = 0

    while start < len(sentences):
        end = start + max_chunk_size
        chunk = " ".join(sentences[start:end]).strip()

        if chunk:
            chunks.append(chunk)


        if end >= len(sentences):
            break

        start = end - overlap

    return chunks