import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

from .data import load_movies
from .semantic_search import cosine_similarity

class MultimodalSearch:
    def __init__(
        self,
        documents: list[dict],
        model_name: str = "clip-ViT-B-32",
    ) -> None:
        self.documents = documents
        self.model = SentenceTransformer(model_name)

        self.texts = [
            f"{doc['title']}: {doc['description']}"
            for doc in documents
        ]

        self.text_embeddings = self.model.encode(
            self.texts,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

    def embed_image(self, image_path: str) -> np.ndarray:
        with Image.open(image_path) as image:
            # 统一为 RGB，兼容灰度或带透明通道的图片
            image = image.convert("RGB")

            embeddings = self.model.encode(
                [image],
                convert_to_numpy=True,
            )

        return embeddings[0]

    def search_with_image(self, image_path: str) -> list[dict]:
        image_embedding = self.embed_image(image_path)

        results = []

        for doc, text_embedding in zip(
            self.documents,
            self.text_embeddings,
        ):
            similarity = cosine_similarity(
                text_embedding,
                image_embedding,
            )

            results.append(
                {
                    "id": doc["id"],
                    "title": doc["title"],
                    "description": doc["description"],
                    "score": float(similarity),
                }
            )

        results.sort(
            key=lambda result: result["score"],
            reverse=True,
        )

        return results[:5]


def verify_image_embedding(image_path: str) -> None:
    search = MultimodalSearch()
    embedding = search.embed_image(image_path)

    print(f"Embedding shape: {embedding.shape[0]} dimensions")

def image_search_command(image_path: str) -> list[dict]:
    documents = load_movies()
    search = MultimodalSearch(documents)

    return search.search_with_image(image_path)