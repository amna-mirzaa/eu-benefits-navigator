"""Local, free embeddings via sentence-transformers (no API key, runs on CPU)."""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from src import config


class Embedder:
    def __init__(self, model_name: str = config.EMBEDDING_MODEL) -> None:
        self._model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        return self._model.get_embedding_dimension()

    def encode(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    def encode_one(self, text: str) -> list[float]:
        return self.encode([text])[0]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return Embedder()
