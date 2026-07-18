"""Pinecone wrapper: index management, upsert, and similarity query."""

from functools import lru_cache
from typing import Optional

from pinecone import Pinecone, ServerlessSpec

from src import config
from src.schemas import SourceChunk


class VectorStore:
    def __init__(self) -> None:
        if not config.has_pinecone_key():
            raise RuntimeError(
                "PINECONE_API_KEY is not set. Add it to your .env file (see .env.example)."
            )
        self._pc = Pinecone(api_key=config.PINECONE_API_KEY)

    def ensure_index(self, dimension: int) -> None:
        existing = {idx["name"] for idx in self._pc.list_indexes()}
        if config.PINECONE_INDEX not in existing:
            self._pc.create_index(
                name=config.PINECONE_INDEX,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
            )

    @property
    def index(self):
        return self._pc.Index(config.PINECONE_INDEX)

    def upsert(self, vectors: list[dict]) -> None:
        """vectors: [{"id": str, "values": [float], "metadata": {...}}, ...]"""
        self.index.upsert(vectors=vectors)

    def query(
        self,
        vector: list[float],
        top_k: int = config.TOP_K_RETRIEVAL,
        country: Optional[str] = None,
    ) -> list[SourceChunk]:
        query_filter = {"country": {"$eq": country}} if country else None
        result = self.index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
            filter=query_filter,
        )
        chunks = []
        for match in result.get("matches", []):
            meta = match.get("metadata", {})
            chunks.append(
                SourceChunk(
                    chunk_id=match["id"],
                    country=meta.get("country", ""),
                    scheme_name=meta.get("scheme_name", ""),
                    category=meta.get("category", ""),
                    source_url=meta.get("source_url", ""),
                    text=meta.get("text", ""),
                    score=match.get("score", 0.0),
                )
            )
        return chunks


@lru_cache(maxsize=1)
def get_vectorstore() -> VectorStore:
    return VectorStore()
