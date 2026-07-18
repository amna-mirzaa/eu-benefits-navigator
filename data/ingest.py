"""Chunk the markdown scheme documents, embed them, and upsert into Pinecone.

Run with: python data/ingest.py
Requires GROQ_API_KEY (not used here) is irrelevant; only PINECONE_API_KEY is
needed since embeddings run locally via sentence-transformers.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.embeddings import get_embedder
from src.vectorstore import get_vectorstore

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)


def parse_frontmatter(raw: str) -> tuple[dict, str]:
    match = FRONTMATTER_RE.match(raw)
    if not match:
        raise ValueError("Missing frontmatter block")
    header, body = match.groups()
    meta = {}
    for line in header.strip().splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta, body.strip()


def chunk_body(body: str) -> list[str]:
    """Split on markdown `##` sections; each section is one retrievable chunk."""
    sections = re.split(r"\n(?=## )", body)
    return [s.strip() for s in sections if s.strip()]


def build_records() -> list[dict]:
    records = []
    for path in sorted(config.SCHEMES_DIR.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        meta, body = parse_frontmatter(raw)
        chunks = chunk_body(body)
        for i, chunk_text in enumerate(chunks):
            records.append(
                {
                    "id": f"{path.stem}-{i}",
                    "text": chunk_text,
                    "country": meta.get("country", ""),
                    "scheme_name": meta.get("scheme_name", ""),
                    "category": meta.get("category", ""),
                    "source_url": meta.get("source_url", ""),
                }
            )
    return records


def main() -> None:
    records = build_records()
    print(f"Built {len(records)} chunks from {len(list(config.SCHEMES_DIR.glob('*.md')))} scheme docs.")

    embedder = get_embedder()
    texts = [r["text"] for r in records]
    vectors = embedder.encode(texts)
    print(f"Embedded with dimension {embedder.dimension}.")

    store = get_vectorstore()
    store.ensure_index(dimension=embedder.dimension)

    upsert_batch = [
        {
            "id": record["id"],
            "values": vector,
            "metadata": {
                "text": record["text"],
                "country": record["country"],
                "scheme_name": record["scheme_name"],
                "category": record["category"],
                "source_url": record["source_url"],
            },
        }
        for record, vector in zip(records, vectors)
    ]
    store.upsert(upsert_batch)
    print(f"Upserted {len(upsert_batch)} vectors into Pinecone index '{config.PINECONE_INDEX}'.")


if __name__ == "__main__":
    main()
