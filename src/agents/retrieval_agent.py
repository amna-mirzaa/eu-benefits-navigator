"""Agent 2: turn the profile into targeted queries and retrieve grounded chunks from Pinecone."""

from src import config
from src.agents.state import NavigatorState
from src.embeddings import get_embedder
from src.schemas import SourceChunk, UserProfile
from src.vectorstore import get_vectorstore

CATEGORY_PROBES = [
    "housing benefit rent or mortgage support",
    "minimum income or unemployment support",
    "healthcare insurance allowance",
    "family or child benefit",
    "energy or fuel allowance",
]


def build_queries(profile: UserProfile) -> list[str]:
    base = (
        f"{profile.country or ''} social benefits for a person "
        f"{f'aged {profile.age}' if profile.age else ''}, "
        f"employment status: {profile.employment_status or 'unspecified'}, "
        f"household: {profile.household or 'unspecified'}, "
        f"income level: {profile.income_band or 'unspecified'}, "
        f"residency status: {profile.residency_status or 'unspecified'}. "
        f"{profile.notes or ''}"
    ).strip()
    return [f"{base} — {probe}" for probe in CATEGORY_PROBES]


def retrieve_chunks(profile: UserProfile, top_k_per_query: int = 4) -> list[SourceChunk]:
    embedder = get_embedder()
    store = get_vectorstore()
    queries = build_queries(profile)
    vectors = embedder.encode(queries)

    seen: dict[str, SourceChunk] = {}
    for vector in vectors:
        matches = store.query(vector, top_k=top_k_per_query, country=profile.country)
        for chunk in matches:
            existing = seen.get(chunk.chunk_id)
            if existing is None or chunk.score > existing.score:
                seen[chunk.chunk_id] = chunk

    ranked = sorted(seen.values(), key=lambda c: c.score, reverse=True)
    return ranked[: config.TOP_K_RETRIEVAL * 2]


def retrieval_node(state: NavigatorState) -> dict:
    profile = state["profile"]
    chunks = retrieve_chunks(profile)
    schemes = sorted({f"{c.country} – {c.scheme_name}" for c in chunks})
    trace = state.get("trace", []) + [
        f"**Retrieval Agent** — queried Pinecone across {len(CATEGORY_PROBES)} category probes, "
        f"retrieved {len(chunks)} grounded chunks covering: {', '.join(schemes) or 'none found'}"
    ]
    return {"retrieved_chunks": chunks, "trace": trace}
