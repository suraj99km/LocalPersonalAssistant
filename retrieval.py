"""Hybrid retrieval: dense (Chroma) + sparse (BM25) + RRF fusion."""

from __future__ import annotations

from typing import Any

from ingestion import _get_collection, embed_text
from sparse_index import get_sparse_index

DENSE_TOP_K = 20
SPARSE_TOP_K = 20
RRF_K = 60
FINAL_TOP_K = 8
# Fetch extra fused hits so parent dedup can still fill FINAL_TOP_K slots
FUSED_CANDIDATE_LIMIT = 40


def _dense_search(query: str, top_k: int = DENSE_TOP_K) -> list[tuple[str, int]]:
    """Return ranked list of (chunk_id, rank) — rank is 0-based."""
    collection = _get_collection()
    if collection.count() == 0:
        return []

    n = min(top_k, collection.count())
    try:
        results = collection.query(
            query_embeddings=[embed_text(query)],
            n_results=n,
            include=["metadatas"],
        )
    except Exception:
        return []

    ids = results["ids"][0] if results["ids"] else []
    return [(chunk_id, rank) for rank, chunk_id in enumerate(ids)]


def _sparse_search(query: str, top_k: int = SPARSE_TOP_K) -> list[tuple[str, int]]:
    """Sparse search via the in-memory BM25 singleton."""
    return get_sparse_index().get_top_k(query, top_k)


def _reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, int]]],
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for chunk_id, rank in ranked:
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _context_text_from_meta(doc: str, meta: dict[str, Any]) -> str:
    """Prefer parent chunk text for the LLM; fall back to child/legacy document text."""
    parent_text = meta.get("parent_text")
    if parent_text and str(parent_text).strip():
        return str(parent_text)
    return doc


def _hydrate_chunks(fused: list[tuple[str, float]], top_k: int) -> list[dict[str, Any]]:
    """
    Hydrate fused child chunk IDs; return up to FINAL_TOP_K unique parents
    with parent-level context text for generation.
    """
    if not fused:
        return []

    candidate_ids = [chunk_id for chunk_id, _ in fused[:FUSED_CANDIDATE_LIMIT]]
    collection = _get_collection()
    try:
        result = collection.get(ids=candidate_ids, include=["documents", "metadatas"])
    except Exception:
        return []

    by_id: dict[str, tuple[str, dict[str, Any]]] = {}
    for chunk_id, doc, meta in zip(
        result["ids"] or [],
        result["documents"] or [],
        result["metadatas"] or [],
    ):
        by_id[chunk_id] = (doc, meta or {})

    chunks: list[dict[str, Any]] = []
    seen_parents: set[str] = set()

    for chunk_id, score in fused:
        if len(chunks) >= top_k:
            break
        if chunk_id not in by_id:
            continue

        doc, meta = by_id[chunk_id]
        parent_id = str(meta.get("parent_id") or chunk_id)
        if parent_id in seen_parents:
            continue
        seen_parents.add(parent_id)

        chunks.append(
            {
                "id": chunk_id,
                "text": _context_text_from_meta(doc, meta),
                "filename": str(meta.get("filename", "unknown")),
                "page": int(meta.get("page", 1)),
                "source": str(meta.get("source", "")),
                "score": score,
                "metadata": meta,
            }
        )
    return chunks


def retrieve(query: str, top_k: int = FINAL_TOP_K) -> list[dict[str, Any]]:
    """
    Hybrid search pipeline:
    dense top-20 + sparse top-20 → RRF (k=60) → top-8 unique parent contexts.
    """
    if not query.strip():
        return []

    dense_ranks = _dense_search(query, DENSE_TOP_K)
    sparse_ranks = _sparse_search(query, SPARSE_TOP_K)

    if not dense_ranks and not sparse_ranks:
        return []

    lists = [r for r in (dense_ranks, sparse_ranks) if r]
    if len(lists) == 1:
        fused = [(cid, 1.0 / (RRF_K + rank + 1)) for cid, rank in lists[0]]
    else:
        fused = _reciprocal_rank_fusion(lists, k=RRF_K)

    return _hydrate_chunks(fused, top_k)
