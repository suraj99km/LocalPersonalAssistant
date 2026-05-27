"""Thread-safe in-memory BM25 index over Chroma child chunk documents."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from rank_bm25 import BM25Okapi

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _load_corpus_from_chroma() -> tuple[list[str], list[str]]:
    """Lazy import avoids circular dependency at module load."""
    from ingestion import _get_collection

    collection = _get_collection()
    count = collection.count()
    if count == 0:
        return [], []

    try:
        result = collection.get(include=["documents"])
    except Exception:
        logger.exception("Failed to load documents from Chroma for BM25 rebuild")
        return [], []

    return (
        result.get("ids") or [],
        result.get("documents") or [],
    )


class SparseIndex:
    """Stateful BM25 over child chunk text; rebuild after ingestion changes."""

    _instance: SparseIndex | None = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._bm25: BM25Okapi | None = None
        self._ids: list[str] = []

    @classmethod
    def get_instance(cls) -> SparseIndex:
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def rebuild_from_chroma(self) -> int:
        """Reload all chunk documents from Chroma and rebuild BM25."""
        ids, documents = _load_corpus_from_chroma()
        with self._lock:
            self._ids = ids
            if not documents:
                self._bm25 = None
                logger.debug("BM25 index cleared (empty corpus)")
                return 0
            tokenized = [doc.lower().split() for doc in documents]
            self._bm25 = BM25Okapi(tokenized)
            logger.info("BM25 index rebuilt (%d chunks)", len(documents))
            return len(documents)

    def get_top_k(self, query: str, k: int) -> list[tuple[str, int]]:
        """Return (chunk_id, rank) for top-k sparse matches; rank is 0-based."""
        if not query.strip():
            return []

        with self._lock:
            if self._bm25 is None or not self._ids:
                return []
            bm25 = self._bm25
            ids = self._ids

        try:
            scores = bm25.get_scores(query.lower().split())
        except Exception:
            logger.exception("BM25 scoring failed")
            return []

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:k]

        return [
            (ids[i], rank)
            for rank, i in enumerate(ranked_indices)
            if scores[i] > 0
        ]


def get_sparse_index() -> SparseIndex:
    return SparseIndex.get_instance()


def refresh_sparse_index() -> int:
    """Rebuild BM25 from Chroma (call after upserts/deletes)."""
    return get_sparse_index().rebuild_from_chroma()
