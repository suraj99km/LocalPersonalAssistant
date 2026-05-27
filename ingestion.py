"""Parse documents, chunk, embed, and index into ChromaDB."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import chromadb
import fitz  # PyMuPDF
from chromadb.config import Settings
from docx import Document
from sentence_transformers import SentenceTransformer

from events import push_ingest_event
from sparse_index import refresh_sparse_index
from utils import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    SUPPORTED_EXTENSIONS,
    ChildChunkRecord,
    chunk_parent_child,
    ensure_dirs,
)

logger = logging.getLogger(__name__)

# Single in-process embedding model (~100 MB) — loaded once
_embedder: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def embed_text(text: str) -> list[float]:
    return get_embedder().encode(text).tolist()


def _get_collection():
    ensure_dirs()
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(name=COLLECTION_NAME)


def _child_chunk_id(record: ChildChunkRecord) -> str:
    return f"{record.parent_id}::child{record.child_index}"


def _delete_source_chunks(collection, source: str) -> None:
    """Remove all chunks for a source before re-indexing."""
    try:
        existing = collection.get(where={"source": source})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
    except Exception:
        logger.debug("No existing chunks for %s", source)


def _index_child_records(
    records: list[ChildChunkRecord],
    *,
    source: str,
    page: int,
    display_name: str,
) -> int:
    if not records:
        return 0

    collection = _get_collection()
    ids: list[str] = []
    documents: list[str] = []
    embeddings: list[list[float]] = []
    metadatas: list[dict[str, Any]] = []

    for record in records:
        chunk_id = _child_chunk_id(record)
        ids.append(chunk_id)
        documents.append(record.child_text)
        embeddings.append(embed_text(record.child_text))
        metadatas.append(
            {
                "source": source,
                "filename": display_name,
                "page": page,
                "parent_id": record.parent_id,
                "parent_text": record.parent_text,
                "parent_index": record.parent_index,
                "child_index": record.child_index,
            }
        )

    try:
        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
    except Exception:
        logger.exception("Chroma upsert failed for %s", display_name)
        return 0

    return len(records)


def _finalize_ingest(display_name: str, chunk_count: int, *, status: str = "done") -> None:
    """Refresh BM25 and notify the UI thread via the shared queue."""
    try:
        refresh_sparse_index()
    except Exception:
        logger.exception("Failed to refresh BM25 index after indexing %s", display_name)

    push_ingest_event(
        {
            "status": status,
            "filename": display_name,
            "chunks": chunk_count,
        }
    )


def extract_pages(file_path: Path) -> list[dict[str, Any]]:
    """Return list of {page, text} dicts per file type."""
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        return _extract_pdf(file_path)
    if ext == ".docx":
        return _extract_docx(file_path)
    if ext in {".txt", ".md", ".markdown"}:
        return _extract_plain(file_path)
    raise ValueError(f"Unsupported file type: {ext}")


def _extract_pdf(file_path: Path) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    with fitz.open(file_path) as doc:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text") or ""
            if text.strip():
                pages.append({"page": page_num + 1, "text": text})
    return pages or [{"page": 1, "text": ""}]


def _extract_docx(file_path: Path) -> list[dict[str, Any]]:
    doc = Document(str(file_path))
    full_text = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return [{"page": 1, "text": full_text}]


def _extract_plain(file_path: Path) -> list[dict[str, Any]]:
    text = file_path.read_text(encoding="utf-8", errors="replace")
    return [{"page": 1, "text": text}]


def process_file(file_path: str | Path) -> int:
    """
    Parse, parent-child chunk, embed children, and upsert all pages of a file.
    Returns total child chunks indexed.
    """
    path = Path(file_path).resolve()
    if not path.is_file():
        logger.warning("File not found: %s", path)
        return 0

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        logger.info("Skipping unsupported file: %s", path)
        return 0

    source = str(path)
    display_name = path.name
    collection = _get_collection()
    _delete_source_chunks(collection, source)

    total = 0
    try:
        pages = extract_pages(path)
    except Exception:
        logger.exception("Failed to parse %s", path)
        push_ingest_event(
            {"status": "error", "filename": display_name, "chunks": 0, "error": "parse_failed"}
        )
        return 0

    for page_data in pages:
        page_num = int(page_data["page"])
        text = page_data["text"]
        records = chunk_parent_child(text, source=source, page=page_num)
        total += _index_child_records(
            records,
            source=source,
            page=page_num,
            display_name=display_name,
        )

    logger.info("Indexed %d child chunks from %s", total, path)
    _finalize_ingest(display_name, total)
    _notify_optional(f"Indexed {display_name} ({total} chunks)")
    return total


def remove_file(file_path: str | Path) -> int:
    """
    Remove all indexed chunks for a file that has been deleted from the KB folder.
    Returns the number of chunks removed (best-effort).
    """
    path = Path(file_path).resolve()
    source = str(path)
    display_name = path.name
    collection = _get_collection()

    removed = 0
    try:
        existing = collection.get(where={"source": source})
        ids = existing.get("ids") or []
        if ids:
            removed = len(ids)
            collection.delete(ids=ids)
        else:
            logger.warning("No indexed chunks found for deleted file %s", source)
    except Exception:
        logger.exception("Failed to remove chunks for %s", source)

    logger.info("Removed %d chunks for deleted file %s", removed, source)
    _finalize_ingest(display_name, removed, status="deleted")
    return removed


def index_system_instructions() -> int:
    """Re-index system instructions so they are searchable in RAG."""
    from memory import SYSTEM_INSTRUCTIONS_FILE

    ensure_dirs()
    if not SYSTEM_INSTRUCTIONS_FILE.exists():
        return 0

    text = SYSTEM_INSTRUCTIONS_FILE.read_text(encoding="utf-8")
    source = str(SYSTEM_INSTRUCTIONS_FILE.resolve())
    display_name = "system_instructions.txt"
    collection = _get_collection()
    _delete_source_chunks(collection, source)

    records = chunk_parent_child(text, source=source, page=1)
    total = _index_child_records(
        records,
        source=source,
        page=1,
        display_name=display_name,
    )
    _finalize_ingest(display_name, total)
    return total


def _notify_optional(message: str) -> None:
    try:
        from plyer import notification

        notification.notify(title="Local Personal Assistant", message=message, timeout=4)
    except Exception:
        pass


def reindex_folder(folder: str | Path) -> int:
    """
    Re-index all supported documents under a folder.
    Returns total chunks indexed.
    """
    base = Path(folder).expanduser().resolve()
    if not base.exists() or not base.is_dir():
        logger.warning("Reindex skipped; not a directory: %s", base)
        return 0

    total = 0
    for file_path in sorted(base.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            total += process_file(file_path)
    return total


def clear_index() -> None:
    """Delete and recreate the Chroma collection (nuclear option)."""
    ensure_dirs()
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        # Collection may not exist yet; ignore.
        pass
    client.get_or_create_collection(name=COLLECTION_NAME)
    refresh_sparse_index()


def cleanup_orphans() -> int:
    """
    Remove chunks whose source file no longer exists on disk.
    Returns number of chunks removed (best-effort).
    """
    collection = _get_collection()
    try:
        result = collection.get(include=["metadatas"])
    except Exception:
        logger.exception("Failed to scan collection for orphans")
        return 0

    metadatas = result.get("metadatas") or []
    sources: set[str] = set()
    for meta in metadatas:
        if isinstance(meta, dict):
            src = meta.get("source")
            if src:
                sources.add(str(src))

    removed_total = 0
    for src in sources:
        if not Path(src).exists():
            removed_total += remove_file(src)
    return removed_total
