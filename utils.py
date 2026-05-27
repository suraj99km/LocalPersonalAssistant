"""Helpers: chunking, paths, and usage metrics logging."""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).resolve().parent
else:
    BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma"
NOTES_DIR = DATA_DIR / "notes"
USAGE_LOG = DATA_DIR / "usage_log.jsonl"
ERROR_LOG = DATA_DIR / "error.log"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "Llama-3.2-3B-Instruct-Q4_K_M.gguf"

_default_kb = Path.home() / "MySpace" / "KnowledgeBase"
KNOWLEDGE_BASE = Path(os.environ.get("KNOWLEDGE_BASE", str(_default_kb))).expanduser()

# Legacy flat chunking (kept for reference / tests)
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Parent-child chunking
PARENT_CHUNK_SIZE = 1800
CHILD_CHUNK_SIZE = 400
CHILD_OVERLAP = 50

COLLECTION_NAME = "knowledge_base"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".markdown"}


@dataclass(frozen=True)
class ChildChunkRecord:
    parent_id: str
    parent_text: str
    child_text: str
    parent_index: int
    child_index: int


def ensure_dirs() -> None:
    """Create runtime data directories and the watched knowledge folder."""
    for raw_path in (DATA_DIR, CHROMA_DIR, NOTES_DIR, KNOWLEDGE_BASE, MODELS_DIR):
        p = Path(raw_path)          # ← force conversion, even if it’s already a Path
        p.mkdir(parents=True, exist_ok=True)

    # Write default system instructions if they don’t exist
    instructions = Path(NOTES_DIR) / "system_instructions.txt"
    if not instructions.exists():
        instructions.write_text(
            "You are a precise assistant. Answer only from the provided context.\n"
            "If the context lacks the answer, say 'I don't know.'\n"
            "Always cite source documents and page numbers."
        )


def set_knowledge_base_path(path: str | Path) -> Path:
    """Update the global knowledge base folder path at runtime."""
    global KNOWLEDGE_BASE
    candidate = Path(path).expanduser()
    KNOWLEDGE_BASE = candidate.resolve()
    ensure_dirs()
    return KNOWLEDGE_BASE


def _merge_paragraphs_to_chunks(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """Split by paragraphs, merge to ~chunk_size chars with overlap."""
    if not text or not text.strip():
        return []

    paragraphs = re.split(r"\n\s*\n", text.strip())
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            overlap_text = current[-overlap:] if len(current) > overlap else current
            current = f"{overlap_text}{para}".strip()
            while len(current) > chunk_size:
                chunks.append(current[:chunk_size])
                current = current[chunk_size - overlap :]

    if current:
        chunks.append(current.strip())

    return [c for c in chunks if c]


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Legacy flat chunking (~500 chars)."""
    return _merge_paragraphs_to_chunks(text, chunk_size, overlap)


def _split_parent_into_children(
    parent_text: str,
    child_size: int = CHILD_CHUNK_SIZE,
    overlap: int = CHILD_OVERLAP,
) -> list[str]:
    """Character-based child splits with overlap within one parent."""
    text = parent_text.strip()
    if not text:
        return []
    if len(text) <= child_size:
        return [text]

    children: list[str] = []
    start = 0
    while start < len(text):
        end = start + child_size
        children.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return [c for c in children if c]


def chunk_parent_child(
    text: str,
    *,
    source: str,
    page: int,
    parent_size: int = PARENT_CHUNK_SIZE,
    child_size: int = CHILD_CHUNK_SIZE,
    child_overlap: int = CHILD_OVERLAP,
) -> list[ChildChunkRecord]:
    """
    Build parent chunks (~1800 chars) then overlapping child chunks (~400 chars).
    Child text is embedded/indexed; parent text is stored in metadata for the LLM.
    """
    parents = _merge_paragraphs_to_chunks(text, parent_size, overlap=0)
    records: list[ChildChunkRecord] = []

    for parent_index, parent_text in enumerate(parents):
        parent_id = f"{source}::p{page}::parent{parent_index}"
        children = _split_parent_into_children(parent_text, child_size, child_overlap)
        for child_index, child_text in enumerate(children):
            records.append(
                ChildChunkRecord(
                    parent_id=parent_id,
                    parent_text=parent_text,
                    child_text=child_text,
                    parent_index=parent_index,
                    child_index=child_index,
                )
            )
    return records


def log_usage(query: str, answer: str, helpful: bool | None) -> None:
    """Append one JSON line to usage_log.jsonl for success metrics."""
    record: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "answer": answer,
        "helpful": helpful,
    }
    with open(USAGE_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
