"""System instructions and note file helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import utils
from utils import NOTES_DIR, ensure_dirs

SYSTEM_INSTRUCTIONS_FILE = NOTES_DIR / "system_instructions.txt"


def get_system_instructions() -> str:
    ensure_dirs()
    if not SYSTEM_INSTRUCTIONS_FILE.exists():
        return ""
    return SYSTEM_INSTRUCTIONS_FILE.read_text(encoding="utf-8").strip()


def save_system_instructions(text: str) -> None:
    ensure_dirs()
    SYSTEM_INSTRUCTIONS_FILE.write_text(text.strip() + "\n", encoding="utf-8")


def save_note_to_knowledge_base(content: str) -> Path:
    """
    Save a timestamped markdown note into ~/KnowledgeBase so the watcher indexes it.
    Returns the path of the created file.
    """
    ensure_dirs()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = utils.KNOWLEDGE_BASE / f"note_{timestamp}.md"
    header = f"# Note — {timestamp} (UTC)\n\n"
    path.write_text(header + content.strip() + "\n", encoding="utf-8")
    return path


def load_system_instructions() -> str:
    """Compatibility alias used by the UI."""
    return get_system_instructions()


def index_system_instructions() -> int:
    """Index the system instruction file for retrieval."""
    from ingestion import process_file

    ensure_dirs()
    if not SYSTEM_INSTRUCTIONS_FILE.exists():
        return 0
    return process_file(SYSTEM_INSTRUCTIONS_FILE)


def save_note(content: str) -> Path:
    """
    Save note content into the active knowledge base folder.
    Falls back to ~/MySpace/KnowledgeBase if KB path is missing.
    """
    kb_raw = utils.KNOWLEDGE_BASE
    if kb_raw is None:
        kb_path = Path.home() / "MySpace" / "KnowledgeBase"
    else:
        kb_path = Path(kb_raw)

    kb_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    note_path = kb_path / f"note_{timestamp}.md"
    note_path.write_text(content.strip() + "\n", encoding="utf-8")
    return note_path