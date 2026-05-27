"""Background filesystem watcher for ~/KnowledgeBase."""

from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler

from utils import KNOWLEDGE_BASE, SUPPORTED_EXTENSIONS, ensure_dirs

logger = logging.getLogger(__name__)

_thread: threading.Thread | None = None


def _create_observer():
    """Prefer native FSEvents on macOS; fall back to polling if unavailable."""
    if sys.platform == "darwin":
        try:
            from watchdog.observers import Observer

            return Observer()
        except Exception:
            pass
    try:
        from watchdog.observers.polling import PollingObserver

        return PollingObserver()
    except ImportError:
        from watchdog.observers import Observer

        return Observer()


class KnowledgeBaseHandler(FileSystemEventHandler):
    """Ingest or remove supported files with light debouncing."""

    def __init__(self, debounce_seconds: float = 1.5) -> None:
        self._debounce = debounce_seconds
        # Map of absolute file path -> (last_event_time, action)
        # action is "upsert" for create/modify, "delete" for deletion.
        self._pending: dict[str, tuple[float, str]] = {}

    def _queue(self, path: Path, action: str) -> None:
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return
        self._pending[str(path.resolve())] = (time.monotonic(), action)

    def _flush(self) -> None:
        from ingestion import process_file, remove_file

        now = time.monotonic()
        ready: list[tuple[str, str]] = []
        for path_str, (ts, action) in list(self._pending.items()):
            if now - ts >= self._debounce:
                ready.append((path_str, action))

        for path_str, action in ready:
            del self._pending[path_str]
            path = Path(path_str)
            try:
                if action == "delete":
                    remove_file(path)
                else:
                    if not path.is_file():
                        continue
                    process_file(path)
            except Exception:
                logger.exception("Watcher failed to handle %s (%s)", path, action)

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._queue(Path(event.src_path), "upsert")

    def on_modified(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._queue(Path(event.src_path), "upsert")

    def on_deleted(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._queue(Path(event.src_path), "delete")


def _run_observer(watch_path: Path) -> None:
    handler = KnowledgeBaseHandler()
    observer = _create_observer()
    observer.schedule(handler, str(watch_path), recursive=True)
    try:
        observer.start()
    except Exception:
        logger.exception("Could not start file watcher; use Upload in the UI instead.")
        return

    logger.info("Watching %s", watch_path)
    try:
        while observer.is_alive():
            handler._flush()
            time.sleep(0.3)
    finally:
        observer.stop()
        observer.join()


def start_watcher_daemon(watch_path: Path | None = None) -> None:
    """Start watchdog in a background daemon thread (safe for Streamlit)."""
    global _thread
    if _thread is not None and _thread.is_alive():
        return

    ensure_dirs()
    path = (watch_path or KNOWLEDGE_BASE).resolve()
    path.mkdir(parents=True, exist_ok=True)

    def target() -> None:
        _run_observer(path)

    _thread = threading.Thread(target=target, name="knowledge-watcher", daemon=True)
    _thread.start()
