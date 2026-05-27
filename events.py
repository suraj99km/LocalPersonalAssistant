"""Thread-safe ingest notifications from background threads to the Streamlit UI."""

from __future__ import annotations

import logging
import queue
from typing import Any

logger = logging.getLogger(__name__)

INGEST_EVENTS: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=200)


def push_ingest_event(payload: dict[str, Any]) -> None:
    """Enqueue a status payload (safe from watcher / ingestion threads)."""
    try:
        INGEST_EVENTS.put_nowait(payload)
    except queue.Full:
        logger.warning("Ingest event queue full; dropping: %s", payload.get("filename"))


def drain_ingest_events() -> list[dict[str, Any]]:
    """Drain all pending events on the main Streamlit thread."""
    events: list[dict[str, Any]] = []
    while True:
        try:
            events.append(INGEST_EVENTS.get_nowait())
        except queue.Empty:
            break
    return events
