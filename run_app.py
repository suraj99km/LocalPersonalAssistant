"""Desktop launcher for Streamlit app."""

from __future__ import annotations

import multiprocessing
import socket
import threading
import time
import webbrowser
from pathlib import Path

from streamlit.web import bootstrap

import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"


def _pick_free_port(preferred: int = 8501) -> int:
    """Pick a free localhost port (prefer 8501)."""
    for port in range(preferred, preferred + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port

    # Fallback: let OS choose ephemeral port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_for_port(host: str, port: int, timeout_s: float = 20.0) -> bool:
    """Wait until TCP port is accepting connections."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def _open_browser_when_ready(url: str, host: str, port: int) -> None:
    def _open() -> None:
        # Give Streamlit some time to start binding.
        _wait_for_port(host, port, timeout_s=25.0)
        webbrowser.open(url)

    threading.Thread(target=_open, daemon=True).start()


def main() -> None:
    multiprocessing.freeze_support()
    app_file = Path(__file__).resolve().parent / "main.py"
    host = "127.0.0.1"
    port = _pick_free_port(8501)
    url = f"http://{host}:{port}"
    _open_browser_when_ready(url, host, port)
    bootstrap.run(
        str(app_file),
        is_hello=False,
        args=[],
        flag_options={
            "server.headless": True,
            "server.port": port,
            "server.address": host,
            "server.enableCORS": False,
            "server.enableXsrfProtection": False,
            "browser.gatherUsageStats": False,
        },
    )


if __name__ == "__main__":
    main()
