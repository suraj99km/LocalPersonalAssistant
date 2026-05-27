"""Desktop launcher for Streamlit app."""

from __future__ import annotations

import multiprocessing
import threading
import time
import webbrowser
from pathlib import Path

from streamlit.web import bootstrap

import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"


def _open_browser_after_delay(url: str, delay_seconds: float = 1.0) -> None:
    def _open() -> None:
        time.sleep(delay_seconds)
        webbrowser.open(url)

    threading.Thread(target=_open, daemon=True).start()


def main() -> None:
    multiprocessing.freeze_support()
    app_file = Path(__file__).resolve().parent / "main.py"
    port = 8501
    _open_browser_after_delay(f"http://localhost:{port}")
    bootstrap.run(
        str(app_file),
        is_hello=False,
        args=[],
        flag_options={
            "server.headless": True,
            "server.port": port,
            "server.enableCORS": False,
            "server.enableXsrfProtection": False,
        },
    )


if __name__ == "__main__":
    main()
