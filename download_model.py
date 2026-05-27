from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

from utils import MODEL_PATH, MODELS_DIR


MODEL_URL = (
    "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/"
    "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)

    def reporthook(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        pct = min(100.0, (downloaded / total_size) * 100.0)
        sys.stdout.write(f"\rDownloading model... {pct:5.1f}%")
        sys.stdout.flush()

    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink(missing_ok=True)  # type: ignore[arg-type]

    urllib.request.urlretrieve(url, tmp, reporthook=reporthook)
    sys.stdout.write("\n")
    sys.stdout.flush()
    tmp.replace(dest)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
        print(f"Model already present: {MODEL_PATH}")
        return

    print(f"Downloading GGUF model to: {MODEL_PATH}")
    _download(MODEL_URL, MODEL_PATH)
    print(f"Done: {MODEL_PATH}")


if __name__ == "__main__":
    main()

