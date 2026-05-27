## Local Private Assistant

A **100% local, private** RAG assistant that turns a folder of documents into a searchable, chat-based knowledge base — **no cloud, no Ollama, no telemetry**.

### Key features
- **Offline + private**: everything runs locally on your machine
- **Streaming answers**: token-by-token output
- **Thinking section**: model reasoning separated from final answer
- **Hybrid retrieval**: dense + BM25 fusion via ChromaDB + rank-bm25
- **Auto-sync**: watches your Knowledge Base folder (adds/updates/deletes)
- **System instructions + quick notes**
- **GPU acceleration (Metal/CUDA)** when available via `llama-cpp-python`

## 📥 Download for macOS

[![Download for macOS](https://img.shields.io/badge/Download-macOS-brightgreen?logo=apple)](https://github.com/suraj99km/LocalPersonalAssistant/releases/latest)

1. Download the `PersonalRAG.dmg` file from the latest release.
2. Open the `.dmg` and drag `PersonalRAG` to your Applications folder.
3. Double-click to launch – your browser will open automatically.
4. No Python, Ollama, or internet required (except for the initial download).

**System Requirements:** macOS 12+ (Monterey or later), 8 GB RAM, ~3.5 GB free disk space.

## Developer setup (run from source)

### Prerequisites
- Python **3.10+**

### Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
python download_model.py
python run_app.py
```

### Knowledge Base folder
By default, documents live in:
- `~/MySpace/KnowledgeBase`

Drop PDFs/DOCX/TXT/MD files there and the app will index them automatically.

## Building the macOS app
See [BUILDING.md](BUILDING.md).

## License
MIT — see [LICENSE](LICENSE).
