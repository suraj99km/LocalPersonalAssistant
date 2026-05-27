# Project Constitution: Personal RAG Assistant (Local RAG)

## 1. Vision & Mission
**Vision**  
A zero‑setup, double‑click desktop application that turns any folder of documents into a private, conversational knowledge base – running entirely offline, with no cloud dependency or telemetry.

**Mission**  
Deliver an open‑source personal AI assistant for consultants, MBAs, and knowledge workers that is:
- **Private** – all data stays local.
- **Fast** – answers in seconds with streaming.
- **Simple** – no install of Python, Ollama, or Docker; just unzip and run.
- **Reliable** – no unexplained UI errors, every button works, graceful degradation when services are missing.

## 2. System Architecture (High‑Level)

┌─────────────────────────────────────────────────────┐
│ Streamlit UI (Browser) │
│ - Chat interface (multi-line input) │
│ - Sidebar (settings, notes, upload, file list) │
└──────────────────┬──────────────────────────────────┘
│ (local network, port 8501)
┌──────────────────▼──────────────────────────────────┐
│ Launcher (run_app.py) │
│ - Starts Streamlit server in-process │
│ - Opens default browser │
└──────────────────┬──────────────────────────────────┘
│
┌──────────────────▼──────────────────────────────────┐
│ Core Application (main.py) │
│ - Manages session state & chat history │
│ - Background watcher thread for auto-indexing │
│ - Handles user interactions (callbacks) │
└──┬───────────────┬───────────────┬──────────────────┘
│ │ │
▼ ▼ ▼
┌────────┐ ┌────────────┐ ┌────────────┐
│Watcher │ │ Retrieval │ │ Generator │
│(watch- │ │(hybrid: │ │(LLM stream)│
│ dog) │ │ dense+sparse│ │ │
└───┬────┘ └─────┬──────┘ └─────┬──────┘
│ │ │
▼ ▼ ▼
┌─────────────────────────────────────────────────────┐
│ Data & Models (Local) │
│ - ChromaDB (vector store) │
│ - GGUF model (in-process via llama-cpp) │
│ - SentenceTransformer (embeddings, in-memory) │
│ - File system: knowledge_base/, data/notes/ │
└─────────────────────────────────────────────────────┘


## 3. Technology Stack (Final, 8 GB‑friendly)
| Layer | Technology | Justification |
|-------|------------|--------------|
| **LLM** | `llama-cpp-python` with Llama-3.2-3B-Instruct Q4_K_M GGUF | In-process, no server needed, ~2 GB RAM |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | 100 MB, 384‑dim, CPU-friendly |
| **Vector DB** | ChromaDB (persistent, local) | Simple API, zero‑config |
| **Sparse retrieval** | `rank_bm25` (in‑memory rebuild per query) | Lightweight, no extra service |
| **UI** | Streamlit 1.31+ (theme: dark, layout: wide) | Fast prototyping, rich widgets |
| **Watcher** | `watchdog` | Cross‑platform file system events |
| **Packaging** | PyInstaller (onedir) with all dependencies bundled | Single folder, double‑click executable |
| **Language** | Python 3.10+ with type hints | Clarity and maintainability |

## 4. Directory Structure (Must Match Exactly)

local-rag/
├── run_app.py # Entry point: starts Streamlit, opens browser
├── main.py # Streamlit UI, session state, callbacks
├── watcher.py # Background folder observer
├── ingestion.py # PDF/DOCX/TXT parsing, chunking, embedding, indexing
├── retrieval.py # Dense + BM25 + RRF fusion
├── generator.py # LLM prompt builder & llama-cpp streaming
├── memory.py # Notes & system instructions file management
├── utils.py # Paths, chunk_text(), log_usage()
├── PROJECT_SPEC.md # This constitution
├── requirements.txt # Exact pip dependencies
├── models/ # (created at build) holds GGUF file
│ └── llama-3.2-3b-instruct.Q4_K_M.gguf
├── data/ # Created at runtime
│ ├── chroma/ # ChromaDB persistence
│ └── notes/ # system_instructions.txt, user note files
├── .streamlit/
│ └── config.toml # Dark theme, wide mode, collapsed sidebar
└── dist/ # PyInstaller output (not in repo)


## 5. Detailed Module Specifications

### 5.1 `run_app.py` – Application Launcher
- **Purpose**: Bootstrap Streamlit server in headless mode and open the default browser. This is the PyInstaller entry point.
- **Responsibilities**:
  - Set Streamlit server options: headless, port 8501, no CORS, no XSRF (for local use).
  - Call `streamlit.web.bootstrap.run(main_script)` with `main.py`.
  - Handle `multiprocessing.freeze_support()` on Windows.
- **No user interaction**; exits when Streamlit stops.

### 5.2 `main.py` – Streamlit UI & Orchestration
- **Purpose**: The central hub. All user interactions are handled here via callbacks. No heavy imports at startup (deferred).
- **Session State Schema** (all keys):
  - `messages`: list of `{"role": "user" | "assistant", "content": str, "sources": list[dict] | None}`.
  - `last_assistant_msg_idx`: int or None (index of last assistant message, for feedback buttons).
  - `feedback_given`: dict mapping message index to bool (helpful/unhelpful). Prevents duplicate feedback.
  - `processing`: bool – True while retrieval+generation is running.
  - `indexing_status`: str – current status of background indexing (“idle”, “indexing N files…”).
  - `show_settings`: bool – toggles settings expander in sidebar.
  - `confirm_clear`: bool – whether to show clear‑conversation confirmation.
- **UI Layout**:
  - **Sidebar** (collapsed by default, toggle with hamburger):
    - “Open Knowledge Base Folder” button → opens file manager.
    - `st.expander(“System Instructions”)`: text area + save button (callback: save & re-index).
    - `st.expander(“Quick Note”)`: text area + save button (callback: save note to KB, triggers indexing).
    - File uploader (drag & drop) – inside an expander or directly.
    - `st.expander(“Clear Conversation”)`: confirmation dialog with “Yes, clear” / “Cancel”.
    - Status indicators: “Indexing…” spinner or text.
  - **Main Chat Area**:
    - Chat history display using `st.chat_message`.
    - Feedback buttons (👍/👎) appear only under the last assistant message, disabled after click.
    - Multi‑line input area at the bottom (custom, see Section 6).
- **Callbacks** (defined as top‑level functions in `main.py`):
  - `on_send()`: triggered when user sends a message. Sets `st.session_state.processing = True`, runs retrieval → generation, appends messages, sets `last_assistant_msg_idx`.
  - `on_save_instructions()`: saves text to file, re-indexes it, shows toast.
  - `on_save_note()`: saves note file, triggers `process_file()` (or watcher), toast.
  - `on_feedback(helpful: bool, msg_idx: int)`: logs to `utils.log_usage()`, marks feedback given.
  - `on_clear_confirmed()`: clears messages, resets feedback dict.
  - `on_file_upload()`: saves uploaded file to KB folder, calls `process_file()` with status.
  - `on_open_kb_folder()`: opens OS file explorer using `webbrowser.open` on the directory path.
- **Error Prevention**:
  - All widget keys must be unique and deterministic.
  - Buttons inside loops use index as part of the key.
  - No inline `if st.button(...)`; all buttons trigger callbacks.
  - Use `st.rerun()` only after state mutations, never from within a callback that already mutated state.

### 5.3 `watcher.py` – Background Folder Watcher
- **Purpose**: Monitor `~/MySpace/KnowledgeBase` (configurable) for new/modified files.
- **Implementation**: `watchdog.observers.Observer` in a daemon thread.
- **Debounce**: 1.5 seconds after last event for a file path before calling `ingestion.process_file()`.
- **Communication**: Sets a thread‑safe flag or uses a queue that `main.py` checks periodically (or simply uses `st.session_state.indexing_status` via a callback that reruns). For MVP, the watcher directly calls `process_file` but ensures it doesn't conflict with UI thread (Streamlit script runs are thread‑safe for data operations? We'll use a lock on Chroma writes if needed, but simple sequential processing is fine because Chroma’s `add` is thread‑safe. Use a `threading.Lock` around Chroma operations just in case).
- **Startup**: Launched once in `main.py` when session state `watcher_started` is False; set to True after starting.

### 5.4 `ingestion.py` – Document Processing
- **Exported functions**: `process_file(file_path: str) -> None`
- **Workflow**:
  1. Determine file type by extension.
  2. Extract text: PDF (PyMuPDF per page), DOCX (full text as single page), TXT/MD (single page).
  3. Chunk text using `utils.chunk_text()` (paragraphs, ~500 chars, 50 overlap).
  4. For each chunk, generate embedding via `embed_model.encode(chunk).tolist()`.
  5. Delete existing chunks for the same `file_path` from Chroma (by metadata query or ID prefix).
  6. Upsert new chunks with IDs: `f"{file_path}::p{page}::c{chunk_index}"`, metadata `{"source": file_path, "filename": os.path.basename(file_path), "page": page, "chunk_index": i}`.
  7. Send desktop notification (plyer) on completion (optional).
- **Globals**: `embed_model` is loaded once at module level (lazy load using a function `get_embed_model()`).
- **Error handling**: Catch and log exceptions; show user‑friendly error toasts via a callback if called from UI.

### 5.5 `retrieval.py` – Hybrid Retrieval
- **Exported function**: `retrieve(query: str, top_k: int = 8) -> list[dict]`
  Each result dict: `{"text": str, "source": str, "page": int, "filename": str}`
- **Steps**:
  1. **Dense search**: Embed query, query Chroma `knowledge_base` collection (n_results=20).
  2. **Sparse search**: Fetch all documents from Chroma (cached? No, just `collection.get()`), tokenize, build BM25 index, retrieve top 20.
  3. **Fusion**: Reciprocal Rank Fusion (RRF) with k=60. Merge lists based on (source, page, chunk_index) key. Sort by combined score, take top `top_k`.
  4. **Hydrate**: Return full text and metadata for the top chunks (using Chroma `get` by IDs).
- **Performance**: BM25 rebuild per query is acceptable for up to 10k chunks. For MVP, it's fine.

### 5.6 `generator.py` – LLM Answer Generation (Streaming)
- **Model**: `llama-cpp-python` Llama instance, loaded once (lazy).
- **Exported generator function**: `stream_answer(system_prompt: str, user_prompt: str) -> Iterator[str]`
- **Prompt Format** (Llama 3.2 Instruct official template):

<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

- **Parameters**: `max_tokens=1024, temperature=0.1, repeat_penalty=1.1, stream=True`
- **Fallback**: If model file missing, raise a clear `FileNotFoundError` that UI catches and shows a friendly message with a link to download the model.

### 5.7 `memory.py` – Notes & Instructions Management
- **Functions**:
- `load_system_instructions() -> str`
- `save_system_instructions(content: str) -> None`
- `save_note(content: str) -> None` (saves timestamped .md in KB folder)
- `index_system_instructions()`: calls `process_file` on the instructions file so it's searchable.
- **Paths**: Uses `utils.DATA_DIR / "notes" / "system_instructions.txt"`.

### 5.8 `utils.py` – Shared Utilities
- **Constants** (from environment or defaults):
- `BASE_DIR` = directory of `run_app.py` (when frozen: `sys.executable` parent).
- `KNOWLEDGE_BASE` = `~/MySpace/KnowledgeBase` (expanduser).
- `CHROMA_PATH` = `BASE_DIR / "data" / "chroma"`.
- `NOTES_PATH` = `BASE_DIR / "data" / "notes"`.
- `USAGE_LOG` = `BASE_DIR / "data" / "usage_log.jsonl"`.
- **Functions**:
- `ensure_dirs()`: create all necessary directories.
- `chunk_text(text: str, chunk_size: int=500, overlap: int=50) -> list[str]`
- `log_usage(query, answer, helpful)`: appends JSON line.

## 6. UI/UX Specification (Pixel‑Perfect Intent)
### 6.1 Multi‑line Chat Input
**Implementation**:
- Use a `st.container()` at the bottom of the chat.
- Inside, two columns: `col1, col2 = st.columns([5,1])`
- `col1` → `st.text_area("Ask anything…", key="user_input", height=120, label_visibility="collapsed")`
- `col2` → `st.button("Send", key="send_btn", use_container_width=True)`
- **Keyboard**: Press Enter to send (JavaScript snippet via `st.components.v1.html` to capture Enter, but Streamlit’s text area sends value on Ctrl+Enter by default; we can use `on_change` callback on text area to detect if the last character is newline and trigger send). Simpler: use a form with `st.form_submit_button` inside the container, which submits on Enter automatically. However, forms block widget reactivity. A reliable pattern: a hidden `st.text_input` for capturing Enter events? That's hacky.
- **Best practice**: Implement a small custom component using `streamlit.components.v1.html` that wraps a textarea and a send button, capturing Enter and Shift+Enter. Or, accept the default text area behavior (Ctrl+Enter) and add a tooltip. For MVP, use a text area with a “Send” button, and advise user: “Ctrl+Enter to add a new line, click Send or press Enter in the button.” We'll use an `on_change` callback on the text area to detect when the user pressed Enter (by checking if `st.session_state.user_input` ends with `\n` and the string is not empty), but this may interfere. A more robust method: use `st.chat_input` for simplicity and just accept single‑line; but requirement is multi‑line. So implement the form approach: wrap the text area and button in `st.form(key="chat_form")`; the form submit button will send on Enter, and Shift+Enter for newline works inside the text area because the form intercepts Enter only when not modifying text? Actually, inside a form, pressing Enter in the text area submits the form unless `ctrl+enter` is used. That might be acceptable. We'll use a form with `clear_on_submit=False` (since Streamlit forms clear automatically?) We'll handle that.
- **Final decision**: Use a `st.form` named “chat_form” containing `st.text_area` (height 120) and `st.form_submit_button(“Send”)`. The text area's value is captured; after submission, clear it by resetting the session state key `"user_input"` to `""` inside the callback. This is clean and native.

### 6.2 Feedback Mechanism
- Under the last assistant message, two buttons: 👍 (key: `f"helpful_{msg_idx}"`) and 👎 (key: `f"not_helpful_{msg_idx}"`).
- Use columns to place them neatly.
- Callbacks: `on_feedback(True, idx)` and `on_feedback(False, idx)`. These set `st.session_state.feedback_given[idx] = True` and call `log_usage()`.
- After feedback given, disable both buttons (set `disabled` based on `st.session_state.feedback_given.get(idx)`).

### 6.3 Progress Indicators
- **Indexing**: When `process_file` is called from UI, use `st.status("Indexing...", expanded=True)` as context manager. Inside, update status: “Extracting text…”, “Chunking…”, “Embedding…”. After completion, change to “Indexing complete!” with a success state. For background watcher, use `st.toast` (if available) or a session state message that displays in the sidebar.
- **Answer generation**: Wrap retrieval+generation in `st.status("Generating answer...")`. Update: “Searching documents…” → “Retrieved 8 relevant chunks” → “Generating answer…” → then streamed text appears in chat (outside status). After completion, close status.

### 6.4 State & Button Reliability
- Every button uses `on_click=callback_name, args=(...)`. No inline logic.
- Widget keys are explicitly unique. For dynamic elements (feedback buttons), include message index.
- Avoid re-creating the same widget key in a rerun by using stable keys based on persistent data (e.g., `f"send_{uuid}"` is bad; use `"send_btn"` always).
- Clear conversation uses a two‑step confirmation: an expander with a red “Clear all messages” button that sets `st.session_state.confirm_clear = True`. If that flag is true, show another small box with “Are you sure? This cannot be undone.” and “Yes, clear” / “Cancel” buttons. The “Yes, clear” callback clears messages and resets the flag.

### 6.5 Empty State
- If no documents are indexed, show a welcoming message in the chat area with a graphic or emoji, and instructions to drop files.
- If Ollama model is missing, show a specific error with a download button (links to the GGUF file or gives CLI command).

### 6.6 Responsive Layout
- Use `st.set_page_config(layout="wide")` already.
- The chat container should use `st.container(height=calc(...))` to scroll, but Streamlit doesn't support fixed height containers well. Instead, we'll let the natural page flow scroll, using `st.empty()` to pin the input area at the bottom? That's complex. Streamlit's native chat works like a message list that grows upward, with input at bottom; that's sufficient. So we'll use `st.chat_message` for each message and the form at the bottom.

## 7. Error Handling Strategy
- **Catch exceptions** in `on_send` and show `st.error` with the exception message and a retry button.
- **Ollama connection error**: wrap generator initialization in try/except, show “LLM not available” with instructions to download model or check path.
- **Chroma errors**: show user-friendly message; rarely occurs.
- **File parsing errors**: skip unsupported files, show warning toast.
- **All log errors** to a `error.log` file in data directory.

## 8. Packaging & Distribution
- Use **PyInstaller** with a spec file:
- Entry point: `run_app.py`
- Add data: `models/*.gguf`, `.streamlit/config.toml`, maybe `data/` directory structure.
- Hidden imports: all modules listed in requirements.
- **onedir** mode for fast startup, not onefile (which extracts to temp and is slow).
- Build commands:
```bash
pyinstaller --name PersonalRAG --add-data "models/*.gguf:models" --add-data ".streamlit/config.toml:.streamlit" --collect-all sentence_transformers --collect-all chromadb --collect-all streamlit --hidden-import llama_cpp --hidden-import ... run_app.py



