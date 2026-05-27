from __future__ import annotations

import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any
import os
import re
import urllib.request

import streamlit as st
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import utils
from events import drain_ingest_events
from memory import load_system_instructions, save_note, save_system_instructions
from utils import ERROR_LOG, MODEL_PATH, ensure_dirs, existing_model_path, log_usage, set_knowledge_base_path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

CHAT_FILE_TYPES = ["pdf", "docx", "txt", "md"]
MAX_MESSAGES = 20  # 10 user/assistant pairs

MODEL_URL = (
    "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/"
    "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
)


def _append_error_log(exc: Exception) -> None:
    ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
    with ERROR_LOG.open("a", encoding="utf-8") as f:
        f.write(f"{exc}\n{traceback.format_exc()}\n")


def _download_model_with_progress(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink()

    progress = st.progress(0, text="Downloading model…")
    status = st.empty()

    def reporthook(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        pct = int(min(100, (downloaded / total_size) * 100))
        progress.progress(pct, text="Downloading model…")
        mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        status.caption(f"{mb:.1f} MB / {total_mb:.1f} MB")

    urllib.request.urlretrieve(MODEL_URL, tmp, reporthook=reporthook)
    tmp.replace(dest)
    progress.progress(100, text="Download complete")
    status.empty()


def _ensure_model_available() -> bool:
    model_on_disk = existing_model_path()
    if model_on_disk.exists() and model_on_disk.stat().st_size > 0:
        return True

    st.warning("The Llama 3.2 model is required to answer questions.")
    st.caption(f"Model path: `{MODEL_PATH}`")
    st.caption("This is a one-time download. After that, everything runs offline.")
    if st.button("Download model", key="download_model_btn", type="primary"):
        try:
            _download_model_with_progress(MODEL_PATH)
            # If a legacy lowercase file exists, prefer canonical by renaming.
            legacy = MODEL_PATH.parent / "llama-3.2-3b-instruct-q4_k_m.gguf"
            if legacy.exists() and not MODEL_PATH.exists():
                legacy.replace(MODEL_PATH)
            st.toast("Model downloaded", icon="✅")
            st.rerun()
        except Exception as exc:
            _append_error_log(exc)
            st.error(f"Failed to download model: {exc}")
    return False


def _init_session() -> None:
    defaults: dict[str, Any] = {
        "messages": [],
        "last_assistant_msg_idx": None,
        "feedback_given": {},
        "processing": False,
        "indexing_status": "idle",
        "confirm_clear": False,
        "confirm_clear_index": False,
        "pending_error": "",
        "system_instructions_input": load_system_instructions(),
        "quick_note_input": "",
        "kb_path_input": str(utils.KNOWLEDGE_BASE),
        "chat_prompt": "",
        "pending_query": "",
        "pending_retrieved_chunks": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def on_clear_index_request() -> None:
    st.session_state.confirm_clear_index = True


def on_clear_index_cancel() -> None:
    st.session_state.confirm_clear_index = False


def on_clear_index_confirmed() -> None:
    try:
        st.session_state.indexing_status = "indexing"
        from ingestion import clear_index, reindex_folder

        clear_index()
        total = reindex_folder(utils.KNOWLEDGE_BASE)
        st.toast("Index cleared successfully", icon="✅")
        st.toast(f"Re-indexed {total} chunks", icon="✅")
    except Exception as exc:
        _append_error_log(exc)
        st.session_state.pending_error = f"Failed to clear index: {exc}"
    finally:
        st.session_state.indexing_status = "idle"
        st.session_state.confirm_clear_index = False


@st.cache_resource
def _start_watcher_once() -> bool:
    try:
        from watcher import start_watcher_daemon

        start_watcher_daemon()
    except Exception as exc:
        logger.warning("Watcher startup failed: %s", exc)
    return True


@st.cache_resource
def _cleanup_orphans_once() -> bool:
    try:
        from ingestion import cleanup_orphans

        removed = cleanup_orphans()
        if removed:
            logger.info("Orphan cleanup removed %d chunks", removed)
    except Exception as exc:
        logger.warning("Orphan cleanup skipped: %s", exc)
    return True


def _clear_pending_error() -> None:
    st.session_state.pending_error = ""


def _normalize_history_window() -> None:
    if len(st.session_state.messages) <= MAX_MESSAGES:
        return
    overflow = len(st.session_state.messages) - MAX_MESSAGES
    st.session_state.messages = st.session_state.messages[overflow:]
    st.session_state.feedback_given = {}
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        st.session_state.last_assistant_msg_idx = len(st.session_state.messages) - 1
    else:
        st.session_state.last_assistant_msg_idx = None


def _save_uploaded_file(uploaded: Any) -> Path:
    ensure_dirs()
    path = utils.KNOWLEDGE_BASE / uploaded.name
    path.write_bytes(uploaded.getbuffer())
    return path


def _drain_ingest_notifications() -> None:
    events = drain_ingest_events()
    if not events:
        if st.session_state.indexing_status != "idle":
            st.session_state.indexing_status = "idle"
        return
    for event in events:
        status = event.get("status", "done")
        filename = event.get("filename", "file")
        chunks = event.get("chunks", 0)
        if status == "error":
            st.toast(f"Failed to index {filename}", icon="⚠️")
        elif status == "deleted":
            st.toast(f"Removed {filename} from index", icon="🗑️")
        else:
            st.toast(f"Indexed {filename} ({chunks} chunks)", icon="✅")
    st.session_state.indexing_status = "idle"


def on_chat_submit() -> None:
    query = st.session_state.chat_prompt.strip()
    if not query or st.session_state.processing:
        return
    st.session_state.pending_query = query
    st.session_state.chat_prompt = ""
    st.session_state.processing = True


def on_save_instructions() -> None:
    try:
        save_system_instructions(st.session_state.system_instructions_input)
        from memory import index_system_instructions

        st.session_state.indexing_status = "indexing"
        chunks = index_system_instructions()
        st.toast(f"System instructions saved ({chunks} chunks indexed)", icon="✅")
    except Exception as exc:
        _append_error_log(exc)
        st.session_state.pending_error = f"Could not save instructions: {exc}"
    finally:
        st.session_state.indexing_status = "idle"


def on_save_note() -> None:
    note = st.session_state.quick_note_input.strip()
    if not note:
        st.toast("Note is empty", icon="⚠️")
        return
    try:
        path = save_note(note)
        from ingestion import process_file

        st.session_state.indexing_status = "indexing"
        chunks = process_file(path)
        st.session_state.quick_note_input = ""
        st.toast(f"Note saved ({chunks} chunks indexed)", icon="✅")
    except Exception as exc:
        _append_error_log(exc)
        st.session_state.pending_error = f"Could not save note: {exc}"
    finally:
        st.session_state.indexing_status = "idle"


def on_file_upload() -> None:
    uploaded_files = st.session_state.get("kb_upload")
    if not uploaded_files:
        return
    if not isinstance(uploaded_files, list):
        uploaded_files = [uploaded_files]
    total = 0
    indexed_files = 0
    try:
        from ingestion import process_file

        st.session_state.indexing_status = "indexing"
        for file_obj in uploaded_files:
            path = _save_uploaded_file(file_obj)
            chunks = process_file(path)
            total += chunks
            indexed_files += 1
        st.toast(f"Indexed {indexed_files} file(s), {total} chunks", icon="✅")
    except Exception as exc:
        _append_error_log(exc)
        st.session_state.pending_error = f"File upload/indexing failed: {exc}"
    finally:
        st.session_state.indexing_status = "idle"
        # Do not assign to st.session_state["kb_upload"] (Streamlit-managed widget state).


def on_feedback(helpful: bool, msg_idx: int) -> None:
    if st.session_state.feedback_given.get(msg_idx, False):
        return
    if msg_idx < 0 or msg_idx >= len(st.session_state.messages):
        return
    assistant_msg = st.session_state.messages[msg_idx]
    user_msg = st.session_state.messages[msg_idx - 1] if msg_idx > 0 else {"content": ""}
    log_usage(
        query=user_msg.get("content", ""),
        answer=assistant_msg.get("content", ""),
        helpful=helpful,
    )
    st.session_state.feedback_given[msg_idx] = True
    st.toast("Feedback saved", icon="✅")


def on_save_response_as_note(msg_idx: int) -> None:
    if msg_idx < 0 or msg_idx >= len(st.session_state.messages):
        return
    msg = st.session_state.messages[msg_idx]
    if msg.get("role") != "assistant":
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = utils.KNOWLEDGE_BASE / f"saved_response_{timestamp}.md"
    file_path.write_text(msg.get("content", "").strip() + "\n", encoding="utf-8")
    st.toast(f"Saved to {file_path.name}", icon="✅")


def on_clear_request() -> None:
    st.session_state.confirm_clear = True


def on_clear_cancel() -> None:
    st.session_state.confirm_clear = False


def on_clear_confirmed() -> None:
    st.session_state.messages = []
    st.session_state.last_assistant_msg_idx = None
    st.session_state.feedback_given = {}
    st.session_state.confirm_clear = False
    st.toast("Conversation cleared", icon="✅")


def _apply_kb_path_to_modules(new_path: Path) -> None:
    set_knowledge_base_path(new_path)
    import memory
    import watcher

    memory.KNOWLEDGE_BASE = utils.KNOWLEDGE_BASE
    watcher.KNOWLEDGE_BASE = utils.KNOWLEDGE_BASE


def _reindex_folder(folder: Path) -> int:
    from ingestion import process_file
    from utils import SUPPORTED_EXTENSIONS

    total_chunks = 0
    for file_path in sorted(folder.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            total_chunks += process_file(file_path)
    return total_chunks


def on_kb_path_change() -> None:
    new_path_raw = (st.session_state.get("kb_path_input", "") or "").strip()
    if not new_path_raw:
        # Ignore empty input and keep the previous valid KB path.
        st.session_state.kb_path_input = str(utils.KNOWLEDGE_BASE)
        return

    try:
        kb_path = Path(new_path_raw).expanduser()
        kb_path.mkdir(parents=True, exist_ok=True)
        set_knowledge_base_path(kb_path)
        st.session_state.kb_path_input = str(utils.KNOWLEDGE_BASE)
        st.success(f"Knowledge Base set to {utils.KNOWLEDGE_BASE}")
    except Exception as exc:
        st.error(f"Path is invalid or could not be created: {exc}")


def _source_caption(sources: list[dict[str, Any]]) -> str:
    seen: set[tuple[str, int]] = set()
    labels: list[str] = []
    for src in sources:
        key = (src.get("filename", "unknown"), int(src.get("page", 1)))
        if key in seen:
            continue
        seen.add(key)
        labels.append(f"📄 {key[0]} (p.{key[1]})")
    return " • ".join(labels)


def _extract_cited_sources(answer_text: str) -> set[tuple[str, int]]:
    """
    Extract citations like [filename, p.1] or [filename, p1] from model output.
    Returns a set of (filename, page).
    """
    cited: set[tuple[str, int]] = set()
    if not answer_text:
        return cited

    # Match: [Some File.pdf, p.12] or [Some File.pdf, p12]
    pattern = re.compile(r"\[([^,\[\]]+?),\s*p\.?\s*(\d+)\]", flags=re.IGNORECASE)
    for m in pattern.finditer(answer_text):
        filename = m.group(1).strip()
        try:
            page = int(m.group(2))
        except Exception:
            continue
        if filename and page > 0:
            cited.add((filename, page))
    return cited


def _render_sidebar() -> None:
    with st.sidebar:
        st.text_input(
            "📁 Knowledge Base Path",
            key="kb_path_input",
            on_change=on_kb_path_change,
        )

        st.markdown("### 📝 Quick Note")
        st.text_area(
            "Quick note",
            key="quick_note_input",
            height=120,
            label_visibility="collapsed",
            placeholder="Capture something important...",
        )
        st.button(
            "Save Note",
            key="save_note_btn",
            on_click=on_save_note,
            use_container_width=True,
        )

        st.markdown("### 📥 Upload")
        st.file_uploader(
            "Upload documents",
            key="kb_upload",
            type=CHAT_FILE_TYPES,
            accept_multiple_files=True,
            on_change=on_file_upload,
            label_visibility="collapsed",
        )

        st.markdown("### ⚙️ Instructions")
        st.text_area(
            "System instructions",
            key="system_instructions_input",
            height=120,
            label_visibility="collapsed",
            placeholder="Set assistant behavior...",
        )
        st.button(
            "Save Instructions",
            key="save_instructions_btn",
            on_click=on_save_instructions,
            use_container_width=True,
        )

        with st.expander("Advanced", expanded=False):
            st.button(
                "⚠️ Clear Entire Index",
                key="clear_index_request_btn",
                use_container_width=True,
                on_click=on_clear_index_request,
            )
            if st.session_state.confirm_clear_index:
                st.warning("This will delete ALL indexed knowledge. Continue?")
                col_yes, col_no = st.columns(2)
                with col_yes:
                    st.button(
                        "Yes, clear index",
                        key="clear_index_confirm_btn",
                        use_container_width=True,
                        on_click=on_clear_index_confirmed,
                    )
                with col_no:
                    st.button(
                        "Cancel",
                        key="clear_index_cancel_btn",
                        use_container_width=True,
                        on_click=on_clear_index_cancel,
                    )

        st.divider()
        st.button(
            "🗑️ Clear Chat",
            key="clear_request_btn",
            use_container_width=True,
            on_click=on_clear_request,
        )
        if st.session_state.confirm_clear:
            st.warning("This will delete current chat history.")
            col_yes, col_no = st.columns(2)
            with col_yes:
                st.button(
                    "Yes, clear",
                    key="clear_confirm_btn",
                    on_click=on_clear_confirmed,
                    use_container_width=True,
                )
            with col_no:
                st.button(
                    "Cancel",
                    key="clear_cancel_btn",
                    on_click=on_clear_cancel,
                    use_container_width=True,
                )

        if st.session_state.indexing_status == "indexing":
            st.caption("⏳ Indexing…")
        else:
            st.caption("● System Status: Ready")

        if "gpu_active" in st.session_state:
            if st.session_state.gpu_active:
                st.caption("● GPU acceleration: Active")
            else:
                st.caption("● GPU acceleration: Inactive (CPU only)")


def _render_messages() -> None:
    if not st.session_state.messages:
        return

    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and msg.get("thinking"):
                st.caption("💭 Thinking")
                st.markdown(msg["thinking"])
                st.divider()

            st.markdown(msg.get("content", ""))

            if msg["role"] == "assistant":
                sources = msg.get("sources") or []
                caption = _source_caption(sources)
                if caption:
                    st.caption(caption)

                st.button(
                    "💾 Save response as note",
                    key=f"save_{idx}",
                    on_click=on_save_response_as_note,
                    args=(idx,),
                )

                if idx == st.session_state.last_assistant_msg_idx:
                    disabled = st.session_state.feedback_given.get(idx, False)
                    col1, col2, _ = st.columns([1, 1, 8])
                    with col1:
                        st.button(
                            "👍",
                            key=f"helpful_{idx}",
                            disabled=disabled,
                            on_click=on_feedback,
                            args=(True, idx),
                        )
                    with col2:
                        st.button(
                            "👎",
                            key=f"not_helpful_{idx}",
                            disabled=disabled,
                            on_click=on_feedback,
                            args=(False, idx),
                        )


def _render_and_process_pending_query() -> None:
    query = st.session_state.pending_query.strip()
    if not query:
        return

    st.session_state.pending_query = ""
    with st.chat_message("user"):
        st.markdown(query)

    assistant_text = ""
    thinking_text = ""
    chunks: list[dict[str, Any]] = []

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching..."):
                from generator import build_messages, is_gpu_active, stream_answer_with_cot
                from retrieval import retrieve

                chunks = retrieve(query, top_k=8)
                system_prompt, user_prompt = build_messages(query, chunks)
                # Model is loaded lazily inside generator; capture acceleration status after first use.
                st.session_state.gpu_active = is_gpu_active()

            phase_placeholder = st.empty()
            phase_placeholder.caption("🤔 Thinking...")
            thinking_parts: list[str] = []
            seen_answer = False

            def _answer_stream() -> Any:
                nonlocal seen_answer
                for token_type, token in stream_answer_with_cot(system_prompt, user_prompt):
                    if token_type == "thinking":
                        if token:
                            thinking_parts.append(token)
                    elif token_type == "answer":
                        if not seen_answer:
                            phase_placeholder.caption("✍️ Writing answer...")
                            seen_answer = True
                        if token:
                            yield token
                    elif token_type == "done":
                        break

            assistant_text = st.write_stream(_answer_stream()) or ""
            thinking_text = "".join(thinking_parts).strip()
            phase_placeholder.empty()

            if thinking_text:
                st.caption("💭 Thinking")
                st.markdown(thinking_text)
                st.divider()
        except FileNotFoundError as exc:
            _append_error_log(exc)
            st.session_state.pending_error = (
                f"{exc} Download the model file and place it under the models folder."
            )
            assistant_text = "I could not load the local model file."
        except Exception as exc:
            _append_error_log(exc)
            st.session_state.pending_error = f"Failed to generate answer: {exc}"
            assistant_text = "Something went wrong while generating the response."
        finally:
            st.session_state.processing = False

        sources = [
            {
                "filename": c.get("filename", "unknown"),
                "page": c.get("page", 1),
                "source": c.get("source", ""),
            }
            for c in chunks
        ]

        # Only show/store sources that were actually cited in the answer text.
        cited = _extract_cited_sources(assistant_text)
        if cited:
            sources = [
                s
                for s in sources
                if (str(s.get("filename", "")).strip(), int(s.get("page", 1))) in cited
            ]

        caption = _source_caption(sources)
        if caption:
            st.caption(caption)

    st.session_state.messages.append({"role": "user", "content": query, "sources": None})
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": assistant_text,
            "sources": sources,
            "thinking": thinking_text,
        }
    )
    st.session_state.last_assistant_msg_idx = len(st.session_state.messages) - 1
    _normalize_history_window()
    st.rerun()


def main() -> None:
    st.set_page_config(page_title="Personal RAG", page_icon="📚", layout="wide")
    ensure_dirs()
    _init_session()
    _start_watcher_once()
    _cleanup_orphans_once()
    _drain_ingest_notifications()

    st.title("Local Personal AI Assistant")
    st.caption("Llama 3.2 · 100% private · runs locally")
    _render_sidebar()

    if st.session_state.pending_error:
        st.error(st.session_state.pending_error)
        st.button("Dismiss", key="dismiss_error_btn", on_click=_clear_pending_error)

    _render_messages()

    # If model is missing, guide the user to download it and disable chat.
    model_ready = _ensure_model_available()
    if model_ready:
        _render_and_process_pending_query()
        st.chat_input(
            "Ask anything about your documents...",
            key="chat_prompt",
            on_submit=on_chat_submit,
            disabled=st.session_state.processing,
        )
    else:
        st.chat_input(
            "Ask anything about your documents...",
            key="chat_prompt",
            disabled=True,
        )


main()