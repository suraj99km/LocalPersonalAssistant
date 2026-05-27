"""Prompt construction and in-process llama-cpp streaming generation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

from llama_cpp import Llama

from memory import load_system_instructions
from utils import MODEL_PATH

BASE_SYSTEM = (
    "You are a precise assistant that answers only using the provided context. "
    "If the context does not contain the answer, say 'I do not know based on the provided documents.' "
    "Cite sources as [filename, p.X] for factual claims."
)
COT_SYSTEM_SUFFIX = (
    "You are a precise research assistant. Before answering, think step-by-step inside "
    "<think>...</think> tags. After that, provide your final answer outside the tags. "
    "Answer ONLY using the provided context. Cite sources."
)

_llm: Llama | None = None
_gpu_active: bool = False


def is_gpu_active() -> bool:
    return _gpu_active


def get_llm() -> Llama:
    """Load the llama.cpp model lazily once per process."""
    global _llm, _gpu_active
    model_file = Path(MODEL_PATH)
    if not model_file.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_file}. "
            "Download Llama-3.2-3B-Instruct-Q4_K_M.gguf using the in-app downloader "
            "or run: python download_model.py"
        )
    if _llm is None:
        try:
            _llm = Llama(
                model_path=str(model_file),
                n_ctx=4096,
                n_threads=max(1, (os.cpu_count() or 4) - 1),
                n_gpu_layers=-1,
                verbose=False,
            )
            _gpu_active = True
        except Exception:
            # Fall back to CPU-only if GPU backend is unavailable.
            _llm = Llama(
                model_path=str(model_file),
                n_ctx=4096,
                n_threads=max(1, (os.cpu_count() or 4) - 1),
                n_gpu_layers=0,
                verbose=False,
            )
            _gpu_active = False
    return _llm


def _build_system_message() -> str:
    custom = load_system_instructions()
    if custom:
        return f"{BASE_SYSTEM}\n\nAdditional instructions:\n{custom}"
    return BASE_SYSTEM


def _format_context(chunks: list[dict]) -> str:
    if not chunks:
        return "(No relevant context retrieved.)"

    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[Document {i}: {chunk.get('filename', 'unknown')}, page {chunk.get('page', 1)}]\n"
            f"{chunk.get('text', '')}"
        )
    return "\n\n".join(parts)


def build_prompt(system_prompt: str, user_prompt: str) -> str:
    return (
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def build_messages(query: str, chunks: list[dict]) -> tuple[str, str]:
    context = _format_context(chunks)
    user_content = f"Context:\n{context}\n\nQuestion: {query}"
    return _build_system_message(), user_content


def stream_answer(system_prompt: str, user_prompt: str) -> Iterator[str]:
    """Stream completion tokens from llama-cpp."""
    prompt = build_prompt(system_prompt, user_prompt)
    llm = get_llm()
    stream = llm(
        prompt,
        max_tokens=1024,
        temperature=0.1,
        repeat_penalty=1.1,
        stream=True,
    )
    for chunk in stream:
        token = chunk.get("choices", [{}])[0].get("text", "")
        if token:
            yield token


def stream_answer_with_cot(system_prompt: str, user_prompt: str) -> Iterator[tuple[str, str | None]]:
    """
    Stream tagged model output as typed events:
    - ("thinking", token)
    - ("answer", token)
    - ("done", None)
    """
    augmented_system_prompt = f"{system_prompt}\n\n{COT_SYSTEM_SUFFIX}"
    prompt = build_prompt(augmented_system_prompt, user_prompt)
    llm = get_llm()
    stream = llm(
        prompt,
        max_tokens=1024,
        temperature=0.1,
        repeat_penalty=1.1,
        stream=True,
    )

    open_tag = "<think>"
    close_tag = "</think>"
    open_keep = len(open_tag) - 1
    close_keep = len(close_tag) - 1

    mode = "pre"  # pre -> thinking -> answer
    buffer = ""
    saw_open_tag = False

    for chunk in stream:
        token = chunk.get("choices", [{}])[0].get("text", "")
        if not token:
            continue
        buffer += token

        while buffer:
            if mode == "pre":
                open_idx = buffer.find(open_tag)
                if open_idx != -1:
                    saw_open_tag = True
                    pre = buffer[:open_idx]
                    if pre:
                        yield ("answer", pre)
                    buffer = buffer[open_idx + len(open_tag) :]
                    mode = "thinking"
                    continue

                # No <think> yet. Keep enough tail to catch split tags.
                if len(buffer) > open_keep:
                    emit = buffer[:-open_keep]
                    if emit:
                        yield ("answer", emit)
                    buffer = buffer[-open_keep:]
                break

            if mode == "thinking":
                close_idx = buffer.find(close_tag)
                if close_idx != -1:
                    thought = buffer[:close_idx]
                    if thought:
                        yield ("thinking", thought)
                    buffer = buffer[close_idx + len(close_tag) :]
                    mode = "answer"
                    continue

                if len(buffer) > close_keep:
                    emit = buffer[:-close_keep]
                    if emit:
                        yield ("thinking", emit)
                    buffer = buffer[-close_keep:]
                break

            # answer mode
            if buffer:
                yield ("answer", buffer)
                buffer = ""

    if buffer:
        if mode == "thinking" and saw_open_tag:
            yield ("thinking", buffer)
        else:
            yield ("answer", buffer)
    yield ("done", None)


def generate_answer(query: str, chunks: list[dict]) -> str:
    """Non-streaming fallback — collects full response."""
    system_prompt, user_prompt = build_messages(query, chunks)
    return "".join(stream_answer(system_prompt, user_prompt))
