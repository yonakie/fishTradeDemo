"""JSONL trace writer for LLM calls.

Each call appends one line to ``logs/trace/{run_id}.jsonl``; the full prompt
and response are stored separately in ``logs/trace/{run_id}_messages.jsonl``
keyed by a SHA-256 hash so the index file stays compact.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config.settings import settings

_LOCK = threading.Lock()


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _trace_dir() -> Path:
    p = Path(settings.log_dir) / "trace"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _hash_messages(messages: Any) -> str:
    blob = json.dumps(messages, sort_keys=True, default=str, ensure_ascii=False)
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


def _append_jsonl(path: Path, record: dict) -> None:
    line = json.dumps(record, ensure_ascii=False, default=str)
    with _LOCK:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def write_llm_trace(
    *,
    run_id: str,
    node: str,
    model_id: str,
    prompt: list[dict],
    response: dict | None,
    usage: dict | None,
    latency_ms: int,
    ok: bool = True,
    error: str | None = None,
) -> str:
    """Persist one LLM call. Returns the messages_hash so callers can correlate."""
    messages_hash = _hash_messages(prompt)

    index_path = _trace_dir() / f"{run_id}.jsonl"
    messages_path = _trace_dir() / f"{run_id}_messages.jsonl"

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    if usage:
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")

    record = {
        "ts": _utcnow_iso(),
        "run_id": run_id,
        "node": node,
        "model_id": model_id,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "latency_ms": latency_ms,
        "endpoint": "chat.completions",
        "messages_hash": messages_hash,
        "ok": ok,
    }
    if error:
        record["error"] = error
    _append_jsonl(index_path, record)

    body = {
        "messages_hash": messages_hash,
        "prompt": prompt,
        "response": response,
    }
    _append_jsonl(messages_path, body)

    return messages_hash


def trace_path_for(run_id: str) -> Path:
    """Return the index trace file path for a given run."""
    return _trace_dir() / f"{run_id}.jsonl"


def iter_trace(run_id: str):
    """Yield each trace record for a run; missing file yields nothing."""
    p = trace_path_for(run_id)
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
