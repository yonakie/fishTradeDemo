"""LLM client: schema injection + factory + tracing wiring (no live calls)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from fishtrade.llm import client as client_mod
from fishtrade.llm.client import _inject_schema_hint, generate_ark_response, reset_ark_client
from fishtrade.llm.factory import resolve_model_id


class _Toy(BaseModel):
    answer: str
    score: int


def test_inject_schema_hint_prepends_system_with_schema():
    msgs = [{"role": "user", "content": "hi"}]
    out = _inject_schema_hint(msgs, _Toy)
    assert out[0]["role"] == "system"
    assert "JSON Schema" in out[0]["content"]
    schema = json.dumps(_Toy.model_json_schema(), ensure_ascii=False)
    assert schema in out[0]["content"]
    # original messages preserved
    assert out[1] == msgs[0]


def test_resolve_model_id_falls_back_to_default(monkeypatch):
    from fishtrade.config import settings as settings_mod

    monkeypatch.setattr(settings_mod.settings, "ark_model_id", "default-ep")
    monkeypatch.setattr(settings_mod.settings, "ark_model_id_research", None)
    assert resolve_model_id("research") == "default-ep"


def test_generate_ark_response_writes_trace(tmp_path: Path, monkeypatch):
    """Mock the Ark client so we can assert the trace file is appended."""
    from fishtrade.config import settings as settings_mod

    monkeypatch.setattr(settings_mod.settings, "log_dir", tmp_path)
    monkeypatch.setattr(settings_mod.settings, "ark_model_id", "ep-mock")

    fake_resp = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hello"))],
        usage=SimpleNamespace(model_dump=lambda: {"prompt_tokens": 7, "completion_tokens": 3}),
        model_dump=lambda: {"id": "fake"},
    )
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_resp

    reset_ark_client()
    with patch.object(client_mod, "create_ark_client", return_value=fake_client):
        out = generate_ark_response(
            messages=[{"role": "user", "content": "say hi"}],
            run_id="run-test",
            node_name="unit_test",
        )
    assert out == "hello"

    trace_file = tmp_path / "trace" / "run-test.jsonl"
    assert trace_file.exists()
    line = trace_file.read_text(encoding="utf-8").strip().splitlines()[0]
    rec = json.loads(line)
    assert rec["run_id"] == "run-test"
    assert rec["node"] == "unit_test"
    assert rec["prompt_tokens"] == 7
    assert rec["ok"] is True
