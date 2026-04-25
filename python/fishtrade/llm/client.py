"""Single Ark exit point for the whole system.

All agents go through :func:`generate_ark_response`.  No agent imports
``openai`` directly — that keeps retries / tracing / schema enforcement in
one place.
"""

from __future__ import annotations

import json
import time
from typing import Any, Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from ..config.settings import settings
from ..observability.trace import write_llm_trace
from .factory import resolve_model_id
from .retry import JSONParseError, ark_retry

T = TypeVar("T", bound=BaseModel)

_CLIENT_SINGLETON: OpenAI | None = None


def create_ark_client() -> OpenAI:
    """Return a process-wide OpenAI client pointed at Ark."""
    global _CLIENT_SINGLETON
    if _CLIENT_SINGLETON is None:
        _CLIENT_SINGLETON = OpenAI(
            api_key=settings.ark_api_key or "missing",
            base_url=settings.ark_base_url,
            timeout=float(settings.ark_timeout_seconds),
            max_retries=0,
        )
    return _CLIENT_SINGLETON


def reset_ark_client() -> None:
    """For tests: drop the cached client so a new one picks up new settings."""
    global _CLIENT_SINGLETON
    _CLIENT_SINGLETON = None


def _schema_hint_for(schema: Type[BaseModel]) -> str:
    body = schema.model_json_schema()
    return (
        "你必须仅返回符合下面 JSON Schema 的 JSON 对象，不要使用 markdown 代码块。\n"
        f"Schema: {json.dumps(body, ensure_ascii=False)}"
    )


def _inject_schema_hint(
    messages: list[dict], schema: Type[BaseModel]
) -> list[dict]:
    """Prepend a system message containing the JSON Schema."""
    hint = {"role": "system", "content": _schema_hint_for(schema)}
    return [hint, *messages]


@ark_retry
def generate_ark_response(
    messages: list[dict],
    *,
    role: str = "default",
    temperature: float = 0.2,
    response_schema: Type[T] | None = None,
    run_id: str = "ad-hoc",
    node_name: str = "unknown",
    extra_kwargs: dict[str, Any] | None = None,
) -> T | str:
    """Call Ark and (optionally) parse the response into a Pydantic model.

    Failures of network / 429 / JSON parsing are retried by :func:`ark_retry`.
    """
    client = create_ark_client()
    model_id = resolve_model_id(role)

    if response_schema is not None:
        messages = _inject_schema_hint(messages, response_schema)

    kwargs: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
    }
    if response_schema is not None:
        kwargs["response_format"] = {"type": "json_object"}
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    t0 = time.perf_counter()
    ok = True
    error: str | None = None
    resp = None
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as exc:
        ok = False
        error = f"{type(exc).__name__}: {exc}"
        latency_ms = int((time.perf_counter() - t0) * 1000)
        write_llm_trace(
            run_id=run_id,
            node=node_name,
            model_id=model_id,
            prompt=messages,
            response=None,
            usage=None,
            latency_ms=latency_ms,
            ok=ok,
            error=error,
        )
        raise

    latency_ms = int((time.perf_counter() - t0) * 1000)
    usage = resp.usage.model_dump() if getattr(resp, "usage", None) else None
    write_llm_trace(
        run_id=run_id,
        node=node_name,
        model_id=model_id,
        prompt=messages,
        response=resp.model_dump(),
        usage=usage,
        latency_ms=latency_ms,
        ok=ok,
        error=error,
    )

    raw = resp.choices[0].message.content or ""
    if response_schema is None:
        return raw

    try:
        return response_schema.model_validate_json(raw)
    except ValidationError as e:
        raise JSONParseError(raw=raw, schema=response_schema, errors=e.errors()) from e
    except json.JSONDecodeError as e:
        raise JSONParseError(raw=raw, schema=response_schema, errors=[str(e)]) from e
