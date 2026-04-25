"""Tenacity-based retry policy for Ark LLM calls.

Three failure modes are retried:
  * APITimeoutError / APIConnectionError — network hiccups
  * RateLimitError                       — Ark 429
  * JSONParseError                       — model returned non-JSON or schema mismatch

After a JSON failure the wrapped function receives an extra system message
asking for strict JSON; this nudges the model on the second attempt.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Type

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

try:
    from openai import APIConnectionError, APITimeoutError, RateLimitError
except Exception:  # pragma: no cover - openai always available, defensive only
    class APITimeoutError(Exception):
        pass

    class APIConnectionError(Exception):
        pass

    class RateLimitError(Exception):
        pass


class JSONParseError(ValueError):
    """Raised when the LLM response cannot be parsed into the expected schema."""

    def __init__(
        self,
        raw: str,
        schema: Type[Any] | None,
        errors: list | None = None,
    ) -> None:
        self.raw = raw
        self.schema = schema
        self.errors = errors or []
        schema_name = schema.__name__ if schema is not None else "?"
        super().__init__(
            f"Failed to parse LLM response into {schema_name}: {self.errors}"
        )


_RETRY_EXCEPTIONS = (
    APITimeoutError,
    APIConnectionError,
    RateLimitError,
    JSONParseError,
)


_REPAIR_NUDGE = (
    "上次返回不是合法的 JSON 对象。请仅返回 JSON 对象，不要使用 markdown 代码块，"
    "不要附加解释文字。务必严格符合提供的 JSON Schema。"
)


def _inject_repair_nudge(messages: list[dict]) -> list[dict]:
    """Append a system reminder so the second attempt yields valid JSON."""
    repaired = list(messages)
    repaired.append({"role": "system", "content": _REPAIR_NUDGE})
    return repaired


def ark_retry(func: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap an Ark caller with retry + JSON repair nudging.

    The decorated function must accept ``messages`` as either positional[0]
    or a keyword argument.
    """

    base = retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(_RETRY_EXCEPTIONS),
    )(func)

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return base(*args, **kwargs)
        except JSONParseError:
            messages = kwargs.get("messages")
            if messages is None and args:
                messages = args[0]
                args = (_inject_repair_nudge(messages),) + args[1:]
            else:
                kwargs["messages"] = _inject_repair_nudge(messages)
            return base(*args, **kwargs)

    return wrapper
