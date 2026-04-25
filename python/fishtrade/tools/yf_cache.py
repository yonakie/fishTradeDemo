"""diskcache-backed key/value store for yfinance responses.

Keys carry the ``ticker``, the endpoint name and the ``as_of_date`` so that
two runs on different dates do not collide (and so that the cache survives
across CLI invocations).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from diskcache import Cache

from ..config.settings import settings


def make_cache_key(ticker: str, endpoint: str, as_of: str, **extra: Any) -> str:
    parts = [endpoint, ticker.upper(), as_of]
    if extra:
        parts.append(json.dumps(extra, sort_keys=True, default=str))
    return "::".join(parts)


class YFCache:
    """Thin wrapper around :class:`diskcache.Cache` with a default TTL.

    Stores arbitrary picklable objects.  The default TTL is taken from
    :class:`fishtrade.config.settings.Settings` (``YF_CACHE_TTL_SECONDS``).
    """

    def __init__(self, cache_dir: str | Path | None = None, ttl: int | None = None):
        if cache_dir is None:
            cache_dir = Path(settings.data_dir) / "cache"
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        self._cache = Cache(str(cache_dir))
        self._ttl = int(ttl if ttl is not None else settings.yf_cache_ttl_seconds)

    @property
    def ttl(self) -> int:
        return self._ttl

    def get(self, key: str, default: Any = None) -> Any:
        return self._cache.get(key, default=default)

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        self._cache.set(key, value, expire=ttl if ttl is not None else self._ttl)

    def has(self, key: str) -> bool:
        return key in self._cache

    def delete(self, key: str) -> None:
        try:
            del self._cache[key]
        except KeyError:
            pass

    def clear(self) -> None:
        self._cache.clear()

    def close(self) -> None:
        self._cache.close()
