"""diskcache wrapper sanity checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from fishtrade.tools.yf_cache import YFCache, make_cache_key


@pytest.fixture
def tmp_cache(tmp_path: Path) -> YFCache:
    cache = YFCache(cache_dir=tmp_path / "cache", ttl=60)
    yield cache
    cache.close()


def test_make_cache_key_includes_endpoint_and_ticker():
    key = make_cache_key("aapl", "info", "2026-04-25")
    assert key.startswith("info::AAPL::2026-04-25")


def test_set_and_get(tmp_cache: YFCache):
    tmp_cache.set("k1", {"a": 1})
    assert tmp_cache.get("k1") == {"a": 1}
    assert tmp_cache.has("k1") is True


def test_get_missing_returns_default(tmp_cache: YFCache):
    assert tmp_cache.get("missing") is None
    assert tmp_cache.get("missing", default=42) == 42


def test_delete(tmp_cache: YFCache):
    tmp_cache.set("x", "y")
    tmp_cache.delete("x")
    assert tmp_cache.get("x") is None


def test_clear(tmp_cache: YFCache):
    tmp_cache.set("a", 1)
    tmp_cache.set("b", 2)
    tmp_cache.clear()
    assert tmp_cache.get("a") is None
    assert tmp_cache.get("b") is None
