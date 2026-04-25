"""Unit tests for technical indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fishtrade.tools.indicators_tech import (
    compute_all_technical,
    compute_atr,
    compute_bollinger,
    compute_macd,
    compute_moving_averages,
    compute_relative_strength,
    compute_rsi,
)


def _ohlcv(n: int, *, trend: float = 0.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = 100.0 * np.cumprod(1.0 + rng.normal(trend, 0.015, size=n))
    highs = closes * (1 + np.abs(rng.normal(0, 0.005, size=n)))
    lows = closes * (1 - np.abs(rng.normal(0, 0.005, size=n)))
    opens = closes * (1 + rng.normal(0, 0.002, size=n))
    vols = rng.integers(1_000_000, 5_000_000, size=n)
    idx = pd.RangeIndex(n)
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": vols},
        index=idx,
    )


def test_compute_macd_returns_dict_for_long_history():
    hist = _ohlcv(80)
    m = compute_macd(hist)
    assert m is not None
    assert {"macd", "signal", "hist", "golden_cross", "death_cross"}.issubset(m)


def test_compute_macd_short_history_returns_none():
    assert compute_macd(_ohlcv(20)) is None


def test_compute_rsi_in_valid_range():
    rsi = compute_rsi(_ohlcv(60))
    assert rsi is not None
    assert 0 <= rsi <= 100


def test_compute_rsi_short_returns_none():
    assert compute_rsi(_ohlcv(5)) is None


def test_compute_moving_averages_for_long_history():
    ma = compute_moving_averages(_ohlcv(220))
    assert ma is not None
    assert ma["sma20"] is not None
    assert ma["sma200"] is not None


def test_compute_moving_averages_short_history_returns_none():
    assert compute_moving_averages(_ohlcv(30)) is None


def test_compute_bollinger():
    b = compute_bollinger(_ohlcv(60))
    assert b is not None
    assert b["lower"] <= b["middle"] <= b["upper"]


def test_compute_atr():
    atr = compute_atr(_ohlcv(60))
    assert atr is not None
    assert atr > 0


def test_compute_relative_strength_with_benchmark():
    a = _ohlcv(60, trend=0.005, seed=1)
    b = _ohlcv(60, trend=0.0, seed=2)
    rs = compute_relative_strength(a, b)
    assert rs is not None


def test_compute_relative_strength_missing_benchmark_returns_none():
    assert compute_relative_strength(_ohlcv(60), None) is None


def test_compute_all_technical_returns_exactly_ten():
    md = {"history": _ohlcv(220), "benchmark_history": _ohlcv(220, seed=2)}
    scores = compute_all_technical(md)
    assert len(scores) == 10
    total = sum(s.score for s in scores)
    assert -10 <= total <= 10


def test_compute_all_technical_with_empty_history_degrades_all():
    scores = compute_all_technical({"history": pd.DataFrame()})
    assert len(scores) == 10
    assert all(s.is_degraded for s in scores)
    assert all(s.score == 0 for s in scores)


def test_compute_all_technical_handles_missing_market_data():
    scores = compute_all_technical({})
    assert len(scores) == 10
