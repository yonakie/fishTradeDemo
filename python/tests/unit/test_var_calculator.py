"""Unit tests for the historical-simulation VaR calculator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fishtrade.tools.var_calculator import compute_var_historical


def _history(n: int, *, seed: int = 42, drift: float = 0.0, vol: float = 0.02) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, size=n)
    prices = 100.0 * np.cumprod(1.0 + rets)
    return pd.DataFrame({"Close": prices})


def test_normal_history_passes():
    hist = _history(252)
    res = compute_var_historical(hist, proposed_position_pct=5.0)
    assert res.passed is True
    assert res.sample_size > 200
    assert res.var_95 > 0
    assert res.method == "historical_simulation"
    # portfolio impact = var_95 * 5%
    assert res.portfolio_impact == pytest.approx(res.var_95 * 0.05, rel=1e-9)


def test_short_history_degrades():
    hist = _history(40)
    res = compute_var_historical(hist, min_samples=60)
    assert res.passed is False
    assert res.fallback_reason is not None
    assert res.var_95 == 0.0
    assert res.portfolio_impact == 0.0


def test_empty_history_degrades():
    res = compute_var_historical(pd.DataFrame())
    assert res.passed is False
    assert res.sample_size == 0


def test_missing_close_column_degrades():
    df = pd.DataFrame({"Open": [1, 2, 3]})
    res = compute_var_historical(df)
    assert res.passed is False
    assert res.fallback_reason is not None


def test_var_is_non_negative_even_with_uniform_gains():
    """All-positive returns ⇒ negative-tail quantile is non-negative ⇒ var_95 floored at 0."""
    hist = pd.DataFrame({"Close": np.linspace(100, 200, 80)})
    res = compute_var_historical(hist)
    assert res.var_95 >= 0


def test_var_zero_position_yields_zero_impact():
    hist = _history(252)
    res = compute_var_historical(hist, proposed_position_pct=0.0)
    assert res.passed is True
    assert res.portfolio_impact == 0.0
