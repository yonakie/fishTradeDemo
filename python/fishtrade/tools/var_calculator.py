"""Historical-simulation VaR — pure function, no LLM."""

from __future__ import annotations

import pandas as pd

from ..models.risk import VarResult


def compute_var_historical(
    history: pd.DataFrame,
    *,
    lookback_days: int = 252,
    confidence: float = 0.95,
    proposed_position_pct: float = 0.0,
    min_samples: int = 60,
) -> VarResult:
    """Compute single-name daily VaR by historical simulation.

    The function is defensive about sparse data: short histories trigger a
    documented degradation rather than an exception.

    Parameters
    ----------
    history:
        OHLCV DataFrame with at least a ``Close`` column.
    lookback_days:
        Number of trailing returns to use (default 252 ≈ 1 trading year).
    confidence:
        e.g. ``0.95`` for VaR(95).
    proposed_position_pct:
        Position size in percent (0..100), used to compute portfolio_impact.
    min_samples:
        Minimum daily returns required; below this we degrade.
    """
    if history is None or not isinstance(history, pd.DataFrame) or history.empty:
        return VarResult(
            var_95=0.0,
            portfolio_impact=0.0,
            passed=False,
            sample_size=0,
            fallback_reason="历史数据为空，VaR 不可计算",
        )

    if "Close" not in history.columns:
        return VarResult(
            var_95=0.0,
            portfolio_impact=0.0,
            passed=False,
            sample_size=0,
            fallback_reason="历史数据缺少 Close 列",
        )

    closes = pd.to_numeric(history["Close"], errors="coerce").dropna()
    if len(closes) < min_samples + 1:
        return VarResult(
            var_95=0.0,
            portfolio_impact=0.0,
            passed=False,
            sample_size=int(len(closes)),
            fallback_reason=f"历史数据 <{min_samples} 个交易日，VaR 不可靠",
        )

    returns = closes.pct_change().dropna().tail(lookback_days)
    if returns.empty:
        return VarResult(
            var_95=0.0,
            portfolio_impact=0.0,
            passed=False,
            sample_size=0,
            fallback_reason="无法生成有效收益率序列",
        )

    quantile = returns.quantile(1.0 - confidence)
    var_95 = float(max(0.0, -quantile))
    portfolio_impact = float(var_95 * max(0.0, proposed_position_pct) / 100.0)

    return VarResult(
        var_95=var_95,
        portfolio_impact=portfolio_impact,
        passed=True,
        sample_size=int(len(returns)),
    )
