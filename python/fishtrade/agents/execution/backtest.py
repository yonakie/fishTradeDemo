"""Backtest execution — simulate fill at as_of_date close, no broker call."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from ...models.execution import ExecutionResult, FillInfo
from ...models.state import GraphState
from ...tools.yf_client import payload_to_df
from ._helpers import build_order, determine_side


def _close_on(history: pd.DataFrame, as_of_date: str) -> float | None:
    if history is None or not isinstance(history, pd.DataFrame) or history.empty:
        return None
    if "Close" not in history.columns:
        return None
    closes = history["Close"].dropna()
    if closes.empty:
        return None
    # Try to match the as_of_date row (string-keyed index after payload_to_df).
    if as_of_date:
        for idx in closes.index:
            if str(idx).startswith(as_of_date):
                return float(closes.loc[idx])
    return float(closes.iloc[-1])


def _coerce_history(payload: Any) -> pd.DataFrame:
    if isinstance(payload, dict):
        return payload_to_df(payload)
    if isinstance(payload, pd.DataFrame):
        return payload
    return pd.DataFrame()


def backtest_node(state: GraphState) -> dict:
    debate = state.get("debate") or {}
    risk = state.get("risk") or {}
    run_input = state.get("input") or {}
    market_data = state.get("market_data") or {}
    info = market_data.get("info") or {}

    pct = float(risk.get("adjusted_position_pct") or 0)
    capital = float(run_input.get("capital") or 0)
    symbol = run_input.get("ticker", "UNKNOWN")
    as_of_date = run_input.get("as_of_date", "")

    history = _coerce_history(market_data.get("history"))
    fill_price = _close_on(history, as_of_date) or float(info.get("regularMarketPrice") or 0)

    if pct <= 0 or capital <= 0 or fill_price <= 0:
        result = ExecutionResult(
            mode="backtest",
            order=None,
            status="skipped",
            error="invalid_inputs_for_order",
        )
        return {"execution": result.model_dump()}

    portfolio_before = state.get("portfolio_before") or {}
    has_position = any(
        (p.get("symbol") if isinstance(p, dict) else None) == symbol
        for p in (portfolio_before.get("positions") or [])
    )
    side = determine_side(debate.get("final_verdict", "BUY"), has_position)

    order = build_order(
        symbol=symbol,
        side=side,
        price=fill_price,
        capital=capital,
        pct=pct,
    )
    fill_info = FillInfo(
        avg_price=fill_price,
        filled_qty=order.qty,
        fill_time=(
            f"{as_of_date}T16:00:00Z"
            if as_of_date
            else datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        ),
    )
    result = ExecutionResult(
        mode="backtest",
        order=order,
        status="filled",
        fill_info=fill_info,
        broker_order_id=f"BT-{symbol}-{as_of_date or 'now'}",
    )
    return {"execution": result.model_dump()}


__all__ = ["backtest_node"]
