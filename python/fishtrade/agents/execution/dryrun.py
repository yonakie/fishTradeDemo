"""Dryrun execution — generate the order, persist to disk, never call broker."""

from __future__ import annotations

import json
from pathlib import Path

from ...config.settings import settings
from ...models.execution import ExecutionResult
from ...models.state import GraphState
from ._helpers import build_order, determine_side


def _orders_dir() -> Path:
    p = Path(settings.log_dir) / "orders"
    p.mkdir(parents=True, exist_ok=True)
    return p


def dryrun_node(state: GraphState) -> dict:
    run_id = state.get("run_id", "ad-hoc")
    debate = state.get("debate") or {}
    risk = state.get("risk") or {}
    market_data = state.get("market_data") or {}
    info = market_data.get("info") or {}
    run_input = state.get("input") or {}

    pct = float(risk.get("adjusted_position_pct") or 0)
    capital = float(run_input.get("capital") or 0)
    price = float(info.get("regularMarketPrice") or 0)
    symbol = run_input.get("ticker", "UNKNOWN")
    portfolio_before = state.get("portfolio_before") or {}
    has_position = any(
        (p.get("symbol") if isinstance(p, dict) else None) == symbol
        for p in (portfolio_before.get("positions") or [])
    )

    side = determine_side(debate.get("final_verdict", "BUY"), has_position)
    if pct <= 0 or capital <= 0 or price <= 0:
        result = ExecutionResult(
            mode="dryrun",
            order=None,
            status="skipped",
            error="invalid_inputs_for_order",
        )
        return {"execution": result.model_dump()}

    order = build_order(
        symbol=symbol,
        side=side,
        price=price,
        capital=capital,
        pct=pct,
    )

    payload = {
        "run_id": run_id,
        "mode": "dryrun",
        "order": order.model_dump(),
    }
    out_path = _orders_dir() / f"{run_id}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    result = ExecutionResult(
        mode="dryrun",
        order=order,
        status="generated",
    )
    return {"execution": result.model_dump()}


__all__ = ["dryrun_node"]
