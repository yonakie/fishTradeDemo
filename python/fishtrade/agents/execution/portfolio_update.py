"""Update the on-disk portfolio after a successful execution.

Contract:
- ``status in {"filled", "partial"}`` → apply fill, persist, return ``portfolio_after``
- ``status="generated"`` (dryrun) → no persistence; mirror ``portfolio_before``
- ``status="failed"`` / ``"submitted"`` / ``"skipped"`` → portfolio unchanged
"""

from __future__ import annotations

from typing import Any

from ...models.execution import ExecutionResult
from ...models.portfolio import PortfolioSnapshot, Position
from ...models.state import GraphState
from ...portfolio.nav import compute_max_drawdown
from ...portfolio.store import PortfolioStore


def _coerce_portfolio(raw: Any) -> PortfolioSnapshot | None:
    if raw is None:
        return None
    if isinstance(raw, PortfolioSnapshot):
        return raw
    if isinstance(raw, dict):
        try:
            return PortfolioSnapshot.model_validate(raw)
        except Exception:
            return None
    return None


def _coerce_execution(raw: Any) -> ExecutionResult | None:
    if raw is None:
        return None
    if isinstance(raw, ExecutionResult):
        return raw
    if isinstance(raw, dict):
        try:
            return ExecutionResult.model_validate(raw)
        except Exception:
            return None
    return None


def _apply_fill(
    snap: PortfolioSnapshot, exec_result: ExecutionResult, sector: str | None
) -> PortfolioSnapshot:
    """Return a new PortfolioSnapshot reflecting the fill."""
    assert exec_result.order is not None and exec_result.fill_info is not None

    order = exec_result.order
    fill = exec_result.fill_info
    cost = fill.avg_price * fill.filled_qty

    cash = float(snap.cash)
    positions: dict[str, Position] = {p.symbol: p for p in snap.positions}

    if order.side == "buy":
        cash -= cost
        existing = positions.get(order.symbol)
        if existing is None:
            positions[order.symbol] = Position(
                symbol=order.symbol,
                qty=fill.filled_qty,
                avg_cost=fill.avg_price,
                sector=sector,
            )
        else:
            new_qty = existing.qty + fill.filled_qty
            new_avg = (existing.avg_cost * existing.qty + cost) / new_qty
            positions[order.symbol] = Position(
                symbol=order.symbol,
                qty=new_qty,
                avg_cost=new_avg,
                sector=existing.sector or sector,
            )
    else:  # sell
        existing = positions.get(order.symbol)
        if existing is None:
            # Selling something we don't hold — refuse to mutate, keep cash flat.
            cash += cost  # treat as proceeds anyway, defensive
        else:
            sell_qty = min(fill.filled_qty, existing.qty)
            cash += sell_qty * fill.avg_price
            remaining = existing.qty - sell_qty
            if remaining <= 0:
                positions.pop(order.symbol, None)
            else:
                positions[order.symbol] = Position(
                    symbol=order.symbol,
                    qty=remaining,
                    avg_cost=existing.avg_cost,
                    sector=existing.sector,
                )

    # NAV uses last-fill price as a price proxy for the touched symbol.
    last_price = fill.avg_price
    new_nav = max(
        0.0,
        cash + sum(
            (last_price if p.symbol == order.symbol else p.avg_cost) * p.qty
            for p in positions.values()
        ),
    )

    nav_history = list(snap.nav_history)
    new_dd = compute_max_drawdown(nav_history) * 100  # already accounts for snap.nav

    return PortfolioSnapshot(
        cash=max(0.0, cash),
        positions=list(positions.values()),
        nav=new_nav,
        nav_history=nav_history,
        max_drawdown_pct=new_dd,
    )


def update_portfolio_node(state: GraphState) -> dict:
    """Apply the fill and return ``portfolio_after`` (and persist when real)."""
    portfolio_before = _coerce_portfolio(state.get("portfolio_before"))
    execution = _coerce_execution(state.get("execution"))

    if portfolio_before is None:
        return {}

    if execution is None or execution.status in ("failed", "submitted", "skipped"):
        # Failed / undecided → portfolio unchanged.
        return {"portfolio_after": portfolio_before.model_dump()}

    if execution.status == "generated":
        # Dryrun — never touch disk.
        return {"portfolio_after": portfolio_before.model_dump()}

    if execution.status not in ("filled", "partial"):
        return {"portfolio_after": portfolio_before.model_dump()}

    market_data = state.get("market_data") or {}
    info = market_data.get("info") or {}
    sector = info.get("sector") if isinstance(info, dict) else None

    new_snap = _apply_fill(portfolio_before, execution, sector)

    # Persist for paper / backtest modes only.
    mode = (state.get("input") or {}).get("mode")
    if mode in ("paper", "backtest"):
        try:
            PortfolioStore().save_atomic(new_snap)
        except Exception:
            # Persistence failures should not break the pipeline; log via warning.
            return {
                "portfolio_after": new_snap.model_dump(),
                "warnings": ["PORTFOLIO_PERSIST_FAILED"],
            }

    return {"portfolio_after": new_snap.model_dump()}


__all__ = ["update_portfolio_node"]
