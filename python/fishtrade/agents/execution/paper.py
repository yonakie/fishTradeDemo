"""Paper-trading execution — Alpaca submit + 30s fill poll, with mock fallback.

Alpaca client is imported lazily so that ``alpaca-py`` stays an optional
dependency; if it's missing or credentials are absent we degrade to a
*mock fill* (so isolation tests can exercise this node without secrets).
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from ...config.settings import settings
from ...models.execution import ExecutionResult, FillInfo
from ...models.state import GraphState
from ...observability.logger import get_logger
from ._helpers import build_order, determine_side

logger = get_logger(__name__)

_POLL_TIMEOUT_S = 30
_POLL_INTERVAL_S = 1.0


def _mock_fill_result(order, broker_order_id: str) -> ExecutionResult:
    """Used when Alpaca SDK / credentials are unavailable (tests, smoke runs)."""
    return ExecutionResult(
        mode="paper",
        order=order,
        status="filled",
        fill_info=FillInfo(
            avg_price=order.limit_price,
            filled_qty=order.qty,
            fill_time=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        ),
        broker_order_id=broker_order_id,
    )


def _try_submit_alpaca(order) -> tuple[Any, str] | None:
    """Submit through alpaca-py; returns (client, order_id) or None on failure."""
    if not settings.has_alpaca_credentials():
        logger.info("paper_node_no_alpaca_credentials")
        return None
    try:
        from alpaca.trading.client import TradingClient  # type: ignore
        from alpaca.trading.enums import OrderSide, TimeInForce  # type: ignore
        from alpaca.trading.requests import LimitOrderRequest  # type: ignore
    except Exception as exc:
        logger.warning("paper_node_alpaca_import_failed", error=str(exc))
        return None

    try:
        client = TradingClient(
            settings.alpaca_api_key,
            settings.alpaca_secret_key,
            paper=True,
        )
        req = LimitOrderRequest(
            symbol=order.symbol,
            qty=order.qty,
            side=OrderSide.BUY if order.side == "buy" else OrderSide.SELL,
            limit_price=order.limit_price,
            time_in_force=TimeInForce.DAY,
        )
        submitted = client.submit_order(req)
        return client, str(getattr(submitted, "id", ""))
    except Exception as exc:
        logger.warning("paper_node_alpaca_submit_failed", error=str(exc))
        return None


def _poll_for_fill(client: Any, order_id: str) -> tuple[str, FillInfo | None]:
    """Poll Alpaca until the order is fully / partially filled or times out."""
    start = time.monotonic()
    while time.monotonic() - start < _POLL_TIMEOUT_S:
        try:
            order = client.get_order_by_id(order_id)
        except Exception as exc:
            logger.warning("paper_node_poll_failed", error=str(exc))
            return "submitted", None
        status = str(getattr(order, "status", "")).lower()
        filled_qty = int(getattr(order, "filled_qty", 0) or 0)
        if status == "filled" and filled_qty > 0:
            avg = float(getattr(order, "filled_avg_price", 0) or 0)
            return "filled", FillInfo(
                avg_price=avg,
                filled_qty=filled_qty,
                fill_time=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            )
        if status == "partially_filled" and filled_qty > 0:
            avg = float(getattr(order, "filled_avg_price", 0) or 0)
            time.sleep(_POLL_INTERVAL_S)
            continue
        time.sleep(_POLL_INTERVAL_S)
    # Final check after timeout
    try:
        order = client.get_order_by_id(order_id)
        filled_qty = int(getattr(order, "filled_qty", 0) or 0)
        if filled_qty > 0:
            avg = float(getattr(order, "filled_avg_price", 0) or 0)
            return "partial", FillInfo(
                avg_price=avg,
                filled_qty=filled_qty,
                fill_time=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            )
    except Exception:
        pass
    return "submitted", None


def paper_node(state: GraphState) -> dict:
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
            mode="paper",
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

    submission = _try_submit_alpaca(order)
    if submission is None:
        # No SDK / no creds → produce a mock fill so the rest of the pipeline runs.
        result = _mock_fill_result(order, broker_order_id="MOCK-PAPER-FILL")
        return {
            "execution": result.model_dump(),
            "warnings": ["PAPER_NODE_MOCK_FILL"],
        }

    client, broker_order_id = submission
    status, fill_info = _poll_for_fill(client, broker_order_id)
    try:
        result = ExecutionResult(
            mode="paper",
            order=order,
            status=status,  # type: ignore[arg-type]
            fill_info=fill_info,
            broker_order_id=broker_order_id,
        )
    except Exception as exc:
        # If status validation fails (e.g. status=="submitted" + fill_info path)
        # mark as failed for safety.
        result = ExecutionResult(
            mode="paper",
            order=order,
            status="failed",
            broker_order_id=broker_order_id,
            error=f"result_assembly_failed: {exc}",
        )
    return {"execution": result.model_dump()}


__all__ = ["paper_node"]
