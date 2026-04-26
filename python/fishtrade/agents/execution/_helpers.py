"""Pure helpers shared by execution nodes."""

from __future__ import annotations

from typing import Literal

from ...config.settings import settings
from ...models.execution import Order


def determine_side(verdict: str, has_position: bool) -> Literal["buy", "sell"]:
    """Map a verdict to a market side, given whether we already hold the symbol."""
    if verdict == "BUY":
        return "buy"
    if verdict == "SELL":
        return "sell" if has_position else "buy"  # never reached if pct==0
    return "buy"


def build_order(
    *,
    symbol: str,
    side: Literal["buy", "sell"],
    price: float,
    capital: float,
    pct: float,
) -> Order:
    """Translate (capital, pct, price) into a limit order with stop guard."""
    target_value = capital * pct / 100
    qty = max(1, int(target_value // max(price, 0.01)))
    if side == "buy":
        return Order(
            symbol=symbol,
            side="buy",
            qty=qty,
            limit_price=round(price * 1.002, 2),
            stop_price=round(price * (1 - settings.risk_stoploss_pct / 100), 2),
        )
    return Order(
        symbol=symbol,
        side="sell",
        qty=qty,
        limit_price=round(price * 0.998, 2),
        stop_price=None,
    )


__all__ = ["build_order", "determine_side"]
