"""Constant risk thresholds (R1-R4).

Mirrors `Settings`'s defaults but exposed as plain constants for use in
pure functions / tests where instantiating Settings is overkill.
"""

from __future__ import annotations

R1_MAX_POSITION_PCT: float = 10.0
R2_MAX_DRAWDOWN_PCT: float = 8.0
R3_VAR95_PORTFOLIO_LIMIT_PCT: float = 2.0
R4_STOPLOSS_PCT: float = 5.0
