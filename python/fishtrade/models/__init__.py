"""Pydantic v2 schemas — system-wide data contracts.

Only schema definitions and field-level validators live here. No business
logic, no external API calls. Downstream modules import models; models
must never import downstream modules.
"""

from .debate import DebateResult, DebateTurn
from .execution import ExecutionResult, FillInfo, Order
from .portfolio import NavSnapshot, PortfolioSnapshot, Position
from .research import IndicatorScore, ResearchReport
from .risk import HardCheckResult, RiskDecision, SoftJudgment, VarResult
from .state import GraphState, MarketDataBundle, ResearchSection, RunInput

__all__ = [
    "DebateResult",
    "DebateTurn",
    "ExecutionResult",
    "FillInfo",
    "GraphState",
    "HardCheckResult",
    "IndicatorScore",
    "MarketDataBundle",
    "NavSnapshot",
    "Order",
    "PortfolioSnapshot",
    "Position",
    "ResearchReport",
    "ResearchSection",
    "RiskDecision",
    "RunInput",
    "SoftJudgment",
    "VarResult",
]
