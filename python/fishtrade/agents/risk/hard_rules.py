"""Hard-rule risk node — short-circuits on R1/R2/R4 failure.

The full risk pipeline is hard → VaR → soft. This node owns hard rules
only; if any fail it returns a *terminal* RiskDecision (decision="reject").
On success it stashes the partial check list in the state (under the
``risk_partial`` key) so the next two nodes can append to it.
"""

from __future__ import annotations

from ...models.state import GraphState
from ._helpers import (
    check_r1_position_limit,
    check_r2_max_drawdown,
    check_r4_stoploss_definable,
    hold_skip_decision,
    reject_decision,
)


def hard_rules_node(state: GraphState) -> dict:
    debate = state.get("debate") or {}
    if not debate:
        # No debate result yet — defensive guard so isolated tests can call this
        # without setting up the full pipeline.
        return {
            "risk": reject_decision(
                hard_checks=[
                    check_r1_position_limit(0.0),
                ],
                var_result=None,
                soft_judgment=None,
                reason="缺少 debate 结果，无法判断仓位",
            ).model_dump()
        }

    final_verdict = debate.get("final_verdict")
    proposed_pct = float(debate.get("proposed_position_pct", 0.0) or 0.0)

    # HOLD / SELL 短路：design 4.3.2 — 直接放行 risk，仓位 0
    if final_verdict in ("HOLD", "SELL") or proposed_pct <= 0:
        return {"risk": hold_skip_decision().model_dump()}

    market_data = state.get("market_data") or {}
    info = market_data.get("info") or {}
    portfolio_before = state.get("portfolio_before") or {}
    nav_history = portfolio_before.get("nav_history") if isinstance(portfolio_before, dict) else None

    checks = [
        check_r1_position_limit(proposed_pct),
        check_r2_max_drawdown(nav_history),
        check_r4_stoploss_definable(info.get("regularMarketPrice")),
    ]
    failed = [c for c in checks if not c.passed]
    if failed:
        reason = "硬规则失败：" + ", ".join(c.rule for c in failed)
        return {
            "risk": reject_decision(
                hard_checks=checks,
                var_result=None,
                soft_judgment=None,
                reason=reason,
            ).model_dump()
        }

    # Pass — store the partial check list for downstream nodes.
    return {"risk_partial": {"hard_checks": [c.model_dump() for c in checks]}}


__all__ = ["hard_rules_node"]
