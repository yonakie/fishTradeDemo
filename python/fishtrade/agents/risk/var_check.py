"""VaR (R3) check — historical simulation, position-weighted."""

from __future__ import annotations

from ...config import thresholds
from ...models.risk import HardCheckResult
from ...models.state import GraphState
from ...tools.var_calculator import compute_var_historical
from ...tools.yf_client import payload_to_df
from ._helpers import reject_decision


def var_check_node(state: GraphState) -> dict:
    risk = state.get("risk") or {}
    if risk and risk.get("decision") == "reject":
        # Already rejected upstream — pass through.
        return {}
    if risk and risk.get("decision") == "approve" and risk.get("adjusted_position_pct", 0) == 0:
        # HOLD/SELL fast-path already approved at 0% — keep it.
        return {}

    debate = state.get("debate") or {}
    proposed_pct = float(debate.get("proposed_position_pct", 0.0) or 0.0)

    market_data = state.get("market_data") or {}
    history_payload = market_data.get("history")
    if isinstance(history_payload, dict):
        history_df = payload_to_df(history_payload)
    else:
        history_df = history_payload

    var_result = compute_var_historical(
        history_df,
        proposed_position_pct=proposed_pct,
    )

    risk_partial = state.get("risk_partial") or {}
    hard_checks_raw = risk_partial.get("hard_checks") or []
    hard_checks = [HardCheckResult.model_validate(c) for c in hard_checks_raw]

    threshold_pct = thresholds.R3_VAR95_PORTFOLIO_LIMIT_PCT
    portfolio_impact_pct = var_result.portfolio_impact * 100
    r3_passed = var_result.passed and portfolio_impact_pct <= threshold_pct

    r3_check = HardCheckResult(
        rule="R3_VAR95",
        passed=r3_passed,
        actual=float(portfolio_impact_pct),
        threshold=threshold_pct,
        detail=(
            f"组合 VaR(95) 影响 {portfolio_impact_pct:.3f}% ≤ 上限 {threshold_pct}%"
            if r3_passed
            else (
                f"组合 VaR(95) 影响 {portfolio_impact_pct:.3f}% > {threshold_pct}%"
                if var_result.passed
                else f"VaR 计算降级: {var_result.fallback_reason or '未知原因'}"
            )
        ),
    )
    hard_checks.append(r3_check)

    if not r3_passed:
        return {
            "risk": reject_decision(
                hard_checks=hard_checks,
                var_result=var_result,
                soft_judgment=None,
                reason="VaR 超限或样本不足",
            ).model_dump()
        }

    # Pass — extend the running partial.
    return {
        "risk_partial": {
            "hard_checks": [c.model_dump() for c in hard_checks],
            "var_result": var_result.model_dump(),
        }
    }


__all__ = ["var_check_node"]
