"""Soft-rule judgement node — LLM with deterministic rule fallback.

Falls back to a pure-rule decision (any flag triggered → reduce position
by 50%) when the LLM call fails. Wired so that the final ``RiskDecision``
is always assembled here and stamped into ``state["risk"]``.
"""

from __future__ import annotations

from typing import Any

from ...llm import JSONParseError, generate_ark_response
from ...llm.prompt_utils import build_soft_risk_prompt
from ...models.risk import (
    HardCheckResult,
    RiskDecision,
    SoftJudgment,
    VarResult,
)
from ...models.state import GraphState
from ._helpers import _coerce_portfolio


def _summarize_holdings(portfolio: Any, this_sector: str | None) -> dict[str, Any]:
    """Summary needed by the LLM to spot CORRELATION_HIGH risk."""
    snap = _coerce_portfolio(portfolio)
    if snap is None:
        return {"total_nav": 0.0, "same_sector_pct": 0.0, "n_positions": 0}
    same_sector_value = 0.0
    total_value = 0.0
    for pos in snap.positions:
        approx_value = pos.qty * pos.avg_cost
        total_value += approx_value
        if (
            this_sector
            and pos.sector
            and pos.sector.lower() == this_sector.lower()
        ):
            same_sector_value += approx_value
    same_sector_pct = (
        (same_sector_value / snap.nav * 100) if snap.nav > 0 else 0.0
    )
    return {
        "total_nav": snap.nav,
        "same_sector_pct": same_sector_pct,
        "n_positions": len(snap.positions),
    }


def _vix_avg(vix_recent: Any) -> float | None:
    if not vix_recent:
        return None
    if isinstance(vix_recent, dict) and "data" in vix_recent:
        # serialised DataFrame payload
        rows = vix_recent.get("data") or []
        cols = vix_recent.get("columns") or []
        if "Close" in cols:
            idx = cols.index("Close")
            vals = [r[idx] for r in rows if r and r[idx] is not None]
            if vals:
                return float(sum(vals) / len(vals))
    if isinstance(vix_recent, dict) and "Close" in vix_recent:
        vals = list(vix_recent["Close"].values()) if isinstance(vix_recent["Close"], dict) else []
        if vals:
            return float(sum(vals) / len(vals))
    return None


def _avg_dollar_volume(info: dict) -> float | None:
    vol = info.get("averageDailyVolume10Day") or info.get("averageDailyVolume3Month")
    price = info.get("regularMarketPrice")
    try:
        if vol and price:
            return float(vol) * float(price)
    except (TypeError, ValueError):
        return None
    return None


def _rule_fallback_judgement(
    *,
    proposed_pct: float,
    flags: list[str],
) -> SoftJudgment:
    """Pure-rule fallback: any flag → reduce 50%; no flags → keep."""
    relevant = [f for f in flags if f != "NONE"]
    if relevant:
        adjusted = round(proposed_pct * 0.5, 4)
        return SoftJudgment(
            flags=relevant,  # type: ignore[arg-type]
            adjustment="reduce",
            adjusted_position_pct=adjusted,
            reasoning=(
                "LLM_FALLBACK_TO_RULES: 触发 "
                + ", ".join(relevant)
                + " → 仓位减半"
            ),
        )
    return SoftJudgment(
        flags=["NONE"],
        adjustment="keep",
        adjusted_position_pct=proposed_pct,
        reasoning="LLM_FALLBACK_TO_RULES: 未触发任何 flag，保持原仓位",
    )


def _detect_rule_flags(
    *,
    vix_avg: float | None,
    same_sector_pct: float,
    avg_dollar_volume: float | None,
) -> list[str]:
    flags: list[str] = []
    if vix_avg is not None and vix_avg > 30:
        flags.append("MARKET_VOLATILE")
    if same_sector_pct > 20:
        flags.append("CORRELATION_HIGH")
    if avg_dollar_volume is not None and avg_dollar_volume < 10_000_000:
        flags.append("LIQUIDITY_LOW")
    return flags or ["NONE"]


def soft_judge_node(state: GraphState) -> dict:
    risk = state.get("risk") or {}
    if risk and risk.get("decision") == "reject":
        return {}
    if risk and risk.get("decision") == "approve" and risk.get("adjusted_position_pct", 0) == 0:
        # HOLD/SELL fast-path already locked in by hard_rules
        return {}

    debate = state.get("debate") or {}
    proposed_pct = float(debate.get("proposed_position_pct", 0.0) or 0.0)

    market_data = state.get("market_data") or {}
    info = market_data.get("info") or {}
    portfolio_before = state.get("portfolio_before")

    sector = info.get("sector")
    holdings = _summarize_holdings(portfolio_before, sector)
    vix_avg = _vix_avg(market_data.get("vix_recent"))
    avg_dollar_vol = _avg_dollar_volume(info)

    rule_flags = _detect_rule_flags(
        vix_avg=vix_avg,
        same_sector_pct=holdings["same_sector_pct"],
        avg_dollar_volume=avg_dollar_vol,
    )

    market_signals = {
        "vix_avg_5d": vix_avg,
        "current_holdings_summary": holdings,
        "stock_avg_dollar_volume": avg_dollar_vol,
        "stock_sector": sector,
        "computed_rule_flags": rule_flags,
    }

    messages = build_soft_risk_prompt(
        ticker=(state.get("input") or {}).get("ticker", "UNKNOWN"),
        debate=debate,
        market_signals=market_signals,
    )

    warnings: list[str] = []
    raw: SoftJudgment | None = None
    try:
        result = generate_ark_response(
            messages,
            role="judge",
            temperature=0.2,
            response_schema=SoftJudgment,
            run_id=state.get("run_id", "ad-hoc"),
            node_name="risk_soft",
        )
    except (JSONParseError, Exception):
        result = None

    if isinstance(result, SoftJudgment):
        raw = result

    if raw is None:
        raw = _rule_fallback_judgement(
            proposed_pct=proposed_pct,
            flags=rule_flags,
        )
        warnings.append("RISK_SOFT_LLM_FALLBACK")

    # Final clamp: enforce the 50% rule even if LLM picked an off-spec value.
    final_pct = raw.adjusted_position_pct
    if raw.adjustment == "reduce":
        expected = round(proposed_pct * 0.5, 4)
        # If the LLM gave something wildly off, override.
        if abs(final_pct - expected) > 0.5:
            final_pct = expected
    if raw.adjustment == "reject":
        final_pct = 0.0
    if raw.adjustment == "keep":
        final_pct = proposed_pct

    soft = SoftJudgment(
        flags=raw.flags,
        adjustment=raw.adjustment,
        adjusted_position_pct=final_pct,
        reasoning=raw.reasoning[:1000],
    )

    risk_partial = state.get("risk_partial") or {}
    hard_checks_raw = risk_partial.get("hard_checks") or []
    var_result_raw = risk_partial.get("var_result")
    hard_checks = [HardCheckResult.model_validate(c) for c in hard_checks_raw]
    if not hard_checks:
        # Defensive: shouldn't happen if pipeline ran in order, but isolation
        # tests may skip the prior nodes.
        from ._helpers import check_r1_position_limit

        hard_checks = [check_r1_position_limit(proposed_pct)]
    var_result = (
        VarResult.model_validate(var_result_raw)
        if var_result_raw
        else VarResult(
            var_95=0.0,
            portfolio_impact=0.0,
            passed=False,
            sample_size=0,
            fallback_reason="缺少 VaR 计算结果（隔离测试场景）",
        )
    )

    if soft.adjustment == "reject":
        decision = RiskDecision(
            decision="reject",
            adjusted_position_pct=0.0,
            hard_checks=hard_checks,
            var_result=var_result,
            soft_judgment=soft,
            reject_reason="软规则裁定 reject: " + soft.reasoning[:200],
        )
    else:
        decision = RiskDecision(
            decision="approve",
            adjusted_position_pct=soft.adjusted_position_pct,
            hard_checks=hard_checks,
            var_result=var_result,
            soft_judgment=soft,
        )

    patch: dict = {"risk": decision.model_dump()}
    if warnings:
        patch["warnings"] = warnings
    return patch


__all__ = ["soft_judge_node"]
