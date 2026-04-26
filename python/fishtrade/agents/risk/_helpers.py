"""Pure helpers shared by the three risk nodes."""

from __future__ import annotations

from typing import Any

from ...config import thresholds
from ...models.portfolio import NavSnapshot, PortfolioSnapshot
from ...models.risk import (
    HardCheckResult,
    RiskDecision,
    SoftJudgment,
    VarResult,
)
from ...portfolio.nav import compute_max_drawdown


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


def _coerce_nav_history(raw: Any) -> list[NavSnapshot]:
    out: list[NavSnapshot] = []
    if not raw:
        return out
    for item in raw:
        if isinstance(item, NavSnapshot):
            out.append(item)
        elif isinstance(item, dict):
            try:
                out.append(NavSnapshot.model_validate(item))
            except Exception:
                continue
    return out


def check_r1_position_limit(proposed_pct: float) -> HardCheckResult:
    threshold = thresholds.R1_MAX_POSITION_PCT
    passed = proposed_pct <= threshold
    return HardCheckResult(
        rule="R1_POSITION_LIMIT",
        passed=passed,
        actual=float(proposed_pct),
        threshold=threshold,
        detail=(
            f"建议仓位 {proposed_pct:.2f}% ≤ 上限 {threshold}%"
            if passed
            else f"建议仓位 {proposed_pct:.2f}% 超出上限 {threshold}%"
        ),
    )


def check_r2_max_drawdown(nav_history_raw: Any) -> HardCheckResult:
    history = _coerce_nav_history(nav_history_raw)
    threshold = thresholds.R2_MAX_DRAWDOWN_PCT
    dd_ratio = compute_max_drawdown(history)  # 0..1
    dd_pct = dd_ratio * 100
    passed = dd_pct <= threshold
    return HardCheckResult(
        rule="R2_MAX_DRAWDOWN",
        passed=passed,
        actual=float(dd_pct),
        threshold=threshold,
        detail=(
            f"当前最大回撤 {dd_pct:.2f}% ≤ {threshold}%（首次运行回撤为 0）"
            if passed
            else f"当前最大回撤 {dd_pct:.2f}% 已超过 {threshold}%"
        ),
    )


def check_r4_stoploss_definable(price: Any) -> HardCheckResult:
    try:
        p = float(price)
    except (TypeError, ValueError):
        p = 0.0
    passed = p > 0
    pct = thresholds.R4_STOPLOSS_PCT
    return HardCheckResult(
        rule="R4_STOPLOSS",
        passed=passed,
        actual=p if passed else None,
        threshold=pct,
        detail=(
            f"现价 {p:.2f}，止损 {pct}% 可计算"
            if passed
            else "无法获取有效现价，止损价不可定义"
        ),
    )


def hold_skip_decision() -> RiskDecision:
    """Special decision used when the debate verdict is HOLD/SELL with 0 position.

    The contract is "approve at 0%": Execution router will then translate
    this into ``status="skipped"``.
    """
    skip_var = VarResult(
        var_95=0.0,
        portfolio_impact=0.0,
        passed=True,
        sample_size=1,
        fallback_reason=None,
    )
    skip_soft = SoftJudgment(
        flags=["NONE"],
        adjustment="keep",
        adjusted_position_pct=0.0,
        reasoning="debate_verdict=HOLD/SELL → 跳过 risk，仓位 0",
    )
    return RiskDecision(
        decision="approve",
        adjusted_position_pct=0.0,
        hard_checks=[
            HardCheckResult(
                rule="R1_POSITION_LIMIT",
                passed=True,
                actual=0.0,
                threshold=thresholds.R1_MAX_POSITION_PCT,
                detail="HOLD/SELL 自动通过 R1（仓位 0）",
            )
        ],
        var_result=skip_var,
        soft_judgment=skip_soft,
        reject_reason=None,
    )


def reject_decision(
    *,
    hard_checks: list[HardCheckResult],
    var_result: VarResult | None,
    soft_judgment: SoftJudgment | None,
    reason: str,
) -> RiskDecision:
    """Build a RiskDecision marked rejected with the supplied evidence."""
    if var_result is None:
        var_result = VarResult(
            var_95=0.0,
            portfolio_impact=0.0,
            passed=False,
            sample_size=0,
            fallback_reason="hard rule failed before VaR could run",
        )
    if soft_judgment is None:
        soft_judgment = SoftJudgment(
            flags=["NONE"],
            adjustment="reject",
            adjusted_position_pct=0.0,
            reasoning="拒绝：上游硬规则或 VaR 失败",
        )
    return RiskDecision(
        decision="reject",
        adjusted_position_pct=0.0,
        hard_checks=hard_checks,
        var_result=var_result,
        soft_judgment=soft_judgment,
        reject_reason=reason,
    )


__all__ = [
    "_coerce_nav_history",
    "_coerce_portfolio",
    "check_r1_position_limit",
    "check_r2_max_drawdown",
    "check_r4_stoploss_definable",
    "hold_skip_decision",
    "reject_decision",
]
