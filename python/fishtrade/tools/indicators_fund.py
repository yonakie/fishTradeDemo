"""Fundamental indicators — 10 pure scorers + the ``compute_all_fundamental`` orchestrator.

Thresholds mirror ``docs/analysismetrics.md`` 第一章. Each scorer takes the
raw value (possibly ``None``) and the ``industry_class`` from
:func:`fishtrade.tools.industry_classifier.classify_industry` and returns
``(score, reasoning)``. Missing inputs degrade gracefully (score=0,
``is_degraded=True``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import pandas as pd

from ..models.research import IndicatorScore
from .industry_classifier import IndustryClass, classify_industry

Score = Literal[-1, 0, 1]
ScorerFn = Callable[[Any, IndustryClass, dict], "ScoreOutcome"]


@dataclass(frozen=True)
class ScoreOutcome:
    score: Score
    reasoning: str
    is_degraded: bool = False
    degrade_reason: str | None = None
    raw_value: Any | None = None


@dataclass(frozen=True)
class IndicatorSpec:
    name: str
    zh: str
    en: str
    fields: tuple[str, ...]
    scorer: ScorerFn


# ---------- helpers --------------------------------------------------------


def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _missing(name: str, reason: str = "yfinance 未返回该字段") -> ScoreOutcome:
    return ScoreOutcome(
        score=0,
        reasoning=f"{name} 数据缺失，按中性处理",
        is_degraded=True,
        degrade_reason=reason,
        raw_value=None,
    )


# ---------- individual scorers --------------------------------------------


def score_pe_ratio(raw: Any, industry: IndustryClass, info: dict) -> ScoreOutcome:
    trailing = _safe_float(raw)
    forward = _safe_float(info.get("forwardPE")) if info else None
    primary = trailing if trailing is not None else forward
    if primary is None:
        return _missing("PE Ratio")

    if primary < 0:
        return ScoreOutcome(
            score=-1,
            reasoning=f"PE 为负 ({primary:.2f})，公司当前亏损",
            raw_value=primary,
        )

    if industry in ("growth",):
        pos_lt, neu_lt = 25, 50
    elif industry in ("financial", "energy", "consumer", "value"):
        pos_lt, neu_lt = 12, 20
    else:
        pos_lt, neu_lt = 12, 20

    if primary < pos_lt:
        return ScoreOutcome(
            score=1,
            reasoning=f"PE {primary:.2f} 低于行业正面阈值 {pos_lt}，估值有吸引力",
            raw_value=primary,
        )
    if primary < neu_lt:
        return ScoreOutcome(
            score=0,
            reasoning=f"PE {primary:.2f} 在行业中性区间 [{pos_lt}, {neu_lt})",
            raw_value=primary,
        )
    return ScoreOutcome(
        score=-1,
        reasoning=f"PE {primary:.2f} 高于 {neu_lt}，估值偏高",
        raw_value=primary,
    )


def score_pb_ratio(raw: Any, industry: IndustryClass, info: dict) -> ScoreOutcome:
    pb = _safe_float(raw)
    if pb is None:
        return _missing("PB Ratio")

    if industry == "financial":
        pos_lt, neu_lt = 1.0, 1.5
    elif industry == "growth":
        pos_lt, neu_lt = 5.0, 15.0
    else:
        pos_lt, neu_lt = 1.0, 3.0

    if pb < pos_lt:
        return ScoreOutcome(
            score=1,
            reasoning=f"PB {pb:.2f} 低于行业正面阈值 {pos_lt}，资产折价",
            raw_value=pb,
        )
    if pb < neu_lt:
        return ScoreOutcome(
            score=0,
            reasoning=f"PB {pb:.2f} 在中性区间 [{pos_lt}, {neu_lt})",
            raw_value=pb,
        )
    return ScoreOutcome(
        score=-1,
        reasoning=f"PB {pb:.2f} 高于 {neu_lt}，相对资产显著溢价",
        raw_value=pb,
    )


def score_ps_ratio(raw: Any, industry: IndustryClass, info: dict) -> ScoreOutcome:
    ps = _safe_float(raw)
    if ps is None:
        return _missing("PS Ratio")

    if industry == "growth":
        pos_lt, neu_lt = 5.0, 15.0
    elif industry == "consumer":
        pos_lt, neu_lt = 0.5, 1.5
    else:
        pos_lt, neu_lt = 3.0, 8.0

    if ps < pos_lt:
        return ScoreOutcome(
            score=1,
            reasoning=f"PS {ps:.2f} 低于 {pos_lt}，营收估值合理",
            raw_value=ps,
        )
    if ps < neu_lt:
        return ScoreOutcome(
            score=0,
            reasoning=f"PS {ps:.2f} 在 [{pos_lt}, {neu_lt})",
            raw_value=ps,
        )
    return ScoreOutcome(
        score=-1,
        reasoning=f"PS {ps:.2f} 高于 {neu_lt}，营收估值偏贵",
        raw_value=ps,
    )


def score_revenue_growth(raw: Any, industry: IndustryClass, info: dict) -> ScoreOutcome:
    rg = _safe_float(raw)
    if rg is None:
        return _missing("Revenue Growth")

    if industry == "growth":
        pos_gt, neu_gt = 0.20, 0.10
    else:
        pos_gt, neu_gt = 0.08, 0.03

    if rg > pos_gt:
        return ScoreOutcome(
            score=1,
            reasoning=f"营收同比增长 {rg*100:.1f}%，超过 {pos_gt*100:.0f}% 正面阈值",
            raw_value=rg,
        )
    if rg > neu_gt:
        return ScoreOutcome(
            score=0,
            reasoning=f"营收同比增长 {rg*100:.1f}%，处于中性区间",
            raw_value=rg,
        )
    return ScoreOutcome(
        score=-1,
        reasoning=f"营收增速 {rg*100:.1f}% 偏弱或下滑",
        raw_value=rg,
    )


def score_gross_margin(raw: Any, industry: IndustryClass, info: dict) -> ScoreOutcome:
    gm = _safe_float(raw)
    if gm is None:
        return _missing("Gross Margin")

    if industry == "growth":
        pos_gt, neu_gt = 0.70, 0.55
    elif industry == "consumer":
        pos_gt, neu_gt = 0.40, 0.25
    elif industry == "energy":
        pos_gt, neu_gt = 0.35, 0.20
    else:
        pos_gt, neu_gt = 0.30, 0.15

    if gm > pos_gt:
        return ScoreOutcome(
            score=1,
            reasoning=f"毛利率 {gm*100:.1f}% 超过行业正面阈值 {pos_gt*100:.0f}%",
            raw_value=gm,
        )
    if gm > neu_gt:
        return ScoreOutcome(
            score=0,
            reasoning=f"毛利率 {gm*100:.1f}% 处于中性区间",
            raw_value=gm,
        )
    return ScoreOutcome(
        score=-1,
        reasoning=f"毛利率 {gm*100:.1f}% 低于行业基准 {neu_gt*100:.0f}%",
        raw_value=gm,
    )


def score_net_margin(raw: Any, industry: IndustryClass, info: dict) -> ScoreOutcome:
    nm = _safe_float(raw)
    if nm is None:
        return _missing("Net Margin")

    if industry in ("growth", "financial"):
        pos_gt, neu_gt = 0.20, 0.10
    elif industry == "consumer":
        pos_gt, neu_gt = 0.08, 0.03
    else:
        pos_gt, neu_gt = 0.10, 0.05

    if nm > pos_gt:
        return ScoreOutcome(
            score=1,
            reasoning=f"净利率 {nm*100:.1f}% 超过 {pos_gt*100:.0f}%，盈利能力强",
            raw_value=nm,
        )
    if nm > neu_gt:
        return ScoreOutcome(
            score=0,
            reasoning=f"净利率 {nm*100:.1f}% 处于中性区间",
            raw_value=nm,
        )
    return ScoreOutcome(
        score=-1,
        reasoning=f"净利率 {nm*100:.1f}% 偏低或亏损",
        raw_value=nm,
    )


def score_roe(raw: Any, industry: IndustryClass, info: dict) -> ScoreOutcome:
    roe = _safe_float(raw)
    if roe is None:
        return _missing("ROE")

    if industry == "financial":
        pos_gt, neu_gt = 0.12, 0.08
    else:
        pos_gt, neu_gt = 0.20, 0.10

    if roe > pos_gt:
        return ScoreOutcome(
            score=1,
            reasoning=f"ROE {roe*100:.1f}% 高于 {pos_gt*100:.0f}%，资本利用率优",
            raw_value=roe,
        )
    if roe > neu_gt:
        return ScoreOutcome(
            score=0,
            reasoning=f"ROE {roe*100:.1f}% 处于中性区间",
            raw_value=roe,
        )
    return ScoreOutcome(
        score=-1,
        reasoning=f"ROE {roe*100:.1f}% 偏低",
        raw_value=roe,
    )


def score_debt_to_equity(raw: Any, industry: IndustryClass, info: dict) -> ScoreOutcome:
    de = _safe_float(raw)
    if de is None:
        return _missing("Debt-to-Equity")

    # yfinance 返回的 debtToEquity 已是百分比 (e.g. 80 = 80%).
    if industry == "growth":
        pos_lt, neu_lt = 30.0, 80.0
    elif industry == "financial":
        # 银行通常 D/E 没意义，按 250 / 400 兜底，避免误判
        pos_lt, neu_lt = 250.0, 400.0
    else:
        pos_lt, neu_lt = 80.0, 150.0

    if de < pos_lt:
        return ScoreOutcome(
            score=1,
            reasoning=f"D/E {de:.1f}% 低于 {pos_lt}%，杠杆健康",
            raw_value=de,
        )
    if de < neu_lt:
        return ScoreOutcome(
            score=0,
            reasoning=f"D/E {de:.1f}% 在中性区间",
            raw_value=de,
        )
    return ScoreOutcome(
        score=-1,
        reasoning=f"D/E {de:.1f}% 高于 {neu_lt}%，偿债压力上升",
        raw_value=de,
    )


def score_free_cashflow(raw: Any, industry: IndustryClass, info: dict) -> ScoreOutcome:
    fcf = _safe_float(raw)
    if fcf is None:
        return _missing("Free Cash Flow")

    market_cap = _safe_float(info.get("marketCap")) if info else None
    if market_cap and market_cap > 0:
        yield_pct = fcf / market_cap
        if yield_pct > 0.05:
            return ScoreOutcome(
                score=1,
                reasoning=f"FCF/MCap = {yield_pct*100:.1f}%，造血能力强",
                raw_value=fcf,
            )
        if yield_pct > 0.02:
            return ScoreOutcome(
                score=0,
                reasoning=f"FCF/MCap = {yield_pct*100:.1f}%，中性",
                raw_value=fcf,
            )
        if yield_pct > 0:
            return ScoreOutcome(
                score=-1,
                reasoning=f"FCF/MCap = {yield_pct*100:.2f}% 偏低",
                raw_value=fcf,
            )

    # No market cap (or non-positive yield) — fall back on sign only.
    if fcf > 0:
        return ScoreOutcome(
            score=1,
            reasoning=f"FCF 为正 ({fcf:.2g})，造血能力存在",
            raw_value=fcf,
        )
    return ScoreOutcome(
        score=-1,
        reasoning=f"FCF 为负 ({fcf:.2g})，公司在烧现金",
        raw_value=fcf,
    )


def score_analyst_upside(raw: Any, industry: IndustryClass, info: dict) -> ScoreOutcome:
    target = _safe_float(raw)
    current = _safe_float(info.get("regularMarketPrice")) if info else None
    if target is None or current is None or current <= 0:
        return _missing("Analyst Upside", "缺少目标价或当前价")

    upside = (target - current) / current
    if upside > 0.20:
        return ScoreOutcome(
            score=1,
            reasoning=f"目标价较现价上行空间 {upside*100:.1f}%",
            raw_value=upside,
        )
    if upside > 0.05:
        return ScoreOutcome(
            score=0,
            reasoning=f"上行空间 {upside*100:.1f}%，中性",
            raw_value=upside,
        )
    return ScoreOutcome(
        score=-1,
        reasoning=f"上行空间仅 {upside*100:.1f}%，下行风险显著",
        raw_value=upside,
    )


# ---------- registry & orchestrator ---------------------------------------


INDICATOR_REGISTRY: dict[str, IndicatorSpec] = {
    "PE_RATIO": IndicatorSpec(
        name="PE_RATIO",
        zh="市盈率",
        en="PE Ratio",
        fields=("trailingPE", "forwardPE"),
        scorer=score_pe_ratio,
    ),
    "PB_RATIO": IndicatorSpec(
        name="PB_RATIO", zh="市净率", en="PB Ratio",
        fields=("priceToBook",), scorer=score_pb_ratio,
    ),
    "PS_RATIO": IndicatorSpec(
        name="PS_RATIO", zh="市销率", en="PS Ratio",
        fields=("priceToSalesTrailing12Months",), scorer=score_ps_ratio,
    ),
    "REVENUE_GROWTH": IndicatorSpec(
        name="REVENUE_GROWTH", zh="营收增速", en="Revenue Growth",
        fields=("revenueGrowth",), scorer=score_revenue_growth,
    ),
    "GROSS_MARGIN": IndicatorSpec(
        name="GROSS_MARGIN", zh="毛利率", en="Gross Margin",
        fields=("grossMargins",), scorer=score_gross_margin,
    ),
    "NET_MARGIN": IndicatorSpec(
        name="NET_MARGIN", zh="净利润率", en="Net Margin",
        fields=("profitMargins",), scorer=score_net_margin,
    ),
    "ROE": IndicatorSpec(
        name="ROE", zh="股本回报率", en="Return on Equity",
        fields=("returnOnEquity",), scorer=score_roe,
    ),
    "DEBT_TO_EQUITY": IndicatorSpec(
        name="DEBT_TO_EQUITY", zh="负债权益比", en="Debt-to-Equity",
        fields=("debtToEquity",), scorer=score_debt_to_equity,
    ),
    "FREE_CASHFLOW": IndicatorSpec(
        name="FREE_CASHFLOW", zh="自由现金流", en="Free Cash Flow",
        fields=("freeCashflow",), scorer=score_free_cashflow,
    ),
    "ANALYST_UPSIDE": IndicatorSpec(
        name="ANALYST_UPSIDE", zh="分析师目标价空间", en="Analyst Upside",
        fields=("targetMeanPrice",), scorer=score_analyst_upside,
    ),
}


def _to_indicator_score(spec: IndicatorSpec, outcome: ScoreOutcome) -> IndicatorScore:
    raw = outcome.raw_value
    if isinstance(raw, (float, int, str)) or raw is None:
        raw_value: float | str | None = raw  # type: ignore[assignment]
    else:
        raw_value = str(raw)
    return IndicatorScore(
        name=spec.name,
        display_name_zh=spec.zh,
        display_name_en=spec.en,
        raw_value=raw_value,
        score=outcome.score,
        reasoning=outcome.reasoning[:400],
        is_degraded=outcome.is_degraded,
        degrade_reason=outcome.degrade_reason,
    )


def compute_all_fundamental(market_data: dict) -> list[IndicatorScore]:
    """Run all 10 fundamental scorers against a market_data bundle.

    ``market_data`` follows :class:`fishtrade.models.state.MarketDataBundle`;
    only ``info`` is required, other fields are tolerated as missing.
    """
    info = (market_data or {}).get("info") or {}
    industry = classify_industry(info)

    out: list[IndicatorScore] = []
    for spec in INDICATOR_REGISTRY.values():
        primary_field = spec.fields[0]
        raw = info.get(primary_field) if info else None
        try:
            outcome = spec.scorer(raw, industry, info)
        except Exception as exc:  # defensive — never propagate
            outcome = ScoreOutcome(
                score=0,
                reasoning=f"{spec.en} 计算异常，已降级",
                is_degraded=True,
                degrade_reason=f"{type(exc).__name__}: {exc}",
                raw_value=None,
            )
        out.append(_to_indicator_score(spec, outcome))

    if len(out) != 10:
        raise RuntimeError(
            f"fundamental indicator count must be 10, got {len(out)}"
        )
    return out
