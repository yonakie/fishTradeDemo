"""Shared helpers for the three research nodes.

Splitting them out keeps the per-facet modules small and lets us test the
fallback / score-rebuild paths in one place.
"""

from __future__ import annotations

from typing import Literal

from ...llm import JSONParseError, generate_ark_response
from ...llm.prompt_utils import build_research_prompt
from ...models.research import IndicatorScore, ResearchReport

Facet = Literal["fundamental", "technical", "sentimental"]
Verdict = Literal["BUY", "HOLD", "SELL"]

# Threshold for marking the whole facet as degraded — matches design 4.1.5.
_FACET_DEGRADE_THRESHOLD = 5


def derive_verdict(total_score: int) -> Verdict:
    """Map total_score → verdict (≥+5 BUY, +1..+4 HOLD, ≤0 SELL)."""
    if total_score >= 5:
        return "BUY"
    if total_score >= 1:
        return "HOLD"
    return "SELL"


def assess_degradation(
    scores: list[IndicatorScore],
) -> tuple[bool, str | None]:
    """Decide whether ≥5 indicators degraded → mark the facet degraded."""
    degraded = [s for s in scores if s.is_degraded]
    if len(degraded) >= _FACET_DEGRADE_THRESHOLD:
        names = ", ".join(s.name for s in degraded)
        return True, f"{len(degraded)}/10 项指标缺数据，已整体降级：{names}"
    return False, None


def _fallback_research_template(
    *,
    facet: Facet,
    ticker: str,
    as_of_date: str,
    industry_class: str | None,
    indicator_scores: list[IndicatorScore],
) -> ResearchReport:
    """Pure-rule report used when the LLM repeatedly fails.

    confidence is locked to 0.3 (or ≤0.4 when degraded) so downstream
    debate / risk can detect the lower trustworthiness.
    """
    total = sum(s.score for s in indicator_scores)
    verdict = derive_verdict(total)
    is_degraded, summary = assess_degradation(indicator_scores)

    positives = [s for s in indicator_scores if s.score == 1]
    negatives = [s for s in indicator_scores if s.score == -1]

    highlights: list[str] = []
    if positives:
        highlights.append(
            "支持因子: " + ", ".join(p.name for p in positives[:5])
        )
    if negatives:
        highlights.append(
            "压制因子: " + ", ".join(n.name for n in negatives[:5])
        )
    highlights.append(
        f"汇总评分 {total} → {verdict}（依据 ≥+5/+1/≤0 阈值表）"
    )
    if is_degraded:
        highlights.append("数据降级触发：confidence 已被压低")
    # ResearchReport requires 3..5 highlights.
    while len(highlights) < 3:
        highlights.append("无额外亮点；按规则模板自动生成")
    highlights = highlights[:5]

    confidence = 0.3 if not is_degraded else 0.25
    return ResearchReport(
        facet=facet,
        ticker=ticker,
        as_of_date=as_of_date,
        indicator_scores=indicator_scores,
        total_score=total,
        verdict=verdict,
        confidence=confidence,
        key_highlights=highlights,
        industry_class=industry_class,
        is_facet_degraded=is_degraded,
        degrade_summary=summary,
    )


def _coerce_llm_report(
    *,
    raw: ResearchReport,
    facet: Facet,
    ticker: str,
    as_of_date: str,
    industry_class: str | None,
    indicator_scores: list[IndicatorScore],
) -> ResearchReport:
    """Force the LLM-produced report to use deterministic scores.

    The LLM is allowed to drift on text / confidence but is *never*
    permitted to alter scores. We therefore rebuild the report using the
    deterministic indicator list and only borrow narrative fields.
    """
    total = sum(s.score for s in indicator_scores)
    verdict = derive_verdict(total)
    is_degraded, summary = assess_degradation(indicator_scores)

    confidence = float(raw.confidence)
    if is_degraded and confidence > 0.4:
        confidence = 0.4

    highlights = list(raw.key_highlights or [])
    while len(highlights) < 3:
        highlights.append(f"汇总 {verdict}（自动补全）")
    highlights = [h[:400] for h in highlights[:5]]

    return ResearchReport(
        facet=facet,
        ticker=ticker,
        as_of_date=as_of_date,
        indicator_scores=indicator_scores,
        total_score=total,
        verdict=verdict,
        confidence=confidence,
        key_highlights=highlights,
        industry_class=industry_class,
        is_facet_degraded=is_degraded,
        degrade_summary=summary,
    )


def run_research_facet(
    *,
    facet: Facet,
    ticker: str,
    as_of_date: str,
    industry_class: str | None,
    indicator_scores: list[IndicatorScore],
    run_id: str,
    node_name: str,
) -> tuple[ResearchReport, list[str]]:
    """Drive one research facet end-to-end.

    Returns ``(report, warnings)`` — warnings is empty on the happy path
    and contains a single ``"<FACET>_LLM_FALLBACK"`` token when the LLM
    failed and the rule-based template was used instead.
    """
    total = sum(s.score for s in indicator_scores)
    verdict = derive_verdict(total)
    is_degraded, _ = assess_degradation(indicator_scores)

    messages = build_research_prompt(
        facet=facet,
        ticker=ticker,
        as_of_date=as_of_date,
        industry_class=industry_class,
        indicator_scores=indicator_scores,
        total_score=total,
        verdict=verdict,
        is_facet_degraded=is_degraded,
    )

    try:
        raw = generate_ark_response(
            messages,
            role="research",
            temperature=0.2,
            response_schema=ResearchReport,
            run_id=run_id,
            node_name=node_name,
        )
    except (JSONParseError, Exception):
        return (
            _fallback_research_template(
                facet=facet,
                ticker=ticker,
                as_of_date=as_of_date,
                industry_class=industry_class,
                indicator_scores=indicator_scores,
            ),
            [f"{facet.upper()}_LLM_FALLBACK"],
        )

    if not isinstance(raw, ResearchReport):
        # Defensive: unexpected return type — fall back to rule template.
        return (
            _fallback_research_template(
                facet=facet,
                ticker=ticker,
                as_of_date=as_of_date,
                industry_class=industry_class,
                indicator_scores=indicator_scores,
            ),
            [f"{facet.upper()}_LLM_FALLBACK"],
        )

    try:
        report = _coerce_llm_report(
            raw=raw,
            facet=facet,
            ticker=ticker,
            as_of_date=as_of_date,
            industry_class=industry_class,
            indicator_scores=indicator_scores,
        )
    except Exception:
        return (
            _fallback_research_template(
                facet=facet,
                ticker=ticker,
                as_of_date=as_of_date,
                industry_class=industry_class,
                indicator_scores=indicator_scores,
            ),
            [f"{facet.upper()}_LLM_FALLBACK"],
        )

    return report, []


__all__ = [
    "_fallback_research_template",
    "assess_degradation",
    "derive_verdict",
    "run_research_facet",
]
