"""Prompt-template loaders and small token-saving helpers.

Templates live in :mod:`fishtrade.llm.prompts` as plain ``.md`` files. We
keep them deliberately short and rely on Pydantic ``response_format`` to
enforce structure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from ..models.debate import DebateResult, DebateTurn
from ..models.research import IndicatorScore, ResearchReport

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(name: str) -> str:
    """Load a Markdown prompt template by file stem (e.g. ``"fundamental"``)."""
    path = _PROMPTS_DIR / f"{name}.md"
    return path.read_text(encoding="utf-8")


def _scores_to_brief(scores: Iterable[IndicatorScore]) -> list[dict]:
    """Compact JSON-friendly view of indicator scores for prompt injection."""
    out: list[dict] = []
    for s in scores:
        out.append(
            {
                "name": s.name,
                "display_name": s.display_name_en,
                "raw_value": s.raw_value,
                "score": s.score,
                "reasoning": s.reasoning,
                "is_degraded": s.is_degraded,
            }
        )
    return out


def build_research_prompt(
    *,
    facet: str,
    ticker: str,
    as_of_date: str,
    industry_class: str | None,
    indicator_scores: list[IndicatorScore],
    total_score: int,
    verdict: str,
    is_facet_degraded: bool,
) -> list[dict]:
    """Build the system+user messages for a research-facet LLM call.

    The LLM is *only* asked for narrative & confidence — scores are pre-
    computed by the deterministic indicator engines.
    """
    template = load_prompt(facet)
    user_payload = {
        "ticker": ticker,
        "as_of_date": as_of_date,
        "facet": facet,
        "industry_class": industry_class,
        "computed_total_score": total_score,
        "expected_verdict": verdict,
        "is_facet_degraded": is_facet_degraded,
        "indicator_scores": _scores_to_brief(indicator_scores),
    }
    return [
        {"role": "system", "content": template},
        {
            "role": "user",
            "content": (
                "请基于下列预先计算好的指标，产出符合 schema 的研究报告。\n"
                "你不能修改 score 或 total_score，只能提供 key_highlights 与 confidence。\n"
                f"输入数据：\n```json\n{json.dumps(user_payload, ensure_ascii=False, default=str)}\n```"
            ),
        },
    ]


def summarize_research_for_debate(report: ResearchReport) -> str:
    """Squeeze a ResearchReport down to the bits debaters actually need.

    Keeps verdict + total score + the 3 strongest indicators; everything
    else is dropped to save tokens.
    """
    if report is None:
        return "(no research)"
    sorted_indicators = sorted(
        report.indicator_scores, key=lambda i: abs(i.score), reverse=True
    )
    top3 = sorted_indicators[:3]
    lines = [
        f"[{report.facet.upper()}] verdict={report.verdict} "
        f"total_score={report.total_score} confidence={report.confidence:.2f} "
        f"degraded={report.is_facet_degraded}",
        "Top indicators:",
    ]
    for ind in top3:
        lines.append(
            f"  - {ind.name} (score={ind.score}, raw={ind.raw_value}): {ind.reasoning}"
        )
    if report.is_facet_degraded:
        lines.append(f"  ! degrade_summary: {report.degrade_summary or 'n/a'}")
    return "\n".join(lines)


def truncate_debate_history(
    turns: list[DebateTurn], keep_last_n: int = 2
) -> list[DebateTurn]:
    """Keep the last ``keep_last_n`` *rounds* (i.e. up to 2*N turns).

    Used to keep the context window from growing as rebuttals accumulate.
    Expected ordering of input ``turns`` is chronological.
    """
    if not turns:
        return []
    if keep_last_n <= 0:
        return []
    rounds_present = sorted({t.round for t in turns})
    if len(rounds_present) <= keep_last_n:
        return list(turns)
    keep_rounds = set(rounds_present[-keep_last_n:])
    return [t for t in turns if t.round in keep_rounds]


def render_debate_history(turns: list[DebateTurn]) -> str:
    """Human-readable rendering of debate turns for prompt injection."""
    if not turns:
        return "(no prior turns)"
    blocks: list[str] = []
    for t in turns:
        blocks.append(
            f"Round {t.round} | {t.role.upper()} -> {t.conclusion}\n"
            f"  Cited: {', '.join(t.cited_indicators)}\n"
            f"  Argument: {t.argument}"
        )
    return "\n\n".join(blocks)


def build_debate_prompt(
    *,
    role: str,
    round_idx: int,
    ticker: str,
    research_summaries: dict[str, str],
    prior_turns: list[DebateTurn],
    valid_indicator_names: list[str],
) -> list[dict]:
    """Build messages for a single bull / bear debate turn."""
    template = load_prompt(f"debate_{role}")
    body = {
        "role": role,
        "round": round_idx,
        "ticker": ticker,
        "research_summaries": research_summaries,
        "valid_indicator_names": valid_indicator_names,
        "prior_turns": render_debate_history(prior_turns),
    }
    return [
        {"role": "system", "content": template},
        {
            "role": "user",
            "content": (
                "请按 schema 返回 JSON：argument, cited_indicators, conclusion。"
                "cited_indicators 必须严格从 valid_indicator_names 列表中选取。\n"
                f"输入：\n```json\n{json.dumps(body, ensure_ascii=False, default=str)}\n```"
            ),
        },
    ]


def build_judge_prompt(
    *,
    ticker: str,
    research_reports: dict[str, ResearchReport | None],
    debate_turns: list[DebateTurn],
    degraded_facets: list[str],
) -> list[dict]:
    """Build messages for the debate judge LLM call."""
    template = load_prompt("debate_judge")
    summaries = {
        facet: (summarize_research_for_debate(r) if r else "(missing)")
        for facet, r in research_reports.items()
    }
    body = {
        "ticker": ticker,
        "research_summaries": summaries,
        "degraded_facets": degraded_facets,
        "debate_turns": [t.model_dump() for t in debate_turns],
    }
    return [
        {"role": "system", "content": template},
        {
            "role": "user",
            "content": (
                "请综合三面研究 + 辩论实录，按 schema 输出 DebateResult。\n"
                "若 degraded_facets 非空，对应面权重置 0；BUY 必须给出 (0,10] 区间的 "
                "proposed_position_pct，HOLD/SELL 必须为 0。\n"
                f"输入：\n```json\n{json.dumps(body, ensure_ascii=False, default=str)}\n```"
            ),
        },
    ]


def build_soft_risk_prompt(
    *,
    ticker: str,
    debate: dict,
    market_signals: dict[str, Any],
) -> list[dict]:
    """Build messages for the soft-judgment LLM call."""
    template = load_prompt("risk_soft")
    body = {
        "ticker": ticker,
        "debate": debate,
        "market_signals": market_signals,
    }
    return [
        {"role": "system", "content": template},
        {
            "role": "user",
            "content": (
                "请输出符合 SoftJudgment schema 的 JSON：flags, adjustment, "
                "adjusted_position_pct, reasoning。\n"
                f"输入：\n```json\n{json.dumps(body, ensure_ascii=False, default=str)}\n```"
            ),
        },
    ]


__all__ = [
    "build_debate_prompt",
    "build_judge_prompt",
    "build_research_prompt",
    "build_soft_risk_prompt",
    "load_prompt",
    "render_debate_history",
    "summarize_research_for_debate",
    "truncate_debate_history",
]
