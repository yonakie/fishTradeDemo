"""Helpers shared by bull / bear / judge debate nodes."""

from __future__ import annotations

from typing import Any, Literal

from ...llm import JSONParseError, generate_ark_response
from ...llm.prompt_utils import (
    build_debate_prompt,
    build_judge_prompt,
    summarize_research_for_debate,
    truncate_debate_history,
)
from ...models.debate import DebateResult, DebateTurn
from ...models.research import ResearchReport

Role = Literal["bull", "bear"]


def coerce_turns(turns: list) -> list[DebateTurn]:
    """Accept list[DebateTurn | dict] from checkpoint and normalise to DebateTurn."""
    out: list[DebateTurn] = []
    for t in turns:
        if isinstance(t, DebateTurn):
            out.append(t)
        elif isinstance(t, dict):
            try:
                out.append(DebateTurn.model_validate(t))
            except Exception:
                pass
    return out


def _coerce_research(raw: Any) -> ResearchReport | None:
    """Accept either an already-parsed ResearchReport or its model_dump dict."""
    if raw is None:
        return None
    if isinstance(raw, ResearchReport):
        return raw
    if isinstance(raw, dict):
        try:
            return ResearchReport.model_validate(raw)
        except Exception:
            return None
    return None


def collect_research(state: dict) -> dict[str, ResearchReport | None]:
    """Pull the three research facets (any may be None)."""
    research = state.get("research") or {}
    return {
        "fundamental": _coerce_research(research.get("fundamental")),
        "technical": _coerce_research(research.get("technical")),
        "sentimental": _coerce_research(research.get("sentimental")),
    }


def collect_summaries(
    research: dict[str, ResearchReport | None]
) -> dict[str, str]:
    """Compact per-facet text summaries for prompt injection."""
    return {
        facet: summarize_research_for_debate(rep) if rep else "(missing)"
        for facet, rep in research.items()
    }


def collect_valid_indicator_names(
    research: dict[str, ResearchReport | None]
) -> list[str]:
    """Union of indicator names across all facets — citations must come from here."""
    out: list[str] = []
    seen: set[str] = set()
    for rep in research.values():
        if rep is None:
            continue
        for ind in rep.indicator_scores:
            if ind.name not in seen:
                seen.add(ind.name)
                out.append(ind.name)
    return out


def previous_turns_of(turns: list[DebateTurn], role: Role) -> list[DebateTurn]:
    return [t for t in turns if t.role == role]


def fallback_turn(
    *,
    role: Role,
    round_idx: int,
    valid_indicator_names: list[str],
    prior_self_turns: list[DebateTurn],
) -> DebateTurn:
    """Reuse the role's last conclusion if available, else neutral HOLD.

    The point is to keep the pipeline moving when the LLM refuses or
    keeps producing invalid JSON — flagged via ``is_fallback=True`` so
    downstream judge / risk / report can call it out.
    """
    if prior_self_turns:
        last = prior_self_turns[-1]
        argument = (
            f"[FALLBACK] LLM 未能按 schema 返回新论点；沿用上一轮 {role} 立场："
            f"{last.argument[:1200]}"
        )
        cited = list(last.cited_indicators) or (
            [valid_indicator_names[0]] if valid_indicator_names else ["UNKNOWN"]
        )
        conclusion = last.conclusion
    else:
        argument = (
            f"[FALLBACK] {role} opening: LLM 解析失败，保守按 HOLD 占位以维持流水线。"
        )
        cited = [valid_indicator_names[0]] if valid_indicator_names else ["UNKNOWN"]
        conclusion = "HOLD"

    return DebateTurn(
        round=round_idx,
        role=role,
        argument=argument[:2000],
        cited_indicators=cited,
        conclusion=conclusion,
        is_fallback=True,
    )


def run_debate_turn(
    *,
    role: Role,
    round_idx: int,
    state: dict,
    run_id: str,
    node_name: str,
) -> tuple[DebateTurn, list[str]]:
    """Execute a single bull/bear turn with truncation + fallback handling."""
    research = collect_research(state)
    summaries = collect_summaries(research)
    valid_indicator_names = collect_valid_indicator_names(research)

    prior_all = coerce_turns(state.get("debate_turns") or [])
    prior_truncated = truncate_debate_history(prior_all, keep_last_n=2)

    messages = build_debate_prompt(
        role=role,
        round_idx=round_idx,
        ticker=(state.get("input") or {}).get("ticker", "UNKNOWN"),
        research_summaries=summaries,
        prior_turns=prior_truncated,
        valid_indicator_names=valid_indicator_names,
    )

    warnings: list[str] = []
    turn: DebateTurn | None = None
    try:
        raw = generate_ark_response(
            messages,
            role="debate",
            temperature=0.7,
            response_schema=DebateTurn,
            run_id=run_id,
            node_name=node_name,
        )
    except (JSONParseError, Exception):
        raw = None

    if isinstance(raw, DebateTurn):
        # Reject citations that wandered outside the indicator catalogue.
        valid_set = set(valid_indicator_names)
        if valid_set and not any(c in valid_set for c in raw.cited_indicators):
            warnings.append(f"DEBATE_{role.upper()}_CITATION_INVALID")
        else:
            try:
                turn = DebateTurn(
                    round=round_idx,
                    role=role,
                    argument=raw.argument[:2000],
                    cited_indicators=[
                        c for c in raw.cited_indicators if (not valid_set) or c in valid_set
                    ] or list(raw.cited_indicators),
                    conclusion=raw.conclusion,
                    is_fallback=False,
                )
            except Exception:
                turn = None

    if turn is None:
        prior_self = previous_turns_of(prior_all, role)
        turn = fallback_turn(
            role=role,
            round_idx=round_idx,
            valid_indicator_names=valid_indicator_names,
            prior_self_turns=prior_self,
        )
        warnings.append(f"DEBATE_{role.upper()}_LLM_FALLBACK")

    return turn, warnings


def _last_real_conclusion(
    debate_turns: list[DebateTurn], role: Role
) -> str | None:
    """Most recent non-fallback conclusion for ``role``, if any."""
    for t in reversed(debate_turns):
        if t.role == role and not t.is_fallback:
            return t.conclusion
    return None


def fallback_debate_result(
    *,
    research: dict[str, ResearchReport | None],
    debate_turns: list[DebateTurn],
    degraded_facets: list[str],
) -> DebateResult:
    """Pure-rule judge with two priority layers.

    1. If the debate produced *real* (non-fallback) bull/bear turns, vote
       on those — they reflect what actually happened in the dialogue.
       This avoids the prior bug where the judge fell back, the bull/bear
       successfully argued BUY/SELL, but the rationale only counted the
       three research verdicts and reported ``BUY=0/SELL=0/HOLD=3``.
    2. Otherwise, fall back to majority vote across research facets.

    A heavily-degraded research panel (≥2 facets degraded) still forces
    a HOLD regardless of the debate.
    """
    if len(degraded_facets) >= 2:
        return DebateResult(
            turns=debate_turns,
            final_verdict="HOLD",
            final_rationale="[FALLBACK] 多面研究降级（≥2），裁判降为 HOLD，仓位 0。",
            confidence=0.2,
            proposed_position_pct=0.0,
            degraded_facets=degraded_facets,  # type: ignore[arg-type]
        )

    bull = _last_real_conclusion(debate_turns, "bull")
    bear = _last_real_conclusion(debate_turns, "bear")
    if bull is not None and bear is not None:
        debate_votes = [bull, bear]
        b = debate_votes.count("BUY")
        s = debate_votes.count("SELL")
        h = debate_votes.count("HOLD")
        if b > s and b >= 1 and s == 0:
            final = "BUY"
            position = 5.0
            rationale = (
                f"[FALLBACK] 辩论结果：bull={bull} / bear={bear}（BUY={b}/SELL={s}/HOLD={h}），"
                "无对立空头，按规则给 5% 仓位。"
            )
        elif s > b and s >= 1 and b == 0:
            final = "SELL"
            position = 0.0
            rationale = (
                f"[FALLBACK] 辩论结果：bull={bull} / bear={bear}（BUY={b}/SELL={s}/HOLD={h}），"
                "无对立多头，按规则 SELL。"
            )
        else:
            final = "HOLD"
            position = 0.0
            rationale = (
                f"[FALLBACK] 辩论双方分歧：bull={bull} / bear={bear}（BUY={b}/SELL={s}/HOLD={h}），"
                "无明确多数，HOLD。"
            )
        return DebateResult(
            turns=debate_turns,
            final_verdict=final,
            final_rationale=rationale,
            confidence=0.3 if not degraded_facets else 0.25,
            proposed_position_pct=position,
            degraded_facets=degraded_facets,  # type: ignore[arg-type]
        )

    verdicts = [r.verdict for r in research.values() if r is not None]
    if not verdicts:
        final = "HOLD"
        position = 0.0
        rationale = "[FALLBACK] 三面研究均缺失；裁判降为 HOLD，仓位 0。"
    else:
        buy = verdicts.count("BUY")
        sell = verdicts.count("SELL")
        if buy > sell and buy >= 2:
            final = "BUY"
            position = 5.0
            rationale = f"[FALLBACK] 三面投票 BUY={buy}/SELL={sell}/HOLD={verdicts.count('HOLD')}，按规则给 5% 仓位。"
        elif sell > buy and sell >= 2:
            final = "SELL"
            position = 0.0
            rationale = f"[FALLBACK] 三面投票偏空：SELL={sell}/BUY={buy}，按规则 SELL。"
        else:
            final = "HOLD"
            position = 0.0
            rationale = f"[FALLBACK] 三面分歧：BUY={buy}/SELL={sell}/HOLD={verdicts.count('HOLD')}，HOLD。"

    return DebateResult(
        turns=debate_turns,
        final_verdict=final,
        final_rationale=rationale,
        confidence=0.3 if not degraded_facets else 0.2,
        proposed_position_pct=position,
        degraded_facets=degraded_facets,  # type: ignore[arg-type]
    )


def run_debate_judge(
    *,
    state: dict,
    run_id: str,
    node_name: str = "debate_judge",
) -> tuple[DebateResult, list[str]]:
    """Drive the judge LLM call with full degradation handling."""
    research = collect_research(state)
    debate_turns: list[DebateTurn] = coerce_turns(state.get("debate_turns") or [])

    degraded_facets = [
        facet
        for facet, rep in research.items()
        if rep is not None and rep.is_facet_degraded
    ]

    # Short-circuit: if all three facets are degraded → skip LLM, force HOLD.
    if all(rep is None or rep.is_facet_degraded for rep in research.values()):
        return (
            fallback_debate_result(
                research=research,
                debate_turns=debate_turns,
                degraded_facets=degraded_facets,
            ),
            ["DEBATE_JUDGE_SKIPPED_ALL_DEGRADED"],
        )

    ticker = (state.get("input") or {}).get("ticker", "UNKNOWN")
    messages = build_judge_prompt(
        ticker=ticker,
        research_reports=research,
        debate_turns=debate_turns,
        degraded_facets=degraded_facets,
    )

    try:
        raw = generate_ark_response(
            messages,
            role="judge",
            temperature=0.2,
            response_schema=DebateResult,
            run_id=run_id,
            node_name=node_name,
        )
    except (JSONParseError, Exception):
        raw = None

    if isinstance(raw, DebateResult):
        try:
            # Force the turns array to match the canonical state — the LLM
            # sometimes drops or reorders entries.
            normalized = DebateResult(
                turns=debate_turns,
                final_verdict=raw.final_verdict,
                final_rationale=raw.final_rationale[:2000],
                confidence=raw.confidence,
                proposed_position_pct=raw.proposed_position_pct,
                degraded_facets=degraded_facets,  # type: ignore[arg-type]
            )
            return normalized, []
        except Exception:
            pass

    return (
        fallback_debate_result(
            research=research,
            debate_turns=debate_turns,
            degraded_facets=degraded_facets,
        ),
        ["DEBATE_JUDGE_LLM_FALLBACK"],
    )


__all__ = [
    "coerce_turns",
    "collect_research",
    "collect_summaries",
    "collect_valid_indicator_names",
    "fallback_debate_result",
    "fallback_turn",
    "run_debate_judge",
    "run_debate_turn",
]
