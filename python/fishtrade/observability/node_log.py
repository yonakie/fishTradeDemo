"""Per-node logging wrapper.

Wraps each LangGraph node so the console shows three structured events
per node: ``node_started``, ``node_completed`` (with derived status +
latency), and a human-readable ``node_content`` event carrying the
fields a frontend would render (research highlights, debate arguments,
risk reasoning, etc.). On exception we emit ``node_failed`` and re-raise.

The status is *derived* from the patch shape (no node code changes
needed):

* ``halt`` — patch sets ``halt_reason``
* ``reject`` — patch's ``risk.decision == "reject"``
* ``skipped`` — patch's ``execution.status == "skipped"``
* ``fallback`` — patch's ``warnings`` contain a token ending in
  ``_FALLBACK`` / ``_SKIPPED_ALL_DEGRADED`` / ``_MOCK_FILL`` etc.
* ``success`` — anything else
"""

from __future__ import annotations

import time
from typing import Any, Callable

from .logger import get_logger

NodeFn = Callable[[dict], dict]

logger = get_logger("fishtrade.node")

_FALLBACK_SUFFIXES = (
    "_FALLBACK",
    "_MOCK_FILL",
    "_SKIPPED_ALL_DEGRADED",
    "RENDER_FAILED",
    "PORTFOLIO_PERSIST_FAILED",
)


def _derive_status(patch: dict) -> str:
    if not isinstance(patch, dict):
        return "success"
    if patch.get("halt_reason"):
        return "halt"
    risk = patch.get("risk")
    if isinstance(risk, dict) and risk.get("decision") == "reject":
        return "reject"
    execution = patch.get("execution")
    if isinstance(execution, dict) and execution.get("status") == "skipped":
        return "skipped"
    warnings = patch.get("warnings") or []
    if isinstance(warnings, list):
        for w in warnings:
            ws = str(w)
            if any(ws.endswith(suf) or suf in ws for suf in _FALLBACK_SUFFIXES):
                return "fallback"
    return "success"


def _trim(text: Any, n: int = 600) -> str:
    s = "" if text is None else str(text)
    return s if len(s) <= n else s[: n - 1] + "…"


# ---------------------------------------------------------------------------
# Per-node content extractors — return a dict logged as ``node_content``.
# Each extractor reads from the patch (preferred) and falls back to state
# when the relevant field is merged-only (e.g. research nodes write
# ``research.<facet>`` but the merged view is in state).
# ---------------------------------------------------------------------------


def _extract_research(facet: str, patch: dict, state: dict) -> dict | None:
    research = (patch.get("research") or {}).get(facet) if isinstance(patch.get("research"), dict) else None
    if not research:
        return None
    return {
        "facet": facet,
        "verdict": research.get("verdict"),
        "confidence": research.get("confidence"),
        "total_score": research.get("total_score"),
        "is_facet_degraded": research.get("is_facet_degraded"),
        "key_highlights": list(research.get("key_highlights") or [])[:5],
    }


def _extract_debate_turns(patch: dict) -> list[dict] | None:
    turns = patch.get("debate_turns")
    if not turns:
        return None
    out: list[dict] = []
    for t in turns:
        if not isinstance(t, dict):
            continue
        out.append(
            {
                "round": t.get("round"),
                "role": t.get("role"),
                "conclusion": t.get("conclusion"),
                "is_fallback": t.get("is_fallback"),
                "cited_indicators": list(t.get("cited_indicators") or [])[:6],
                "argument": _trim(t.get("argument"), 800),
            }
        )
    return out or None


def _extract_judge(patch: dict) -> dict | None:
    debate = patch.get("debate")
    if not isinstance(debate, dict):
        return None
    return {
        "final_verdict": debate.get("final_verdict"),
        "confidence": debate.get("confidence"),
        "proposed_position_pct": debate.get("proposed_position_pct"),
        "degraded_facets": list(debate.get("degraded_facets") or []),
        "final_rationale": _trim(debate.get("final_rationale"), 800),
    }


def _extract_risk(patch: dict) -> dict | None:
    risk = patch.get("risk")
    if isinstance(risk, dict):
        soft = risk.get("soft_judgment") or {}
        return {
            "decision": risk.get("decision"),
            "adjusted_position_pct": risk.get("adjusted_position_pct"),
            "reject_reason": risk.get("reject_reason"),
            "soft_flags": list(soft.get("flags") or []),
            "soft_adjustment": soft.get("adjustment"),
            "soft_reasoning": _trim(soft.get("reasoning"), 600),
        }
    partial = patch.get("risk_partial")
    if isinstance(partial, dict):
        # Intermediate hard / VaR pass — surface a brief progress line.
        out: dict[str, Any] = {"stage": "partial"}
        if "hard_checks" in partial:
            out["hard_checks_passed"] = sum(
                1 for c in partial["hard_checks"] if isinstance(c, dict) and c.get("passed")
            )
            out["hard_checks_total"] = len(partial["hard_checks"])
        if "var_result" in partial:
            vr = partial["var_result"] or {}
            out["var_passed"] = vr.get("passed")
            out["var_portfolio_impact_pct"] = (
                round(float(vr.get("portfolio_impact") or 0) * 100, 4)
            )
        return out
    return None


def _extract_execution(patch: dict) -> dict | None:
    execution = patch.get("execution")
    if not isinstance(execution, dict):
        return None
    order = execution.get("order") or {}
    fill = execution.get("fill_info") or {}
    out: dict[str, Any] = {
        "mode": execution.get("mode"),
        "status": execution.get("status"),
        "error": execution.get("error"),
    }
    if order:
        out["order"] = {
            "symbol": order.get("symbol"),
            "side": order.get("side"),
            "qty": order.get("qty"),
            "limit_price": order.get("limit_price"),
        }
    if fill:
        out["fill"] = {
            "avg_price": fill.get("avg_price"),
            "filled_qty": fill.get("filled_qty"),
        }
    return out


def _extract_portfolio(patch: dict) -> dict | None:
    snap = patch.get("portfolio_after")
    if not isinstance(snap, dict):
        return None
    return {
        "nav": snap.get("nav"),
        "cash": snap.get("cash"),
        "n_positions": len(snap.get("positions") or []),
        "max_drawdown_pct": snap.get("max_drawdown_pct"),
    }


def _extract_validate(patch: dict) -> dict | None:
    if patch.get("halt_reason"):
        return {"halt_reason": patch.get("halt_reason")}
    return None


def _extract_fetch_market(patch: dict) -> dict | None:
    md = patch.get("market_data") or {}
    if not md:
        return None
    fw = md.get("fetch_warnings") or []
    return {
        "has_history": md.get("history") is not None,
        "has_financials": md.get("financials") is not None,
        "fetch_warnings": list(fw),
    }


# Map node name → content extractor. Plumbing-only nodes (hitl_gate /
# execute_router_dispatch / render_report) skip content emission.
_EXTRACTORS: dict[str, Callable[[dict, dict], dict | None]] = {
    "validate_input": lambda p, s: _extract_validate(p),
    "fetch_market": lambda p, s: _extract_fetch_market(p),
    "research_fund": lambda p, s: _extract_research("fundamental", p, s),
    "research_tech": lambda p, s: _extract_research("technical", p, s),
    "research_sent": lambda p, s: _extract_research("sentimental", p, s),
    "debate_open_bull": lambda p, s: {"turns": _extract_debate_turns(p)} if _extract_debate_turns(p) else None,
    "debate_open_bear": lambda p, s: {"turns": _extract_debate_turns(p)} if _extract_debate_turns(p) else None,
    "debate_rebuttal": lambda p, s: {"turns": _extract_debate_turns(p)} if _extract_debate_turns(p) else None,
    "debate_judge": lambda p, s: _extract_judge(p),
    "risk_hard": lambda p, s: _extract_risk(p),
    "risk_var": lambda p, s: _extract_risk(p),
    "risk_soft": lambda p, s: _extract_risk(p),
    "execute_dryrun": lambda p, s: _extract_execution(p),
    "execute_paper": lambda p, s: _extract_execution(p),
    "execute_backtest": lambda p, s: _extract_execution(p),
    "skip_execution": lambda p, s: _extract_execution(p),
    "update_portfolio": lambda p, s: _extract_portfolio(p),
}


def wrap_node(name: str, fn: NodeFn) -> NodeFn:
    """Return ``fn`` wrapped with start/complete/content/failed logging."""

    extractor = _EXTRACTORS.get(name)

    def _wrapped(state: dict) -> dict:
        run_id = state.get("run_id", "ad-hoc") if isinstance(state, dict) else "ad-hoc"
        logger.info("node_started", node=name, run_id=run_id)
        t0 = time.perf_counter()
        try:
            patch = fn(state)
        except Exception as exc:  # noqa: BLE001
            latency_ms = int((time.perf_counter() - t0) * 1000)
            logger.error(
                "node_failed",
                node=name,
                run_id=run_id,
                latency_ms=latency_ms,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise

        latency_ms = int((time.perf_counter() - t0) * 1000)
        patch_dict = patch if isinstance(patch, dict) else {}
        status = _derive_status(patch_dict)
        new_warnings = patch_dict.get("warnings") or []
        logger.info(
            "node_completed",
            node=name,
            run_id=run_id,
            status=status,
            latency_ms=latency_ms,
            warnings=list(new_warnings) if isinstance(new_warnings, list) else [],
        )

        if extractor is not None:
            try:
                content = extractor(patch_dict, state if isinstance(state, dict) else {})
            except Exception as exc:  # noqa: BLE001 — never let logging break a run
                logger.warning("node_content_extract_failed", node=name, error=str(exc))
                content = None
            if content:
                logger.info("node_content", node=name, run_id=run_id, **content)

        return patch

    _wrapped.__name__ = getattr(fn, "__name__", name)
    _wrapped.__qualname__ = getattr(fn, "__qualname__", name)
    return _wrapped


__all__ = ["wrap_node"]
