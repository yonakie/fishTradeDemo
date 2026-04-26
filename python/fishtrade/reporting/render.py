"""Markdown report rendering for the multi-agent decision pipeline.

Renders the final ``GraphState`` (or its ``BuilderState`` superset) into
either a Chinese, English, or bilingual Markdown document with the eight
sections from design 6.2:

1. Header (ticker / as_of_date / run_id / mode / final_verdict)
2. 三面打分 (Research scores — three 10-row tables)
3. 辩论实录 (Debate turns + judge rationale)
4. 风控判定 (Hard rules + VaR + soft flags)
5. 执行记录 (Order + fill_info + status)
6. Portfolio 变更 (before / after)
7. Warnings (degradation flags)
8. Trace 链接 (path to JSONL)

Templates live next to this module under ``templates/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from ..config.settings import settings
from ..observability.trace import trace_path_for

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def _make_env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=False,
        keep_trailing_newline=True,
        undefined=StrictUndefined,
    )
    env.filters["safe_str"] = lambda v: "—" if v is None else str(v)
    env.filters["pct"] = lambda v: "—" if v is None else f"{float(v):.2f}%"
    return env


def _build_context(state: Mapping[str, Any]) -> dict[str, Any]:
    """Project the runtime state into a flat dict the Jinja templates can iterate.

    All accessors are defensive — when the pipeline halts early (invalid
    ticker, hard-rule reject, HITL rejected), several keys may be missing.
    """
    run_input = dict(state.get("input") or {})
    research = dict(state.get("research") or {})
    debate = dict(state.get("debate") or {})
    risk = dict(state.get("risk") or {})
    execution = state.get("execution")
    portfolio_before = state.get("portfolio_before")
    portfolio_after = state.get("portfolio_after")
    warnings = list(state.get("warnings") or [])
    run_id = state.get("run_id") or "ad-hoc"
    halt_reason = state.get("halt_reason")

    return {
        "run_id": run_id,
        "ticker": run_input.get("ticker", "—"),
        "capital": run_input.get("capital"),
        "mode": run_input.get("mode", "—"),
        "as_of_date": run_input.get("as_of_date", "—"),
        "language": run_input.get("language", "bilingual"),
        "debate_rounds": run_input.get("debate_rounds", 0),
        "hitl_enabled": bool(run_input.get("hitl", False)),
        "halt_reason": halt_reason,
        "research_fundamental": research.get("fundamental"),
        "research_technical": research.get("technical"),
        "research_sentimental": research.get("sentimental"),
        "debate": debate or None,
        "risk": risk or None,
        "execution": execution,
        "portfolio_before": portfolio_before,
        "portfolio_after": portfolio_after,
        "warnings": warnings,
        "trace_path": str(trace_path_for(run_id)),
    }


def render_report(
    state: Mapping[str, Any],
    language: str = "bilingual",
) -> str:
    """Render the report as a Markdown string. Pure function — no I/O."""
    env = _make_env()
    ctx = _build_context(state)

    if language == "bilingual":
        zh = env.get_template("report_zh.md.j2").render(s=ctx)
        en = env.get_template("report_en.md.j2").render(s=ctx)
        return f"# 中文版\n\n{zh}\n\n---\n\n# English\n\n{en}\n"
    if language == "zh":
        return env.get_template("report_zh.md.j2").render(s=ctx)
    if language == "en":
        return env.get_template("report_en.md.j2").render(s=ctx)
    raise ValueError(f"unsupported language: {language!r}")


def _resolve_report_path(
    *, ticker: str, as_of_date: str, run_id: str, report_dir: Path
) -> Path:
    base = report_dir / f"{ticker}-{as_of_date}.md"
    if not base.exists():
        return base
    short = run_id.replace("e2e-", "").replace("run-", "")[:8] or "x"
    return report_dir / f"{ticker}-{as_of_date}-{short}.md"


def write_report(
    state: Mapping[str, Any],
    *,
    language: str = "bilingual",
    report_dir: Path | str | None = None,
) -> Path:
    """Render and write the report to ``<report_dir>/<ticker>-<as_of>[-<short>].md``.

    Returns the resolved path. Raises on render error so callers can
    record ``RENDER_FAILED`` warnings.
    """
    text = render_report(state, language=language)
    target_dir = Path(report_dir) if report_dir else Path(settings.report_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    run_input = state.get("input") or {}
    path = _resolve_report_path(
        ticker=str(run_input.get("ticker") or "UNKNOWN"),
        as_of_date=str(run_input.get("as_of_date") or "unknown"),
        run_id=str(state.get("run_id") or "adhoc"),
        report_dir=target_dir,
    )
    path.write_text(text, encoding="utf-8")
    return path


__all__ = ["render_report", "write_report"]
