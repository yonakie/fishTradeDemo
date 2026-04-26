"""Typer-based CLI entry point — see design 6.1.

Subcommands
-----------

* ``doctor``  — environment self-check (Ark / Alpaca keys, dirs, yfinance)
* ``run``     — single-ticker decision run (dryrun / paper / backtest)
* ``resume``  — resume a HITL-paused run with approved/rejected decision
* ``replay``  — re-render the report from existing trace WITHOUT calling LLM

Hard rule: ``--mode paper`` requires Alpaca creds — checked at startup so
nothing executes against the live paper sandbox without a key.
"""

from __future__ import annotations

import json
import re
import sys
import uuid
from datetime import date as date_cls
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .config.settings import settings
from .graph import build_graph, open_sqlite_checkpointer
from .observability.logger import configure_logging, get_logger
from .observability.trace import iter_trace, trace_path_for
from .reporting.render import write_report

app = typer.Typer(
    add_completion=False,
    help="fishtrade — 多 Agent 量化决策 CLI",
    no_args_is_help=True,
)
console = Console()
err_console = Console(stderr=True)
logger = get_logger(__name__)

_TICKER_RE = re.compile(r"^[A-Z.\-]{1,6}$")


# =============================================================================
# Helpers
# =============================================================================


def _exit(code: int, msg: str | None = None) -> None:
    if msg:
        err_console.print(msg)
    raise typer.Exit(code=code)


def _validate_ticker_or_die(ticker: str) -> str:
    """Apply the design 6.1 regex literally — lowercase fails, no auto-upper.

    Lowercase tickers fail the same regex yfinance applies, and
    silently upper-casing would mask the user's typo before
    ``validate_input_node`` ever runs.
    """
    norm = ticker.strip()
    if not _TICKER_RE.match(norm):
        _exit(2, f"[red]INVALID_TICKER[/red]: {ticker!r} 不符合 ^[A-Z.\\-]{{1,6}}$")
    return norm


def _validate_mode_credentials_or_die(mode: str) -> None:
    if mode == "paper" and not settings.has_alpaca_credentials():
        _exit(
            2,
            "[red]MISSING_ALPACA_KEY[/red]: paper 模式需要 ALPACA_API_KEY/SECRET_KEY，"
            "请检查 .env 文件",
        )


def _ensure_ark_or_die() -> None:
    if not settings.has_ark_key():
        _exit(2, "[red]MISSING_ARK_KEY[/red]: 请在 .env 中设置 ARK_API_KEY")


def _initial_state(
    *,
    ticker: str,
    capital: float,
    mode: str,
    debate_rounds: int,
    as_of_date: str,
    language: str,
    hitl: bool,
    run_id: str,
) -> dict:
    return {
        "input": {
            "ticker": ticker,
            "capital": capital,
            "mode": mode,
            "debate_rounds": debate_rounds,
            "as_of_date": as_of_date,
            "language": language,
            "hitl": hitl,
        },
        "research": {},
        "debate_turns": [],
        "warnings": [],
        "tokens_total": 0,
        "latency_ms_total": 0,
        "run_id": run_id,
    }


def _load_portfolio_into_state(state: dict, *, capital: float) -> dict:
    """Lazy-import to keep ``cli`` importable without portfolio."""
    from .portfolio.store import PortfolioStore

    snap = PortfolioStore().load(capital_default=capital)
    state["portfolio_before"] = snap.model_dump()
    return state


def _persist_portfolio_after(final: dict) -> None:
    """Write the post-run portfolio + nav history. Best-effort."""
    from .models.portfolio import PortfolioSnapshot
    from .portfolio.store import PortfolioStore

    portfolio_after = final.get("portfolio_after")
    if not portfolio_after:
        return
    try:
        snap = PortfolioSnapshot.model_validate(portfolio_after)
    except Exception as exc:  # noqa: BLE001
        logger.warning("persist_portfolio_failed", error=str(exc))
        return
    store = PortfolioStore()
    store.save_atomic(snap)
    if snap.nav_history:
        # Re-write the nav_history as the source of truth so subsequent runs
        # see the same drawdown surface the just-finished run computed.
        store.overwrite_nav_history(snap.nav_history)


def _print_summary(final: dict) -> None:
    table = Table(title="决策摘要", show_header=True)
    table.add_column("字段")
    table.add_column("值")
    run_input = final.get("input") or {}
    table.add_row("Ticker", str(run_input.get("ticker", "—")))
    table.add_row("Mode", str(run_input.get("mode", "—")))
    table.add_row("Run ID", str(final.get("run_id", "—")))
    debate = final.get("debate") or {}
    if debate:
        table.add_row("Verdict", str(debate.get("final_verdict", "—")))
        table.add_row("Proposed %", f"{float(debate.get('proposed_position_pct', 0.0)):.2f}")
    risk = final.get("risk") or {}
    if risk:
        table.add_row("Risk", f"{risk.get('decision')}@{risk.get('adjusted_position_pct')}%")
        if risk.get("reject_reason"):
            table.add_row("Reject reason", str(risk.get("reject_reason")))
    execution = final.get("execution") or {}
    if execution:
        table.add_row("Execution", f"{execution.get('mode')} / {execution.get('status')}")
    if final.get("halt_reason"):
        table.add_row("Halt", str(final.get("halt_reason")))
    warnings = final.get("warnings") or []
    if warnings:
        table.add_row("Warnings", ", ".join(warnings[:6]) + ("..." if len(warnings) > 6 else ""))
    console.print(table)


def _hitl_prompt(snap_values: dict) -> str:
    risk = snap_values.get("risk") or {}
    debate = snap_values.get("debate") or {}
    pct = risk.get("adjusted_position_pct", 0.0)
    verdict = debate.get("final_verdict", "—")
    rej = risk.get("reject_reason")
    body = (
        f"\n[bold]HITL 审批[/bold]\n"
        f"  最终判定：{verdict}\n"
        f"  风控建议仓位：{pct}%\n"
    )
    if rej:
        body += f"  拒绝原因（仅供参考）：{rej}\n"
    body += "\n是否批准执行？(y/N)"
    console.print(body)
    return typer.prompt("> ", default="N")


# =============================================================================
# doctor
# =============================================================================


@app.command(help="自检环境：Ark/Alpaca key、目录可写、yfinance 可达")
def doctor(
    skip_yfinance: bool = typer.Option(
        False, "--skip-yfinance", help="跳过真实 yfinance 网络探测"
    ),
) -> None:
    configure_logging(settings.log_level)
    table = Table(title="环境自检", show_header=True)
    table.add_column("检查项")
    table.add_column("结果")
    table.add_column("详情")

    has_ark = settings.has_ark_key()
    table.add_row(
        "ARK_API_KEY", "OK" if has_ark else "FAIL", "已配置" if has_ark else "请在 .env 中设置"
    )

    has_alpaca = settings.has_alpaca_credentials()
    table.add_row(
        "ALPACA creds (paper only)",
        "OK" if has_alpaca else "WARN",
        f"base_url={settings.alpaca_base_url}",
    )

    for name, path in (
        ("data_dir", Path(settings.data_dir)),
        ("log_dir", Path(settings.log_dir)),
        ("report_dir", Path(settings.report_dir)),
    ):
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            table.add_row(name, "OK", str(path))
        except Exception as exc:  # noqa: BLE001
            table.add_row(name, "FAIL", f"{path}: {exc}")

    if skip_yfinance:
        table.add_row("yfinance", "SKIP", "--skip-yfinance")
    else:
        try:
            from .tools.yf_client import YFinanceClient

            info = YFinanceClient().get_info("AAPL")
            table.add_row(
                "yfinance",
                "OK",
                f"AAPL price={info.get('regularMarketPrice')}",
            )
        except Exception as exc:  # noqa: BLE001
            table.add_row("yfinance", "WARN", f"探测失败：{exc}")

    console.print(table)


# =============================================================================
# run
# =============================================================================


@app.command(help="单标的决策运行")
def run(
    ticker: str = typer.Option(..., "--ticker"),
    capital: float = typer.Option(100_000.0, "--capital"),
    mode: str = typer.Option("dryrun", "--mode"),
    debate_rounds: int = typer.Option(1, "--debate-rounds", min=0, max=3),
    as_of_date: Optional[str] = typer.Option(None, "--as-of-date"),
    language: str = typer.Option("bilingual", "--language"),
    hitl: bool = typer.Option(False, "--hitl"),
) -> None:
    if mode not in {"dryrun", "paper", "backtest"}:
        _exit(2, f"[red]invalid --mode {mode}[/red]")
    if language not in {"zh", "en", "bilingual"}:
        _exit(2, f"[red]invalid --language {language}[/red]")

    norm_ticker = _validate_ticker_or_die(ticker)
    _ensure_ark_or_die()
    _validate_mode_credentials_or_die(mode)

    configure_logging(settings.log_level)
    as_of = as_of_date or date_cls.today().isoformat()
    run_id = f"run-{norm_ticker}-{as_of}-{uuid.uuid4().hex[:8]}"

    state = _initial_state(
        ticker=norm_ticker,
        capital=capital,
        mode=mode,
        debate_rounds=debate_rounds,
        as_of_date=as_of,
        language=language,
        hitl=hitl,
        run_id=run_id,
    )
    _load_portfolio_into_state(state, capital=capital)

    saver = open_sqlite_checkpointer()
    try:
        # ``interrupt_before=[]`` skips the HITL pause entirely when the user
        # didn't ask for it. The default in build_graph (``["hitl_gate"]``) is
        # the right shape only for hitl-enabled runs.
        graph = build_graph(
            checkpointer=saver,
            interrupt_before=["hitl_gate"] if hitl else [],
        )
        cfg = {"configurable": {"thread_id": run_id}}
        first = graph.invoke(state, config=cfg)
        snap = graph.get_state(cfg)

        if hitl and snap.next == ("hitl_gate",):
            decision = _hitl_prompt(snap.values).strip().lower()
            choice = "approved" if decision in {"y", "yes"} else "rejected"
            console.print(f"\n[bold]HITL decision:[/bold] {choice}")
            graph.update_state(cfg, {"hitl_decision": choice})
            final = graph.invoke(None, config=cfg)
        else:
            final = first

        _persist_portfolio_after(final)
        _print_summary(final)
        report_path = (
            Path(settings.report_dir) / f"{norm_ticker}-{as_of}.md"
        )
        if report_path.exists():
            console.print(f"\n[green]报告已生成：[/green] {report_path}")
    finally:
        try:
            saver.conn.close()
        except Exception:  # noqa: BLE001
            pass


# =============================================================================
# resume
# =============================================================================


@app.command(help="恢复一个 HITL 暂停的 run")
def resume(
    run_id: str = typer.Option(..., "--run-id"),
    decision: str = typer.Option(..., "--decision"),
) -> None:
    if decision not in {"approved", "rejected"}:
        _exit(2, "[red]--decision 只能是 approved 或 rejected[/red]")

    configure_logging(settings.log_level)
    saver = open_sqlite_checkpointer()
    try:
        graph = build_graph(checkpointer=saver)
        cfg = {"configurable": {"thread_id": run_id}}
        snap = graph.get_state(cfg)
        if not snap or not snap.next:
            _exit(2, f"[red]run-id {run_id!r} 未找到或不在挂起态[/red]")
        if "hitl_gate" not in snap.next:
            _exit(2, f"[red]run-id {run_id!r} 当前 next={snap.next} 不是 hitl_gate[/red]")
        graph.update_state(cfg, {"hitl_decision": decision})
        final = graph.invoke(None, config=cfg)
        _persist_portfolio_after(final)
        _print_summary(final)
    finally:
        try:
            saver.conn.close()
        except Exception:  # noqa: BLE001
            pass


# =============================================================================
# replay
# =============================================================================


@app.command(help="从 trace 重建报告（不重新调用 LLM）")
def replay(
    run_id: str = typer.Option(..., "--run-id"),
    language: str = typer.Option("bilingual", "--language"),
) -> None:
    """Re-render a report from saved trace without invoking the LLM.

    The CLI never calls ``generate_ark_response`` here — it loads the
    last persisted portfolio snapshot for context and reads the trace
    index file to compose a minimal state.
    """
    if language not in {"zh", "en", "bilingual"}:
        _exit(2, f"[red]invalid --language {language}[/red]")

    configure_logging(settings.log_level)
    trace_file = trace_path_for(run_id)
    if not trace_file.exists():
        _exit(2, f"[red]找不到 trace 文件 {trace_file}[/red]")

    # Try to recover the rich state from a SQLite checkpoint if present;
    # this lets replay produce the SAME report the original run did.
    state: dict = {
        "input": {
            "ticker": "REPLAY",
            "capital": 0.0,
            "mode": "dryrun",
            "debate_rounds": 0,
            "as_of_date": date_cls.today().isoformat(),
            "language": language,
            "hitl": False,
        },
        "run_id": run_id,
        "research": {},
        "debate_turns": [],
        "warnings": ["REPLAY_FROM_TRACE"],
        "tokens_total": 0,
        "latency_ms_total": 0,
    }
    saver = None
    try:
        saver = open_sqlite_checkpointer()
        graph = build_graph(checkpointer=saver)
        cfg = {"configurable": {"thread_id": run_id}}
        snap = graph.get_state(cfg)
        if snap and snap.values:
            state = dict(snap.values)
            state["warnings"] = list(state.get("warnings") or []) + ["REPLAY_FROM_TRACE"]
    except Exception as exc:  # noqa: BLE001
        logger.info("replay_no_checkpoint", run_id=run_id, error=str(exc))
    finally:
        if saver is not None:
            try:
                saver.conn.close()
            except Exception:  # noqa: BLE001
                pass

    # Synthesise a tiny replay summary block in the warnings so the report
    # makes it obvious this came from trace.
    trace_records = list(iter_trace(run_id))
    state["warnings"] = list(state.get("warnings") or []) + [
        f"REPLAY_TRACE_RECORDS:{len(trace_records)}"
    ]

    path = write_report(state, language=language)
    console.print(f"[green]报告已重建：[/green] {path}")


# =============================================================================
# Entrypoint
# =============================================================================


def main() -> None:
    app()


if __name__ == "__main__":
    main()
