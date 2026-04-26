"""H-extra — ``--mode paper`` without Alpaca creds → MISSING_ALPACA_KEY.

Per design 6.1: the check fires at startup, before the graph is built.
"""

from __future__ import annotations

from typer.testing import CliRunner

from fishtrade.cli import app
from fishtrade.config import settings as settings_mod


runner = CliRunner()


def test_paper_mode_without_alpaca_key_fails_fast(monkeypatch):
    # Force the singleton to look unconfigured for Alpaca, even if local
    # .env contains creds.
    monkeypatch.setattr(settings_mod.settings, "alpaca_api_key", "")
    monkeypatch.setattr(settings_mod.settings, "alpaca_secret_key", "")
    # Ensure ARK appears configured so we exercise the alpaca branch.
    monkeypatch.setattr(settings_mod.settings, "ark_api_key", "test-mock-key-not-real")
    monkeypatch.setattr(settings_mod.settings, "ark_model_id", "mock-model")

    result = runner.invoke(
        app,
        ["run", "--ticker", "AAPL", "--mode", "paper"],
        catch_exceptions=False,
    )
    assert result.exit_code != 0
    assert "MISSING_ALPACA_KEY" in result.output


def test_replay_does_not_call_llm(monkeypatch, tmp_path):
    """Replay rebuilds report from trace WITHOUT touching Ark."""
    from fishtrade import cli as cli_mod
    from fishtrade.observability import trace as trace_mod
    from fishtrade.config import settings as settings_mod

    call_counter = {"n": 0}

    def _boom(*args, **kwargs):
        call_counter["n"] += 1
        raise RuntimeError("LLM should not be called during replay")

    # Patch every Ark call-site — replay must not invoke any of them.
    for target in (
        "fishtrade.agents.research._common.generate_ark_response",
        "fishtrade.agents.debate._common.generate_ark_response",
        "fishtrade.agents.risk.soft_judge.generate_ark_response",
    ):
        monkeypatch.setattr(target, _boom)

    # Redirect log_dir + report_dir into tmp.
    monkeypatch.setattr(settings_mod.settings, "log_dir", tmp_path / "logs")
    monkeypatch.setattr(settings_mod.settings, "report_dir", tmp_path / "reports")

    # Seed a trace file so replay finds something to summarise.
    trace_dir = tmp_path / "logs" / "trace"
    trace_dir.mkdir(parents=True)
    (trace_dir / "fixture-run.jsonl").write_text(
        '{"ts":"2026-04-25T00:00:00Z","run_id":"fixture-run","node":"x"}\n',
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["replay", "--run-id", "fixture-run", "--language", "zh"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    # Most importantly: zero LLM calls.
    assert call_counter["n"] == 0
    # Report file produced.
    files = list((tmp_path / "reports").glob("*.md"))
    assert len(files) == 1
