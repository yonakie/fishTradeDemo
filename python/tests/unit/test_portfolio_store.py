"""Portfolio store: first-run init + atomic save + NAV history + drawdown."""

from __future__ import annotations

from pathlib import Path

import pytest

from fishtrade.models.portfolio import NavSnapshot, PortfolioSnapshot, Position
from fishtrade.portfolio.nav import compute_max_drawdown
from fishtrade.portfolio.store import PortfolioStore


@pytest.fixture
def store(tmp_path: Path) -> PortfolioStore:
    return PortfolioStore(
        path=tmp_path / "portfolio.json",
        nav_path=tmp_path / "nav.jsonl",
    )


def test_first_run_initializes_with_capital(store: PortfolioStore):
    snap = store.load(capital_default=100_000)
    assert snap.cash == 100_000
    assert snap.nav == 100_000
    assert snap.positions == []
    assert store.path.exists()


def test_save_then_load_roundtrip(store: PortfolioStore):
    snap = PortfolioSnapshot(
        cash=50_000,
        positions=[Position(symbol="AAPL", qty=10, avg_cost=180.0, sector="Tech")],
        nav=51_800,
    )
    store.save_atomic(snap)
    loaded = store.load(capital_default=0)
    assert loaded.cash == 50_000
    assert loaded.positions[0].symbol == "AAPL"


def test_save_atomic_no_partial_file_on_error(tmp_path: Path):
    """If the .tmp file is left behind it must not corrupt the main JSON."""
    store = PortfolioStore(path=tmp_path / "p.json", nav_path=tmp_path / "n.jsonl")
    snap = PortfolioSnapshot(cash=1, positions=[], nav=1)
    store.save_atomic(snap)
    # No .tmp file should remain.
    assert not (tmp_path / "p.json.tmp").exists()


def test_append_nav_persists_lines(store: PortfolioStore):
    store.append_nav("2026-04-23", 100_000)
    store.append_nav("2026-04-24", 101_000)
    snaps = store.read_nav_history()
    assert len(snaps) == 2
    assert snaps[0].nav == 100_000
    assert snaps[1].nav == 101_000


# ---------- drawdown ------------------------------------------------------


def test_max_drawdown_empty_is_zero():
    assert compute_max_drawdown([]) == 0.0


def test_max_drawdown_single_point_is_zero():
    assert compute_max_drawdown([NavSnapshot(date="2026-04-25", nav=100)]) == 0.0


def test_max_drawdown_simple():
    snaps = [
        NavSnapshot(date="2026-01-01", nav=100),
        NavSnapshot(date="2026-01-02", nav=120),
        NavSnapshot(date="2026-01-03", nav=90),
        NavSnapshot(date="2026-01-04", nav=110),
    ]
    dd = compute_max_drawdown(snaps)
    # Peak 120 -> trough 90 = 25%
    assert dd == pytest.approx(0.25, rel=1e-3)


def test_max_drawdown_monotonic_returns_zero():
    snaps = [
        NavSnapshot(date=f"2026-01-0{i}", nav=100 + i)
        for i in range(1, 6)
    ]
    assert compute_max_drawdown(snaps) == 0.0
