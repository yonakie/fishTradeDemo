"""H7 — Portfolio store auto-initialises on first run.

Uses an empty temp directory for ``data_dir`` so the store has no prior
JSON file. Loading must seed an initial snapshot and persist it.
"""

from __future__ import annotations

from pathlib import Path

from fishtrade.portfolio.store import PortfolioStore


def test_portfolio_store_initialises_when_missing(tmp_path: Path):
    store = PortfolioStore(
        path=tmp_path / "portfolio.json",
        nav_path=tmp_path / "nav.jsonl",
    )
    assert not (tmp_path / "portfolio.json").exists()

    snap = store.load(capital_default=50_000.0)
    assert snap.cash == 50_000.0
    assert snap.nav == 50_000.0
    assert snap.positions == []

    # Persisted to disk so the next run picks it up unchanged.
    assert (tmp_path / "portfolio.json").exists()
    snap2 = store.load(capital_default=99_999.0)
    assert snap2.nav == 50_000.0  # capital_default ignored on second load
