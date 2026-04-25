"""Portfolio JSON store with copy-on-write atomic saves.

Path layout::

    data/portfolio.json          # current snapshot (PortfolioSnapshot)
    data/nav_history.jsonl       # one NavSnapshot per line (append-only)
"""

from __future__ import annotations

import json
from pathlib import Path

from ..config.settings import settings
from ..models.portfolio import NavSnapshot, PortfolioSnapshot


def _atomic_write(path: Path, payload: str) -> None:
    """Write ``payload`` to ``path`` via a ``.tmp`` rename — no half-files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)


class PortfolioStore:
    """Read/write the on-disk portfolio with optional dependency injection."""

    def __init__(
        self,
        path: Path | str | None = None,
        nav_path: Path | str | None = None,
    ) -> None:
        data_dir = Path(settings.data_dir)
        self.path = Path(path) if path else data_dir / "portfolio.json"
        self.nav_path = Path(nav_path) if nav_path else data_dir / "nav_history.jsonl"

    # ---------- snapshot ------------------------------------------------

    def load(self, capital_default: float) -> PortfolioSnapshot:
        if not self.path.exists():
            snap = PortfolioSnapshot(
                cash=capital_default, positions=[], nav=capital_default
            )
            self.save_atomic(snap)
            return snap
        return PortfolioSnapshot.model_validate_json(
            self.path.read_text(encoding="utf-8")
        )

    def save_atomic(self, snap: PortfolioSnapshot) -> None:
        _atomic_write(self.path, snap.model_dump_json(indent=2))

    # ---------- nav history --------------------------------------------

    def append_nav(self, date: str, nav: float) -> NavSnapshot:
        entry = NavSnapshot(date=date, nav=nav)
        self.nav_path.parent.mkdir(parents=True, exist_ok=True)
        with self.nav_path.open("a", encoding="utf-8") as fh:
            fh.write(entry.model_dump_json() + "\n")
        return entry

    def read_nav_history(self) -> list[NavSnapshot]:
        if not self.nav_path.exists():
            return []
        out: list[NavSnapshot] = []
        for line in self.nav_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(NavSnapshot.model_validate_json(line))
            except Exception:
                continue
        return out

    def overwrite_nav_history(self, snaps: list[NavSnapshot]) -> None:
        """Used by tests / migrations.  Atomic via rewrite."""
        body = "\n".join(s.model_dump_json() for s in snaps) + ("\n" if snaps else "")
        _atomic_write(self.nav_path, body)
