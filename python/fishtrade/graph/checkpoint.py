"""Checkpoint factory: a sqlite-backed saver for HITL pause/resume.

LangGraph's :class:`SqliteSaver` takes a raw ``sqlite3.Connection``. We
own the connection lifecycle here so the file is created at a known
location (``data/checkpoints.sqlite`` by default) and ``check_same_thread``
is set to ``False`` — LangGraph dispatches node calls across worker
threads, and the connection is reused.

The CLI wires ``build_graph(checkpointer=...)``; ``cli.py`` (Session 5)
will call :func:`open_sqlite_checkpointer` and tear it down with
``with`` semantics. Tests inject ``MemorySaver`` directly via the same
constructor parameter and skip this module entirely.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from ..config.settings import settings


def default_checkpoint_path() -> Path:
    """``<data_dir>/checkpoints.sqlite`` — created on first use."""
    return Path(settings.data_dir) / "checkpoints.sqlite"


def open_sqlite_checkpointer(path: Path | str | None = None) -> SqliteSaver:
    """Open (and ``setup``) a SqliteSaver bound to ``path``.

    The returned saver owns the underlying ``sqlite3.Connection``. Callers
    must close it via ``saver.conn.close()`` when shutting the CLI down,
    or use a ``with`` block around the build_graph call.
    """
    target = Path(path) if path is not None else default_checkpoint_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


__all__ = ["default_checkpoint_path", "open_sqlite_checkpointer"]
