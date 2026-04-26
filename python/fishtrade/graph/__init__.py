"""Graph-layer package — workflow assembly + checkpoint plumbing.

Public surface:

* :func:`build_graph` — compile and return the multi-agent graph.
* :func:`open_sqlite_checkpointer` — sqlite-backed saver factory for HITL.

``BuilderState`` and ``_RiskPartial`` are intentionally not exported;
they are private graph-implementation types (see Session-4 brief).
"""

from .builder import build_graph
from .checkpoint import default_checkpoint_path, open_sqlite_checkpointer

__all__ = [
    "build_graph",
    "default_checkpoint_path",
    "open_sqlite_checkpointer",
]
