"""H6 — short history (< 60 trading days) → VaR sample-size degradation.

We feed a 30-day price history. The VaR calculator returns
``passed=False`` with a ``fallback_reason`` mentioning "<60", which
flips the overall risk decision to ``reject``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from langgraph.checkpoint.memory import MemorySaver

from fishtrade.graph import build_graph
from fishtrade.tools.yf_client import _df_to_payload  # type: ignore
from tests.integration._fixtures import make_market_data


def _short_history(n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    # Allocate slack and trim — pandas can return one fewer date when the
    # end falls on a weekend (the fixture pattern in tests/integration/_fixtures).
    dates = pd.date_range(end="2026-04-25", periods=n + 10, freq="B")[-n:]
    n = len(dates)
    rets = rng.normal(0.0005, 0.012, size=n)
    closes = 100 * np.exp(np.cumsum(rets))
    return pd.DataFrame(
        {
            "Open": closes * 0.998,
            "High": closes * 1.005,
            "Low": closes * 0.993,
            "Close": closes,
            "Volume": rng.integers(1_000_000, 5_000_000, size=n),
        },
        index=dates,
    )


def test_short_history_degrades_var_check(patch_ark, initial_state):
    patch_ark(judge_verdict="BUY", judge_pct=7.0)
    md = make_market_data()
    md["history"] = _df_to_payload(_short_history())
    graph = build_graph(checkpointer=MemorySaver(), interrupt_before=[])

    final = graph.invoke(
        initial_state(market_data=md, run_id="h6-short"),
        config={"configurable": {"thread_id": "h6-short"}},
    )
    assert final["risk"]["decision"] == "reject"
    var_res = final["risk"]["var_result"]
    assert var_res["passed"] is False
    assert "60" in (var_res.get("fallback_reason") or "")
