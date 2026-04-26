"""H3 — R3 (VaR 95) rejects when single-name VaR is too large.

We feed a synthetic price history that drops every other day by ~5%.
Historical-simulation VaR(95) on that series sits well above the 2%
portfolio-impact ceiling once we propose a 7% position.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from langgraph.checkpoint.memory import MemorySaver

from fishtrade.graph import build_graph
from fishtrade.tools.yf_client import _df_to_payload  # type: ignore

from tests.integration._fixtures import make_market_data


def _violent_history(n: int = 250, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end="2026-04-25", periods=n + 10, freq="B")[-n:]
    n = len(dates)
    # σ=0.50 daily — daily VaR(95) ≈ 0.8 → portfolio impact at 7% pct ≈ 5.6%,
    # well over the 2% R3 ceiling.
    rets = rng.normal(loc=0.0, scale=0.50, size=n)
    closes = 100.0 * np.exp(np.cumsum(rets))
    return pd.DataFrame(
        {
            "Open": closes * 0.99,
            "High": closes * 1.10,
            "Low": closes * 0.90,
            "Close": closes,
            "Volume": rng.integers(1_000_000, 5_000_000, size=n),
        },
        index=dates,
    )


def test_r3_rejects_when_var_exceeds_portfolio_limit(patch_ark, initial_state):
    patch_ark(judge_verdict="BUY", judge_pct=7.0)
    md = make_market_data()
    md["history"] = _df_to_payload(_violent_history())
    graph = build_graph(checkpointer=MemorySaver(), interrupt_before=[])

    final = graph.invoke(
        initial_state(market_data=md, run_id="h3-var"),
        config={"configurable": {"thread_id": "h3-var"}},
    )
    assert final["risk"]["decision"] == "reject"
    # The var_check_node sets reject_reason to a VaR-related string.
    assert "VaR" in (final["risk"].get("reject_reason") or "")
