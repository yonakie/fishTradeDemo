"""D5 acceptance: pull real AAPL data and run all three indicator compute_all_*.

Run with::

    python -m scripts.d5_aapl

The script never raises on degraded indicators — that *is* the contract: a
real network call may legitimately drop options chain or insider txs and we
must keep going.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from fishtrade.tools.indicators_fund import compute_all_fundamental
from fishtrade.tools.indicators_sent import compute_all_sentimental
from fishtrade.tools.indicators_tech import compute_all_technical
from fishtrade.tools.industry_classifier import classify_industry
from fishtrade.tools.var_calculator import compute_var_historical
from fishtrade.tools.yf_client import YFinanceClient


def _fmt(score) -> dict[str, Any]:
    return {
        "name": score.name,
        "raw": score.raw_value,
        "score": score.score,
        "degraded": score.is_degraded,
        "reason": (score.degrade_reason if score.is_degraded else score.reasoning)[:80],
    }


def main(ticker: str = "AAPL") -> int:
    client = YFinanceClient()

    print(f">>> fetching {ticker} info / history / SPY benchmark ...")
    info = client.get_info(ticker)
    history = client.get_history(ticker, period="1y")
    benchmark = client.get_history("SPY", period="1y")
    options = client.get_option_chain_safe(ticker)
    holders = client.get_institutional_holders_safe(ticker)
    insider = client.get_insider_transactions_safe(ticker)
    upgrades = client.get_upgrades_downgrades_safe(ticker)
    earnings = client.get_earnings_dates_safe(ticker)

    bundle = {
        "info": info,
        "history": history,
        "benchmark_history": benchmark,
        "options_chain": options,
        "institutional_holders": holders,
        "insider_transactions": insider,
        "upgrades_downgrades": upgrades,
        "earnings_dates": earnings,
    }

    industry = classify_industry(info)
    print(f"sector={info.get('sector')!r}  industry_class={industry}")
    print(f"history rows={len(history)}, benchmark rows={len(benchmark)}")

    funda = compute_all_fundamental(bundle)
    tech = compute_all_technical(bundle)
    sent = compute_all_sentimental(bundle)

    for label, scores in [("FUNDAMENTAL", funda), ("TECHNICAL", tech), ("SENTIMENTAL", sent)]:
        total = sum(s.score for s in scores)
        deg = sum(1 for s in scores if s.is_degraded)
        verdict = "BUY" if total >= 5 else ("HOLD" if total >= 1 else "SELL")
        print(f"\n=== {label}  total={total:+d}  degraded={deg}/10  verdict={verdict} ===")
        for s in scores:
            print(f"  {s.name:20s} score={s.score:+d} degraded={s.is_degraded} :: {s.reasoning[:80]}")
        assert len(scores) == 10, f"{label} returned {len(scores)} scores"

    var = compute_var_historical(history, proposed_position_pct=5.0)
    print(f"\nVaR(95) historical: {var.var_95*100:.2f}% sample={var.sample_size} passed={var.passed}")

    # JSON dump for inspection
    print("\n[json]")
    print(json.dumps(
        {
            "fundamental": [_fmt(s) for s in funda],
            "technical":   [_fmt(s) for s in tech],
            "sentimental": [_fmt(s) for s in sent],
            "totals": {
                "fundamental": sum(s.score for s in funda),
                "technical":   sum(s.score for s in tech),
                "sentimental": sum(s.score for s in sent),
            },
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(*(sys.argv[1:] or ["AAPL"])))
