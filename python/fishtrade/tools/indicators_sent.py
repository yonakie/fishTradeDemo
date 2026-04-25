"""Sentiment indicators (10 items).

Inputs come from a :class:`fishtrade.models.state.MarketDataBundle`.  Most
fields are best-effort: we degrade gracefully when yfinance does not return
the underlying object.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..models.research import IndicatorScore


@dataclass(frozen=True)
class _Out:
    score: int
    reasoning: str
    raw_value: float | str | None = None
    is_degraded: bool = False
    degrade_reason: str | None = None


def _missing(zh: str, reason: str = "yfinance 未返回该字段") -> _Out:
    return _Out(
        score=0,
        reasoning=f"{zh} 数据缺失，按中性处理",
        is_degraded=True,
        degrade_reason=reason,
    )


def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except (TypeError, ValueError):
        pass
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def _payload_to_df(payload: Any) -> pd.DataFrame | None:
    if payload is None:
        return None
    if isinstance(payload, pd.DataFrame):
        return payload
    if isinstance(payload, dict) and "data" in payload:
        from .yf_client import payload_to_df

        df = payload_to_df(payload)
        return df if not df.empty else None
    return None


# ---------- individual scorers --------------------------------------------


def compute_short_float(info: dict) -> _Out:
    sf = _safe_float(info.get("shortPercentOfFloat"))
    sr = _safe_float(info.get("shortRatio"))
    if sf is None and sr is None:
        return _missing("Short Float")
    pieces = []
    score = 0
    if sf is not None:
        # yfinance returns fraction (0.05 = 5%)
        if sf < 0.05:
            score += 1
            pieces.append(f"Short Float {sf*100:.1f}% < 5%")
        elif sf > 0.20:
            score -= 1
            pieces.append(f"Short Float {sf*100:.1f}% > 20%")
        else:
            pieces.append(f"Short Float {sf*100:.1f}% 中性")
    if sr is not None:
        if sr < 3:
            score += 1
            pieces.append(f"Short Ratio {sr:.1f} < 3 天")
        elif sr > 10:
            score -= 1
            pieces.append(f"Short Ratio {sr:.1f} > 10 天")
        else:
            pieces.append(f"Short Ratio {sr:.1f} 中性")
    final = max(-1, min(1, score))
    return _Out(score=final, reasoning="；".join(pieces), raw_value=sf if sf is not None else sr)


def compute_insider_tx(insider: Any) -> _Out:
    df = _payload_to_df(insider)
    if df is None or df.empty:
        return _missing("Insider Transactions")
    cols = {c.lower(): c for c in df.columns}
    val_col = cols.get("value") or cols.get("transactionvalue") or cols.get("net amount")
    type_col = cols.get("transaction") or cols.get("transactiontype")
    if val_col is None or type_col is None:
        return _Out(
            score=0,
            reasoning=f"近期内部交易 {len(df)} 条，结构不可解析，按中性处理",
            is_degraded=True,
            degrade_reason="缺少 transaction/value 字段",
            raw_value=str(len(df)),
        )

    try:
        net = 0.0
        for _, row in df.iterrows():
            v = _safe_float(row[val_col]) or 0.0
            kind = str(row[type_col]).lower()
            if "buy" in kind or "purchase" in kind:
                net += v
            elif "sell" in kind or "sale" in kind:
                net -= v
    except Exception as exc:
        return _Out(
            score=0,
            reasoning="内部交易聚合失败，按中性处理",
            is_degraded=True,
            degrade_reason=str(exc),
        )

    if net > 0:
        return _Out(score=1, reasoning=f"内部人净买入 ${net:,.0f}", raw_value=net)
    if net < 0:
        return _Out(score=-1, reasoning=f"内部人净卖出 ${-net:,.0f}", raw_value=net)
    return _Out(score=0, reasoning="内部人买卖基本平衡", raw_value=0.0)


def compute_institutional_hold(info: dict, holders: Any) -> _Out:
    pct = _safe_float(info.get("heldPercentInstitutions"))
    if pct is None:
        return _missing("Institutional Holdings")
    if pct > 0.70:
        return _Out(score=1, reasoning=f"机构持股 {pct*100:.1f}% > 70%", raw_value=pct)
    if pct < 0.30:
        return _Out(score=-1, reasoning=f"机构持股仅 {pct*100:.1f}% < 30%", raw_value=pct)
    return _Out(score=0, reasoning=f"机构持股 {pct*100:.1f}% 处于中性区间", raw_value=pct)


def compute_analyst_rating(info: dict, upgrades: Any) -> _Out:
    mean = _safe_float(info.get("recommendationMean"))
    if mean is None:
        df = _payload_to_df(upgrades)
        if df is None or df.empty:
            return _missing("Analyst Rating")
        return _Out(
            score=0,
            reasoning=f"近期评级动作 {len(df)} 条，但平均评级缺失",
            is_degraded=True,
            degrade_reason="recommendationMean 缺失",
            raw_value=str(len(df)),
        )
    if mean <= 2.0:
        return _Out(score=1, reasoning=f"分析师平均评级 {mean:.2f} (Buy 倾向)", raw_value=mean)
    if mean >= 3.5:
        return _Out(score=-1, reasoning=f"分析师平均评级 {mean:.2f} (Hold/Sell 倾向)", raw_value=mean)
    return _Out(score=0, reasoning=f"分析师平均评级 {mean:.2f} 中性", raw_value=mean)


def compute_options_pcr(option_chain: Any) -> _Out:
    if not option_chain or not isinstance(option_chain, dict):
        return _missing("Options P/C Ratio", "期权链不可用")
    calls_payload = option_chain.get("calls")
    puts_payload = option_chain.get("puts")
    calls_df = _payload_to_df(calls_payload)
    puts_df = _payload_to_df(puts_payload)
    if calls_df is None or puts_df is None or calls_df.empty or puts_df.empty:
        return _missing("Options P/C Ratio", "期权链空")

    call_vol = _safe_float(calls_df.get("volume", pd.Series([0])).sum()) or 0.0
    put_vol = _safe_float(puts_df.get("volume", pd.Series([0])).sum()) or 0.0
    if call_vol <= 0:
        return _Out(
            score=0,
            reasoning="Calls 成交量为 0，无法计算 P/C",
            is_degraded=True,
            degrade_reason="call_volume=0",
        )
    pcr = put_vol / call_vol
    if pcr < 0.7:
        return _Out(score=1, reasoning=f"P/C={pcr:.2f} 偏乐观", raw_value=pcr)
    if pcr > 1.2:
        return _Out(score=-1, reasoning=f"P/C={pcr:.2f} 偏悲观", raw_value=pcr)
    return _Out(score=0, reasoning=f"P/C={pcr:.2f} 中性", raw_value=pcr)


def compute_buyback(info: dict) -> _Out:
    shares_change = _safe_float(info.get("sharesPercentSharesOut"))
    repurchase = _safe_float(info.get("netSharesRepurchased"))
    if repurchase is not None:
        if repurchase > 0:
            return _Out(score=1, reasoning=f"回购 {repurchase:,.0f} 股", raw_value=repurchase)
        if repurchase < 0:
            return _Out(score=-1, reasoning=f"净增发 {-repurchase:,.0f} 股", raw_value=repurchase)
    # Fallback: buyback yield via marketCap proxy
    market_cap = _safe_float(info.get("marketCap"))
    if shares_change is not None and market_cap:
        if shares_change < 0:
            return _Out(
                score=1,
                reasoning=f"流通股变化 {shares_change*100:.2f}%，呈现回购迹象",
                raw_value=shares_change,
            )
    return _missing("Share Buyback", "yfinance 未提供回购口径")


def compute_dividend(info: dict) -> _Out:
    yield_pct = _safe_float(info.get("dividendYield"))
    cont_growth = _safe_float(info.get("fiveYearAvgDividendYield"))
    payout = _safe_float(info.get("payoutRatio"))
    if yield_pct is None and cont_growth is None:
        # Growth stocks may legitimately have no dividend; report neutral
        return _Out(score=0, reasoning="无股息（可能为成长股）", raw_value=0.0)
    if yield_pct is not None:
        # yfinance flip-flops between fraction (0.024 = 2.4%) and percent (2.4 = 2.4%).
        # Heuristic: dividend yields below 0.10 are unambiguously fraction (a 10%+
        # yield-as-fraction would be implausible); 0.10..1.0 and >1.0 are percent.
        if abs(yield_pct) < 0.10:
            normalized = yield_pct
        else:
            normalized = yield_pct / 100.0
        if 0.03 <= normalized <= 0.06 and (payout is None or payout < 0.8):
            return _Out(
                score=1,
                reasoning=f"股息率 {normalized*100:.2f}% 处于健康区间",
                raw_value=normalized,
            )
        if normalized < 0.005:
            return _Out(
                score=0,
                reasoning=f"股息率 {normalized*100:.2f}%（成长股偏好）",
                raw_value=normalized,
            )
        if payout is not None and payout > 0.9:
            return _Out(
                score=-1,
                reasoning=f"股息率 {normalized*100:.2f}%，但派息比 {payout*100:.0f}% 偏高",
                raw_value=normalized,
            )
        return _Out(
            score=0,
            reasoning=f"股息率 {normalized*100:.2f}%",
            raw_value=normalized,
        )
    return _missing("Dividend")


def compute_retail_social() -> _Out:
    """Reddit / Twitter 数据 yfinance 无法获取，按设计永远降级。"""
    return _Out(
        score=0,
        reasoning="社交媒体情绪数据源未接入 (placeholder)",
        is_degraded=True,
        degrade_reason="yfinance 不提供社交媒体数据",
        raw_value=None,
    )


def compute_52week_position(info: dict) -> _Out:
    high = _safe_float(info.get("fiftyTwoWeekHigh"))
    low = _safe_float(info.get("fiftyTwoWeekLow"))
    price = _safe_float(info.get("regularMarketPrice"))
    if None in (high, low, price) or high <= low:
        return _missing("52 周位置")
    pos = (price - low) / (high - low)
    if pos >= 0.7:
        return _Out(score=1, reasoning=f"位于 52 周高位区 ({pos*100:.0f}%)", raw_value=pos)
    if pos <= 0.2:
        return _Out(score=-1, reasoning=f"位于 52 周低位区 ({pos*100:.0f}%)", raw_value=pos)
    return _Out(score=0, reasoning=f"位于 52 周中性区域 ({pos*100:.0f}%)", raw_value=pos)


def compute_earnings_beat(earnings_dates: Any) -> _Out:
    df = _payload_to_df(earnings_dates)
    if df is None or df.empty:
        return _missing("Earnings Beat")
    cols = {c.lower(): c for c in df.columns}
    surprise_col = (
        cols.get("surprise(%)")
        or cols.get("surprise %")
        or cols.get("surprise")
    )
    if surprise_col is None:
        return _Out(
            score=0,
            reasoning=f"近 {len(df)} 次财报记录，但缺少 surprise 字段",
            is_degraded=True,
            degrade_reason="surprise 列缺失",
            raw_value=str(len(df)),
        )

    surprises = (
        df[surprise_col]
        .apply(lambda x: _safe_float(x))
        .dropna()
        .head(4)
    )
    if surprises.empty:
        return _missing("Earnings Beat", "surprise 全部为空")

    beats = int((surprises > 0).sum())
    misses = int((surprises < 0).sum())
    avg = float(surprises.mean())
    if beats >= 3 and avg > 5:
        return _Out(
            score=1,
            reasoning=f"近 4 次中 {beats} 次 Beat，均值 {avg:.1f}%",
            raw_value=avg,
        )
    if misses >= 2:
        return _Out(
            score=-1,
            reasoning=f"近 4 次中 {misses} 次 Miss，均值 {avg:.1f}%",
            raw_value=avg,
        )
    return _Out(
        score=0,
        reasoning=f"近期财报表现混合 (Beat {beats}, Miss {misses})",
        raw_value=avg,
    )


# ---------- orchestrator ---------------------------------------------------


def _to_indicator_score(name: str, zh: str, en: str, outcome: _Out) -> IndicatorScore:
    raw = outcome.raw_value
    if raw is not None and not isinstance(raw, (float, int, str)):
        raw = str(raw)
    return IndicatorScore(
        name=name,
        display_name_zh=zh,
        display_name_en=en,
        raw_value=raw,
        score=outcome.score,  # type: ignore[arg-type]
        reasoning=outcome.reasoning[:400],
        is_degraded=outcome.is_degraded,
        degrade_reason=outcome.degrade_reason,
    )


def compute_all_sentimental(market_data: dict) -> list[IndicatorScore]:
    md = market_data or {}
    info = md.get("info") or {}
    holders = md.get("institutional_holders")
    insider = md.get("insider_transactions")
    upgrades = md.get("upgrades_downgrades")
    options = md.get("options_chain")
    earnings = md.get("earnings_dates")

    out: list[IndicatorScore] = []
    out.append(_to_indicator_score("SHORT_FLOAT", "做空比例", "Short Float", compute_short_float(info)))
    out.append(_to_indicator_score("INSIDER_TX", "内部人交易", "Insider Transactions", compute_insider_tx(insider)))
    out.append(_to_indicator_score("INSTITUTIONAL_HOLD", "机构持股", "Institutional Holdings", compute_institutional_hold(info, holders)))
    out.append(_to_indicator_score("ANALYST_RATING", "分析师评级", "Analyst Rating", compute_analyst_rating(info, upgrades)))
    out.append(_to_indicator_score("OPTIONS_PCR", "期权 P/C", "Options P/C Ratio", compute_options_pcr(options)))
    out.append(_to_indicator_score("BUYBACK", "股票回购", "Share Buyback", compute_buyback(info)))
    out.append(_to_indicator_score("DIVIDEND", "股息", "Dividend", compute_dividend(info)))
    out.append(_to_indicator_score("RETAIL_SOCIAL", "散户/社交媒体", "Retail/Social", compute_retail_social()))
    out.append(_to_indicator_score("WEEK52_POSITION", "52 周位置", "52-Week Position", compute_52week_position(info)))
    out.append(_to_indicator_score("EARNINGS_BEAT", "财报超预期", "Earnings Beat", compute_earnings_beat(earnings)))

    if len(out) != 10:
        raise RuntimeError(f"sentimental indicator count must be 10, got {len(out)}")
    return out
