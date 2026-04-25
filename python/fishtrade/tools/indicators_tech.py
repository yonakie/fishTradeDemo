"""Technical indicators — pure functions over OHLCV DataFrames.

Computation uses ``pandas-ta`` where available and falls back to manual
implementations for the small number of helpers it doesn't ship.

Each scorer is independent and degrade-safe: short / empty histories yield
a degraded :class:`fishtrade.models.research.IndicatorScore` rather than
raising.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..models.research import IndicatorScore

try:
    import pandas_ta as ta  # type: ignore
    _HAS_TA = True
except Exception:  # pragma: no cover
    ta = None  # type: ignore
    _HAS_TA = False


@dataclass(frozen=True)
class _Out:
    score: int
    reasoning: str
    raw_value: float | str | None = None
    is_degraded: bool = False
    degrade_reason: str | None = None


def _missing(zh_name: str, reason: str = "历史数据不足") -> _Out:
    return _Out(
        score=0,
        reasoning=f"{zh_name} 数据不足，按中性处理",
        is_degraded=True,
        degrade_reason=reason,
    )


def _ensure_close(history: pd.DataFrame | None) -> pd.Series | None:
    if history is None or not isinstance(history, pd.DataFrame) or history.empty:
        return None
    if "Close" not in history.columns:
        return None
    closes = pd.to_numeric(history["Close"], errors="coerce").dropna()
    return closes if not closes.empty else None


def _safe_last(series: pd.Series | None) -> float | None:
    if series is None or series.empty:
        return None
    val = series.iloc[-1]
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    return float(val)


# ---------- raw computers (exposed for tests / future agents) -------------


def compute_macd(history: pd.DataFrame) -> dict | None:
    closes = _ensure_close(history)
    if closes is None or len(closes) < 35:
        return None
    if _HAS_TA:
        df = ta.macd(closes)
        if df is None or df.empty:
            return None
        macd = df.iloc[:, 0]
        signal = df.iloc[:, 2]
        hist = df.iloc[:, 1]
    else:  # pragma: no cover
        ema12 = closes.ewm(span=12, adjust=False).mean()
        ema26 = closes.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal

    if macd.empty or signal.empty:
        return None
    last_macd = _safe_last(macd)
    last_sig = _safe_last(signal)
    last_hist = _safe_last(hist)
    if None in (last_macd, last_sig, last_hist):
        return None
    prev_macd = float(macd.iloc[-2]) if len(macd) >= 2 else last_macd
    prev_sig = float(signal.iloc[-2]) if len(signal) >= 2 else last_sig
    golden_cross = prev_macd <= prev_sig and last_macd > last_sig
    death_cross = prev_macd >= prev_sig and last_macd < last_sig
    return {
        "macd": last_macd,
        "signal": last_sig,
        "hist": last_hist,
        "golden_cross": golden_cross,
        "death_cross": death_cross,
    }


def compute_rsi(history: pd.DataFrame, n: int = 14) -> float | None:
    closes = _ensure_close(history)
    if closes is None or len(closes) < n + 1:
        return None
    if _HAS_TA:
        rsi = ta.rsi(closes, length=n)
    else:  # pragma: no cover
        delta = closes.diff()
        up = delta.clip(lower=0).rolling(n).mean()
        down = (-delta.clip(upper=0)).rolling(n).mean()
        rs = up / down.replace(0, pd.NA)
        rsi = 100 - 100 / (1 + rs)
    return _safe_last(rsi)


def compute_moving_averages(history: pd.DataFrame) -> dict | None:
    closes = _ensure_close(history)
    if closes is None or len(closes) < 50:
        return None
    out: dict[str, Any] = {"price": float(closes.iloc[-1])}
    for window in (20, 50, 200):
        if len(closes) >= window:
            out[f"sma{window}"] = float(closes.rolling(window).mean().iloc[-1])
        else:
            out[f"sma{window}"] = None
    return out


def compute_bollinger(history: pd.DataFrame, n: int = 20, k: float = 2.0) -> dict | None:
    closes = _ensure_close(history)
    if closes is None or len(closes) < n:
        return None
    sma = closes.rolling(n).mean()
    std = closes.rolling(n).std(ddof=0)
    upper = sma + k * std
    lower = sma - k * std
    last_close = _safe_last(closes)
    last_sma = _safe_last(sma)
    last_up = _safe_last(upper)
    last_lo = _safe_last(lower)
    if None in (last_close, last_sma, last_up, last_lo):
        return None
    width = (last_up - last_lo) / last_sma if last_sma else 0.0
    return {
        "upper": last_up,
        "middle": last_sma,
        "lower": last_lo,
        "width": width,
        "price": last_close,
    }


def compute_volume_profile(history: pd.DataFrame) -> dict | None:
    if history is None or "Volume" not in history.columns or len(history) < 20:
        return None
    vol = pd.to_numeric(history["Volume"], errors="coerce").dropna()
    closes = _ensure_close(history)
    if closes is None or vol.empty:
        return None
    avg_20 = float(vol.tail(20).mean())
    last_vol = float(vol.iloc[-1])
    last_ret = float(closes.pct_change().iloc[-1]) if len(closes) >= 2 else 0.0
    rel = last_vol / avg_20 if avg_20 > 0 else 0.0
    return {"avg_20": avg_20, "last_volume": last_vol, "rel_volume": rel, "last_return": last_ret}


def compute_atr(history: pd.DataFrame, n: int = 14) -> float | None:
    if history is None or len(history) < n + 1:
        return None
    cols = {c.lower() for c in history.columns}
    if not {"high", "low", "close"}.issubset(cols):
        return None
    high = pd.to_numeric(history["High"], errors="coerce")
    low = pd.to_numeric(history["Low"], errors="coerce")
    close = pd.to_numeric(history["Close"], errors="coerce")
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(n).mean()
    return _safe_last(atr)


def compute_fibonacci(history: pd.DataFrame, lookback: int = 60) -> dict | None:
    closes = _ensure_close(history)
    if closes is None or len(closes) < lookback:
        return None
    window = closes.tail(lookback)
    high = float(window.max())
    low = float(window.min())
    last = float(window.iloc[-1])
    rng = max(high - low, 1e-9)
    levels = {
        "0.236": high - 0.236 * rng,
        "0.382": high - 0.382 * rng,
        "0.500": high - 0.500 * rng,
        "0.618": high - 0.618 * rng,
    }
    pos = (last - low) / rng
    return {"high": high, "low": low, "last": last, "levels": levels, "position_pct": pos}


def compute_relative_strength(
    history: pd.DataFrame, benchmark_history: pd.DataFrame | None
) -> float | None:
    closes = _ensure_close(history)
    bench = _ensure_close(benchmark_history) if benchmark_history is not None else None
    if closes is None or bench is None or len(closes) < 21 or len(bench) < 21:
        return None
    stock_ret = closes.iloc[-1] / closes.iloc[-21] - 1.0
    bench_ret = bench.iloc[-1] / bench.iloc[-21] - 1.0
    return float(stock_ret - bench_ret)


def detect_price_pattern(history: pd.DataFrame) -> str | None:
    closes = _ensure_close(history)
    if closes is None or len(closes) < 40:
        return None
    last = closes.tail(40)
    high1 = float(last.iloc[:20].max())
    high2 = float(last.iloc[20:].max())
    low1 = float(last.iloc[:20].min())
    low2 = float(last.iloc[20:].min())
    if abs(high1 - high2) / max(high1, 1e-9) < 0.03 and last.iloc[-1] < min(high1, high2) * 0.97:
        return "double_top"
    if abs(low1 - low2) / max(low1, 1e-9) < 0.03 and last.iloc[-1] > max(low1, low2) * 1.03:
        return "double_bottom"
    return None


def compute_support_resistance(history: pd.DataFrame, lookback: int = 60) -> dict | None:
    closes = _ensure_close(history)
    if closes is None or len(closes) < lookback:
        return None
    window = closes.tail(lookback)
    last = float(window.iloc[-1])
    support = float(window.min())
    resistance = float(window.max())
    if last <= 0:
        return None
    return {
        "last": last,
        "support": support,
        "resistance": resistance,
        "support_dist_pct": (last - support) / last,
        "resistance_dist_pct": (resistance - last) / last,
    }


# ---------- scorers --------------------------------------------------------


def _score_macd(history: pd.DataFrame) -> _Out:
    m = compute_macd(history)
    if m is None:
        return _missing("MACD")
    if m["golden_cross"] or m["hist"] > 0:
        return _Out(
            score=1,
            reasoning=(
                f"MACD={m['macd']:.3f} > Signal={m['signal']:.3f}，"
                f"hist={m['hist']:.3f}，多头信号"
            ),
            raw_value=m["macd"],
        )
    if m["death_cross"] or m["hist"] < 0:
        return _Out(
            score=-1,
            reasoning=(
                f"MACD={m['macd']:.3f} < Signal={m['signal']:.3f}，"
                f"hist={m['hist']:.3f}，空头信号"
            ),
            raw_value=m["macd"],
        )
    return _Out(score=0, reasoning="MACD 在零轴附近震荡", raw_value=m["macd"])


def _score_rsi(history: pd.DataFrame) -> _Out:
    rsi = compute_rsi(history)
    if rsi is None:
        return _missing("RSI 14")
    if rsi < 30:
        return _Out(score=1, reasoning=f"RSI {rsi:.1f} 处于超卖区间", raw_value=rsi)
    if rsi > 70:
        return _Out(score=-1, reasoning=f"RSI {rsi:.1f} 处于超买区间", raw_value=rsi)
    return _Out(score=0, reasoning=f"RSI {rsi:.1f} 处于中性区间", raw_value=rsi)


def _score_moving_averages(history: pd.DataFrame) -> _Out:
    ma = compute_moving_averages(history)
    if ma is None:
        return _missing("均线系统")
    price = ma["price"]
    s20, s50, s200 = ma.get("sma20"), ma.get("sma50"), ma.get("sma200")
    if None in (s20, s50, s200):
        return _Out(
            score=0,
            reasoning=f"价格 {price:.2f}，均线数据不全",
            raw_value=price,
            is_degraded=True,
            degrade_reason="缺少 SMA200 数据",
        )
    if price > s20 > s50 > s200:
        return _Out(score=1, reasoning="价格 > SMA20 > SMA50 > SMA200，多头排列", raw_value=price)
    if price < s20 < s50 < s200:
        return _Out(score=-1, reasoning="均线呈空头排列", raw_value=price)
    return _Out(score=0, reasoning="均线交织，趋势不明", raw_value=price)


def _score_bollinger(history: pd.DataFrame) -> _Out:
    b = compute_bollinger(history)
    if b is None:
        return _missing("布林带")
    price = b["price"]
    if price <= b["lower"]:
        return _Out(score=1, reasoning=f"价格 {price:.2f} 触及下轨，反弹概率上升", raw_value=price)
    if price >= b["upper"]:
        return _Out(score=-1, reasoning=f"价格 {price:.2f} 触及上轨，回调风险", raw_value=price)
    return _Out(score=0, reasoning=f"价格 {price:.2f} 在布林带中部", raw_value=price)


def _score_volume(history: pd.DataFrame) -> _Out:
    v = compute_volume_profile(history)
    if v is None:
        return _missing("成交量")
    rel = v["rel_volume"]
    last_ret = v["last_return"]
    if last_ret > 0 and rel > 1.5:
        return _Out(score=1, reasoning=f"上涨放量，rel_vol={rel:.2f}", raw_value=rel)
    if last_ret < 0 and rel > 1.5:
        return _Out(score=-1, reasoning=f"下跌放量，rel_vol={rel:.2f}，资金出逃", raw_value=rel)
    return _Out(score=0, reasoning=f"量价配合一般，rel_vol={rel:.2f}", raw_value=rel)


def _score_atr(history: pd.DataFrame) -> _Out:
    atr = compute_atr(history)
    if atr is None:
        return _missing("ATR 14")
    closes = _ensure_close(history)
    last = float(closes.iloc[-1]) if closes is not None else None
    if last is None or last <= 0:
        return _missing("ATR 14", "价格无效")
    rel = atr / last
    if rel < 0.02:
        return _Out(score=1, reasoning=f"ATR/Price={rel*100:.2f}%，波动可控", raw_value=atr)
    if rel < 0.04:
        return _Out(score=0, reasoning=f"ATR/Price={rel*100:.2f}%，中性", raw_value=atr)
    return _Out(score=-1, reasoning=f"ATR/Price={rel*100:.2f}%，高波动", raw_value=atr)


def _score_fibonacci(history: pd.DataFrame) -> _Out:
    f = compute_fibonacci(history)
    if f is None:
        return _missing("斐波那契")
    pos = f["position_pct"]
    if 0.55 <= pos <= 0.75:
        return _Out(score=1, reasoning=f"价格位于回调 38.2% 附近 (pos={pos:.2f})", raw_value=pos)
    if pos < 0.382:
        return _Out(score=-1, reasoning=f"价格跌破 61.8% 回撤位 (pos={pos:.2f})", raw_value=pos)
    return _Out(score=0, reasoning=f"价格位于斐波那契中性区域 (pos={pos:.2f})", raw_value=pos)


def _score_relative_strength(market_data: dict) -> _Out:
    history = market_data.get("history")
    bench = market_data.get("benchmark_history")
    if isinstance(history, dict):
        from .yf_client import payload_to_df

        history = payload_to_df(history)
    if isinstance(bench, dict):
        from .yf_client import payload_to_df

        bench = payload_to_df(bench)
    rs = compute_relative_strength(history, bench)
    if rs is None:
        return _missing("相对强弱", "缺少基准 SPY 数据")
    if rs > 0.05:
        return _Out(score=1, reasoning=f"近月跑赢大盘 {rs*100:.1f}%", raw_value=rs)
    if rs < -0.05:
        return _Out(score=-1, reasoning=f"近月跑输大盘 {rs*100:.1f}%", raw_value=rs)
    return _Out(score=0, reasoning=f"与大盘表现接近 (rs={rs*100:.1f}%)", raw_value=rs)


def _score_price_pattern(history: pd.DataFrame) -> _Out:
    p = detect_price_pattern(history)
    if p is None and (history is None or len(history) < 40):
        return _missing("价格形态")
    if p == "double_bottom":
        return _Out(score=1, reasoning="识别到双底形态", raw_value=p)
    if p == "double_top":
        return _Out(score=-1, reasoning="识别到双顶形态", raw_value=p)
    return _Out(score=0, reasoning="未识别到反转形态", raw_value="none")


def _score_support_resistance(history: pd.DataFrame) -> _Out:
    sr = compute_support_resistance(history)
    if sr is None:
        return _missing("支撑/压力位")
    sd = sr["support_dist_pct"]
    rd = sr["resistance_dist_pct"]
    if sd >= 0 and rd > 2 * abs(sd) and rd > 0.05:
        return _Out(score=1, reasoning=f"距支撑近 ({sd*100:.1f}%)，距压力远 ({rd*100:.1f}%)", raw_value=sd)
    if rd < 0.02:
        return _Out(score=-1, reasoning=f"逼近压力位，仅剩 {rd*100:.1f}% 空间", raw_value=rd)
    return _Out(score=0, reasoning=f"位置中性 (S {sd*100:.1f}%, R {rd*100:.1f}%)", raw_value=sd)


# ---------- orchestrator ---------------------------------------------------


_SCORERS: list[tuple[str, str, str, Any]] = [
    ("MACD", "MACD", "MACD", _score_macd),
    ("RSI_14", "RSI 14", "RSI 14", _score_rsi),
    ("MOVING_AVERAGES", "均线系统", "Moving Averages", _score_moving_averages),
    ("BOLLINGER", "布林带", "Bollinger Bands", _score_bollinger),
    ("VOLUME_TREND", "成交量趋势", "Volume Trend", _score_volume),
    ("ATR_14", "ATR 14", "ATR 14", _score_atr),
    ("FIBONACCI", "斐波那契回撤", "Fibonacci", _score_fibonacci),
    ("RELATIVE_STRENGTH", "相对强弱", "Relative Strength", "rs"),
    ("PRICE_PATTERN", "价格形态", "Price Pattern", _score_price_pattern),
    ("SUPPORT_RESISTANCE", "支撑压力位", "Support/Resistance", _score_support_resistance),
]


def compute_all_technical(market_data: dict) -> list[IndicatorScore]:
    history = (market_data or {}).get("history")
    if isinstance(history, dict):
        from .yf_client import payload_to_df

        history = payload_to_df(history)

    out: list[IndicatorScore] = []
    for name, zh, en, fn in _SCORERS:
        try:
            if fn == "rs":
                outcome = _score_relative_strength(market_data or {})
            else:
                outcome = fn(history)
        except Exception as exc:  # defensive
            outcome = _Out(
                score=0,
                reasoning=f"{en} 计算异常，已降级",
                is_degraded=True,
                degrade_reason=f"{type(exc).__name__}: {exc}",
            )
        raw = outcome.raw_value
        if raw is not None and not isinstance(raw, (float, int, str)):
            raw = str(raw)
        out.append(
            IndicatorScore(
                name=name,
                display_name_zh=zh,
                display_name_en=en,
                raw_value=raw,
                score=outcome.score,  # type: ignore[arg-type]
                reasoning=outcome.reasoning[:400],
                is_degraded=outcome.is_degraded,
                degrade_reason=outcome.degrade_reason,
            )
        )

    if len(out) != 10:
        raise RuntimeError(f"technical indicator count must be 10, got {len(out)}")
    return out
