"""NAV time-series helpers: rolling series + max drawdown."""

from __future__ import annotations

from ..models.portfolio import NavSnapshot


def compute_max_drawdown(nav_series: list[NavSnapshot]) -> float:
    """Peak-to-trough drawdown as a positive percent (e.g. 0.087 = 8.7%).

    Returns ``0.0`` for empty / single-point series; this is the documented
    "first run" behaviour and lets R2 pass automatically.
    """
    if not nav_series or len(nav_series) < 2:
        return 0.0

    peak = nav_series[0].nav
    max_dd = 0.0
    for snap in nav_series:
        if snap.nav > peak:
            peak = snap.nav
        if peak <= 0:
            continue
        dd = (peak - snap.nav) / peak
        if dd > max_dd:
            max_dd = dd
    return float(max_dd)


def rolling_nav_values(snaps: list[NavSnapshot]) -> list[float]:
    """Convenience: pull ``nav`` floats out of a NavSnapshot list."""
    return [s.nav for s in snaps]
