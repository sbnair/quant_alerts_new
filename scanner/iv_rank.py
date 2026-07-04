"""
V2 real IV Rank, computed from the IVHistory table that V1 has been
accumulating on every scan.

    iv_rank        = (current − min) / (max − min) × 100  over the lookback
    iv_percentile  = % of history days with IV below current

Both need IV_RANK_MIN_OBS distinct days of history before they're trusted;
until then they stay None and the IV_RANK_EXTREME flag falls back to the
V1 IV/HV-ratio heuristic.
"""

from datetime import date, timedelta

import config
from models import IVHistory


def iv_ranks(market_type: str, current_ivs: dict[str, float]) -> dict[str, dict]:
    """{symbol: {iv_rank, iv_percentile}} for symbols with enough history.
    current_ivs: today's ATM IV per symbol (%); must run inside app context."""
    if not current_ivs:
        return {}
    cutoff = date.today() - timedelta(days=config.IV_RANK_LOOKBACK_DAYS)
    rows = (IVHistory.query
            .filter(IVHistory.market_type == market_type,
                    IVHistory.symbol.in_(list(current_ivs)),
                    IVHistory.date >= cutoff,
                    IVHistory.date < date.today())   # today's row is the input, not history
            .all())
    by_sym: dict[str, list[float]] = {}
    for r in rows:
        by_sym.setdefault(r.symbol, []).append(r.atm_iv)

    out = {}
    for sym, iv in current_ivs.items():
        hist = by_sym.get(sym, [])
        if iv is None or len(hist) < config.IV_RANK_MIN_OBS:
            continue
        lo, hi = min(hist), max(hist)
        rank = None if hi <= lo else max(0.0, min(100.0, (iv - lo) / (hi - lo) * 100))
        pctl = sum(1 for h in hist if h < iv) / len(hist) * 100
        out[sym] = {"iv_rank": None if rank is None else round(rank, 1),
                    "iv_percentile": round(pctl, 1)}
    return out
