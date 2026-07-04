"""
V1.5 market-level context.

INDIA  — India VIX level + 1-day change, GIFT Nifty last vs NIFTY previous
         close → implied index gap %. GIFT Nifty has no reliable free feed,
         so the ticker is configurable and a failed fetch just omits the gap.
US     — VIX + ES futures implied gap vs SPX previous close.
CRYPTO — BTC 24h change + funding as the market's risk dial.

The context yields a score multiplier: each "hit" (VIX elevated, VIX spiking,
material index-futures gap) boosts every symbol's score by
CONTEXT_BOOST_PER_HIT so more names clear the alert gate on risk-heavy
mornings. Ranks are unaffected (uniform multiplier); the gate is the point.
"""

import pandas as pd
import requests
import yfinance as yf

import config
from models import MARKET_INDIA, MARKET_CRYPTO, MARKET_US


def _last_two_closes(ticker: str) -> tuple[float | None, float | None]:
    """(last close, previous close) from daily history; (None, None) on failure."""
    try:
        h = yf.Ticker(ticker).history(period="10d", interval="1d")
        c = h["Close"].dropna()
        if len(c) >= 2:
            return float(c.iloc[-1]), float(c.iloc[-2])
        if len(c) == 1:
            return float(c.iloc[-1]), None
    except Exception:
        pass
    return None, None


def _live_price(ticker: str) -> float | None:
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        p = (info.get("preMarketPrice") or info.get("postMarketPrice")
             or info.get("regularMarketPrice"))
        if p:
            return float(p)
    except Exception:
        pass
    try:
        fi = yf.Ticker(ticker).fast_info
        p = fi.get("last_price") if hasattr(fi, "get") else getattr(fi, "last_price", None)
        return float(p) if p else None
    except Exception:
        return None


def _vix_block(ticker: str) -> dict:
    last, prev = _last_two_closes(ticker)
    live = _live_price(ticker) or last
    out = {"vix": live, "vix_prev_close": prev}
    if live and prev:
        out["vix_change_pct"] = (live - prev) / prev * 100
    else:
        out["vix_change_pct"] = None
    return out


def _implied_gap(fut_ticker: str, cash_ticker: str) -> dict:
    """Index-futures last vs cash previous close → implied opening gap %."""
    fut = _live_price(fut_ticker)
    if fut is None:
        fut, _ = _last_two_closes(fut_ticker)
    _, cash_prev = _last_two_closes(cash_ticker)
    # when history returned only today's bar, fall back to the last close
    if cash_prev is None:
        cash_last, _ = _last_two_closes(cash_ticker)
        cash_prev = cash_last
    gap = None
    if fut and cash_prev:
        g = (fut - cash_prev) / cash_prev * 100
        if abs(g) < 10:            # sanity: discard stale-feed artifacts
            gap = g
    return {"futures_last": fut, "cash_prev_close": cash_prev,
            "implied_gap_pct": gap}


def _india_context() -> dict:
    ctx = {"market": MARKET_INDIA}
    ctx.update(_vix_block(config.INDIA_VIX_TICKER))
    ctx["vix_name"] = "India VIX"
    g = _implied_gap(config.GIFT_NIFTY_TICKER, "^NSEI")
    ctx["gift_nifty"] = g["futures_last"]
    ctx["nifty_prev_close"] = g["cash_prev_close"]
    ctx["implied_gap_pct"] = g["implied_gap_pct"]
    ctx["gap_source"] = "GIFT Nifty"
    return ctx


def _us_context() -> dict:
    ctx = {"market": MARKET_US}
    ctx.update(_vix_block(config.US_VIX_TICKER))
    ctx["vix_name"] = "VIX"
    g = _implied_gap(config.US_FUTURES_TICKER, "^GSPC")
    ctx["es_futures"] = g["futures_last"]
    ctx["spx_prev_close"] = g["cash_prev_close"]
    ctx["implied_gap_pct"] = g["implied_gap_pct"]
    ctx["gap_source"] = "ES futures"
    return ctx


def _crypto_context() -> dict:
    ctx = {"market": MARKET_CRYPTO, "vix": None, "vix_change_pct": None,
           "vix_name": None, "gap_source": "BTC 24h"}
    try:
        t = requests.get("https://fapi.binance.com/fapi/v1/ticker/24hr",
                         params={"symbol": "BTCUSDT"}, timeout=10).json()
        ctx["btc_price"] = float(t["lastPrice"])
        ctx["implied_gap_pct"] = float(t["priceChangePercent"])
    except Exception:
        ctx["btc_price"] = ctx["implied_gap_pct"] = None
    try:
        p = requests.get("https://fapi.binance.com/fapi/v1/premiumIndex",
                         params={"symbol": "BTCUSDT"}, timeout=10).json()
        ctx["btc_funding"] = float(p.get("lastFundingRate", 0))
    except Exception:
        ctx["btc_funding"] = None
    return ctx


def _multiplier(ctx: dict) -> tuple[float, list[str]]:
    """Uniform score multiplier + the human-readable reasons behind it."""
    hits = []
    vix, vchg = ctx.get("vix"), ctx.get("vix_change_pct")
    if vix is not None and vix >= config.VIX_HIGH_LEVEL:
        hits.append(f"{ctx.get('vix_name', 'VIX')} elevated ({vix:.1f})")
    if vchg is not None and vchg >= config.VIX_SPIKE_PCT:
        hits.append(f"{ctx.get('vix_name', 'VIX')} spiking ({vchg:+.1f}%)")
    gap = ctx.get("implied_gap_pct")
    if gap is not None and abs(gap) >= config.INDEX_GAP_ALERT_PCT \
            and ctx["market"] != MARKET_CRYPTO:
        hits.append(f"{ctx.get('gap_source')} implies {gap:+.2f}% open")
    if ctx["market"] == MARKET_CRYPTO:
        if gap is not None and abs(gap) >= 3.0:
            hits.append(f"BTC {gap:+.1f}% in 24h")
        f = ctx.get("btc_funding")
        if f is not None and abs(f) >= config.CRYPTO_FUNDING_ALERT:
            hits.append(f"BTC funding extreme ({f * 100:.3f}%)")
    mult = min(1.0 + config.CONTEXT_BOOST_PER_HIT * len(hits),
               config.CONTEXT_MULT_MAX)
    return mult, hits


_BUILDERS = {MARKET_INDIA: _india_context, MARKET_US: _us_context,
             MARKET_CRYPTO: _crypto_context}


def get_context(market_type: str) -> dict:
    """Market context dict incl. score_multiplier + context_notes.
    Never raises — a dead feed degrades to a neutral context."""
    if not config.ENABLE_MARKET_CONTEXT:
        return {"market": market_type, "score_multiplier": 1.0, "context_notes": []}
    try:
        ctx = _BUILDERS[market_type]()
    except Exception as e:
        ctx = {"market": market_type, "error": str(e)}
    mult, hits = _multiplier(ctx)
    ctx["score_multiplier"] = round(mult, 3)
    ctx["context_notes"] = hits
    # JSON-safe
    return {k: (None if isinstance(v, float) and pd.isna(v) else v)
            for k, v in ctx.items()}
