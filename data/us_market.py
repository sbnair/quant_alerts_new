"""
US per-symbol features via yfinance: nearest-expiry option chain gives the
straddle-implied move, ATM IV, PCR and gamma regime; daily history gives
HV / ATR / gap features.
"""

import math
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf

import config
from gamma_lite import gamma_regime
from data.india_market import price_features  # same OHLCV math


def batch_history(symbols: list[str], period: str = "1y") -> dict[str, pd.DataFrame]:
    data = yf.download(symbols, period=period, auto_adjust=False,
                       group_by="ticker", progress=False, threads=True)
    out = {}
    for sym in symbols:
        try:
            df = data[sym].dropna(how="all") if len(symbols) > 1 else data.dropna(how="all")
            if not df.empty:
                out[sym] = df
        except KeyError:
            pass
    return out


def fetch_symbol(symbol: str, hist: pd.DataFrame | None) -> dict:
    tk = yf.Ticker(symbol)

    expiries = list(tk.options)
    if not expiries:
        raise ValueError("no listed options")
    expiry = expiries[0]
    chain = tk.option_chain(expiry)
    calls, puts = chain.calls, chain.puts

    spot = 0.0
    try:
        # pre/post-market indicative price when the session is closed
        info = tk.info
        spot = float(info.get("preMarketPrice") or info.get("postMarketPrice")
                     or info.get("regularMarketPrice") or 0)
    except Exception:
        pass
    if spot <= 0:
        fi = getattr(tk, "fast_info", {})
        spot = float(fi.get("last_price") or 0) if fi else 0.0
    if spot <= 0 and hist is not None and not hist.empty:
        spot = float(hist["Close"].dropna().iloc[-1])
    if spot <= 0:
        raise ValueError("no spot price")

    atm_c = calls.iloc[(calls["strike"] - spot).abs().argsort()].iloc[0]
    atm_p = puts.iloc[(puts["strike"] - spot).abs().argsort()].iloc[0]
    straddle = float(atm_c.get("lastPrice", 0)) + float(atm_p.get("lastPrice", 0))

    # Yahoo's per-leg IVs are frequently degenerate (0DTE, pre-market), so the
    # primary ATM IV is the Brenner–Subrahmanyam value implied by the traded
    # ATM straddle price: sigma ≈ straddle / (0.8 · S · √T). Chain median is
    # only a fallback when there is no straddle print.
    atm_iv = None

    days = max((pd.Timestamp(expiry) - pd.Timestamp(date.today())).days, 1)
    implied_move = straddle / spot * 100 / math.sqrt(days) if straddle > 0 else None

    if straddle > 0:
        t_years = days / 252
        atm_iv = float(straddle / spot / (0.8 * math.sqrt(t_years)) * 100)
    else:
        band = pd.concat([calls, puts])
        band = band[(band["strike"] - spot).abs() / spot <= 0.02]
        ivs = pd.to_numeric(band["impliedVolatility"], errors="coerce").dropna()
        ivs = ivs[ivs > 0.05]
        if len(ivs) >= 4:
            atm_iv = float(ivs.median() * 100)

    ce_oi = float(calls["openInterest"].fillna(0).sum())
    pe_oi = float(puts["openInterest"].fillna(0).sum())

    rows = []
    for df_side, side in ((calls, "call"), (puts, "put")):
        d = df_side[["strike", "openInterest", "impliedVolatility"]].copy()
        d["type"] = side
        d["expiry"] = expiry
        d = d.rename(columns={"openInterest": "openInterest",
                              "impliedVolatility": "impliedVolatility"})
        rows.append(d)
    opt_df = pd.concat(rows, ignore_index=True).dropna(subset=["strike"])
    opt_df["openInterest"] = opt_df["openInterest"].fillna(0)
    opt_df["impliedVolatility"] = opt_df["impliedVolatility"].replace(0, np.nan).fillna(0.35)

    regime, flip = (None, float("nan"))
    try:
        regime, flip = gamma_regime(opt_df, spot, contract_size=100)
    except Exception:
        pass

    feats = {
        "symbol": symbol, "spot": spot, "expiry": expiry,
        "implied_move": implied_move, "atm_iv": atm_iv,
        "pcr": pe_oi / ce_oi if ce_oi > 0 else None,
        "oi_total": ce_oi + pe_oi,
        "regime": regime,
        "gamma_flip": None if (flip is None or np.isnan(flip)) else float(flip),
    }

    if hist is not None and not hist.empty:
        pf = price_features(hist)
        feats.update(pf)
        prev = pf.get("prev_close")
        if prev:
            gap = (spot - prev) / prev * 100
            feats["gap_pct"] = gap if abs(gap) < 25 else None
        if atm_iv and pf.get("hv_long"):
            feats["iv_hv_ratio"] = atm_iv / pf["hv_long"]
    return feats
