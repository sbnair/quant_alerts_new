"""
India F&O per-symbol feature extraction for the volatility scanner.

One Dhan option-chain request (nearest expiry) per symbol gives us:
spot, straddle-implied move, ATM IV, PCR, total OI + overnight OI change
(chain legs carry previous_oi), and the gamma regime via gamma_lite.
Daily OHLCV history comes from a single batched yfinance download.
"""

import math
from datetime import datetime, date

import numpy as np
import pandas as pd
import yfinance as yf

import config
from data import dhan_client
from gamma_lite import gamma_regime

YF_INDEX_MAP = {
    "NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK",
    "FINNIFTY": "NIFTY_FIN_SERVICE.NS", "SENSEX": "^BSESN",
}


def yf_ticker(symbol: str) -> str:
    return YF_INDEX_MAP.get(symbol.upper(), f"{symbol.upper()}.NS")


def batch_history(symbols: list[str], period: str = "1y") -> dict[str, pd.DataFrame]:
    """One yfinance download for the whole universe → {symbol: OHLCV df}."""
    tickers = [yf_ticker(s) for s in symbols]
    data = yf.download(tickers, period=period, auto_adjust=False,
                       group_by="ticker", progress=False, threads=True)
    out = {}
    for sym, tk in zip(symbols, tickers):
        try:
            df = data[tk].dropna(how="all") if len(tickers) > 1 else data.dropna(how="all")
            if not df.empty:
                out[sym] = df
        except KeyError:
            pass
    return out


def price_features(hist: pd.DataFrame) -> dict:
    """HV / ATR / 52-week features from a daily OHLCV frame.
    Today's partial bar (present during market hours) is dropped so
    prev_close is genuinely the previous session's close."""
    if not hist.empty:
        last_day = pd.Timestamp(hist.index[-1]).date()
        if last_day >= date.today():
            hist = hist.iloc[:-1]
    c = hist["Close"].dropna()
    if len(c) < 25:
        return {}
    ret = np.log(c / c.shift(1)).dropna()
    hv5  = float(ret.tail(5).std()  * math.sqrt(252) * 100)
    hv20 = float(ret.tail(20).std() * math.sqrt(252) * 100)

    h, l, pc = hist["High"], hist["Low"], hist["Close"].shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr14 = float(tr.rolling(14).mean().dropna().iloc[-1])
    close = float(c.iloc[-1])

    hi52, lo52 = float(c.max()), float(c.min())
    dist_hi = (hi52 - close) / close * 100
    dist_lo = (close - lo52) / close * 100

    return {
        "prev_close": close,          # last completed session close
        "hv_short": hv5, "hv_long": hv20,
        "hv_ratio": hv5 / hv20 if hv20 > 0 else None,
        "atr_pct": atr14 / close * 100 if close > 0 else None,
        "dist_52w_pct": min(dist_hi, dist_lo),
    }


def _chain_features(sec_id: int, segment: str, lot_size: int) -> dict:
    """Fetch nearest-expiry chain from Dhan and derive option features."""
    expiries = dhan_client.expiry_list(sec_id, segment)
    if not expiries:
        raise ValueError("no expiries")
    expiry = expiries[0]
    chain = dhan_client.option_chain(sec_id, segment, expiry)

    spot = float(chain.get("last_price", 0))
    oc = chain.get("oc", {})
    if spot <= 0 or not oc:
        raise ValueError("empty chain")

    rows, ce_oi, pe_oi, prev_oi = [], 0.0, 0.0, 0.0
    strikes = {}
    for strike_str, legs in oc.items():
        try:
            strike = float(strike_str)
        except ValueError:
            continue
        strikes[strike] = legs
        for key, side in (("ce", "call"), ("pe", "put")):
            leg = legs.get(key) or {}
            oi = float(leg.get("oi", 0))
            iv = float(leg.get("implied_volatility", 0)) / 100.0
            if side == "call":
                ce_oi += oi
            else:
                pe_oi += oi
            prev_oi += float(leg.get("previous_oi", 0))
            if oi >= config.MIN_OPEN_INTEREST and iv > 0:
                rows.append({"strike": strike, "type": side, "openInterest": oi,
                             "impliedVolatility": iv, "expiry": expiry})

    atm = min(strikes, key=lambda k: abs(k - spot))
    ce, pe = strikes[atm].get("ce") or {}, strikes[atm].get("pe") or {}
    straddle = float(ce.get("last_price", 0)) + float(pe.get("last_price", 0))
    ivs = [v for v in (float(ce.get("implied_volatility", 0)),
                       float(pe.get("implied_volatility", 0))) if v > 0]
    atm_iv = sum(ivs) / len(ivs) if ivs else None

    # Straddle premium ≈ expected move to expiry; scale to 1-day via sqrt(T)
    days = max((pd.Timestamp(expiry) - pd.Timestamp(date.today())).days, 1)
    implied_move = (straddle / spot * 100 / math.sqrt(days)) if straddle > 0 else None

    total_oi = ce_oi + pe_oi
    oi_change = ((total_oi - prev_oi) / prev_oi * 100) if prev_oi > 0 else None

    regime, flip = (None, float("nan"))
    df = pd.DataFrame(rows)
    if not df.empty:
        try:
            regime, flip = gamma_regime(df, spot, contract_size=lot_size)
        except Exception:
            pass

    return {
        "spot": spot, "expiry": expiry, "implied_move": implied_move,
        "atm_iv": atm_iv, "pcr": pe_oi / ce_oi if ce_oi > 0 else None,
        "oi_total": total_oi, "oi_change_pct": oi_change,
        "regime": regime,
        "gamma_flip": None if (flip is None or np.isnan(flip)) else float(flip),
    }


def fetch_symbol(symbol: str, security_id: str, segment: str, lot_size: int,
                 hist: pd.DataFrame | None) -> dict:
    """Full feature dict for one India symbol. Raises on hard failure."""
    sec_id = int(security_id)
    try:
        feats = _chain_features(sec_id, segment, lot_size)
    except Exception:
        # Some underlyings answer on the equity segment instead of NSE_FNO
        alt = "NSE_EQ" if segment == "NSE_FNO" else "NSE_FNO"
        feats = _chain_features(sec_id, alt, lot_size)

    if hist is not None and not hist.empty:
        pf = price_features(hist)
        feats.update(pf)
        prev = pf.get("prev_close")
        if prev and feats["spot"] > 0:
            gap = (feats["spot"] - prev) / prev * 100
            # yfinance can lag; a >20% "gap" is almost surely stale data
            feats["gap_pct"] = gap if abs(gap) < 20 else None
        if feats.get("atm_iv") and pf.get("hv_long"):
            feats["iv_hv_ratio"] = feats["atm_iv"] / pf["hv_long"]

    feats["symbol"] = symbol
    return feats
