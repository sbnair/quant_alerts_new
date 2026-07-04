"""
Crypto per-symbol features from Binance USDT-perpetual futures (public FAPI).
For BTC/ETH/BNB a straddle-implied move is added from the options venue (EAPI).
"""

import math
from datetime import date

import numpy as np
import pandas as pd
import requests

import config

FAPI = "https://fapi.binance.com"
EAPI = "https://eapi.binance.com"
EAPI_ASSETS = {"BTC", "ETH", "BNB"}


def _get(base: str, path: str, params: dict | None = None):
    r = requests.get(f"{base}{path}", params=params or {}, timeout=15)
    r.raise_for_status()
    return r.json()


def ticker_24h() -> dict[str, dict]:
    return {t["symbol"]: t for t in _get(FAPI, "/fapi/v1/ticker/24hr")}


def _klines_features(symbol: str) -> dict:
    kl = _get(FAPI, "/fapi/v1/klines",
              {"symbol": symbol, "interval": "1d", "limit": 365})
    if len(kl) < 25:
        return {}
    df = pd.DataFrame(kl, columns=["ot", "o", "h", "l", "c", "v", "ct", "qv",
                                   "n", "tb", "tq", "ig"])
    for col in ("o", "h", "l", "c"):
        df[col] = pd.to_numeric(df[col])
    c = df["c"]
    ret = np.log(c / c.shift(1)).dropna()
    hv5  = float(ret.tail(5).std()  * math.sqrt(365) * 100)
    hv20 = float(ret.tail(20).std() * math.sqrt(365) * 100)
    tr = pd.concat([df["h"] - df["l"], (df["h"] - c.shift(1)).abs(),
                    (df["l"] - c.shift(1)).abs()], axis=1).max(axis=1)
    atr14 = float(tr.rolling(14).mean().dropna().iloc[-1])
    close = float(c.iloc[-1])
    hi, lo = float(c.max()), float(c.min())
    # crypto trades 24/7: "gap" = move since previous daily close (UTC)
    prev_close = float(c.iloc[-2])
    return {
        "prev_close": prev_close,
        "hv_short": hv5, "hv_long": hv20,
        "hv_ratio": hv5 / hv20 if hv20 > 0 else None,
        "atr_pct": atr14 / close * 100,
        "dist_52w_pct": min((hi - close) / close * 100, (close - lo) / close * 100),
    }


def _oi_change(symbol: str) -> tuple[float | None, float | None]:
    """(current OI, % change vs ~24h ago) from openInterestHist."""
    try:
        hist = _get(FAPI, "/futures/data/openInterestHist",
                    {"symbol": symbol, "period": "1h", "limit": 25})
        if len(hist) >= 2:
            now = float(hist[-1]["sumOpenInterest"])
            ago = float(hist[0]["sumOpenInterest"])
            return now, (now - ago) / ago * 100 if ago > 0 else None
    except Exception:
        pass
    return None, None


def _funding(symbol: str) -> float | None:
    try:
        d = _get(FAPI, "/fapi/v1/premiumIndex", {"symbol": symbol})
        return float(d.get("lastFundingRate", 0))
    except Exception:
        return None


_eapi_marks_cache: list | None = None


def _implied_move(asset: str, spot: float) -> tuple[float | None, float | None]:
    """(1-day implied move %, ATM IV %) from Binance options nearest expiry."""
    global _eapi_marks_cache
    if asset not in EAPI_ASSETS:
        return None, None
    try:
        if _eapi_marks_cache is None:
            _eapi_marks_cache = _get(EAPI, "/eapi/v1/mark")
        rows = []
        for d in _eapi_marks_cache:
            sym = d.get("symbol", "")
            if not sym.startswith(f"{asset}-"):
                continue
            p = sym.split("-")
            rows.append({"expiry": f"20{p[1][:2]}-{p[1][2:4]}-{p[1][4:]}",
                         "strike": float(p[2]), "side": p[3],
                         "mark": float(d.get("markPrice", 0)),
                         "iv": float(d.get("markIV", 0))})
        if not rows:
            return None, None
        df = pd.DataFrame(rows)
        expiry = df["expiry"].min()
        near = df[df["expiry"] == expiry]
        atm = near.iloc[(near["strike"] - spot).abs().argsort()]["strike"].iloc[0]
        legs = near[near["strike"] == atm]
        call = legs[legs["side"] == "C"]["mark"].sum()
        put  = legs[legs["side"] == "P"]["mark"].sum()
        iv   = float(legs["iv"].mean() * 100) or None
        days = max((pd.Timestamp(expiry) - pd.Timestamp(date.today())).days, 1)
        move = (call + put) / spot * 100 / math.sqrt(days) if call + put > 0 else None
        return move, iv
    except Exception:
        return None, None


def reset_cache():
    global _eapi_marks_cache
    _eapi_marks_cache = None


def fetch_symbol(symbol: str, tick: dict) -> dict:
    """Feature dict for one Binance USDT-perp (tick = its 24h ticker entry)."""
    spot = float(tick["lastPrice"])
    feats = {"symbol": symbol, "spot": spot}
    feats.update(_klines_features(symbol))

    prev = feats.get("prev_close")
    if prev:
        feats["gap_pct"] = (spot - prev) / prev * 100

    oi, oi_chg = _oi_change(symbol)
    feats["oi_total"] = oi
    feats["oi_change_pct"] = oi_chg
    feats["funding_rate"] = _funding(symbol)

    asset = symbol.replace("USDT", "")
    move, iv = _implied_move(asset, spot)
    feats["implied_move"] = move
    feats["atm_iv"] = iv
    if iv and feats.get("hv_long"):
        feats["iv_hv_ratio"] = iv / feats["hv_long"]
    return feats
