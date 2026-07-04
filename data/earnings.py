"""
V1.5 earnings-calendar EVENT_DAY override.

For US and India symbols, ask yfinance for the next earnings date; a symbol
whose earnings fall today or within EVENT_DAY_LOOKAHEAD days gets
event_day=True, which adds the EVENT_DAY flag and exempts the name from the
alert score gate (an event name should alert even on a quiet tape).

Lookups are cached to instance/earnings_cache.json for the calendar day so
repeated scans (premarket + postopen + manual) don't refetch the calendar.
"""

import json
import os
from datetime import date, datetime

import yfinance as yf

import config

_cache: dict | None = None


def _load_cache() -> dict:
    global _cache
    if _cache is not None:
        return _cache
    try:
        with open(config.EARNINGS_CACHE) as f:
            data = json.load(f)
        if data.get("date") == date.today().isoformat():
            _cache = data
            return _cache
    except Exception:
        pass
    _cache = {"date": date.today().isoformat(), "symbols": {}}
    return _cache


def _save_cache():
    if _cache is None:
        return
    try:
        os.makedirs(os.path.dirname(config.EARNINGS_CACHE), exist_ok=True)
        with open(config.EARNINGS_CACHE, "w") as f:
            json.dump(_cache, f)
    except Exception:
        pass


def _next_earnings_date(yf_symbol: str):
    """Next earnings date via yfinance calendar; None when unknown."""
    try:
        cal = yf.Ticker(yf_symbol).calendar
        dates = (cal or {}).get("Earnings Date") or []
        dates = [d for d in dates if d is not None]
        if dates:
            d = min(dates)
            if isinstance(d, datetime):
                d = d.date()
            return d
    except Exception:
        pass
    return None


def is_event_day(symbol: str, yf_symbol: str | None = None) -> bool:
    """True when earnings are today or within EVENT_DAY_LOOKAHEAD days."""
    if not config.ENABLE_EARNINGS:
        return False
    cache = _load_cache()
    key = symbol.upper()
    if key in cache["symbols"]:
        iso = cache["symbols"][key]
    else:
        d = _next_earnings_date(yf_symbol or symbol)
        iso = d.isoformat() if d else ""
        cache["symbols"][key] = iso
        _save_cache()
    if not iso:
        return False
    try:
        delta = (date.fromisoformat(iso) - date.today()).days
    except ValueError:
        return False
    return 0 <= delta <= config.EVENT_DAY_LOOKAHEAD
