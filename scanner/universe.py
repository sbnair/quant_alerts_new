"""
F&O universe builders for each market.

INDIA — parsed from the Dhan instrument master CSV: every NSE stock with a
listed stock future (FUTSTK), joined to its NSE equity security id (that id +
segment NSE_FNO is what Dhan's option-chain API wants for the underlying).
Cached to instance/fno_universe.csv; refresh with build_india_universe(force=True).

CRYPTO — top-N Binance USDT perpetuals by 24h quote volume (fetched live).

US — static list of highly liquid optionable names from config.
"""

import io
import os

import pandas as pd
import requests

import config

# Most liquid F&O names scanned first so a limited universe still covers
# the stocks that matter; the rest of the master follows alphabetically.
INDIA_PRIORITY = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "SBIN", "INFY", "TCS", "AXISBANK",
    "KOTAKBANK", "LT", "ITC", "BHARTIARTL", "TATAMOTORS", "TATASTEEL",
    "ADANIENT", "ADANIPORTS", "BAJFINANCE", "BAJAJFINSV", "MARUTI", "M&M",
    "HINDUNILVR", "SUNPHARMA", "TITAN", "ULTRACEMCO", "WIPRO", "HCLTECH",
    "TECHM", "POWERGRID", "NTPC", "ONGC", "COALINDIA", "JSWSTEEL", "HINDALCO",
    "VEDL", "DLF", "INDUSINDBK", "BANKBARODA", "PNB", "CANBK", "IDFCFIRSTB",
    "ZOMATO", "PAYTM", "HAL", "BEL", "IRCTC", "LICHSGFIN", "DIXON",
    "TRENT", "JIOFIN", "BHEL", "SAIL",
]


def build_india_universe(force: bool = False) -> pd.DataFrame:
    """
    Returns DataFrame: symbol, security_id, segment, lot_size, kind (INDEX|STOCK).
    Indices from config.DHAN_INDEX_MAP; stocks from the Dhan instrument master.
    """
    cache = config.UNIVERSE_CACHE
    if not force and os.path.exists(cache):
        return pd.read_csv(cache, dtype={"security_id": str})

    print("[UNIVERSE] Downloading Dhan instrument master (one-time, ~large file)...")
    resp = requests.get(config.DHAN_INSTRUMENT_MASTER_URL, timeout=120)
    resp.raise_for_status()
    master = pd.read_csv(io.BytesIO(resp.content), low_memory=False)

    # Stock futures → the set of F&O underlyings + lot sizes
    futs = master[
        (master["SEM_EXM_EXCH_ID"] == "NSE")
        & (master["SEM_INSTRUMENT_NAME"] == "FUTSTK")
    ].copy()
    futs["underlying"] = futs["SEM_TRADING_SYMBOL"].str.split("-").str[0]
    lot_by_sym = (
        futs.sort_values("SEM_EXPIRY_DATE")
        .groupby("underlying")["SEM_LOT_UNITS"].first()
    )

    # NSE equity rows → underlying security ids
    eq = master[
        (master["SEM_EXM_EXCH_ID"] == "NSE")
        & (master["SEM_INSTRUMENT_NAME"] == "EQUITY")
        & (master["SEM_SERIES"].astype(str) == "EQ")
    ][["SEM_TRADING_SYMBOL", "SEM_SMST_SECURITY_ID"]].drop_duplicates("SEM_TRADING_SYMBOL")
    eq_id = dict(zip(eq["SEM_TRADING_SYMBOL"], eq["SEM_SMST_SECURITY_ID"].astype(str)))

    rows = [
        {"symbol": s, "security_id": meta["security_id"], "segment": meta["segment"],
         "lot_size": meta["lot_size"], "kind": "INDEX"}
        for s, meta in config.DHAN_INDEX_MAP.items()
    ]
    stock_syms = sorted(lot_by_sym.index)
    ordered = [s for s in INDIA_PRIORITY if s in stock_syms] + \
              [s for s in stock_syms if s not in INDIA_PRIORITY]
    for sym in ordered:
        sid = eq_id.get(sym)
        if not sid:
            continue
        rows.append({"symbol": sym, "security_id": sid, "segment": "NSE_FNO",
                     "lot_size": int(lot_by_sym[sym]), "kind": "STOCK"})

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    df.to_csv(cache, index=False)
    print(f"[UNIVERSE] {len(df)} India F&O instruments cached → {cache}")
    return df


def india_universe(limit: int | None = None) -> pd.DataFrame:
    df = build_india_universe()
    limit = limit or config.INDIA_UNIVERSE_LIMIT
    idx = df[df["kind"] == "INDEX"]
    stocks = df[df["kind"] == "STOCK"].head(limit)
    return pd.concat([idx, stocks], ignore_index=True)


def crypto_universe(limit: int | None = None) -> list[str]:
    """Top-N Binance USDT-perp symbols by 24h quote volume."""
    limit = limit or config.CRYPTO_UNIVERSE_LIMIT
    tickers = requests.get("https://fapi.binance.com/fapi/v1/ticker/24hr",
                           timeout=15).json()
    usdt = [t for t in tickers
            if t["symbol"].endswith("USDT") and not t["symbol"].endswith("_USDT")]
    usdt.sort(key=lambda t: float(t.get("quoteVolume", 0)), reverse=True)
    return [t["symbol"] for t in usdt[:limit]]


def us_universe() -> list[str]:
    return list(config.US_UNIVERSE)
