"""
Configuration for the Volatility Scanner (quant_alert_1 — V1 + V1.5 + V2).
Secrets (Dhan / Binance / Telegram / Discord) are loaded from ~/quant_4/.env
so keys live in one place. A local .env in this directory can override them.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load shared keys first, then local overrides (local wins)
load_dotenv(Path.home() / "quant_4" / ".env")
load_dotenv(Path(__file__).parent / ".env", override=True)

BASE_DIR = Path(__file__).parent


def _bool(key, default="false"):
    return os.getenv(key, default).lower() in ("true", "1", "yes")

def _float(key, default):
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):
        return float(default)

def _int(key, default):
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        return int(default)


# ── Credentials ───────────────────────────────────────────────────────────────
DHAN_CLIENT_ID      = os.getenv("DHAN_CLIENT_ID", "")
DHAN_ACCESS_TOKEN   = os.getenv("DHAN_ACCESS_TOKEN", "")
BINANCE_API_KEY     = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET  = os.getenv("BINANCE_API_SECRET", "")

ENABLE_TELEGRAM     = _bool("ENABLE_TELEGRAM", "true")
ENABLE_DISCORD      = _bool("ENABLE_DISCORD", "true")
TELEGRAM_BOT_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID    = os.getenv("TELEGRAM_CHAT_ID", "")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

# ── Flask ─────────────────────────────────────────────────────────────────────
FLASK_SECRET_KEY    = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")
FLASK_PORT          = _int("VOLSCAN_PORT", 5051)           # 5050 belongs to quant_alert
FLASK_DEBUG         = _bool("FLASK_DEBUG", "false")
DB_PATH             = "volscan.db"

# ── Scan parameters ───────────────────────────────────────────────────────────
RISK_FREE_RATE      = _float("RISK_FREE_RATE", 0.065)
MIN_OPEN_INTEREST   = _int("MIN_OPEN_INTEREST", 1)
STRIKE_RANGE_PCT    = _float("STRIKE_RANGE_PCT", 0.20)

# Dhan option-chain API is rate-limited (~1 req / 3 s) — pace universe scans.
DHAN_REQUEST_GAP_SEC = _float("DHAN_REQUEST_GAP_SEC", 3.1)

# How many India F&O stocks (by futures OI liquidity) to scan besides indices.
INDIA_UNIVERSE_LIMIT = _int("INDIA_UNIVERSE_LIMIT", 60)

# Crypto universe: top-N Binance USDT-perps by 24h quote volume
CRYPTO_UNIVERSE_LIMIT = _int("CRYPTO_UNIVERSE_LIMIT", 40)
CRYPTO_SCAN_INTERVAL_MIN = _int("CRYPTO_SCAN_INTERVAL_MIN", 30)

# US universe: highly liquid optionable indices/ETFs/mega-caps (yfinance)
US_UNIVERSE = [
    "SPY", "QQQ", "IWM", "DIA",
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD",
    "NFLX", "AVGO", "JPM", "BAC", "XOM", "CVX", "BA", "COIN",
    "PLTR", "SMCI", "MSTR", "INTC", "MU", "CRM", "ORCL", "UNH",
]

# ── Alert thresholds (V1 rule-based) ─────────────────────────────────────────
TOP_N_ALERTS            = _int("TOP_N_ALERTS", 10)
GAP_ALERT_PCT           = _float("GAP_ALERT_PCT", 2.0)
IMPLIED_MOVE_ALERT_PCT  = _float("IMPLIED_MOVE_ALERT_PCT", 3.0)
IV_HV_RATIO_ALERT       = _float("IV_HV_RATIO_ALERT", 1.5)
HV_RATIO_ALERT          = _float("HV_RATIO_ALERT", 1.8)
CRYPTO_FUNDING_ALERT    = _float("CRYPTO_FUNDING_ALERT", 0.0008)
CRYPTO_OI_CHANGE_ALERT  = _float("CRYPTO_OI_CHANGE_ALERT", 15.0)
MIN_SCORE_FOR_ALERT     = _float("MIN_SCORE_FOR_ALERT", 60.0)

# Composite score weights (V1) — normalized percentile features
SCORE_WEIGHTS = {
    "implied_move":   0.30,
    "gap_pct":        0.20,
    "iv_hv_ratio":    0.15,
    "hv_ratio":       0.15,
    "oi_change":      0.10,
    "negative_gamma": 0.10,
}

# ── V1.5: market-level context ────────────────────────────────────────────────
ENABLE_MARKET_CONTEXT   = _bool("ENABLE_MARKET_CONTEXT", "true")
INDIA_VIX_TICKER        = os.getenv("INDIA_VIX_TICKER", "^INDIAVIX")
# GIFT Nifty has no stable free feed; ticker is configurable and failures are
# tolerated (context simply omits the implied gap). NIFTY futures proxy works
# on some yfinance builds.
GIFT_NIFTY_TICKER       = os.getenv("GIFT_NIFTY_TICKER", "GIFTNIFTY.NS")
US_VIX_TICKER           = os.getenv("US_VIX_TICKER", "^VIX")
US_FUTURES_TICKER       = os.getenv("US_FUTURES_TICKER", "ES=F")

# Score adjustment: a risk-on morning (VIX spike / big index futures gap)
# multiplies every symbol's score so more names clear the alert gate.
VIX_HIGH_LEVEL          = _float("VIX_HIGH_LEVEL", 18.0)    # India VIX / VIX "elevated"
VIX_SPIKE_PCT           = _float("VIX_SPIKE_PCT", 8.0)      # 1-day VIX change considered a spike
INDEX_GAP_ALERT_PCT     = _float("INDEX_GAP_ALERT_PCT", 0.75)  # |implied index gap| considered material
CONTEXT_BOOST_PER_HIT   = _float("CONTEXT_BOOST_PER_HIT", 0.10)
CONTEXT_MULT_MAX        = _float("CONTEXT_MULT_MAX", 1.30)

# ── V1.5: earnings-calendar EVENT_DAY override ───────────────────────────────
ENABLE_EARNINGS         = _bool("ENABLE_EARNINGS", "true")
EVENT_DAY_LOOKAHEAD     = _int("EVENT_DAY_LOOKAHEAD", 1)    # earnings today or within N days
EARNINGS_CACHE          = str(BASE_DIR / "instance" / "earnings_cache.json")

# ── V1.5: futures OI buildup classification ─────────────────────────────────
# Minimum |price change| / |OI change| for a classification to be meaningful.
BUILDUP_MIN_PRICE_PCT   = _float("BUILDUP_MIN_PRICE_PCT", 0.25)
BUILDUP_MIN_OI_PCT      = _float("BUILDUP_MIN_OI_PCT", 2.0)

# ── V2: IV Rank ───────────────────────────────────────────────────────────────
IV_RANK_LOOKBACK_DAYS   = _int("IV_RANK_LOOKBACK_DAYS", 252)
IV_RANK_MIN_OBS         = _int("IV_RANK_MIN_OBS", 20)       # history needed before rank is trusted
IV_RANK_EXTREME_LEVEL   = _float("IV_RANK_EXTREME_LEVEL", 80.0)

# ── V2: LightGBM ranker ──────────────────────────────────────────────────────
USE_ML_SCORE            = _bool("USE_ML_SCORE", "true")     # only kicks in once a model exists
ML_MODEL_PATH           = str(BASE_DIR / "instance" / "lgbm_ranker.txt")
ML_META_PATH            = str(BASE_DIR / "instance" / "lgbm_ranker.meta.json")
ML_MIN_TRAINING_ROWS    = _int("ML_MIN_TRAINING_ROWS", 500)
ML_BLEND                = _float("ML_BLEND", 0.7)           # ml weight; (1-blend) stays on V1 score
ML_AUTO_RETRAIN         = _bool("ML_AUTO_RETRAIN", "true")  # weekly retrain job (Sunday)

# ── Dhan static metadata ──────────────────────────────────────────────────────
DHAN_INDEX_MAP = {
    "NIFTY":      {"security_id": "13", "segment": "IDX_I",   "lot_size": 75},
    "BANKNIFTY":  {"security_id": "25", "segment": "IDX_I",   "lot_size": 30},
    "FINNIFTY":   {"security_id": "27", "segment": "IDX_I",   "lot_size": 65},
    "SENSEX":     {"security_id": "1",  "segment": "BSE_FNO", "lot_size": 10},
}

DHAN_INSTRUMENT_MASTER_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"
UNIVERSE_CACHE = "instance/fno_universe.csv"

IST_TZ = "Asia/Kolkata"
