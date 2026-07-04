"""
V2 LightGBM ranker.

Training data: every ScanResult's features_json snapshot, labeled with the
*realized next-session range* ((high−low)/prev_close in %) fetched after the
fact — yfinance for India/US, Binance klines for crypto. Labels live in the
ml_labels table so each row is fetched once.

At scan time the trained model predicts the expected next-day range for each
symbol; predictions are percentile-scaled 0–100 within the scan (the same
scale V1 uses) and blended with the rule-based score by config.ML_BLEND.
Everything degrades gracefully: no lightgbm / no model file / too little
history → pure V1 scoring.
"""

import json
import os
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

import config
from models import db, ScanResult, MLLabel, \
    MARKET_INDIA, MARKET_CRYPTO, MARKET_US

try:
    import lightgbm as lgb
except ImportError:          # model features simply switch off
    lgb = None

# Numeric features pulled from features_json (missing → NaN; LightGBM handles it)
NUMERIC_FEATURES = [
    "implied_move", "gap_pct", "abs_gap_pct", "atm_iv", "iv_hv_ratio",
    "hv_short", "hv_long", "hv_ratio", "atr_pct", "dist_52w_pct", "pcr",
    "oi_change_pct", "abs_oi_change_pct", "funding_rate",
    "iv_rank", "iv_percentile",
]
BINARY_FEATURES = ["negative_gamma", "event_day"]
MARKET_DUMMIES  = ["mkt_india", "mkt_crypto", "mkt_us"]
ALL_FEATURES    = NUMERIC_FEATURES + BINARY_FEATURES + MARKET_DUMMIES

_booster = None
_booster_mtime = None


# ── feature matrix ────────────────────────────────────────────────────────────

def build_matrix(df: pd.DataFrame, market_type: str) -> pd.DataFrame:
    """Model input matrix from a scored/collected frame (one row per symbol)."""
    x = pd.DataFrame(index=df.index)
    for f in NUMERIC_FEATURES:
        if f == "abs_gap_pct":
            x[f] = pd.to_numeric(df.get("gap_pct"), errors="coerce").abs()
        elif f == "abs_oi_change_pct":
            x[f] = pd.to_numeric(df.get("oi_change_pct"), errors="coerce").abs()
        else:
            x[f] = pd.to_numeric(df.get(f), errors="coerce")
    x["negative_gamma"] = (df.get("regime") == "Negative Gamma").astype(float) \
        if "regime" in df else 0.0
    ev = df.get("event_day")
    x["event_day"] = ev.fillna(False).astype(float) if ev is not None else 0.0
    x["mkt_india"]  = float(market_type == MARKET_INDIA)
    x["mkt_crypto"] = float(market_type == MARKET_CRYPTO)
    x["mkt_us"]     = float(market_type == MARKET_US)
    return x[ALL_FEATURES]


# ── scan-time prediction ─────────────────────────────────────────────────────

def _load_booster():
    """Cached booster, reloaded when the model file changes on disk."""
    global _booster, _booster_mtime
    if lgb is None or not os.path.exists(config.ML_MODEL_PATH):
        return None
    mtime = os.path.getmtime(config.ML_MODEL_PATH)
    if _booster is None or mtime != _booster_mtime:
        try:
            _booster = lgb.Booster(model_file=config.ML_MODEL_PATH)
            _booster_mtime = mtime
        except Exception as e:
            print(f"[ML] failed to load model: {e}")
            return None
    return _booster


def available() -> bool:
    return config.USE_ML_SCORE and _load_booster() is not None


def predict_scores(df: pd.DataFrame, market_type: str) -> pd.Series | None:
    """Predicted next-day range → percentile-scaled 0–100 score, or None."""
    booster = _load_booster()
    if booster is None or df.empty:
        return None
    try:
        x = build_matrix(df, market_type)
        pred = booster.predict(x[booster.feature_name()])
        s = pd.Series(pred, index=df.index)
        return (s.rank(pct=True) * 100).round(1)
    except Exception as e:
        print(f"[ML] predict failed: {e}")
        return None


def model_info() -> dict:
    try:
        with open(config.ML_META_PATH) as f:
            meta = json.load(f)
    except Exception:
        meta = {}
    meta["model_file"] = os.path.exists(config.ML_MODEL_PATH)
    meta["lightgbm"] = lgb is not None
    meta["active"] = available()
    return meta


# ── label backfill ───────────────────────────────────────────────────────────

def _yf_history_batch(symbols: list[str], to_yf) -> dict[str, pd.DataFrame]:
    import yfinance as yf
    tickers = [to_yf(s) for s in symbols]
    data = yf.download(tickers, period="6mo", auto_adjust=False,
                       group_by="ticker", progress=False, threads=True)
    out = {}
    for sym, tk in zip(symbols, tickers):
        try:
            d = data[tk].dropna(how="all") if len(tickers) > 1 else data.dropna(how="all")
            if not d.empty:
                out[sym] = d
        except KeyError:
            pass
    return out


def _binance_daily(symbol: str) -> pd.DataFrame | None:
    try:
        kl = requests.get("https://fapi.binance.com/fapi/v1/klines",
                          params={"symbol": symbol, "interval": "1d", "limit": 180},
                          timeout=15).json()
        df = pd.DataFrame(kl, columns=["ot", "o", "h", "l", "c", "v", "ct", "qv",
                                       "n", "tb", "tq", "ig"])
        df.index = pd.to_datetime(df["ot"], unit="ms")
        for col, name in (("h", "High"), ("l", "Low"), ("c", "Close")):
            df[name] = pd.to_numeric(df[col])
        return df[["High", "Low", "Close"]]
    except Exception:
        return None


def _label_from_hist(hist: pd.DataFrame, scan_date: date,
                     complete_before: date | None = None):
    """(next_range_pct, next_absret_pct) from the first bar after scan_date.
    Bars on/after complete_before are still-forming sessions and never used."""
    if complete_before is not None:
        hist = hist[pd.to_datetime(hist.index).date < complete_before]
    days = pd.to_datetime(hist.index).date
    after = np.where(days > scan_date)[0]
    at_or_before = np.where(days <= scan_date)[0]
    if len(after) == 0 or len(at_or_before) == 0:
        return None, None
    nxt, prv = hist.iloc[after[0]], hist.iloc[at_or_before[-1]]
    prev_close = float(prv["Close"])
    if not prev_close or np.isnan(prev_close):
        return None, None
    rng = (float(nxt["High"]) - float(nxt["Low"])) / prev_close * 100
    ret = abs(float(nxt["Close"]) / prev_close - 1) * 100
    if np.isnan(rng):
        return None, None
    return rng, ret


def backfill_labels(app, max_age_days: int = 120) -> int:
    """Label every unlabeled ScanResult old enough for next-day data to exist."""
    from data.india_market import yf_ticker as india_yf

    with app.app_context():
        cutoff = datetime.now().date() - timedelta(days=max_age_days)
        labeled = {r[0] for r in db.session.query(MLLabel.result_id).all()}
        rows = (ScanResult.query
                .filter(ScanResult.scanned_at < datetime.combine(date.today(), datetime.min.time()))
                .all())
        todo = [r for r in rows
                if r.id not in labeled and r.scanned_at.date() >= cutoff]
        if not todo:
            return 0

        by_market: dict[str, set] = {}
        for r in todo:
            by_market.setdefault(r.market_type, set()).add(r.symbol)

        hists: dict[tuple, pd.DataFrame] = {}
        if MARKET_INDIA in by_market:
            for s, h in _yf_history_batch(sorted(by_market[MARKET_INDIA]), india_yf).items():
                hists[(MARKET_INDIA, s)] = h
        if MARKET_US in by_market:
            for s, h in _yf_history_batch(sorted(by_market[MARKET_US]), lambda x: x).items():
                hists[(MARKET_US, s)] = h
        for s in sorted(by_market.get(MARKET_CRYPTO, [])):
            h = _binance_daily(s)
            if h is not None:
                hists[(MARKET_CRYPTO, s)] = h

        # a "next session" bar only counts once it has closed: crypto daily
        # bars roll at 00:00 UTC; stock bars are partial on their own date
        utc_today = datetime.now(timezone.utc).date()
        n = 0
        for r in todo:
            hist = hists.get((r.market_type, r.symbol))
            if hist is None:
                continue
            cutoff = utc_today if r.market_type == MARKET_CRYPTO else date.today()
            rng, ret = _label_from_hist(hist, r.scanned_at.date(),
                                        complete_before=cutoff)
            if rng is None:
                continue
            db.session.add(MLLabel(result_id=r.id, market_type=r.market_type,
                                   symbol=r.symbol, scan_date=r.scanned_at.date(),
                                   next_range_pct=rng, next_absret_pct=ret))
            n += 1
        db.session.commit()
        print(f"[ML] backfilled {n} labels ({len(todo)} candidates)")
        return n


# ── training ─────────────────────────────────────────────────────────────────

def _training_frame(app) -> pd.DataFrame:
    with app.app_context():
        q = (db.session.query(ScanResult, MLLabel)
             .join(MLLabel, MLLabel.result_id == ScanResult.id)
             .filter(MLLabel.next_range_pct.isnot(None)))
        recs = []
        for res, lab in q.all():
            try:
                feats = json.loads(res.features_json or "{}")
            except json.JSONDecodeError:
                feats = {}
            feats.update({"regime": res.regime, "event_day": res.event_day,
                          "iv_rank": res.iv_rank, "iv_percentile": res.iv_percentile,
                          "_market": res.market_type, "_date": lab.scan_date,
                          "_y": lab.next_range_pct})
            recs.append(feats)
    return pd.DataFrame(recs)


def train(app) -> dict:
    """Backfill labels, train, save model + meta. Returns metrics dict."""
    if lgb is None:
        return {"error": "lightgbm not installed"}
    backfill_labels(app)
    df = _training_frame(app)
    if len(df) < config.ML_MIN_TRAINING_ROWS:
        msg = f"only {len(df)} labeled rows; need {config.ML_MIN_TRAINING_ROWS}"
        print(f"[ML] not training — {msg}")
        return {"skipped": msg, "n_rows": len(df)}

    parts = [build_matrix(df[df["_market"] == m], m)
             for m in (MARKET_INDIA, MARKET_CRYPTO, MARKET_US)
             if (df["_market"] == m).any()]
    x = pd.concat(parts).sort_index()
    y = df["_y"].clip(upper=df["_y"].quantile(0.995))  # tame blow-up outliers

    # time-ordered split: newest 20% is the validation set
    order = df["_date"].argsort().values
    n_val = max(len(df) // 5, 50)
    tr, va = order[:-n_val], order[-n_val:]

    params = {"objective": "regression", "metric": "mae",
              "learning_rate": 0.05, "num_leaves": 31,
              "min_data_in_leaf": 20, "bagging_fraction": 0.9,
              "bagging_freq": 1, "feature_fraction": 0.9,
              "seed": 42, "verbose": -1}
    dtrain = lgb.Dataset(x.iloc[tr], label=y.iloc[tr])
    dval = lgb.Dataset(x.iloc[va], label=y.iloc[va], reference=dtrain)
    model = lgb.train(params, dtrain, num_boost_round=500,
                      valid_sets=[dval],
                      callbacks=[lgb.early_stopping(50, verbose=False)])

    pred = model.predict(x.iloc[va], num_iteration=model.best_iteration)
    mae = float(np.mean(np.abs(pred - y.iloc[va].values)))
    spearman = float(pd.Series(pred).corr(
        pd.Series(y.iloc[va].values), method="spearman"))

    os.makedirs(os.path.dirname(config.ML_MODEL_PATH), exist_ok=True)
    model.save_model(config.ML_MODEL_PATH, num_iteration=model.best_iteration)
    meta = {"trained_at": datetime.now().isoformat(timespec="seconds"),
            "n_rows": len(df), "n_val": int(n_val),
            "val_mae": round(mae, 3), "val_spearman": round(spearman, 3),
            "features": ALL_FEATURES,
            "best_iteration": int(model.best_iteration or 0)}
    with open(config.ML_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    global _booster, _booster_mtime
    _booster = None  # force reload of the fresh file
    print(f"[ML] trained on {len(df)} rows — val MAE {mae:.2f}%, "
          f"rank corr {spearman:.3f} → {config.ML_MODEL_PATH}")
    return meta
