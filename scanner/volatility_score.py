"""
Composite volatility score (0–100) + threshold flags.

V1   — every feature percentile-ranked cross-sectionally within the scan,
       weighted per config.SCORE_WEIGHTS, summed.
V1.5 — futures OI buildup classification (price × OI direction), EVENT_DAY
       earnings flag, and a market-context multiplier (India VIX / GIFT
       Nifty / VIX / ES) applied uniformly so risk-heavy mornings push more
       names over the alert gate.
V2   — real IV Rank (from IVHistory) feeds IV_RANK_EXTREME; when a trained
       LightGBM model exists, its predicted next-day range (percentile-scaled)
       is blended with the V1 score by config.ML_BLEND.
"""

import numpy as np
import pandas as pd

import config
from scanner import ml_model


def _pct_rank(series: pd.Series) -> pd.Series:
    """Percentile rank 0–1; NaNs get the neutral 0.5 so they neither boost nor sink."""
    r = series.rank(pct=True)
    return r.fillna(0.5)


def classify_buildup(price_chg, oi_chg) -> str | None:
    """V1.5 futures positioning read from price direction × OI direction.

        price ↑  OI ↑   Long Buildup      (new longs — trend conviction)
        price ↓  OI ↑   Short Buildup     (new shorts — bearish conviction)
        price ↑  OI ↓   Short Covering    (shorts exiting — squeeze fuel)
        price ↓  OI ↓   Long Unwinding    (longs exiting)
    """
    if price_chg is None or oi_chg is None \
            or pd.isna(price_chg) or pd.isna(oi_chg):
        return None
    if abs(price_chg) < config.BUILDUP_MIN_PRICE_PCT \
            or abs(oi_chg) < config.BUILDUP_MIN_OI_PCT:
        return None
    if oi_chg > 0:
        return "Long Buildup" if price_chg > 0 else "Short Buildup"
    return "Short Covering" if price_chg > 0 else "Long Unwinding"


def score_universe(features: list[dict], market_type: str | None = None,
                   context: dict | None = None) -> pd.DataFrame:
    """features: list of per-symbol dicts. Returns df with score/rank/flags added."""
    df = pd.DataFrame(features)
    if df.empty:
        return df

    for col in ("implied_move", "gap_pct", "iv_hv_ratio", "hv_ratio",
                "oi_change_pct", "funding_rate", "atm_iv", "pcr",
                "iv_rank", "iv_percentile"):
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "regime" not in df.columns:
        df["regime"] = None
    if "event_day" not in df.columns:
        df["event_day"] = False
    df["event_day"] = df["event_day"].fillna(False).astype(bool)

    # V1.5 — futures OI buildup (India: overnight ΔOI, crypto: 24h ΔOI)
    df["oi_buildup"] = df.apply(
        lambda r: classify_buildup(r.get("gap_pct"), r.get("oi_change_pct")), axis=1)

    # V1 rule-based percentile score
    w = config.SCORE_WEIGHTS
    parts = (
        _pct_rank(df["implied_move"])          * w["implied_move"]
        + _pct_rank(df["gap_pct"].abs())       * w["gap_pct"]
        + _pct_rank(df["iv_hv_ratio"])         * w["iv_hv_ratio"]
        + _pct_rank(df["hv_ratio"])            * w["hv_ratio"]
        + _pct_rank(df["oi_change_pct"].abs()) * w["oi_change"]
        + (df["regime"] == "Negative Gamma").astype(float) * w["negative_gamma"]
    )
    df["score_v1"] = (parts / sum(w.values()) * 100).round(1)

    # V2 — LightGBM predicted-range score, blended when a model is live
    ml_scores = ml_model.predict_scores(df, market_type or "") \
        if config.USE_ML_SCORE else None
    if ml_scores is not None:
        df["score_ml"] = ml_scores
        df["score"] = (config.ML_BLEND * df["score_ml"]
                       + (1 - config.ML_BLEND) * df["score_v1"]).round(1)
        df["score_source"] = "ml" if config.ML_BLEND >= 1.0 else "blend"
    else:
        df["score_ml"] = np.nan
        df["score"] = df["score_v1"]
        df["score_source"] = "v1"

    # V1.5 — market-context multiplier (uniform: gates change, ranks don't)
    mult = float((context or {}).get("score_multiplier") or 1.0)
    if mult != 1.0:
        df["score"] = (df["score"] * mult).clip(upper=100).round(1)

    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    df["flags"] = df.apply(_flags, axis=1)
    return df


def _flags(row) -> str:
    f = []
    gap = row.get("gap_pct")
    if gap is not None and not pd.isna(gap) and abs(gap) >= config.GAP_ALERT_PCT:
        f.append("GAP_RISK")
    im = row.get("implied_move")
    if im is not None and not pd.isna(im) and im >= config.IMPLIED_MOVE_ALERT_PCT:
        f.append("IMPLIED_MOVE_SPIKE")

    # V2 real IV Rank when history allows; V1 IV/HV heuristic otherwise
    ivr = row.get("iv_rank")
    ivhv = row.get("iv_hv_ratio")
    if ivr is not None and not pd.isna(ivr):
        if ivr >= config.IV_RANK_EXTREME_LEVEL:
            f.append("IV_RANK_EXTREME")
    elif ivhv is not None and not pd.isna(ivhv) and ivhv >= config.IV_HV_RATIO_ALERT:
        f.append("IV_RANK_EXTREME")

    hvr = row.get("hv_ratio")
    if hvr is not None and not pd.isna(hvr) and hvr >= config.HV_RATIO_ALERT:
        f.append("HOT_REALIZED_VOL")
    if row.get("regime") == "Negative Gamma" and gap is not None \
            and not pd.isna(gap) and abs(gap) >= 1.0:
        f.append("NEGATIVE_GAMMA_GAP")
    fr = row.get("funding_rate")
    if fr is not None and not pd.isna(fr) and abs(fr) >= config.CRYPTO_FUNDING_ALERT:
        f.append("FUNDING_EXTREME")
    oic = row.get("oi_change_pct")
    if oic is not None and not pd.isna(oic) and abs(oic) >= config.CRYPTO_OI_CHANGE_ALERT:
        f.append("OI_SPIKE")

    # V1.5 flags
    if row.get("event_day"):
        f.append("EVENT_DAY")
    bu = row.get("oi_buildup")
    if bu == "Short Buildup":
        f.append("SHORT_BUILDUP")
    elif bu == "Short Covering":
        f.append("SHORT_COVERING")
    return ",".join(f)


# Severity of each flag when it becomes an individual alert
FLAG_SEVERITY = {
    "NEGATIVE_GAMMA_GAP":  "CRITICAL",
    "GAP_RISK":            "HIGH",
    "IMPLIED_MOVE_SPIKE":  "HIGH",
    "EVENT_DAY":           "HIGH",
    "IV_RANK_EXTREME":     "MEDIUM",
    "HOT_REALIZED_VOL":    "MEDIUM",
    "OI_SPIKE":            "MEDIUM",
    "FUNDING_EXTREME":     "MEDIUM",
    "SHORT_BUILDUP":       "MEDIUM",
    "SHORT_COVERING":      "MEDIUM",
}

FLAG_GUIDANCE = {
    "NEGATIVE_GAMMA_GAP": "Negative gamma + gap: dealer hedging amplifies moves. Expect wide range; size down, avoid market orders at open.",
    "GAP_RISK":           "Large overnight gap. Expect elevated opening volatility; use wider stops.",
    "IMPLIED_MOVE_SPIKE": "Options price an outsized one-day move. Expect wide range; size positions accordingly.",
    "EVENT_DAY":          "Earnings today/imminent — expect an outsized post-result move; options are event-priced, spreads widen.",
    "IV_RANK_EXTREME":    "IV at the top of its own range — the market expects movement (possible event day).",
    "HOT_REALIZED_VOL":   "Short-term realized vol is running well above baseline; the hot streak tends to persist.",
    "OI_SPIKE":           "Unusual open-interest shift — positioning is changing fast.",
    "FUNDING_EXTREME":    "Extreme funding rate — crowded leverage; squeeze risk elevated.",
    "SHORT_BUILDUP":      "Price down on rising OI — fresh shorts with conviction; downside continuation risk, but squeeze-prone if reclaimed.",
    "SHORT_COVERING":     "Price up on falling OI — shorts exiting; sharp but potentially short-lived upside bursts.",
}

# Flags exempt from the MIN_SCORE_FOR_ALERT gate
GATE_EXEMPT_FLAGS = {"NEGATIVE_GAMMA_GAP", "EVENT_DAY"}
