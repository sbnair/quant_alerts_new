"""
Minimal gamma-regime math extracted from quant_4's gamma_engine —
just enough to classify Positive/Negative gamma and locate the flip,
without the charting/action layers.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import norm

import config


def bs_gamma(S, K, T, r, sigma):
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0 or np.isnan(sigma):
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def gamma_regime(options_df: pd.DataFrame, spot: float, contract_size: int = 1):
    """
    options_df columns: strike, type, openInterest, impliedVolatility, expiry.
    Returns (regime_str, gamma_flip_or_nan).
    """
    today = pd.Timestamp(datetime.now()).normalize()
    lo, hi = spot * (1 - config.STRIKE_RANGE_PCT), spot * (1 + config.STRIKE_RANGE_PCT)
    df = options_df[(options_df["strike"] >= lo) & (options_df["strike"] <= hi)].copy()
    if df.empty:
        return None, np.nan

    df["T"] = ((pd.to_datetime(df["expiry"]) - today).dt.days / 365).clip(lower=1 / 365)
    df["gamma"] = df.apply(
        lambda x: bs_gamma(spot, x["strike"], x["T"], config.RISK_FREE_RATE,
                           x["impliedVolatility"]), axis=1)
    df["signed_gex"] = np.where(
        df["type"] == "call", 1, -1
    ) * df["gamma"] * df["openInterest"] * contract_size * spot ** 2 / 100

    by_strike = (df.groupby("strike")["signed_gex"].sum()
                   .sort_index().cumsum().reset_index())
    by_strike.columns = ["strike", "agg_gex"]

    flip = np.nan
    for i in range(1, len(by_strike)):
        y1, y2 = by_strike.loc[i - 1, "agg_gex"], by_strike.loc[i, "agg_gex"]
        if (y1 <= 0 <= y2) or (y1 >= 0 >= y2):
            x1, x2 = by_strike.loc[i - 1, "strike"], by_strike.loc[i, "strike"]
            flip = x1 + (0 - y1) * (x2 - x1) / (y2 - y1) if y2 != y1 else x1
            break

    net_total = float(df["signed_gex"].sum())
    if not np.isnan(flip):
        regime = "Positive Gamma" if spot > flip else "Negative Gamma"
    else:
        regime = "Positive Gamma" if net_total > 0 else "Negative Gamma"
    return regime, flip
