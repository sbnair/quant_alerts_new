"""
Scan orchestration: collect a market's universe, enrich with V1.5 context
(earnings EVENT_DAY, market-level VIX/futures context) and V2 IV Rank,
score (V1 percentile weights blended with the LightGBM model when trained),
persist to DB, and hand alerts to the alert engine.
"""

import json
import traceback
from datetime import datetime, timezone

import pandas as pd

import config
from models import db, ScanRun, ScanResult, IVHistory, \
    MARKET_INDIA, MARKET_CRYPTO, MARKET_US
from scanner import universe as uni
from scanner import iv_rank
from scanner.volatility_score import score_universe
from data import india_market, crypto_market, us_market, market_context, earnings
from alerts import alert_engine

NUMERIC_FIELDS = [
    "spot", "prev_close", "gap_pct", "implied_move", "atm_iv", "iv_hv_ratio",
    "hv_short", "hv_long", "hv_ratio", "atr_pct", "dist_52w_pct", "pcr",
    "oi_total", "oi_change_pct", "gamma_flip", "funding_rate",
    "iv_rank", "iv_percentile", "score_v1", "score_ml",
]
STRING_FIELDS = ["regime", "oi_buildup", "score_source"]


def _collect_india(progress) -> list[dict]:
    u = uni.india_universe()
    hists = india_market.batch_history(u["symbol"].tolist())
    feats = []
    for i, row in u.iterrows():
        progress(f"India {i + 1}/{len(u)}: {row['symbol']}")
        try:
            f = india_market.fetch_symbol(
                row["symbol"], row["security_id"], row["segment"],
                int(row["lot_size"]), hists.get(row["symbol"]))
            # V1.5 earnings flag (indices have no earnings)
            f["event_day"] = False if row["kind"] == "INDEX" else \
                earnings.is_event_day(row["symbol"],
                                      india_market.yf_ticker(row["symbol"]))
            feats.append(f)
        except Exception as e:
            progress(f"  skip {row['symbol']}: {e}")
    return feats


def _collect_crypto(progress) -> list[dict]:
    crypto_market.reset_cache()
    symbols = uni.crypto_universe()
    ticks = crypto_market.ticker_24h()
    feats = []
    for i, sym in enumerate(symbols):
        progress(f"Crypto {i + 1}/{len(symbols)}: {sym}")
        try:
            feats.append(crypto_market.fetch_symbol(sym, ticks[sym]))
        except Exception as e:
            progress(f"  skip {sym}: {e}")
    return feats


def _collect_us(progress) -> list[dict]:
    symbols = uni.us_universe()
    hists = us_market.batch_history(symbols)
    feats = []
    for i, sym in enumerate(symbols):
        progress(f"US {i + 1}/{len(symbols)}: {sym}")
        try:
            f = us_market.fetch_symbol(sym, hists.get(sym))
            f["event_day"] = earnings.is_event_day(sym)
            feats.append(f)
        except Exception as e:
            progress(f"  skip {sym}: {e}")
    return feats


COLLECTORS = {
    MARKET_INDIA:  _collect_india,
    MARKET_CRYPTO: _collect_crypto,
    MARKET_US:     _collect_us,
}


def run_scan(app, market_type: str, scan_kind: str = "manual",
             notify: bool = True) -> int | None:
    """Run one full scan for a market. Returns the ScanRun id (or None on failure)."""
    def progress(msg):
        print(f"[SCAN:{market_type}] {msg}")

    with app.app_context():
        run = ScanRun(market_type=market_type, scan_kind=scan_kind)
        db.session.add(run)
        db.session.commit()
        run_id = run.id

    try:
        context = market_context.get_context(market_type)
        if context.get("context_notes"):
            progress(f"context: {'; '.join(context['context_notes'])} "
                     f"(score ×{context['score_multiplier']})")

        feats = COLLECTORS[market_type](progress)

        # V2 — merge real IV Rank from accumulated IVHistory (needs app context)
        with app.app_context():
            ranks = iv_rank.iv_ranks(
                market_type,
                {f["symbol"]: f.get("atm_iv") for f in feats
                 if f.get("atm_iv") is not None})
        for f in feats:
            f.update(ranks.get(f["symbol"], {}))

        scored = score_universe(feats, market_type, context)
    except Exception:
        traceback.print_exc()
        with app.app_context():
            run = db.session.get(ScanRun, run_id)
            run.status, run.finished_at = "failed", datetime.now(timezone.utc)
            db.session.commit()
        return None

    with app.app_context():
        run = db.session.get(ScanRun, run_id)
        run.context_json = json.dumps(context, default=str)
        today = datetime.now(timezone.utc).date()
        for _, row in scored.iterrows():
            rec = ScanResult(run_id=run_id, market_type=market_type,
                             symbol=row["symbol"],
                             score=row.get("score"), rank=int(row["rank"]),
                             event_day=bool(row.get("event_day")),
                             flags=row.get("flags") or "")
            for f in NUMERIC_FIELDS:
                v = row.get(f)
                if v is not None and not pd.isna(v):
                    setattr(rec, f, float(v))
            for f in STRING_FIELDS:
                v = row.get(f)
                if v is not None and not (not isinstance(v, str) and pd.isna(v)):
                    setattr(rec, f, str(v))
            raw = {k: (None if pd.isna(v) else v) for k, v in row.items()
                   if isinstance(v, (int, float, str, type(None)))}
            rec.features_json = json.dumps(raw, default=str)
            db.session.add(rec)

            # accumulate ATM IV history — the raw material for IV Rank
            iv = row.get("atm_iv")
            if iv is not None and not pd.isna(iv):
                exists = IVHistory.query.filter_by(symbol=row["symbol"],
                                                   date=today).first()
                if exists:
                    exists.atm_iv = float(iv)
                else:
                    db.session.add(IVHistory(symbol=row["symbol"],
                                             market_type=market_type,
                                             date=today, atm_iv=float(iv)))

        run.status, run.finished_at = "done", datetime.now(timezone.utc)
        run.n_symbols = len(scored)
        db.session.commit()

        if notify and not scored.empty:
            alert_engine.process_scan(scored, market_type, scan_kind, context)

    progress(f"done — {len(scored)} symbols scored "
             f"(source: {scored['score_source'].iloc[0] if not scored.empty else 'n/a'})")
    return run_id
