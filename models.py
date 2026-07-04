"""
Database models for the Volatility Scanner (V1 + V1.5 + V2 columns).
"""

from datetime import datetime, timezone

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

MARKET_INDIA  = "INDIA"
MARKET_CRYPTO = "CRYPTO"
MARKET_US     = "US"


def utcnow():
    return datetime.now(timezone.utc)


class ScanRun(db.Model):
    """One scan pass across a market's universe."""
    __tablename__ = "scan_runs"

    id          = db.Column(db.Integer, primary_key=True)
    market_type = db.Column(db.String(10), nullable=False, index=True)
    scan_kind   = db.Column(db.String(20), nullable=False)  # premarket | postopen | intraday | manual
    started_at  = db.Column(db.DateTime, default=utcnow)
    finished_at = db.Column(db.DateTime)
    n_symbols   = db.Column(db.Integer, default=0)
    n_errors    = db.Column(db.Integer, default=0)
    status      = db.Column(db.String(15), default="running")  # running | done | failed
    context_json = db.Column(db.Text)                          # V1.5 market-level context snapshot

    results = db.relationship("ScanResult", backref="run", lazy="dynamic")


class ScanResult(db.Model):
    """Per-symbol feature snapshot + composite score for one scan run."""
    __tablename__ = "scan_results"

    id           = db.Column(db.Integer, primary_key=True)
    run_id       = db.Column(db.Integer, db.ForeignKey("scan_runs.id"), index=True)
    market_type  = db.Column(db.String(10), nullable=False, index=True)
    symbol       = db.Column(db.String(30), nullable=False, index=True)
    scanned_at   = db.Column(db.DateTime, default=utcnow)

    spot         = db.Column(db.Float)
    prev_close   = db.Column(db.Float)
    gap_pct      = db.Column(db.Float)          # (spot - prev_close)/prev_close * 100
    implied_move = db.Column(db.Float)          # straddle-implied 1-day move %
    atm_iv       = db.Column(db.Float)          # ATM IV %
    iv_hv_ratio  = db.Column(db.Float)          # ATM IV vs 20d realized vol
    hv_short     = db.Column(db.Float)          # 5d realized vol (annualized %)
    hv_long      = db.Column(db.Float)          # 20d realized vol (annualized %)
    hv_ratio     = db.Column(db.Float)          # hv_short / hv_long
    atr_pct      = db.Column(db.Float)          # 14d ATR as % of close
    dist_52w_pct = db.Column(db.Float)          # distance from nearer 52w extreme, %
    pcr          = db.Column(db.Float)          # put-call OI ratio
    oi_total     = db.Column(db.Float)
    oi_change_pct = db.Column(db.Float)         # overnight (India) / 24h (crypto)
    regime       = db.Column(db.String(20))     # Positive/Negative Gamma (or None)
    gamma_flip   = db.Column(db.Float)
    funding_rate = db.Column(db.Float)          # crypto only

    # V1.5
    oi_buildup   = db.Column(db.String(20))     # Long/Short Buildup, Short Covering, Long Unwinding
    event_day    = db.Column(db.Boolean, default=False)  # earnings today / imminent

    # V2
    iv_rank      = db.Column(db.Float)          # 0–100 vs own IV history (min/max)
    iv_percentile = db.Column(db.Float)         # 0–100 % of history days below current IV
    score_v1     = db.Column(db.Float)          # rule-based percentile score
    score_ml     = db.Column(db.Float)          # LightGBM-derived score (null until model exists)
    score_source = db.Column(db.String(10))     # v1 | ml | blend

    features_json = db.Column(db.Text)          # full raw snapshot for model training

    score        = db.Column(db.Float, index=True)  # final composite 0–100 (context-adjusted)
    rank         = db.Column(db.Integer)
    flags        = db.Column(db.String(200))    # comma-separated triggered flags


class Alert(db.Model):
    __tablename__ = "alerts"

    id          = db.Column(db.Integer, primary_key=True)
    created_at  = db.Column(db.DateTime, default=utcnow, index=True)
    market_type = db.Column(db.String(10), nullable=False, index=True)
    symbol      = db.Column(db.String(30), nullable=False, index=True)
    alert_type  = db.Column(db.String(40), nullable=False)
    severity    = db.Column(db.String(10), default="MEDIUM")
    message     = db.Column(db.Text)
    score       = db.Column(db.Float)
    sent_telegram = db.Column(db.Boolean, default=False)
    sent_discord  = db.Column(db.Boolean, default=False)

    def to_dict(self):
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "market_type": self.market_type,
            "symbol": self.symbol,
            "alert_type": self.alert_type,
            "severity": self.severity,
            "message": self.message,
            "score": self.score,
        }


class IVHistory(db.Model):
    """Daily ATM IV log per symbol — the raw material for the V2 IV Rank."""
    __tablename__ = "iv_history"

    id      = db.Column(db.Integer, primary_key=True)
    symbol  = db.Column(db.String(30), nullable=False, index=True)
    market_type = db.Column(db.String(10), nullable=False)
    date    = db.Column(db.Date, nullable=False, index=True)
    atm_iv  = db.Column(db.Float, nullable=False)

    __table_args__ = (db.UniqueConstraint("symbol", "date", name="uq_iv_symbol_date"),)


class MLLabel(db.Model):
    """V2 training label: realized next-day range for a scan_result row.
    Backfilled by train_model.py / the nightly label job once the following
    session's OHLC is available."""
    __tablename__ = "ml_labels"

    id             = db.Column(db.Integer, primary_key=True)
    result_id      = db.Column(db.Integer, db.ForeignKey("scan_results.id"),
                               unique=True, index=True)
    market_type    = db.Column(db.String(10), index=True)
    symbol         = db.Column(db.String(30), index=True)
    scan_date      = db.Column(db.Date, index=True)
    next_range_pct = db.Column(db.Float)   # next session (high-low)/prev_close * 100
    next_absret_pct = db.Column(db.Float)  # |next close/close - 1| * 100
