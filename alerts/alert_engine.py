"""
Alert engine: after each scan, send one ranked digest (with V1.5 market
context header) plus individual alerts for flagged symbols above the score
floor. NEGATIVE_GAMMA_GAP and EVENT_DAY bypass the floor. Dedupe: the same
symbol+flag fires at most once per market per calendar day (UTC).
"""

from datetime import datetime, timezone

import pandas as pd

import config
from models import db, Alert
from scanner.volatility_score import FLAG_SEVERITY, FLAG_GUIDANCE, GATE_EXEMPT_FLAGS
from alerts import telegram_alert, discord_alert


def _row_dict(row) -> dict:
    d = {}
    for k in ("symbol", "spot", "gap_pct", "implied_move", "atm_iv",
              "iv_hv_ratio", "hv_ratio", "regime", "score", "rank", "flags",
              "funding_rate", "oi_change_pct",
              "iv_rank", "oi_buildup", "event_day", "score_source"):
        v = row.get(k)
        d[k] = None if (v is not None and not isinstance(v, (str, bool))
                        and pd.isna(v)) else v
    return d


def _already_sent_today(market_type: str, symbol: str, alert_type: str) -> bool:
    start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0,
                                               microsecond=0)
    return db.session.query(Alert.id).filter(
        Alert.market_type == market_type,
        Alert.symbol == symbol,
        Alert.alert_type == alert_type,
        Alert.created_at >= start,
    ).first() is not None


def process_scan(scored: pd.DataFrame, market_type: str, scan_kind: str,
                 context: dict | None = None):
    """Called inside app context by scan_engine after results are persisted."""
    top = scored.head(config.TOP_N_ALERTS)
    top_rows = [_row_dict(r) for _, r in top.iterrows()]

    # 1. ranked digest (once per market per scan)
    tg = telegram_alert.send_digest(top_rows, market_type, scan_kind, context)
    dc = discord_alert.send_digest(top_rows, market_type, scan_kind, context)
    db.session.add(Alert(
        market_type=market_type, symbol="*",
        alert_type=f"VOL_SCORE_TOP_{len(top_rows)}", severity="INFO",
        message="; ".join(f"{r['rank']}.{r['symbol']} {r['score']:.0f}"
                          for r in top_rows),
        score=top_rows[0]["score"] if top_rows else None,
        sent_telegram=tg, sent_discord=dc,
    ))

    # 2. individual flag alerts for names above the score floor
    for _, row in scored.iterrows():
        flags = (row.get("flags") or "").split(",")
        flags = [f for f in flags if f]
        if not flags:
            continue
        score = row.get("score") or 0
        if score < config.MIN_SCORE_FOR_ALERT and \
                not (GATE_EXEMPT_FLAGS & set(flags)):
            continue
        rd = _row_dict(row)
        for flag in flags:
            if _already_sent_today(market_type, row["symbol"], flag):
                continue
            alert = {
                "symbol": row["symbol"],
                "market_type": market_type,
                "alert_type": flag,
                "severity": FLAG_SEVERITY.get(flag, "MEDIUM"),
                "message": FLAG_GUIDANCE.get(flag, ""),
                "row": rd,
            }
            tg = telegram_alert.send_alert(alert)
            dc = discord_alert.send_alert(alert)
            db.session.add(Alert(
                market_type=market_type, symbol=row["symbol"],
                alert_type=flag, severity=alert["severity"],
                message=alert["message"], score=score,
                sent_telegram=tg, sent_discord=dc,
            ))
    db.session.commit()
