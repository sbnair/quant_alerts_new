"""
Volatility Scanner (V1 + V1.5 + V2) — Flask UI + JSON API.

Pages:   /            dashboard (ranked table per market, latest run + context)
         /alerts      alert feed
API:     /api/scan/<market>            latest scored results
         /api/scan/<market>/run  POST  trigger manual scan
         /api/alerts                   recent alerts
         /api/status                   scheduler + run + ML model status
         /api/context/<market>         live V1.5 market context
         /api/ml/train           POST  backfill labels + retrain LightGBM
"""

import json
import os
from datetime import datetime, timezone

from flask import Flask, jsonify, render_template, request

import config
import scheduler
from models import db, ScanRun, ScanResult, Alert, \
    MARKET_INDIA, MARKET_CRYPTO, MARKET_US
from scanner import ml_model
from data import market_context

VALID_MARKETS = {MARKET_INDIA, MARKET_CRYPTO, MARKET_US}

app = Flask(__name__)
app.config["SECRET_KEY"] = config.FLASK_SECRET_KEY
os.makedirs("instance", exist_ok=True)
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{config.DB_PATH}"
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {"connect_args": {"timeout": 30}}

db.init_app(app)
with app.app_context():
    db.create_all()


def _latest_run(market_type: str) -> ScanRun | None:
    return (ScanRun.query.filter_by(market_type=market_type, status="done")
            .order_by(ScanRun.id.desc()).first())


def _run_results(run: ScanRun) -> list[dict]:
    rows = (ScanResult.query.filter_by(run_id=run.id)
            .order_by(ScanResult.rank).all())
    out = []
    for r in rows:
        out.append({
            "rank": r.rank, "symbol": r.symbol, "score": r.score,
            "score_v1": r.score_v1, "score_ml": r.score_ml,
            "score_source": r.score_source,
            "spot": r.spot, "gap_pct": r.gap_pct,
            "implied_move": r.implied_move, "atm_iv": r.atm_iv,
            "iv_rank": r.iv_rank, "iv_percentile": r.iv_percentile,
            "iv_hv_ratio": r.iv_hv_ratio, "hv_ratio": r.hv_ratio,
            "pcr": r.pcr, "oi_change_pct": r.oi_change_pct,
            "oi_buildup": r.oi_buildup, "event_day": r.event_day,
            "regime": r.regime, "funding_rate": r.funding_rate,
            "flags": r.flags or "",
        })
    return out


def _run_context(run: ScanRun | None) -> dict:
    if not run or not run.context_json:
        return {}
    try:
        return json.loads(run.context_json)
    except json.JSONDecodeError:
        return {}


# ── Pages ─────────────────────────────────────────────────────────────────────

@app.route("/")
def dashboard():
    market = request.args.get("market", MARKET_INDIA).upper()
    if market not in VALID_MARKETS:
        market = MARKET_INDIA
    run = _latest_run(market)
    results = _run_results(run) if run else []
    return render_template("dashboard.html",
                           market=market, markets=sorted(VALID_MARKETS),
                           run=run, results=results,
                           context=_run_context(run),
                           model=ml_model.model_info(),
                           scanning=scheduler.is_running(market))


@app.route("/alerts")
def alerts_page():
    alerts = Alert.query.order_by(Alert.id.desc()).limit(200).all()
    return render_template("alerts.html", alerts=alerts,
                           markets=sorted(VALID_MARKETS))


# ── API ───────────────────────────────────────────────────────────────────────

@app.route("/api/scan/<market>")
def api_scan(market):
    market = market.upper()
    if market not in VALID_MARKETS:
        return jsonify({"error": "unknown market"}), 404
    run = _latest_run(market)
    if not run:
        return jsonify({"run": None, "results": [],
                        "scanning": scheduler.is_running(market)})
    return jsonify({
        "run": {"id": run.id, "kind": run.scan_kind,
                "finished_at": run.finished_at.isoformat() if run.finished_at else None,
                "n_symbols": run.n_symbols,
                "context": _run_context(run)},
        "results": _run_results(run),
        "scanning": scheduler.is_running(market),
    })


@app.route("/api/scan/<market>/run", methods=["POST"])
def api_scan_run(market):
    market = market.upper()
    if market not in VALID_MARKETS:
        return jsonify({"error": "unknown market"}), 404
    notify = request.args.get("notify", "true").lower() != "false"
    started = scheduler.scan_async(app, market, "manual", notify=notify)
    return jsonify({"started": started,
                    "already_running": not started}), (202 if started else 409)


@app.route("/api/alerts")
def api_alerts():
    q = Alert.query.order_by(Alert.id.desc())
    market = request.args.get("market")
    if market:
        q = q.filter_by(market_type=market.upper())
    return jsonify([a.to_dict() for a in q.limit(100).all()])


@app.route("/api/context/<market>")
def api_context(market):
    market = market.upper()
    if market not in VALID_MARKETS:
        return jsonify({"error": "unknown market"}), 404
    return jsonify(market_context.get_context(market))


@app.route("/api/ml/train", methods=["POST"])
def api_ml_train():
    return jsonify(ml_model.train(app))


@app.route("/api/status")
def api_status():
    return jsonify({
        "now": datetime.now(timezone.utc).isoformat(),
        "jobs": scheduler.jobs_info(),
        "scanning": {m: scheduler.is_running(m) for m in VALID_MARKETS},
        "channels": {"telegram": bool(config.TELEGRAM_BOT_TOKEN and config.ENABLE_TELEGRAM),
                     "discord": bool(config.DISCORD_WEBHOOK_URL and config.ENABLE_DISCORD)},
        "ml_model": ml_model.model_info(),
    })


if __name__ == "__main__":
    scheduler.start(app)
    app.run(host="0.0.0.0", port=config.FLASK_PORT, debug=config.FLASK_DEBUG,
            use_reloader=False)
