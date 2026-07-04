"""
APScheduler jobs:
  INDIA  — pre-market scan 09:07 IST and post-open rescan 09:20 IST (Mon–Fri)
  US     — pre-market scan 09:10 America/New_York (Mon–Fri)
  CRYPTO — every CRYPTO_SCAN_INTERVAL_MIN minutes, 24/7
  ML     — V2: nightly label backfill (22:30 UTC) + weekly retrain
           (Sunday 02:00 UTC) when ML_AUTO_RETRAIN is on.
Manual scans from the UI run through the same run_scan entry point.
"""

import threading

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

import config
from models import MARKET_INDIA, MARKET_CRYPTO, MARKET_US
from scanner.scan_engine import run_scan
from scanner import ml_model

_scheduler: BackgroundScheduler | None = None
_running: dict[str, bool] = {}
_lock = threading.Lock()


def is_running(market_type: str) -> bool:
    return _running.get(market_type, False)


def scan_async(app, market_type: str, scan_kind: str = "manual",
               notify: bool = True) -> bool:
    """Kick off a scan in a worker thread; refuse if one is already running."""
    with _lock:
        if _running.get(market_type):
            return False
        _running[market_type] = True

    def work():
        try:
            run_scan(app, market_type, scan_kind, notify=notify)
        finally:
            with _lock:
                _running[market_type] = False

    threading.Thread(target=work, daemon=True).start()
    return True


def start(app):
    global _scheduler
    if _scheduler is not None:
        return _scheduler

    sched = BackgroundScheduler()

    sched.add_job(lambda: scan_async(app, MARKET_INDIA, "premarket"),
                  CronTrigger(day_of_week="mon-fri", hour=9, minute=7,
                              timezone=config.IST_TZ),
                  id="india_premarket", name="India pre-market scan")
    sched.add_job(lambda: scan_async(app, MARKET_INDIA, "postopen"),
                  CronTrigger(day_of_week="mon-fri", hour=9, minute=20,
                              timezone=config.IST_TZ),
                  id="india_postopen", name="India post-open rescan")
    sched.add_job(lambda: scan_async(app, MARKET_US, "premarket"),
                  CronTrigger(day_of_week="mon-fri", hour=9, minute=10,
                              timezone="America/New_York"),
                  id="us_premarket", name="US pre-market scan")
    sched.add_job(lambda: scan_async(app, MARKET_CRYPTO, "intraday"),
                  IntervalTrigger(minutes=config.CRYPTO_SCAN_INTERVAL_MIN),
                  id="crypto_interval", name="Crypto interval scan")

    if config.ML_AUTO_RETRAIN:
        # label yesterday's rows once all three sessions have closed
        sched.add_job(lambda: ml_model.backfill_labels(app),
                      CronTrigger(hour=22, minute=30, timezone="UTC"),
                      id="ml_labels", name="V2 label backfill")
        sched.add_job(lambda: ml_model.train(app),
                      CronTrigger(day_of_week="sun", hour=2, minute=0,
                                  timezone="UTC"),
                      id="ml_retrain", name="V2 weekly LightGBM retrain")

    sched.start()
    _scheduler = sched
    print(f"[SCHEDULER] started — India 09:07/09:20 IST, US 09:10 ET, "
          f"crypto every {config.CRYPTO_SCAN_INTERVAL_MIN}m, "
          f"ML retrain {'on (Sun 02:00 UTC)' if config.ML_AUTO_RETRAIN else 'off'}")
    return sched


def jobs_info() -> list[dict]:
    if _scheduler is None:
        return []
    return [{"id": j.id, "name": j.name,
             "next_run": str(j.next_run_time) if j.next_run_time else None}
            for j in _scheduler.get_jobs()]
