"""
Rate-limited Dhan API wrapper.
Dhan's option-chain endpoint allows ~1 request / 3 s — every call through
this module is paced globally so a universe scan never trips the limit.
"""

import threading
import time

from dhanhq import dhanhq as DhanHQ
try:
    from dhanhq import DhanContext
except ImportError:
    DhanContext = None

import config

_dhan = None
_lock = threading.Lock()
_last_call = 0.0


def client() -> DhanHQ:
    global _dhan
    if _dhan is None:
        if DhanContext is not None:
            _dhan = DhanHQ(DhanContext(config.DHAN_CLIENT_ID, config.DHAN_ACCESS_TOKEN))
        else:
            _dhan = DhanHQ(config.DHAN_CLIENT_ID, config.DHAN_ACCESS_TOKEN)
    return _dhan


def _paced(fn, *args, **kwargs):
    """Serialize + pace all Dhan calls across threads."""
    global _last_call
    with _lock:
        wait = config.DHAN_REQUEST_GAP_SEC - (time.time() - _last_call)
        if wait > 0:
            time.sleep(wait)
        try:
            return fn(*args, **kwargs)
        finally:
            _last_call = time.time()


def expiry_list(security_id: int, segment: str) -> list[str]:
    r = _paced(client().expiry_list,
               under_security_id=security_id, under_exchange_segment=segment)
    if r.get("status") != "success":
        raise ValueError(f"Dhan expiry_list failed: {r.get('remarks', r)}")
    return sorted(r["data"]["data"])


def option_chain(security_id: int, segment: str, expiry: str) -> dict:
    """Returns {"last_price": float, "oc": {strike: {"ce": {...}, "pe": {...}}}}"""
    r = _paced(client().option_chain,
               under_security_id=security_id, under_exchange_segment=segment,
               expiry=expiry)
    if r.get("status") != "success":
        raise ValueError(f"Dhan option_chain failed: {r.get('remarks', r)}")
    return r["data"]["data"]
