"""
Telegram sender — individual volatility alerts + ranked morning digest.
(Bot taken from quant_alert; extended with the V1.5 market-context header,
IV Rank and OI-buildup lines.)
"""

import requests

import config

API = "https://api.telegram.org/bot{token}/sendMessage"

SEVERITY_EMOJI = {"CRITICAL": "🚨", "HIGH": "⚠️", "MEDIUM": "📊", "INFO": "ℹ️"}
MARKET_EMOJI   = {"INDIA": "🇮🇳", "CRYPTO": "🪙", "US": "🇺🇸"}


def _send_text(text: str) -> bool:
    if not config.ENABLE_TELEGRAM:
        return False
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        print("[Telegram] Missing BOT_TOKEN or CHAT_ID")
        return False
    try:
        r = requests.post(API.format(token=config.TELEGRAM_BOT_TOKEN), json={
            "chat_id": config.TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": "Markdown",
        }, timeout=10)
        r.raise_for_status()
        return True
    except Exception as e:
        print(f"[Telegram] send error: {e}")
        return False


def _fv(v, digits=2, suffix=""):
    if v is None:
        return "N/A"
    try:
        return f"{float(v):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return str(v)


def _context_lines(context: dict | None) -> list[str]:
    """V1.5 market-context header for the digest."""
    if not context:
        return []
    lines = []
    vix, vname = context.get("vix"), context.get("vix_name")
    if vix is not None and vname:
        chg = context.get("vix_change_pct")
        chg_s = f" ({chg:+.1f}%)" if chg is not None else ""
        lines.append(f"{vname}: *{vix:.2f}*{chg_s}")
    gap = context.get("implied_gap_pct")
    if gap is not None and context.get("gap_source"):
        lines.append(f"{context['gap_source']}: *{gap:+.2f}%*")
    mult = context.get("score_multiplier")
    if mult and mult != 1.0:
        lines.append(f"⚡ risk context boost ×{mult:g} — "
                     + "; ".join(context.get("context_notes", [])))
    return lines


def send_alert(alert: dict) -> bool:
    """alert keys: symbol, market_type, alert_type, severity, message, row(dict)"""
    row = alert.get("row", {})
    em = SEVERITY_EMOJI.get(alert.get("severity", "MEDIUM"), "📌")
    mk = MARKET_EMOJI.get(alert.get("market_type", ""), "")
    lines = [
        f"{em} *{alert['alert_type'].replace('_', ' ')}* {mk} `{alert['symbol']}`",
        "",
        alert.get("message", ""),
        "",
        "```",
        f"Spot:         {_fv(row.get('spot'))}",
        f"Gap:          {_fv(row.get('gap_pct'), 2, '%')}",
        f"Implied Move: {_fv(row.get('implied_move'), 2, '%/day')}",
        f"ATM IV:       {_fv(row.get('atm_iv'), 1, '%')}",
        f"IV Rank:      {_fv(row.get('iv_rank'), 0) if row.get('iv_rank') is not None else 'building history'}",
        f"IV/HV:        {_fv(row.get('iv_hv_ratio'))}",
        f"HV5/HV20:     {_fv(row.get('hv_ratio'))}",
        f"Regime:       {row.get('regime') or 'N/A'}",
        f"OI Buildup:   {row.get('oi_buildup') or 'N/A'}",
        f"Vol Score:    {_fv(row.get('score'), 1)}/100 ({row.get('score_source') or 'v1'})",
        "```",
    ]
    if row.get("event_day"):
        lines.insert(3, "📅 *Earnings today / imminent*")
    return _send_text("\n".join(lines))


def send_digest(scored_rows: list[dict], market_type: str, scan_kind: str,
                context: dict | None = None) -> bool:
    """Ranked top-N table as a single message, with market-context header."""
    mk = MARKET_EMOJI.get(market_type, "")
    title = {"premarket": "PRE-MARKET VOLATILITY SCAN",
             "postopen": "POST-OPEN RESCAN",
             "intraday": "VOLATILITY SCAN"}.get(scan_kind, "VOLATILITY SCAN")
    lines = [
        f"📋 *{title}* {mk} {market_type}",
        f"_Top {len(scored_rows)} by expected-volatility score_",
    ]
    ctx = _context_lines(context)
    if ctx:
        lines += ["", *ctx]
    lines += [
        "",
        "```",
        f"{'#':>2} {'SYMBOL':<12} {'SCORE':>5} {'GAP%':>6} {'IMPL%':>6} {'IVR':>3} FLAGS",
    ]
    for r in scored_rows:
        gap = f"{r['gap_pct']:+.1f}" if r.get("gap_pct") is not None else "  n/a"
        im  = f"{r['implied_move']:.1f}" if r.get("implied_move") is not None else "  n/a"
        ivr = f"{r['iv_rank']:.0f}" if r.get("iv_rank") is not None else "  –"
        flags = (r.get("flags") or "").replace("_", "")[:22]
        lines.append(f"{r['rank']:>2} {r['symbol']:<12} {r['score']:>5.1f} "
                     f"{gap:>6} {im:>6} {ivr:>3} {flags}")
    lines += ["```", "",
              "_Risk guidance: high-score names → expect wide ranges; "
              "size down, use wider stops, avoid market orders at the open._"]
    return _send_text("\n".join(lines))
