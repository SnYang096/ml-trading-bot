"""Central Telegram notifications for mlbot monitoring and live ops.

All Python paths share ``send_telegram_message`` + creds from
``GRAFANA_ALERT_TELEGRAM_*`` / ``/opt/quant-engine/monitoring/.env``.

Consumers:
  - ``scheduler.post_run_hooks`` → ``notify_cadence_result`` (watchdog/drift ALERT)
  - ``staleness_check`` → ``notify_stale_cadences`` (缺勤)
  - ``rebalance_cockpit_run`` → rebalance NAV band alerts
  - ``account_telegram_watch`` → equity move + new exchange positions
  - ``scripts/monitoring/monitor_telegram_notify.sh`` (systemd OnFailure, curl)
  - ``deploy/monitoring/scripts/quant_telegram_notify.sh`` (infra units, curl)
  - Grafana Unified Alerting → ``telegram-quant-ops`` contact point
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_file() -> Path:
    raw = os.environ.get("MLBOT_MONITORING_ENV") or os.environ.get(
        "QUANT_MONITORING_ENV", "/opt/quant-engine/monitoring/.env"
    )
    return Path(raw)


def _load_telegram_creds() -> tuple[str, str]:
    env_path = _env_file()
    token = os.environ.get("GRAFANA_ALERT_TELEGRAM_BOT_TOKEN", "")
    chat = os.environ.get("GRAFANA_ALERT_TELEGRAM_CHAT_ID", "")
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip("'\"")
            if k == "GRAFANA_ALERT_TELEGRAM_BOT_TOKEN" and v:
                token = v
            if k == "GRAFANA_ALERT_TELEGRAM_CHAT_ID" and v:
                chat = v
    return token, chat


def _cooldown_ok(stamp_key: str, cooldown_sec: int) -> bool:
    stamp_dir = Path(os.environ.get("MLBOT_TG_NOTIFY_STAMP_DIR", "/tmp"))
    stamp = stamp_dir / f"mlbot_monitor_py_{stamp_key.replace('/', '_')}.stamp"
    if stamp.is_file():
        age = time.time() - stamp.stat().st_mtime
        if age < cooldown_sec:
            return False
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.touch()
    return True


def send_telegram_message(
    text: str,
    *,
    stamp_key: str = "default",
    cooldown_sec: Optional[int] = None,
    token: Optional[str] = None,
    chat_id: Optional[str] = None,
    skip_cooldown: bool = False,
) -> bool:
    env_token, env_chat = _load_telegram_creds()
    token = (token or env_token or "").strip()
    chat = (chat_id or env_chat or "").strip()
    if not token or not chat:
        print("monitor_telegram: not configured", flush=True)
        return False
    cd = int(
        cooldown_sec
        if cooldown_sec is not None
        else os.environ.get("MLBOT_TG_NOTIFY_COOLDOWN_SEC", "600")
    )
    if not skip_cooldown and not _cooldown_ok(stamp_key, cd):
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode(
        {
            "chat_id": chat,
            "text": text,
            "disable_web_page_preview": "true",
        }
    ).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=15) as resp:
        resp.read()
    print(f"monitor_telegram: sent ({stamp_key})", flush=True)
    return True


def _alert_events_for_run(
    registry_db: Path,
    *,
    cadence: str,
    run_ts: str,
) -> List[Dict[str, str]]:
    if not registry_db.is_file():
        return []
    conn = sqlite3.connect(registry_db)
    try:
        cur = conn.execute(
            """
            SELECT source, strategy, status, detail_json
            FROM monitor_event
            WHERE cadence = ? AND run_ts = ? AND status = 'ALERT'
            ORDER BY source, strategy
            """,
            (cadence, run_ts),
        )
        rows = []
        for source, strategy, status, detail in cur.fetchall():
            rows.append(
                {
                    "source": str(source),
                    "strategy": str(strategy),
                    "status": str(status),
                    "detail": str(detail)[:200],
                }
            )
        return rows
    finally:
        conn.close()


def format_alert_message(
    *,
    cadence: str,
    card: Dict[str, Any],
    alert_events: List[Dict[str, str]],
    host: Optional[str] = None,
) -> str:
    import socket

    host = host or socket.gethostname()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        f"⚠️ mlbot monitor {cadence} {card.get('display_status', 'ALERT')} on {host}",
        f"time: {ts}",
        f"run_ts: {card.get('run_ts')}",
        f"exit: {card.get('exit_code')}",
    ]
    if card.get("watchdog_any_alert"):
        lines.append("watchdog: ALERT")
    if card.get("drift_any_alert"):
        lines.append("drift: ALERT")
    if card.get("output_dir"):
        lines.append(f"out: {card['output_dir']}")
    if card.get("missed"):
        age = card.get("age_hours")
        max_h = card.get("max_age_hours")
        lines.append(f"staleness: {age:.1f}h > {max_h:.0f}h" if age is not None else "staleness: no run")
    if alert_events:
        lines.append("strategies:")
        for ev in alert_events[:12]:
            lines.append(f"  - {ev['source']}/{ev['strategy']}: {ev['status']}")
        if len(alert_events) > 12:
            lines.append(f"  … +{len(alert_events) - 12} more")
    return "\n".join(lines)


def should_notify_cadence_result(
    *,
    exit_code: int,
    index_row: Dict[str, Any],
) -> bool:
    status = str(index_row.get("status") or "")
    business_alert = status == "ALERT" or bool(
        index_row.get("watchdog_any_alert") or index_row.get("drift_any_alert")
    )
    return exit_code != 0 or business_alert


def notify_cadence_result(
    *,
    cadence: str,
    exit_code: int,
    index_row: Dict[str, Any],
    registry_db: Path,
    force: bool = False,
) -> bool:
    """Send TG when run failed or business ALERT (not only systemd crash)."""
    if not force and not should_notify_cadence_result(
        exit_code=exit_code, index_row=index_row
    ):
        return False
    status = str(index_row.get("status") or "")
    business_alert = status == "ALERT" or bool(
        index_row.get("watchdog_any_alert") or index_row.get("drift_any_alert")
    )

    run_ts = str(index_row.get("run_ts") or "")
    events = _alert_events_for_run(registry_db, cadence=cadence, run_ts=run_ts)
    from src.monitoring.staleness import evaluate_cadence_health

    card = evaluate_cadence_health(cadence, index_row, max_age_hours=1e9)
    card["display_status"] = "ALERT" if business_alert or exit_code else status
    msg = format_alert_message(cadence=cadence, card=card, alert_events=events)
    return send_telegram_message(msg, stamp_key=f"alert:{cadence}:{run_ts}")


def format_account_equity_change_message(
    *,
    scope: str,
    anchor: float,
    current: float,
    threshold_pct: float,
) -> Optional[str]:
    if anchor <= 0.0 or current <= 0.0:
        return None
    delta = current - anchor
    pct = delta / anchor
    if abs(pct) < threshold_pct:
        return None
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    direction = "↑" if pct > 0 else "↓"
    return (
        f"📊 Multi-leg 账户权益变动 {direction}\n"
        f"scope: {scope}\n"
        f"time: {ts}\n"
        f"anchor: {anchor:,.2f} USDT\n"
        f"now: {current:,.2f} USDT\n"
        f"Δ: {delta:+,.2f} USDT ({pct:+.2%})\n"
        f"threshold: {threshold_pct:.1%}"
    )


def format_account_open_position_message(*, scope: str, keys: List[str]) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    legs = ", ".join(_format_position_key(k) for k in keys)
    return (
        f"📈 Multi-leg 新开仓\n"
        f"scope: {scope}\n"
        f"time: {ts}\n"
        f"legs: {legs}"
    )


def _format_position_key(key: str) -> str:
    sym, _, side = key.partition(":")
    return f"{sym} {side.upper()}"


def send_account_watch_alerts(
    messages: List[str],
    *,
    scope: str = "multi_leg",
    force_notify: bool = False,
) -> int:
    """Send equity / open-position messages; returns count sent."""
    sent = 0
    for msg in messages:
        kind = "equity" if "权益" in msg else "open"
        ok = send_telegram_message(
            msg,
            stamp_key=f"acct:{scope}:{kind}",
            cooldown_sec=300 if not force_notify else 0,
            skip_cooldown=force_notify,
        )
        if ok:
            sent += 1
    return sent


def notify_stale_cadences(
    stale_cards: List[Dict[str, Any]],
) -> bool:
    if not stale_cards:
        return False
    import socket

    host = socket.gethostname()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [f"⚠️ mlbot monitor 缺勤 on {host}", f"time: {ts}", "overdue:"]
    for c in stale_cards:
        age = c.get("age_hours")
        max_h = c.get("max_age_hours")
        if age is None:
            lines.append(f"  - {c['cadence']}: never run")
        else:
            lines.append(f"  - {c['cadence']}: {age:.0f}h ago (limit {max_h:.0f}h)")
    return send_telegram_message("\n".join(lines), stamp_key="staleness")
