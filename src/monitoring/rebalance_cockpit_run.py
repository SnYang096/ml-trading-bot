"""Scheduled rebalance cockpit check: persist monitor_event + optional Telegram."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from mlbot_console.config import ConsoleSettings
from mlbot_console.services.rebalance_advisor import build_regime_cockpit
from src.monitoring.store import (
    DEFAULT_REGISTRY_DB,
    init_registry_db,
    update_monitoring_index,
)
from src.monitoring.telegram import send_telegram_message

CADENCE = "rebalance_4h"
SOURCE = "rebalance_cockpit"
STRATEGY = "_portfolio"


def _run_ts_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _monitor_status(alert: str) -> str:
    if alert in {"WATCH", "REBALANCE_SUGGEST"}:
        return "ALERT"
    return "OK"


def _exit_code(alert: str) -> int:
    if alert == "REBALANCE_SUGGEST":
        return 2
    if alert == "WATCH":
        return 1
    return 0


def _compact_detail(payload: Dict[str, Any]) -> Dict[str, Any]:
    alloc = payload.get("allocation") or {}
    scopes = []
    for row in alloc.get("scopes") or []:
        if not isinstance(row, dict):
            continue
        scopes.append(
            {
                "scope": row.get("scope"),
                "label": row.get("label"),
                "nav_pct": row.get("nav_pct"),
                "status": row.get("status"),
                "target": (row.get("band") or {}).get("target"),
            }
        )
    return {
        "alert": alloc.get("alert"),
        "composite": payload.get("composite"),
        "total_nav_usdt": alloc.get("total_nav_usdt"),
        "scopes": scopes,
        "suggestions": alloc.get("suggestions") or [],
        "feature_bus": payload.get("feature_bus"),
        "symbol": payload.get("symbol"),
        "as_of": payload.get("as_of"),
    }


def _write_artifacts(
    output_dir: Path,
    *,
    payload: Dict[str, Any],
    alert: str,
    run_ts: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    cockpit_path = output_dir / "cockpit.json"
    cockpit_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    hb = {
        "task": SOURCE,
        "cadence": CADENCE,
        "status": _monitor_status(alert),
        "alert": alert,
        "run_ts": run_ts,
        "composite": (payload.get("composite") or {}).get("label"),
        "output_dir": str(output_dir),
        "cockpit_path": str(cockpit_path),
    }
    (output_dir / "heartbeat.json").write_text(
        json.dumps(hb, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return cockpit_path


def upsert_rebalance_monitor_event(
    *,
    run_ts: str,
    status: str,
    detail: Dict[str, Any],
    output_dir: Path,
    cockpit_path: Path,
    db_path: Optional[Path] = None,
) -> None:
    db = init_registry_db(db_path or DEFAULT_REGISTRY_DB)
    now = datetime.now(timezone.utc).isoformat()
    eid = f"{CADENCE}:{run_ts}:{SOURCE}:{STRATEGY}"
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO monitor_event
            (id, cadence, source, strategy, status, detail_json, report_path, run_ts, output_dir, ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                eid,
                CADENCE,
                SOURCE,
                STRATEGY,
                status,
                json.dumps(detail, ensure_ascii=False),
                str(cockpit_path),
                run_ts,
                str(output_dir),
                now,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def format_rebalance_telegram_message(
    *,
    payload: Dict[str, Any],
    alert: str,
    run_ts: str,
    host: Optional[str] = None,
) -> str:
    import socket

    host = host or socket.gethostname()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    alloc = payload.get("allocation") or {}
    composite = (payload.get("composite") or {}).get("label_title") or (
        payload.get("composite") or {}
    ).get("label")
    lines = [
        f"⚠️ Regime Cockpit {alert} on {host}",
        f"time: {ts}",
        f"run_ts: {run_ts}",
        f"symbol: {payload.get('symbol')}",
        f"composite: {composite}",
        f"total_nav: {alloc.get('total_nav_usdt')}",
    ]
    if payload.get("feature_bus", {}).get("stale"):
        lines.append("feature_bus: STALE")
    for row in alloc.get("scopes") or []:
        if not isinstance(row, dict):
            continue
        st = row.get("status")
        if st and st != "OK":
            nav = row.get("nav_pct")
            nav_s = f"{float(nav):.0%}" if nav is not None else "—"
            lines.append(f"  {row.get('label')}: {nav_s} [{st}]")
    for sug in (alloc.get("suggestions") or [])[:4]:
        lines.append(f"→ {sug}")
    return "\n".join(lines)


def notify_rebalance_if_needed(
    *,
    payload: Dict[str, Any],
    alert: str,
    run_ts: str,
    dry_run: bool = False,
) -> bool:
    if alert == "OK":
        return False
    msg = format_rebalance_telegram_message(
        payload=payload, alert=alert, run_ts=run_ts
    )
    if dry_run:
        print(msg, flush=True)
        return True
    return send_telegram_message(msg, stamp_key=f"rebalance:cockpit:{alert}")


def run_rebalance_cockpit_check(
    *,
    settings: Optional[ConsoleSettings] = None,
    symbol: str = "BTCUSDT",
    window_days: int = 7,
    run_ts: Optional[str] = None,
    output_root: Optional[Path] = None,
    registry_db: Optional[Path] = None,
    dry_run: bool = False,
    skip_telegram: bool = False,
    skip_index: bool = False,
) -> Dict[str, Any]:
    """Build cockpit, write artifacts, index CMS, optional TG. Returns summary dict."""
    cfg = settings or ConsoleSettings.from_env()
    ts = run_ts or _run_ts_now()
    root = output_root or (cfg.repo_root / "results" / "monitoring" / CADENCE)
    output_dir = Path(root) / ts

    payload = build_regime_cockpit(
        strategies_root=cfg.strategies_root,
        project_root=cfg.repo_root,
        feature_bus_root=cfg.feature_bus_root,
        symbol=symbol,
        window_days=window_days,
    )
    alert = str((payload.get("allocation") or {}).get("alert") or "OK")
    status = _monitor_status(alert)
    exit_code = _exit_code(alert)
    detail = _compact_detail(payload)

    if dry_run:
        return {
            "dry_run": True,
            "run_ts": ts,
            "alert": alert,
            "status": status,
            "exit_code": exit_code,
            "detail": detail,
            "output_dir": str(output_dir),
        }

    cockpit_path = _write_artifacts(output_dir, payload=payload, alert=alert, run_ts=ts)
    upsert_rebalance_monitor_event(
        run_ts=ts,
        status=status,
        detail=detail,
        output_dir=output_dir,
        cockpit_path=cockpit_path,
        db_path=registry_db,
    )
    if not skip_index:
        update_monitoring_index(
            cadence=CADENCE,
            run_ts=ts,
            exit_code=exit_code,
            output_dir=output_dir,
            manifest_path="scripts/monitoring/rebalance_cockpit_check.py",
            index_path=cfg.repo_root / "results" / "monitoring" / "index.json",
        )
    tg_sent = False
    if not skip_telegram:
        tg_sent = notify_rebalance_if_needed(
            payload=payload, alert=alert, run_ts=ts, dry_run=False
        )

    return {
        "run_ts": ts,
        "alert": alert,
        "status": status,
        "exit_code": exit_code,
        "output_dir": str(output_dir),
        "cockpit_path": str(cockpit_path),
        "telegram_sent": tg_sent,
        "detail": detail,
    }


def load_latest_rebalance_event(
    registry_db: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    db = registry_db or DEFAULT_REGISTRY_DB
    if not Path(db).is_file():
        return None
    conn = sqlite3.connect(db)
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            """
            SELECT cadence, source, strategy, status, detail_json, report_path,
                   run_ts, output_dir, ts
            FROM monitor_event
            WHERE cadence = ? AND source = ?
            ORDER BY ts DESC LIMIT 1
            """,
            (CADENCE, SOURCE),
        )
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()
