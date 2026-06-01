"""CMS dashboard aggregation (cards + strategy alerts)."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.monitoring.staleness import build_cadence_cards, list_stale_cadences
from src.monitoring.store import load_monitoring_index, load_schedules


def load_monitor_events(
    registry_db: Path,
    *,
    cadence: Optional[str] = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    if not registry_db.is_file():
        return []
    sql = (
        "SELECT cadence, source, strategy, status, report_path, run_ts, output_dir, ts "
        "FROM monitor_event"
    )
    params: list[Any] = []
    if cadence:
        sql += " WHERE cadence = ?"
        params.append(cadence)
    sql += " ORDER BY ts DESC LIMIT ?"
    params.append(int(limit))
    conn = sqlite3.connect(registry_db)
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(sql, tuple(params))
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def strategy_alerts_by_cadence(
    events: List[Dict[str, Any]],
    cards: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, str]]]:
    """Latest run_ts per cadence → ALERT strategy rows."""
    run_ts_by: Dict[str, str] = {}
    for c in cards:
        if c.get("run_ts"):
            run_ts_by[str(c["cadence"])] = str(c["run_ts"])
    out: Dict[str, List[Dict[str, str]]] = {k: [] for k in run_ts_by}
    for ev in events:
        cad = str(ev.get("cadence") or "")
        if cad not in run_ts_by or str(ev.get("run_ts")) != run_ts_by[cad]:
            continue
        if str(ev.get("status")) != "ALERT":
            continue
        out[cad].append(
            {
                "source": str(ev.get("source") or ""),
                "strategy": str(ev.get("strategy") or ""),
            }
        )
    return out


def build_monitoring_dashboard(
    repo_root: Path,
    registry_db: Path,
    *,
    schedules_path: Optional[Path] = None,
) -> Dict[str, Any]:
    sched_path = schedules_path or (repo_root / "config" / "monitoring" / "schedules.yaml")
    schedules_cfg = load_schedules(sched_path) if sched_path.is_file() else {}
    index = load_monitoring_index(repo_root)
    events = load_monitor_events(registry_db, limit=300)
    cards = build_cadence_cards(index, schedules_cfg)
    stale = list_stale_cadences(index, schedules_cfg)
    alerts = strategy_alerts_by_cadence(events, cards)
    return {
        "index_updated_at": index.get("updated_at"),
        "cards": cards,
        "stale_cadences": [c["cadence"] for c in stale],
        "strategy_alerts": alerts,
        "summary": {
            "any_alert": any(c.get("display_status") == "ALERT" for c in cards),
            "any_missed": any(c.get("display_status") == "MISSED" for c in cards),
            "n_cards": len(cards),
        },
    }
