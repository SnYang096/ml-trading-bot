"""CMS dashboard aggregation (cards + strategy alerts)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.monitoring.staleness import build_cadence_cards, list_stale_cadences
from src.monitoring.store import load_monitoring_index, load_schedules

CADENCE_DISPLAY_ORDER = (
    "daily",
    "weekly",
    "weekly_c",
    "monthly",
    "monthly_c",
    "quarterly",
    "yearly",
)


def _cadence_sort_key(cadence: str) -> int:
    try:
        return CADENCE_DISPLAY_ORDER.index(cadence)
    except ValueError:
        return len(CADENCE_DISPLAY_ORDER)


def sort_cadence_cards(cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(cards, key=lambda c: _cadence_sort_key(str(c.get("cadence") or "")))


def messages_from_monitor_detail(
    detail_json: Optional[str],
    *,
    source: str,
    strategy: str,
    status: str,
) -> List[str]:
    """Human-readable lines from monitor_event.detail_json."""
    if not detail_json:
        return []
    try:
        body = json.loads(detail_json)
    except json.JSONDecodeError:
        return []
    if not isinstance(body, dict):
        return []

    lines: List[str] = []
    for a in body.get("alerts") or []:
        if a:
            lines.append(str(a))

    if status in ("NO_PLATEAUS", "BASELINE_MISSING"):
        skipped = body.get("skipped")
        if skipped:
            lines.append(str(skipped))

    for it in body.get("items") or []:
        if not isinstance(it, dict):
            continue
        if it.get("kind") == "regime_shares":
            cur = it.get("current") or it.get("shares") or {}
            if isinstance(cur, dict) and cur:
                parts = ", ".join(f"{k}={float(v):.1%}" for k, v in sorted(cur.items()))
                lines.append(f"regime mix: {parts}")

    if strategy == "_factor_health":
        for it in body.get("items") or []:
            if not isinstance(it, dict):
                continue
            if it.get("skipped"):
                lines.append(f"{it.get('kind', 'skip')}: {it['skipped']}")
        if not lines:
            for it in body.get("items") or []:
                if not isinstance(it, dict):
                    continue
                if it.get("kind") == "psi" and it.get("psi") is not None:
                    feat = it.get("feature", "?")
                    lines.append(f"PSI {feat}: {float(it['psi']):.3f}")

    return lines


def enrich_cards_with_details(
    cards: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Attach alert_details / uncalibrated_details per cadence card."""
    run_ts_by = {
        str(c["cadence"]): str(c["run_ts"])
        for c in cards
        if c.get("run_ts")
    }
    alert_lines: Dict[str, List[str]] = {k: [] for k in run_ts_by}
    uncal_lines: Dict[str, List[str]] = {k: [] for k in run_ts_by}

    for ev in events:
        cad = str(ev.get("cadence") or "")
        if cad not in run_ts_by or str(ev.get("run_ts")) != run_ts_by[cad]:
            continue
        source = str(ev.get("source") or "")
        strategy = str(ev.get("strategy") or "")
        status = str(ev.get("status") or "")
        msgs = messages_from_monitor_detail(
            ev.get("detail_json"),
            source=source,
            strategy=strategy,
            status=status,
        )
        if not msgs:
            continue
        label = strategy if strategy != "_factor_health" else "因子健康 (PSI/IC)"
        prefixed = [f"[{label}] {m}" for m in msgs]
        if status == "ALERT":
            alert_lines[cad].extend(prefixed)
        elif status == "NO_PLATEAUS":
            uncal_lines[cad].extend(prefixed)

    out: List[Dict[str, Any]] = []
    for card in cards:
        cad = str(card.get("cadence") or "")
        enriched = dict(card)
        if alert_lines.get(cad):
            enriched["alert_details"] = alert_lines[cad]
        if uncal_lines.get(cad):
            enriched["uncalibrated_details"] = uncal_lines[cad]
        out.append(enriched)
    return out


def load_monitor_events(
    registry_db: Path,
    *,
    cadence: Optional[str] = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    if not registry_db.is_file():
        return []
    sql = (
        "SELECT cadence, source, strategy, status, detail_json, report_path, run_ts, "
        "output_dir, ts FROM monitor_event"
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
        strategy = str(ev.get("strategy") or "")
        out[cad].append(
            {
                "source": str(ev.get("source") or ""),
                "strategy": strategy,
                "messages": messages_from_monitor_detail(
                    ev.get("detail_json"),
                    source=str(ev.get("source") or ""),
                    strategy=strategy,
                    status="ALERT",
                ),
            }
        )
    return out


def strategy_uncalibrated_by_cadence(
    events: List[Dict[str, Any]],
    cards: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, str]]]:
    """Latest run_ts per cadence → NO_PLATEAUS drift rows (not TG-worthy, but visible)."""
    run_ts_by: Dict[str, str] = {}
    for c in cards:
        if c.get("run_ts"):
            run_ts_by[str(c["cadence"])] = str(c["run_ts"])
    out: Dict[str, List[Dict[str, str]]] = {k: [] for k in run_ts_by}
    for ev in events:
        cad = str(ev.get("cadence") or "")
        if cad not in run_ts_by or str(ev.get("run_ts")) != run_ts_by[cad]:
            continue
        if str(ev.get("status")) != "NO_PLATEAUS":
            continue
        strategy = str(ev.get("strategy") or "")
        out[cad].append(
            {
                "source": str(ev.get("source") or ""),
                "strategy": strategy,
                "messages": messages_from_monitor_detail(
                    ev.get("detail_json"),
                    source=str(ev.get("source") or ""),
                    strategy=strategy,
                    status="NO_PLATEAUS",
                ),
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
    cards = enrich_cards_with_details(
        sort_cadence_cards(build_cadence_cards(index, schedules_cfg)),
        events,
    )
    stale = list_stale_cadences(index, schedules_cfg)
    alerts = strategy_alerts_by_cadence(events, cards)
    uncalibrated = strategy_uncalibrated_by_cadence(events, cards)
    return {
        "index_updated_at": index.get("updated_at"),
        "cards": cards,
        "stale_cadences": [c["cadence"] for c in stale],
        "strategy_alerts": alerts,
        "strategy_uncalibrated": uncalibrated,
        "summary": {
            "any_alert": any(c.get("display_status") == "ALERT" for c in cards),
            "any_missed": any(c.get("display_status") == "MISSED" for c in cards),
            "any_uncalibrated": any(
                bool(v) for v in uncalibrated.values()
            )
            or any(c.get("drift_no_plateaus") for c in cards),
            "n_cards": len(cards),
        },
    }
