"""Cadence staleness (missed cron) detection for CMS and Telegram."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_STALENESS_HOURS: Dict[str, float] = {
    "daily": 36.0,
    "weekly": 8.0 * 24.0,
    "monthly": 35.0 * 24.0,
    "quarterly": 100.0 * 24.0,
    "yearly": 400.0 * 24.0,
}


def parse_run_ts(run_ts: str) -> Optional[datetime]:
    raw = str(run_ts or "").strip()
    if not raw:
        return None
    for fmt in ("%Y%m%d_%H%M%S", "%Y%m%d_%H%M"):
        try:
            dt = datetime.strptime(raw, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def hours_since_run(run_ts: str, *, now: Optional[datetime] = None) -> Optional[float]:
    dt = parse_run_ts(run_ts)
    if dt is None:
        return None
    now = now or datetime.now(timezone.utc)
    return max(0.0, (now - dt).total_seconds() / 3600.0)


def resolve_staleness_hours(
    schedules_cfg: Dict[str, Any],
) -> Dict[str, float]:
    raw = schedules_cfg.get("staleness_hours") or {}
    out = dict(DEFAULT_STALENESS_HOURS)
    if isinstance(raw, dict):
        for k, v in raw.items():
            try:
                out[str(k)] = float(v)
            except (TypeError, ValueError):
                continue
    return out


def evaluate_cadence_health(
    cadence: str,
    row: Optional[Dict[str, Any]],
    *,
    max_age_hours: float,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Return card row: OK | ALERT | MISSED."""
    if not row:
        return {
            "cadence": cadence,
            "display_status": "MISSED",
            "missed": True,
            "stale": True,
            "age_hours": None,
            "max_age_hours": max_age_hours,
            "run_ts": None,
            "exit_code": None,
            "watchdog_any_alert": None,
            "drift_any_alert": None,
            "output_dir": None,
            "updated_at": None,
        }

    run_ts = str(row.get("run_ts") or "")
    age = hours_since_run(run_ts, now=now)
    stale = age is None or age > max_age_hours
    run_status = str(row.get("status") or "OK")
    missed = stale
    if run_status == "ALERT":
        display = "ALERT"
    elif missed:
        display = "MISSED"
    else:
        display = "OK"

    return {
        "cadence": cadence,
        "display_status": display,
        "missed": missed,
        "stale": stale,
        "age_hours": age,
        "max_age_hours": max_age_hours,
        "run_ts": run_ts,
        "exit_code": row.get("exit_code"),
        "watchdog_any_alert": row.get("watchdog_any_alert"),
        "drift_any_alert": row.get("drift_any_alert"),
        "output_dir": row.get("output_dir"),
        "updated_at": row.get("updated_at"),
        "manifest": row.get("manifest"),
    }


def list_stale_cadences(
    index: Dict[str, Any],
    schedules_cfg: Dict[str, Any],
    *,
    now: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    cadences = (schedules_cfg.get("schedules") or {}).keys()
    limits = resolve_staleness_hours(schedules_cfg)
    rows = index.get("cadences") or {}
    stale: List[Dict[str, Any]] = []
    for cadence in sorted(cadences):
        card = evaluate_cadence_health(
            cadence,
            rows.get(cadence) if isinstance(rows, dict) else None,
            max_age_hours=limits.get(cadence, 7 * 24),
            now=now,
        )
        if card["missed"]:
            stale.append(card)
    return stale


def build_cadence_cards(
    index: Dict[str, Any],
    schedules_cfg: Dict[str, Any],
    *,
    now: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    cadences = (schedules_cfg.get("schedules") or {}).keys()
    limits = resolve_staleness_hours(schedules_cfg)
    rows = index.get("cadences") or {}
    cards: List[Dict[str, Any]] = []
    for cadence in sorted(cadences):
        row = rows.get(cadence) if isinstance(rows, dict) else None
        cards.append(
            evaluate_cadence_health(
                cadence,
                row if isinstance(row, dict) else None,
                max_age_hours=limits.get(cadence, 7 * 24),
                now=now,
            )
        )
    return cards
