"""Cadence staleness check + optional Telegram (缺勤)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.monitoring.store import PROJECT_ROOT, load_monitoring_index, load_schedules
from src.monitoring.staleness import list_stale_cadences
from src.monitoring.telegram import notify_stale_cadences


def run_staleness_check(
    *,
    repo_root: Optional[Path] = None,
    schedules_path: Optional[Path] = None,
    dry_run: bool = False,
) -> int:
    root = repo_root or PROJECT_ROOT
    sched_path = schedules_path or (root / "config/monitoring/schedules.yaml")
    if not sched_path.is_absolute():
        sched_path = (root / sched_path).resolve()
    cfg = load_schedules(sched_path)
    index = load_monitoring_index(root)
    stale = list_stale_cadences(index, cfg)
    if not stale:
        print("monitor staleness: all cadences OK")
        return 0
    for c in stale:
        age = c.get("age_hours")
        print(f"  STALE {c['cadence']}: age_hours={age}")
    if dry_run:
        return 1
    notify_stale_cadences(stale)
    return 1
