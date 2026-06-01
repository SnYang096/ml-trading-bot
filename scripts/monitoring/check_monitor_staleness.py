#!/usr/bin/env python3
"""CLI: cadence staleness / 缺勤 check (thin wrapper)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.monitoring.staleness_check import run_staleness_check


def main() -> int:
    p = argparse.ArgumentParser(description="Check monitor cadence staleness")
    p.add_argument("--schedules", default="config/monitoring/schedules.yaml")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    sched_path = Path(args.schedules)
    if not sched_path.is_absolute():
        sched_path = (PROJECT_ROOT / sched_path).resolve()
    return run_staleness_check(
        repo_root=PROJECT_ROOT,
        schedules_path=sched_path,
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    raise SystemExit(main())
