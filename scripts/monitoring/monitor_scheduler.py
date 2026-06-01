#!/usr/bin/env python3
"""CLI: run monitor cadence schedules (thin wrapper over src.monitoring.scheduler)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.monitoring.run_monitor_manifest import _load_manifest, execute_manifest
from src.monitoring.scheduler import list_cadences, run_all_due, run_cadence


def main() -> int:
    p = argparse.ArgumentParser(
        description="Monitor scheduler (cadence manifests + CMS index)"
    )
    p.add_argument("--cadence", default="")
    p.add_argument("--all", action="store_true")
    p.add_argument("--schedules", default="config/monitoring/schedules.yaml")
    p.add_argument("--run-ts", default="")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-index", action="store_true")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()

    sched_path = Path(args.schedules)
    if not sched_path.is_absolute():
        sched_path = (PROJECT_ROOT / sched_path).resolve()

    if args.list:
        for c in list_cadences(sched_path):
            print(c)
        return 0

    kwargs = {
        "execute_manifest": execute_manifest,
        "load_manifest": _load_manifest,
        "schedules_path": sched_path,
        "dry_run": bool(args.dry_run),
    }
    try:
        if args.all:
            return run_all_due(**kwargs)
        cadence = str(args.cadence).strip()
        if not cadence:
            p.error("specify --cadence <name> or --all")
        return run_cadence(
            cadence,
            **kwargs,
            run_ts=str(args.run_ts).strip() or None,
            skip_index=bool(args.skip_index),
        )
    except (KeyError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
