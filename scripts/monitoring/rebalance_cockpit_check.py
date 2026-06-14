#!/usr/bin/env python3
"""Scheduled Regime Cockpit / rebalance alert (T2d).

Writes monitor_event + results/monitoring/index.json; optional Telegram.

Usage:
  python scripts/monitoring/rebalance_cockpit_check.py
  python scripts/monitoring/rebalance_cockpit_check.py --dry-run
  mlbot monitor rebalance-check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.monitoring.rebalance_cockpit_run import run_rebalance_cockpit_check


def main() -> int:
    p = argparse.ArgumentParser(description="Regime Cockpit rebalance check (T2d)")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--window-days", type=int, default=7)
    p.add_argument("--run-ts", default="")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-telegram", action="store_true")
    p.add_argument("--skip-index", action="store_true")
    args = p.parse_args()

    summary = run_rebalance_cockpit_check(
        symbol=str(args.symbol).upper(),
        window_days=int(args.window_days),
        run_ts=str(args.run_ts).strip() or None,
        dry_run=bool(args.dry_run),
        skip_telegram=bool(args.skip_telegram),
        skip_index=bool(args.skip_index),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    # Business alert level lives in monitor_event / index, not process exit (avoid systemd OnFailure).
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
