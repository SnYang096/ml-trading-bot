#!/usr/bin/env python3
"""Poll multi-leg account; Telegram on >=3% equity move or new exchange positions.

Preferred entry: ``mlbot monitor account-watch`` (same creds as other monitor TG).

Env:
  GRAFANA_ALERT_TELEGRAM_BOT_TOKEN / GRAFANA_ALERT_TELEGRAM_CHAT_ID
  MULTI_LEG_BINANCE_FUTURES_TESTNET_* (default) or MULTI_LEG_BINANCE_FUTURES_* (--mainnet)
  MLBOT_ACCOUNT_TG_CHANGE_PCT=0.03
  MLBOT_ACCOUNT_TG_STATE=data/monitoring/multi_leg_account_tg_state.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.monitoring.account_telegram_watch import run_account_watch_once


def main() -> int:
    p = argparse.ArgumentParser(description="Multi-leg account Telegram watcher")
    p.add_argument("--once", action="store_true")
    p.add_argument("--loop", action="store_true")
    p.add_argument("--interval-seconds", type=float, default=60.0)
    p.add_argument("--mainnet", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force-notify", action="store_true")
    p.add_argument(
        "--change-pct",
        type=float,
        default=None,
        help="Equity move threshold (default env MLBOT_ACCOUNT_TG_CHANGE_PCT or 0.03)",
    )
    p.add_argument(
        "--state-path",
        default=os.getenv(
            "MLBOT_ACCOUNT_TG_STATE",
            "data/monitoring/multi_leg_account_tg_state.json",
        ),
    )
    args = p.parse_args()
    if not args.once and not args.loop:
        args.once = True

    state_path = Path(str(args.state_path))
    testnet = not bool(args.mainnet)

    def _tick() -> dict:
        return run_account_watch_once(
            state_path=state_path,
            testnet=testnet,
            change_threshold_pct=args.change_pct,
            dry_run=bool(args.dry_run),
            force_notify=bool(args.force_notify),
        )

    if args.once:
        summary = _tick()
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    while True:
        summary = _tick()
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        time.sleep(max(5.0, float(args.interval_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
