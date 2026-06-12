#!/usr/bin/env python3
"""Capture daily Binance wallet/equity snapshots for account equity curves.

Run once per UTC day (idempotent upsert for same date). Suggested cron (VPS):

  5 0 * * * docker exec mlbot-business-console python3 /app/scripts/snapshot_account_equity.py

Or via monitoring manifest step ``account-equity-snapshot`` in daily_health.yaml.

DB path: ``SETTINGS.account_snapshot_db`` → ``live_data/db/account_equity.db``
(override with ``MLBOT_ACCOUNT_SNAPSHOT_DB``). VPS after deploy::

  docker exec mlbot-business-console python3 /app/scripts/snapshot_account_equity.py
  sudo systemctl enable --now mlbot-monitor-daily.timer

Historical backfill cannot restore past exchange balances; use
``scripts/backfill_account_equity_snapshots.py --from/--to`` to stamp today's
values per UTC day from enable date forward.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def main() -> None:
    from mlbot_console.config import SETTINGS
    from mlbot_console.services.account_equity_snapshots import (
        capture_daily_account_snapshots,
    )
    from mlbot_console.services.mark_prices import fetch_mark_prices
    from mlbot_console.services.universe import load_universe_symbols

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--db-path",
        default=os.getenv(
            "MLBOT_ACCOUNT_SNAPSHOT_DB",
            str(SETTINGS.account_snapshot_db),
        ),
    )
    p.add_argument(
        "--date",
        default="",
        help="UTC YYYY-MM-DD (default: today)",
    )
    args = p.parse_args()

    symbols = load_universe_symbols(SETTINGS.universe_yaml)
    marks = fetch_mark_prices(SETTINGS.feature_bus_root, symbols)
    report = capture_daily_account_snapshots(
        Path(args.db_path),
        mark_prices=marks,
        snapshot_date=str(args.date).strip() or None,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
