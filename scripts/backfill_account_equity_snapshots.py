#!/usr/bin/env python3
"""Write today's exchange wallet/equity snapshot for each day in --from/--to.

Cannot reconstruct historical exchange balances; each date upserts the **current**
Binance values (idempotent per UTC day). Use after enabling snapshot timer so
curves accumulate from deploy day forward.

VPS one-shot after deploy:

  docker exec mlbot-business-console python3 /app/scripts/snapshot_account_equity.py
  docker exec mlbot-business-console python3 /app/scripts/backfill_account_equity_snapshots.py --from 2026-06-10 --to 2026-06-10

DB path defaults to ``SETTINGS.account_snapshot_db`` (``live_data/db/account_equity.db``).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _parse_date(raw: str) -> date:
    return datetime.strptime(str(raw).strip(), "%Y-%m-%d").date()


def main() -> None:
    from mlbot_console.config import SETTINGS
    from mlbot_console.services.account_equity_snapshots import (
        capture_daily_account_snapshots,
    )
    from mlbot_console.services.mark_prices import fetch_mark_prices
    from mlbot_console.services.universe import load_universe_symbols

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--from", dest="from_date", required=True, help="UTC YYYY-MM-DD")
    p.add_argument("--to", dest="to_date", required=True, help="UTC YYYY-MM-DD")
    p.add_argument(
        "--db-path",
        default=os.getenv(
            "MLBOT_ACCOUNT_SNAPSHOT_DB",
            str(SETTINGS.account_snapshot_db),
        ),
    )
    args = p.parse_args()

    start = _parse_date(args.from_date)
    end = _parse_date(args.to_date)
    if end < start:
        raise SystemExit("--to must be >= --from")

    symbols = load_universe_symbols(SETTINGS.universe_yaml)
    marks = fetch_mark_prices(SETTINGS.feature_bus_root, symbols)
    db_path = Path(args.db_path)

    reports = []
    d = start
    while d <= end:
        report = capture_daily_account_snapshots(
            db_path,
            mark_prices=marks,
            snapshot_date=d.isoformat(),
        )
        reports.append(report)
        d += timedelta(days=1)

    print(
        json.dumps(
            {
                "db_path": str(db_path),
                "from": start.isoformat(),
                "to": end.isoformat(),
                "days": len(reports),
                "reports": reports,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
