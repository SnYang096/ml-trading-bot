#!/usr/bin/env python3
"""Periodic A/B/C account reconciliation: engine state + PnL vs exchange equity.

Exit 0 when all checks pass; exit 1 when any issue is found (for systemd/TG hooks).

Example:
  PYTHONPATH=src python scripts/reconcile_account_pnl.py
  PYTHONPATH=src python scripts/reconcile_account_pnl.py --json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from mlbot_console.config import SETTINGS
from mlbot_console.services.account_reconciliation import reconcile_all_accounts
from mlbot_console.services.mark_prices import fetch_mark_prices
from mlbot_console.services.universe import load_universe_symbols

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconcile ABC accounts vs exchange")
    parser.add_argument("--symbol", default="*", help="Symbol filter (* = all)")
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=0,
        help="Realized PnL lookback (0 = all history)",
    )
    parser.add_argument("--json", action="store_true", help="Print full JSON report")
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Only warnings/errors"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    symbols = load_universe_symbols(SETTINGS.universe_yaml)
    marks = fetch_mark_prices(SETTINGS.feature_bus_root, symbols)

    report = reconcile_all_accounts(
        trend_db=SETTINGS.trend_order_db,
        spot_db=SETTINGS.spot_order_db,
        spot_ledger_db=SETTINGS.spot_ledger_db,
        multi_leg_db=SETTINGS.multi_leg_db,
        feature_bus_root=SETTINGS.feature_bus_root,
        mark_prices=marks,
        symbol=args.symbol,
        lookback_days=int(args.lookback_days),
    )

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        totals = (report.get("pnl") or {}).get("totals") or {}
        logger.info(
            "account reconcile: ok=%s issues=%d equity=%s local_upnl=%s ex_upnl=%s",
            report.get("ok"),
            len(report.get("issues") or []),
            totals.get("exchange_equity_usdt"),
            totals.get("local_unrealized_pnl"),
            totals.get("exchange_unrealized_usdt"),
        )
        for issue in report.get("issues") or []:
            layer = issue.get("layer", "?")
            scope = issue.get("scope", "?")
            kind = issue.get("kind", "?")
            msg = issue.get("message") or issue
            logger.warning("[%s/%s] %s: %s", layer, scope, kind, msg)

    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
