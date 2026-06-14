#!/usr/bin/env python3
"""Probe Binance USDT 24h gainers + profit_satellite weekly deploy sizing.

Usage:
    python scripts/profit_satellite_probe.py
    python scripts/profit_satellite_probe.py --profit-pool-usdt 5000 --limit 10
    python scripts/profit_satellite_probe.py --format json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.market_momentum.binance_spot_24h import (  # noqa: E402
    fetch_usdt_24h_gainers,
    weekly_deploy_usdt,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="profit_satellite Binance 24h probe")
    p.add_argument("--limit", type=int, default=20, help="top N gainers")
    p.add_argument(
        "--min-quote-volume",
        type=float,
        default=1_000_000.0,
        help="min 24h quote volume (USDT)",
    )
    p.add_argument(
        "--profit-pool-usdt",
        type=float,
        default=0.0,
        help="spot profit pool for deploy sizing (0 = skip)",
    )
    p.add_argument(
        "--deploy-frac",
        type=float,
        default=0.01,
        help="weekly deploy fraction of profit pool (default 0.01 = 1%%)",
    )
    p.add_argument(
        "--single-coin-cap-usdt",
        type=float,
        default=None,
        help="optional single-coin NAV cap",
    )
    p.add_argument(
        "--format",
        choices=("table", "json"),
        default="table",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    rows = fetch_usdt_24h_gainers(
        limit=args.limit,
        min_quote_volume_usdt=args.min_quote_volume,
    )
    as_of = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    top1 = rows[0] if rows else None
    deploy = None
    if args.profit_pool_usdt > 0:
        deploy = weekly_deploy_usdt(
            args.profit_pool_usdt,
            deploy_frac=args.deploy_frac,
            single_coin_cap_usdt=args.single_coin_cap_usdt,
        )

    if args.format == "json":
        payload = {
            "ok": True,
            "provider": "binance_spot",
            "endpoint": "GET /api/v3/ticker/24hr",
            "window": "24h_rolling",
            "as_of": as_of,
            "filters": {
                "quote": "USDT",
                "min_quote_volume_usdt": args.min_quote_volume,
                "symbol_status": "TRADING",
            },
            "rows": [
                {
                    "rank": r.rank,
                    "symbol": r.symbol,
                    "base": r.base,
                    "price_change_pct_24h": round(r.price_change_pct, 4),
                    "quote_volume_usdt_24h": round(r.quote_volume_usdt, 2),
                    "last_price": r.last_price,
                }
                for r in rows
            ],
            "top1_pick": (
                {
                    "symbol": top1.symbol,
                    "price_change_pct_24h": round(top1.price_change_pct, 4),
                }
                if top1
                else None
            ),
            "deploy": (
                {
                    "profit_pool_usdt": args.profit_pool_usdt,
                    "deploy_frac": args.deploy_frac,
                    "weekly_deploy_usdt": round(deploy, 2),
                }
                if deploy is not None
                else None
            ),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print()
    print("profit_satellite probe — Binance Spot USDT 24h gainers")
    print(f"as_of (UTC): {as_of}")
    print(f"filters: TRADING, quoteVolume >= {args.min_quote_volume:,.0f} USDT")
    print()
    print(f"{'#':>3}  {'symbol':<14} {'24h%':>10} {'quoteVol(USDT)':>18} {'last':>14}")
    print("-" * 64)
    for r in rows:
        print(
            f"{r.rank:>3}  {r.symbol:<14} {r.price_change_pct:>9.2f}% "
            f"{r.quote_volume_usdt:>18,.0f} {r.last_price:>14.8g}"
        )
    if top1:
        print()
        print(f"Top1 pick: {top1.symbol} (+{top1.price_change_pct:.2f}% / 24h)")
    if deploy is not None:
        print(
            f"Weekly deploy: profit_pool={args.profit_pool_usdt:,.2f} USDT "
            f"× {args.deploy_frac:.2%} → {deploy:,.2f} USDT"
        )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
