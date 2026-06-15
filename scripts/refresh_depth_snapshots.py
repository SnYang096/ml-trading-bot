#!/usr/bin/env python3
"""Incremental depth snapshot refresh (one poll per symbol, cron-friendly)."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from src.data_tools.download_depth_snapshots import DepthSnapshotDownloader

logger = logging.getLogger(__name__)


def _default_symbols() -> list[str]:
    raw = os.getenv("MLBOT_LIVE_SYMBOLS", "")
    if raw.strip():
        return [s.strip().upper() for s in raw.split(",") if s.strip()]
    return ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "HYPEUSDT"]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(
        description="Refresh depth wall snapshots (one poll/symbol)"
    )
    ap.add_argument(
        "--symbols", default="", help="Comma-separated; default MLBOT_LIVE_SYMBOLS"
    )
    ap.add_argument("--depth-limit", type=int, default=1000)
    ap.add_argument("--bucket-pct", type=float, default=0.005)
    ap.add_argument("--parquet-dir", default="data/orderbook/parquet")
    args = ap.parse_args()

    symbols = (
        [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        if args.symbols
        else _default_symbols()
    )
    dl = DepthSnapshotDownloader(
        parquet_dir=Path(args.parquet_dir),
        depth_limit=int(args.depth_limit),
        bucket_pct=float(args.bucket_pct),
    )
    ok = 0
    for sym in symbols:
        try:
            snap = dl.poll_once(sym)
            path = dl.append_snapshot(sym, snap)
            ok += 1
            logger.info(
                "  %s bid_wall=$%.0f ask_wall=$%.0f → %s",
                sym,
                snap["wall_bid_notional_usd_max"].iloc[0],
                snap["wall_ask_notional_usd_max"].iloc[0],
                path.name,
            )
        except Exception as e:
            logger.warning("  %s depth refresh failed: %s", sym, e)
    logger.info("depth refresh done: ok=%d/%d", ok, len(symbols))
    sys.exit(0 if ok > 0 else 1)


if __name__ == "__main__":
    main()
