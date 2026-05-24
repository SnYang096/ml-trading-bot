#!/usr/bin/env python3
"""Merge recent live_storage 1m bars into feature-bus rolling snapshots.

Use after a WS outage when ``bars_1min`` bus parquets have holes but the
archive (``live/<flow>/data/bars/<SYMBOL>/<date>.parquet``) is complete.
Archive rows win on duplicate timestamps.

Backfill is "non-shrinking": ``merge_bars_1m`` is called with
``preserve_history=True`` so the existing bus rows are never tailed below
their current count, even if this script's ``--max-rows`` is smaller than
the online publisher's effective cap.

Example (production paths, run from repo root inside the publisher image):

  python3 scripts/sync_feature_bus_bars_from_archive.py \\
    --live-storage-base /app/live/highcap/data \\
    --feature-bus-root /app/live/shared_feature_bus \\
    --symbols BNBUSDT,ETHUSDT,BTCUSDT,SOLUSDT,XRPUSDT,ADAUSDT \\
    --lookback-hours 168
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from src.live_data_stream.auto_gap_fill import sync_archive_bars_to_feature_bus
from src.live_data_stream.feature_bus import FeatureBusWriter
from src.live_data_stream.feature_storage import StorageManager

logger = logging.getLogger(__name__)


def _parse_symbols(raw: str) -> list[str]:
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--live-storage-base",
        default=os.getenv("MLBOT_LIVE_STORAGE_BASE", "live/highcap/data"),
    )
    p.add_argument(
        "--feature-bus-root",
        default=os.getenv("MLBOT_FEATURE_BUS_ROOT", "live/shared_feature_bus"),
    )
    p.add_argument(
        "--symbols",
        default=os.getenv("MLBOT_LIVE_SYMBOLS", "BTCUSDT,ETHUSDT,BNBUSDT"),
    )
    p.add_argument("--lookback-hours", type=float, default=168.0)
    p.add_argument(
        "--max-rows",
        type=int,
        default=10080,
        help=(
            "Writer cap; only takes effect when preserve_history=False on the"
            " merge path. Backfill itself is non-shrinking, so this argument"
            " mostly affects newly-created bus parquets."
        ),
    )
    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    symbols = _parse_symbols(args.symbols)
    if not symbols:
        logger.error("no symbols")
        return 1

    storage = StorageManager(Path(args.live_storage_base))
    writer = FeatureBusWriter(args.feature_bus_root, max_rows=int(args.max_rows))
    synced = sync_archive_bars_to_feature_bus(
        storage,
        writer,
        symbols,
        lookback_hours=args.lookback_hours,
    )
    logger.info(
        "feature-bus archive sync done symbols=%s lookback=%.1fh rows_merged=%d",
        ",".join(symbols),
        args.lookback_hours,
        synced,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
