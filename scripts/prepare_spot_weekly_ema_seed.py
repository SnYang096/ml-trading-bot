#!/usr/bin/env python3
"""Download Binance Vision spot 1d klines and build weekly EMA200 seed parquets."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.live_data_stream.spot_weekly_ema_seed import (  # noqa: E402
    macro_seeds_ready,
    prepare_spot_weekly_ema_seed,
)

logger = logging.getLogger(__name__)


def _parse_symbols(raw: str) -> list[str]:
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


def _resolve_symbols(args: argparse.Namespace) -> list[str]:
    from src.live_data_stream.universe_symbols import resolve_symbols_csv

    csv = resolve_symbols_csv(
        cli_symbols=args.symbols,
        universe=str(args.universe),
        env_symbols=None,
        project_root=PROJECT_ROOT,
    )
    return _parse_symbols(csv)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--symbols",
        default=None,
        help=(
            "Comma-separated spot symbols. Default: live/{universe}/universe.yaml keys."
        ),
    )
    p.add_argument(
        "--universe",
        default="highcap",
        help="Universe name for universe.yaml lookup when --symbols is omitted.",
    )
    p.add_argument(
        "--kline-root",
        default="live/highcap/data/macro/spot_klines",
        help="Cache root for Vision spot 1d ZIPs",
    )
    p.add_argument(
        "--seed-root",
        default="live/highcap/data/macro/spot_weekly_ema200",
        help="Output directory for weekly EMA seed parquets",
    )
    p.add_argument(
        "--start-date",
        default="2017-01-01",
        help="First calendar day to include (YYYY-MM-DD)",
    )
    p.add_argument("--ema-span-weeks", type=int, default=200)
    p.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download ZIPs even if cached",
    )
    p.add_argument("--refresh-recent-days", type=int, default=45)
    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    symbols = _resolve_symbols(args)
    logger.info("macro seed symbols=%s", ",".join(symbols))
    seed_root = Path(args.seed_root)
    start = date.fromisoformat(str(args.start_date))
    written = prepare_spot_weekly_ema_seed(
        symbols,
        kline_root=Path(args.kline_root),
        seed_root=seed_root,
        start_date=start,
        ema_span_weeks=int(args.ema_span_weeks),
        force_download=bool(args.force_download),
        refresh_recent_days=int(args.refresh_recent_days),
    )
    ready, missing = macro_seeds_ready(symbols, seed_root)
    if not ready:
        logger.error(
            "macro seed incomplete: missing or empty EMA for %s (wrote %d/%d)",
            ",".join(missing),
            len(written),
            len(symbols),
        )
        return 1
    for sym, path in sorted(written.items()):
        logger.info("seed ready: %s -> %s", sym, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
