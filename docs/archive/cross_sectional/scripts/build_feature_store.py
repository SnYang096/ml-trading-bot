#!/usr/bin/env python3
"""
Build a monthly-partitioned FeatureStore for cross-sectional workflows (no ticks).

Writes:
  feature_store/<layer>/<symbol>/<timeframe>/YYYY-MM.parquet

These partitions can be reused across:
  - `mlbot cross-section rank`
  - `mlbot cross-section factor-eval` (FeatureStore source)
  - `mlbot cross-section pipeline` (FeatureStore source)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.cross_sectional.feature_store_builder import (  # noqa: E402
    CSFeatureStoreBuildConfig,
    build_feature_store_for_symbols,
    load_factor_set,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build CS FeatureStore (monthly parquet) with caching."
    )
    p.add_argument(
        "--symbols",
        required=True,
        help="Comma-separated symbols (e.g., BTCUSDT,ETHUSDT)",
    )
    p.add_argument("--timeframe", default="240T", help="Timeframe (e.g., 240T)")
    p.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    p.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    p.add_argument(
        "--data-path",
        default="data/parquet_data",
        help="Raw parquet root (default: data/parquet_data)",
    )

    p.add_argument(
        "--factor-set-yaml", required=True, help="YAML containing factor_sets"
    )
    p.add_argument("--factor-set", required=True, help="Factor set name to compute")
    p.add_argument(
        "--feature-deps",
        default="config/feature_dependencies.yaml",
        help="Feature dependencies YAML",
    )

    p.add_argument(
        "--features-store-root", default="feature_store", help="FeatureStore root"
    )
    p.add_argument(
        "--features-store-layer",
        default=None,
        help="Optional layer name (default: hashed)",
    )
    p.add_argument(
        "--warmup-bars",
        type=int,
        default=600,
        help="Warmup bars before each month (rolling features)",
    )
    p.add_argument(
        "--include-ohlcv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include OHLCV in store",
    )
    p.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overwrite existing month files",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    symbols: List[str] = [
        s.strip().upper() for s in str(args.symbols).split(",") if s.strip()
    ]
    factors = load_factor_set(
        factor_set_yaml=str(args.factor_set_yaml), factor_set=str(args.factor_set)
    )

    cfg = CSFeatureStoreBuildConfig(
        data_path=str(args.data_path),
        features_store_root=str(args.features_store_root),
        features_store_layer=str(args.features_store_layer or ""),
        timeframe=str(args.timeframe),
        start_date=str(args.start_date),
        end_date=str(args.end_date),
        warmup_bars=int(args.warmup_bars),
        include_ohlcv=bool(args.include_ohlcv),
        overwrite=bool(args.overwrite),
    )

    layer = build_feature_store_for_symbols(
        symbols=symbols,
        desired_output_cols=factors,
        feature_deps_path=str(args.feature_deps),
        cfg=cfg,
    )
    print(f"✅ CS FeatureStore build done. layer={layer}")


if __name__ == "__main__":
    main()
