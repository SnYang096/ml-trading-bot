#!/usr/bin/env python3
"""
Model-agnostic FeatureStore builder.

This is intended to sit *above* tree/nn:
- build once (monthly partitions + warmup)
- tree and nn both read from the same FeatureStore dataset (layer)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_tools.data_utils import load_raw_data  # noqa: E402
from src.data_tools.universe_config import load_universe_config  # noqa: E402
from src.feature_store.feature_store import FeatureStore, FeatureStoreSpec  # noqa: E402
from src.feature_store.layer_naming import resolve_layer_name  # noqa: E402
from src.features.loader.strategy_feature_loader import (
    StrategyFeatureLoader,
)  # noqa: E402
from src.time_series_model.strategy_config import StrategyConfigLoader  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build monthly FeatureStore from a config directory."
    )
    p.add_argument(
        "--config", required=True, help="Config directory containing features.yaml."
    )
    p.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated symbols. If not provided, will use --universe-config if specified.",
    )
    p.add_argument(
        "--universe-config",
        default=None,
        help="Path to universe config YAML (e.g., config/download/crypto_4h_token_universe_groups.yaml). "
        "If provided and --symbols is not set, will load all symbols from the config.",
    )
    p.add_argument(
        "--universe-set",
        default="starter_a",
        help="Universe set name to use from universe config (default: starter_a).",
    )
    p.add_argument(
        "--universe-groups",
        default=None,
        help="Comma-separated groups to include (e.g., 'highcap,alt'). If not specified, includes all groups.",
    )
    p.add_argument("--timeframe", required=True, help="Timeframe (e.g., 240T).")
    p.add_argument("--data-path", default="data/parquet_data")
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    p.add_argument("--root", default="feature_store", help="FeatureStore root dir.")
    p.add_argument(
        "--layer",
        default=None,
        help="FeatureStore layer (dataset id). If not specified, auto-generated from config content. "
        "You can pass a versioned name like heavy_v6 for manual invalidation.",
    )
    p.add_argument("--warmup-months", type=int, default=1)
    p.add_argument("--warmup-bars", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_dir = Path(args.config).resolve()
    loader = StrategyConfigLoader(cfg_dir)
    cfg = loader.load()

    # Resolve symbols: from --symbols or --universe-config
    if args.symbols:
        symbols = [s.strip() for s in str(args.symbols).split(",") if s.strip()]
    elif args.universe_config:
        # Load from universe config
        universe_cfg = load_universe_config(args.universe_config)
        groups = (
            [g.strip() for g in args.universe_groups.split(",") if g.strip()]
            if args.universe_groups
            else None
        )
        symbols = universe_cfg.resolve_symbols_usdt(
            universe_set=args.universe_set, groups=groups
        )
        print(
            f"   📋 Loaded {len(symbols)} symbols from universe config: {args.universe_config}"
        )
        if groups:
            print(f"   📋 Groups: {', '.join(groups)}")
        print(
            f"   📋 Symbols: {', '.join(symbols[:10])}{'...' if len(symbols) > 10 else ''}"
        )
    else:
        raise ValueError(
            "Either --symbols or --universe-config must be provided. "
            "Use --universe-config to load all symbols from config/download/crypto_4h_token_universe_groups.yaml"
        )

    if not symbols:
        raise ValueError("No symbols resolved. Check --symbols or --universe-config.")

    root = Path(args.root).resolve()
    store = FeatureStore(root)

    # Auto-generate layer name if not specified (unified handling for both CLI and direct script calls)
    layer = resolve_layer_name(args.layer, cfg_dir)
    warmup_months = max(0, int(args.warmup_months))
    warmup_bars = max(0, int(args.warmup_bars))

    # IMPORTANT: disable FeatureComputer's own monthly cache so warmup context can flow across month boundaries.
    feature_loader = StrategyFeatureLoader(use_monthly_cache=False)
    feature_cache_version = getattr(feature_loader.computer, "cache_version", None)
    requested = cfg.features.requested_features

    # Global statistics
    stats = {
        "symbols_processed": 0,
        "symbols_failed": 0,
        "months_skipped": 0,
        "months_built": 0,
        "months_failed": 0,
        "failed_symbols": [],
        "failed_months": [],
    }

    for sym_idx, sym in enumerate(symbols, 1):
        print(f"\n{'='*60}")
        print(f"📊 Processing symbol {sym_idx}/{len(symbols)}: {sym}")
        print(f"{'='*60}")

        try:
            df_raw = load_raw_data(
                data_path=args.data_path,
                symbol=sym,
                start_date=args.start_date,
                end_date=args.end_date,
                timeframe=args.timeframe,
            )
            if df_raw.empty:
                print(f"  ⚠️  No raw data loaded for symbol={sym}, skipping")
                stats["symbols_failed"] += 1
                stats["failed_symbols"].append((sym, "No raw data"))
                continue
            if "symbol" not in df_raw.columns:
                df_raw["symbol"] = sym
            df_raw = df_raw.sort_index()

            spec = FeatureStoreSpec(
                layer=str(layer), symbol=str(sym), timeframe=str(args.timeframe)
            )
            monthly_groups = df_raw.groupby(pd.Grouper(freq="M"))
            base_cols = ["open", "high", "low", "close", "volume", "_symbol", "symbol"]

            # Parse start_date and end_date for month filtering
            start_ts = pd.Timestamp(args.start_date) if args.start_date else None
            end_ts = pd.Timestamp(args.end_date) if args.end_date else None

            # Count months to process
            all_months = []
            for period, df_month in monthly_groups:
                if df_month.empty:
                    continue
                month_start = df_month.index.min()
                month_end = df_month.index.max()
                if start_ts is not None and month_end < start_ts:
                    continue
                if end_ts is not None and month_start > end_ts:
                    continue
                all_months.append(period.strftime("%Y-%m"))

            # Check which months exist
            existing_months = [m for m in all_months if store.has_month(spec, m)]
            missing_months = [m for m in all_months if not store.has_month(spec, m)]

            stats["months_skipped"] += len(existing_months)

            if existing_months:
                print(
                    f"  ⏭️  Skipping {len(existing_months)} existing month(s): {', '.join(existing_months[:5])}{'...' if len(existing_months) > 5 else ''}"
                )
            if missing_months:
                print(
                    f"  🔨 Building {len(missing_months)} missing month(s): {', '.join(missing_months[:5])}{'...' if len(missing_months) > 5 else ''}"
                )
            elif not existing_months and not missing_months:
                print(f"  ⚠️  No months to process for {sym}")

            # Process each month with error handling
            for period, df_month in monthly_groups:
                if df_month.empty:
                    continue

                month_start = df_month.index.min()
                month_end = df_month.index.max()

                # Filter months by start_date and end_date if provided
                if start_ts is not None and month_end < start_ts:
                    continue  # Skip months before start_date
                if end_ts is not None and month_start > end_ts:
                    continue  # Skip months after end_date

                month_str = period.strftime("%Y-%m")
                if store.has_month(spec, month_str):
                    continue

                print(f"\n  📅 Building {sym} {month_str}...")

                try:
                    if warmup_months > 0:
                        warmup_start = pd.Timestamp(month_start) - pd.DateOffset(
                            months=warmup_months
                        )
                        df_window = df_raw.loc[
                            (df_raw.index >= warmup_start) & (df_raw.index <= month_end)
                        ]
                    elif warmup_bars > 0:
                        pos_end = df_raw.index.searchsorted(month_start, side="left")
                        pos_start = max(0, pos_end - warmup_bars)
                        df_window = df_raw.iloc[pos_start:].loc[:month_end]
                    else:
                        df_window = df_raw.loc[
                            (df_raw.index >= month_start) & (df_raw.index <= month_end)
                        ]

                    df_feats_window = feature_loader.load_features_from_requested(
                        df_window, requested_features=requested, fit=True
                    )
                    if "symbol" not in df_feats_window.columns:
                        df_feats_window["symbol"] = sym
                    df_feats_month = df_feats_window.loc[
                        (df_feats_window.index >= month_start)
                        & (df_feats_window.index <= month_end)
                    ]

                    # Extract feature columns (all columns except base columns)
                    feature_cols = [
                        c for c in df_feats_month.columns if c not in base_cols
                    ]

                    store.write_month(
                        spec,
                        month_str,
                        df_feats_month,
                        base_columns=base_cols,
                        feature_columns=feature_cols,
                        overwrite=False,
                        metadata={
                            "config_dir": str(cfg_dir),
                            "warmup_months": warmup_months,
                            "warmup_bars": warmup_bars,
                            "requested_features": requested,
                            "feature_cache_version": feature_cache_version,
                        },
                    )
                    stats["months_built"] += 1
                    print(f"  ✅ Successfully built {sym} {month_str}")
                except Exception as e:
                    stats["months_failed"] += 1
                    error_msg = str(e)
                    stats["failed_months"].append((sym, month_str, error_msg))
                    print(f"  ❌ Failed to build {sym} {month_str}: {error_msg}")
                    import traceback

                    print(f"     Traceback: {traceback.format_exc()}")
                    # Continue to next month instead of crashing

            stats["symbols_processed"] += 1
            print(
                f"✅ Completed {sym}: {len(existing_months)} skipped, {len(missing_months)} built"
            )

        except Exception as e:
            stats["symbols_failed"] += 1
            error_msg = str(e)
            stats["failed_symbols"].append((sym, error_msg))
            print(f"  ❌ Failed to process symbol {sym}: {error_msg}")
            import traceback

            print(f"     Traceback: {traceback.format_exc()}")
            # Continue to next symbol instead of crashing

    # Print summary statistics
    print(f"\n{'='*60}")
    print("📊 Build Summary")
    print(f"{'='*60}")
    print(f"  ✅ Symbols processed: {stats['symbols_processed']}/{len(symbols)}")
    print(f"  ❌ Symbols failed: {stats['symbols_failed']}")
    print(f"  ⏭️  Months skipped (already exist): {stats['months_skipped']}")
    print(f"  🔨 Months built: {stats['months_built']}")
    print(f"  ❌ Months failed: {stats['months_failed']}")

    if stats["failed_symbols"]:
        print(f"\n  ⚠️  Failed symbols ({len(stats['failed_symbols'])}):")
        for sym, error in stats["failed_symbols"][:10]:
            print(f"     - {sym}: {error[:100]}")
        if len(stats["failed_symbols"]) > 10:
            print(f"     ... and {len(stats['failed_symbols']) - 10} more")

    if stats["failed_months"]:
        print(f"\n  ⚠️  Failed months ({len(stats['failed_months'])}):")
        for sym, month, error in stats["failed_months"][:10]:
            print(f"     - {sym} {month}: {error[:100]}")
        if len(stats["failed_months"]) > 10:
            print(f"     ... and {len(stats['failed_months']) - 10} more")

    if stats["months_failed"] > 0 or stats["symbols_failed"] > 0:
        print(
            f"\n  💡 Tip: Re-run the command to retry failed months (existing months will be skipped)"
        )

    # Save metadata with statistics
    meta = {
        "config_dir": str(cfg_dir),
        "timeframe": str(args.timeframe),
        "symbols": symbols,
        "layer": str(layer),
        "warmup_months": warmup_months,
        "warmup_bars": warmup_bars,
        "build_stats": stats,
    }
    (root / f"{layer}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"\n✅ Saved meta: {root / f'{layer}.meta.json'}")


if __name__ == "__main__":
    main()
