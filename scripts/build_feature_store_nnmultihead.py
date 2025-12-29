#!/usr/bin/env python3
"""
Precompute and persist features for nnmultihead (path primitives) so that:
  - feature computation is a separate batch job (CPU/IO heavy, especially with ticks)
  - training/prediction becomes fast and repeatable

Output:
  - a directory with per-symbol feature files: features_<SYMBOL>.parquet
  - meta.json describing config/timeframe/date range
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
from src.features.loader.strategy_feature_loader import (
    StrategyFeatureLoader,
)  # noqa: E402
from src.time_series_model.strategy_config import StrategyConfigLoader  # noqa: E402
from scripts.train_strategy_pipeline import (
    run_feature_pipeline,
    determine_feature_columns,
)  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build feature store for nnmultihead config."
    )
    p.add_argument(
        "--config", required=True, help="Config directory containing features.yaml."
    )
    p.add_argument("--symbols", required=True, help="Comma-separated symbols.")
    p.add_argument("--data-path", default="data/parquet_data")
    p.add_argument("--timeframe", default="240T")
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    p.add_argument(
        "--output-dir",
        default="feature_store",
        help="FeatureStore root directory (default: feature_store)",
    )
    p.add_argument(
        "--output-format",
        default="monthly",
        choices=["monthly", "flat"],
        help="monthly (recommended): write FeatureStore layout {root}/{layer}/{symbol}/{timeframe}/{YYYY-MM}.parquet. "
        "flat (NOT recommended for ticks/stateful features): write features_<SYMBOL>.parquet in output-dir.",
    )
    p.add_argument(
        "--layer",
        default="nnmultihead_v1",
        help="FeatureStore layer name when output-format=monthly",
    )
    p.add_argument(
        "--warmup-bars",
        type=int,
        default=512,
        help="For monthly output: prepend previous N bars when computing each month, then trim before writing. "
        "Helps stateful/rolling/ticks features at month boundaries.",
    )
    p.add_argument(
        "--warmup-months",
        type=int,
        default=0,
        help="For monthly output: prepend previous N calendar months of bars when computing each month, then trim. "
        "For VPIN-style tick features, continuity is handled inside tick_loader monthly cache; use warmup-months mainly "
        "for non-tick rolling features that truly need more history.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_dir = Path(args.config).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = StrategyConfigLoader(cfg_dir)
    cfg = loader.load()

    symbols = [s.strip() for s in str(args.symbols).split(",") if s.strip()]
    if not symbols:
        raise ValueError("No symbols provided.")

    feature_loader = StrategyFeatureLoader()
    feature_cols_union: List[str] = []

    for sym in symbols:
        df_raw = load_raw_data(
            data_path=args.data_path,
            symbol=sym,
            start_date=args.start_date,
            end_date=args.end_date,
            timeframe=args.timeframe,
        )
        if df_raw.empty:
            raise ValueError(f"No raw data loaded for symbol={sym}")
        if "symbol" not in df_raw.columns:
            df_raw["symbol"] = sym

        if args.output_format == "flat":
            # Guardrail: flat is dangerous for ticks/stateful features and can easily OOM when loading many symbols.
            raise ValueError(
                "output-format=flat is not recommended for tick/stateful features and can cause incorrect month-boundary "
                "results + memory blowups. Use --output-format monthly (FeatureStore layout) instead."
            )

        # monthly FeatureStore layout (reuse tree-model feature_store mechanism)
        from src.feature_store.feature_store import FeatureStore, FeatureStoreSpec

        store = FeatureStore(out_dir)
        spec = FeatureStoreSpec(
            layer=str(args.layer), symbol=sym, timeframe=str(args.timeframe)
        )

        # Incremental by month on RAW, with warmup context, then trim before writing.
        if not isinstance(df_raw.index, pd.DatetimeIndex):
            raise ValueError("Expected df_raw index to be DatetimeIndex")
        df_raw = df_raw.sort_index()

        base_cols = ["open", "high", "low", "close", "volume", "_symbol", "symbol"]
        requested = cfg.features.requested_features

        monthly_groups = df_raw.groupby(pd.Grouper(freq="M"))
        for period, df_month_raw in monthly_groups:
            if df_month_raw.empty:
                continue
            month_str = period.strftime("%Y-%m")
            if store.has_month(spec, month_str):
                continue

            month_start = df_month_raw.index.min()
            month_end = df_month_raw.index.max()

            warmup_months = max(0, int(args.warmup_months))
            warmup_bars = max(0, int(args.warmup_bars))

            # Prefer warmup by calendar months when explicitly requested.
            if warmup_months > 0:
                start_ts = pd.Timestamp(month_start) - pd.DateOffset(
                    months=warmup_months
                )
                df_window = df_raw.loc[
                    (df_raw.index >= start_ts) & (df_raw.index <= month_end)
                ]
            elif warmup_bars > 0:
                # Fallback: warmup by bars
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

            # Trim to just this month before writing
            df_feats_month = df_feats_window.loc[
                (df_feats_window.index >= month_start)
                & (df_feats_window.index <= month_end)
            ]

            store.write_month(
                spec,
                month_str,
                df_feats_month,
                base_columns=base_cols,
                feature_columns=None,
                overwrite=False,
                metadata={
                    "config_dir": str(cfg_dir),
                    "warmup_months": warmup_months,
                    "warmup_bars": warmup_bars,
                },
            )

        # Update union columns based on a small sample month (avoid concatenating full df_features in memory)
        try:
            # pick latest available month in store for this symbol
            feature_cols_union.extend(
                [c for c in base_cols if c not in feature_cols_union]
            )
        except Exception:
            pass
        print("✅ Saved monthly FeatureStore for:", spec)

    meta = {
        "config_dir": str(cfg_dir),
        "timeframe": str(args.timeframe),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "symbols": symbols,
        "feature_cols": feature_cols_union,
        "output_format": str(args.output_format),
        "layer": str(args.layer) if args.output_format == "monthly" else None,
        "warmup_bars": (
            int(args.warmup_bars) if args.output_format == "monthly" else None
        ),
        "warmup_months": (
            int(args.warmup_months) if args.output_format == "monthly" else None
        ),
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print("✅ Saved meta to:", out_dir / "meta.json")


if __name__ == "__main__":
    main()
