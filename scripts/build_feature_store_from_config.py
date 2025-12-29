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
from src.feature_store.feature_store import FeatureStore, FeatureStoreSpec  # noqa: E402
from src.feature_store.layer_naming import default_layer_from_config  # noqa: E402
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
    p.add_argument("--symbols", required=True, help="Comma-separated symbols.")
    p.add_argument("--timeframe", required=True, help="Timeframe (e.g., 240T).")
    p.add_argument("--data-path", default="data/parquet_data")
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    p.add_argument("--root", default="feature_store", help="FeatureStore root dir.")
    p.add_argument(
        "--layer",
        default="AUTO",
        help="FeatureStore layer (dataset id). Default=AUTO (derived from config content). "
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

    symbols = [s.strip() for s in str(args.symbols).split(",") if s.strip()]
    if not symbols:
        raise ValueError("No symbols provided.")

    root = Path(args.root).resolve()
    store = FeatureStore(root)

    layer = (
        default_layer_from_config(cfg_dir)
        if str(args.layer).upper() == "AUTO"
        else str(args.layer)
    )
    warmup_months = max(0, int(args.warmup_months))
    warmup_bars = max(0, int(args.warmup_bars))

    # IMPORTANT: disable FeatureComputer's own monthly cache so warmup context can flow across month boundaries.
    feature_loader = StrategyFeatureLoader(use_monthly_cache=False)
    requested = cfg.features.requested_features

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
        df_raw = df_raw.sort_index()

        spec = FeatureStoreSpec(
            layer=str(layer), symbol=str(sym), timeframe=str(args.timeframe)
        )
        monthly_groups = df_raw.groupby(pd.Grouper(freq="M"))
        base_cols = ["open", "high", "low", "close", "volume", "_symbol", "symbol"]

        for period, df_month in monthly_groups:
            if df_month.empty:
                continue
            month_str = period.strftime("%Y-%m")
            if store.has_month(spec, month_str):
                continue

            month_start = df_month.index.min()
            month_end = df_month.index.max()

            if warmup_months > 0:
                start_ts = pd.Timestamp(month_start) - pd.DateOffset(
                    months=warmup_months
                )
                df_window = df_raw.loc[
                    (df_raw.index >= start_ts) & (df_raw.index <= month_end)
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
                    "requested_features": requested,
                },
            )
        print("✅ Saved FeatureStore:", spec)

    meta = {
        "config_dir": str(cfg_dir),
        "timeframe": str(args.timeframe),
        "symbols": symbols,
        "layer": str(layer),
        "warmup_months": warmup_months,
        "warmup_bars": warmup_bars,
    }
    (root / f"{layer}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print("✅ Saved meta:", root / f"{layer}.meta.json")


if __name__ == "__main__":
    main()
