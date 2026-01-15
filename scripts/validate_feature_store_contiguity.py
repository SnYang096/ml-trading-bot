#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# Ensure repo root is importable when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_tools.data_utils import load_raw_data
from src.feature_store.feature_store import FeatureStore, FeatureStoreSpec
from src.features.loader.strategy_feature_loader import StrategyFeatureLoader
from src.time_series_model.strategy_config.loader import StrategyConfigLoader


BASE_COLS = {"open", "high", "low", "close", "volume", "_symbol", "symbol"}


def _month_range(start: str, end: str) -> List[str]:
    s = pd.Timestamp(start).to_period("M").to_timestamp()
    e = pd.Timestamp(end).to_period("M").to_timestamp()
    months = []
    cur = s
    while cur <= e:
        months.append(cur.strftime("%Y-%m"))
        cur = (cur + pd.offsets.MonthBegin(1)).to_period("M").to_timestamp()
    return months


def _read_feature_store_df(
    *,
    root: Path,
    layer: str,
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    store = FeatureStore(root)
    spec = FeatureStoreSpec(layer=layer, symbol=symbol, timeframe=timeframe)
    months = _month_range(start_date, end_date)
    parts = []
    for m in months:
        if not store.has_month(spec, m):
            continue
        df_m = store.read_month(spec, m)
        parts.append(df_m)
    if not parts:
        raise ValueError(
            f"No FeatureStore months found for {symbol} {timeframe} {layer}"
        )
    df = pd.concat(parts, axis=0).sort_index()
    idx = pd.to_datetime(df.index, utc=True, errors="coerce")
    df.index = idx
    df = df[~df.index.isna()]
    df = df[
        (df.index >= pd.Timestamp(start_date, tz="UTC"))
        & (df.index <= pd.Timestamp(end_date, tz="UTC"))
    ]
    return df


def _select_numeric_cols(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        if c in BASE_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Validate FeatureStore contiguity (no month-boundary reset) by comparing against direct compute."
    )
    ap.add_argument("--config", required=True, help="Derived nnmultihead config dir")
    ap.add_argument("--feature-store-root", default="feature_store")
    ap.add_argument("--layer", required=True)
    ap.add_argument("--symbols", required=True, help="Comma-separated symbols")
    ap.add_argument("--timeframe", default="240T")
    ap.add_argument("--data-path", default="data/parquet_data")
    ap.add_argument("--start-date", required=True)
    ap.add_argument("--end-date", required=True)
    ap.add_argument("--warmup-months", type=int, default=3)
    ap.add_argument("--max-nan-rate", type=float, default=0.01)
    ap.add_argument("--tol-abs", type=float, default=1e-6)
    ap.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable disk/memory cache for fresh recompute.",
    )
    ap.add_argument(
        "--check-prefixes",
        default="hilbert_,spectrum_,hurst_",
        help="Comma-separated prefixes to enforce low NaN rate.",
    )
    ap.add_argument(
        "--compare-prefixes",
        default="hilbert_,spectrum_,hurst_",
        help="Comma-separated prefixes to compare for value equality.",
    )
    ap.add_argument("--out", default=None, help="Optional JSON output path.")
    args = ap.parse_args()

    cfg_dir = Path(args.config).resolve()
    loader = StrategyConfigLoader(cfg_dir)
    cfg = loader.load()

    feature_loader = StrategyFeatureLoader(
        use_monthly_cache=False,
        use_disk_cache=not args.no_cache,
        use_memory_cache=not args.no_cache,
    )
    requested = cfg.features.requested_features

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    prefixes = [p.strip() for p in args.check_prefixes.split(",") if p.strip()]
    cmp_prefixes = [p.strip() for p in args.compare_prefixes.split(",") if p.strip()]

    results = {"symbols": {}, "status": "ok"}

    for sym in symbols:
        start_date = pd.Timestamp(args.start_date, tz="UTC")
        end_date = pd.Timestamp(args.end_date, tz="UTC")
        warm_start = (
            start_date - pd.DateOffset(months=int(args.warmup_months))
        ).strftime("%Y-%m-%d")
        df_raw = load_raw_data(
            data_path=args.data_path,
            symbol=sym,
            start_date=warm_start,
            end_date=args.end_date,
            timeframe=args.timeframe,
        )
        df_raw = df_raw.sort_index()

        # Direct compute on full window (contiguous)
        df_full = feature_loader.load_features_from_requested(
            df_raw, requested_features=requested, fit=True
        )
        if "symbol" not in df_full.columns:
            df_full["symbol"] = sym
        idx = pd.to_datetime(df_full.index, utc=True, errors="coerce")
        df_full.index = idx
        df_full = df_full[~df_full.index.isna()]

        df_full = df_full[(df_full.index >= start_date) & (df_full.index <= end_date)]

        # FeatureStore snapshot (monthly)
        df_fs = _read_feature_store_df(
            root=Path(args.feature_store_root).resolve(),
            layer=args.layer,
            symbol=sym,
            timeframe=args.timeframe,
            start_date=args.start_date,
            end_date=args.end_date,
        )

        # Align index
        df_full = df_full.sort_index()
        df_fs = df_fs.sort_index()
        idx = df_full.index.intersection(df_fs.index)
        df_full = df_full.loc[idx]
        df_fs = df_fs.loc[idx]

        cols_all = sorted(
            set(_select_numeric_cols(df_full)).intersection(_select_numeric_cols(df_fs))
        )
        cols = [c for c in cols_all if any(c.startswith(p) for p in cmp_prefixes)]
        if not cols:
            raise ValueError(
                f"No comparable feature columns found for {sym} with prefixes={cmp_prefixes}"
            )

        max_abs_diff = 0.0
        per_col = {}
        for c in cols:
            a = pd.to_numeric(df_full[c], errors="coerce")
            b = pd.to_numeric(df_fs[c], errors="coerce")
            diff = (a - b).abs()
            diff = diff.replace([np.inf, -np.inf], np.nan)
            col_max = float(diff.max(skipna=True)) if diff.notna().any() else 0.0
            max_abs_diff = max(max_abs_diff, col_max)
            per_col[c] = col_max
        top_diffs = sorted(per_col.items(), key=lambda kv: kv[1], reverse=True)[:5]

        # NaN rate checks for key prefixes
        nan_checks = {}
        for p in prefixes:
            cols_p = [c for c in cols if c.startswith(p)]
            if not cols_p:
                continue
            nan_rate = float(df_fs[cols_p].isna().mean().mean())
            nan_checks[p] = nan_rate
            if nan_rate > args.max_nan_rate:
                results["status"] = "fail"

        if max_abs_diff > float(args.tol_abs):
            results["status"] = "fail"

        results["symbols"][sym] = {
            "rows": int(len(df_fs)),
            "max_abs_diff": max_abs_diff,
            "top_diffs": top_diffs,
            "nan_rate_by_prefix": nan_checks,
        }

    if args.out:
        Path(args.out).write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    if results["status"] != "ok":
        raise SystemExit(
            f"Validation failed: {json.dumps(results, ensure_ascii=False)}"
        )

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
