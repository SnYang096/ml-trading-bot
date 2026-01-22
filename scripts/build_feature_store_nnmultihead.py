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
    _ensure_ticks_configured,
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

    # TaskSpec-only enforcement (no legacy config mode).
    if not (cfg_dir / "derived_from_task_spec.json").exists():
        raise SystemExit(
            "ERROR: nnmultihead is TaskSpec-only.\n"
            f"Config dir is not TaskSpec-derived: {cfg_dir}\n"
            "Please run via `mlbot nnmultihead build-feature-store --task-spec ...` (recommended),\n"
            "or materialize first via `mlbot nnmultihead materialize-config-from-task-spec --task-spec ...`."
        )

    loader = StrategyConfigLoader(cfg_dir)
    cfg = loader.load()

    symbols = [s.strip() for s in str(args.symbols).split(",") if s.strip()]
    if not symbols:
        raise ValueError("No symbols provided.")

    # Use monthly cache + warmup to speed up computation without month-boundary NaNs.
    feature_loader = StrategyFeatureLoader(
        use_monthly_cache=True,
        monthly_warmup_months=int(args.warmup_months or 0),
    )
    feature_cache_version = getattr(feature_loader.computer, "cache_version", None)
    feature_cols_union: List[str] = []

    for sym in symbols:
        print(
            f"📥 Load raw: symbol={sym} timeframe={args.timeframe} "
            f"range=[{args.start_date or 'min'}..{args.end_date or 'max'}]",
            flush=True,
        )
        # If warmup_months is requested, extend the raw-load window backwards so
        # month-level rolling features don't reset with NaNs every month.
        load_start_date = args.start_date
        try:
            warmup_months = max(0, int(args.warmup_months))
        except Exception:
            warmup_months = 0
        if load_start_date and warmup_months > 0:
            load_start_date = (
                pd.Timestamp(load_start_date) - pd.DateOffset(months=warmup_months)
            ).strftime("%Y-%m-%d")

        df_raw = load_raw_data(
            data_path=args.data_path,
            symbol=sym,
            start_date=load_start_date,
            end_date=args.end_date,
            timeframe=args.timeframe,
        )
        print(f"📦 Raw loaded: symbol={sym} rows={len(df_raw)}", flush=True)
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

        # Configure tick data for features that require it (e.g., vpin)
        if not df_raw.empty:
            start_ts = df_raw.index.min().isoformat()
            end_ts = df_raw.index.max().isoformat()
            try:
                # requested can be a list or a dict
                if isinstance(requested, dict):
                    requested_features_list = requested.get("required", [])
                    if isinstance(requested.get("optional_blocks"), dict):
                        # optional_blocks is a dict of {block_name: [feature_nodes]}
                        for block_features in requested.get(
                            "optional_blocks", {}
                        ).values():
                            if isinstance(block_features, list):
                                requested_features_list.extend(block_features)
                elif isinstance(requested, list):
                    requested_features_list = requested
                else:
                    requested_features_list = []

                if requested_features_list:
                    _ensure_ticks_configured(
                        feature_loader=feature_loader,
                        symbol=sym,
                        data_path=args.data_path,
                        start_ts=start_ts,
                        end_ts=end_ts,
                        requested_features=requested_features_list,
                    )
                    print(f"✅ Configured tick data for {sym}", flush=True)
            except ValueError as e:
                # If tick data is missing but not required, continue (will fail later if actually needed)
                print(f"⚠️  Tick configuration warning for {sym}: {e}", flush=True)
            except Exception as e:
                print(f"⚠️  Tick configuration error for {sym}: {e}", flush=True)

        # Precompute expected output columns for requested feature nodes
        features_cfg = feature_loader.feature_deps.get("features", {}) or {}
        expected_cols: List[str] = []
        for feature_name in requested:
            if feature_name in features_cfg:
                outs = features_cfg[feature_name].get("output_columns", [feature_name])
                if isinstance(outs, list):
                    expected_cols.extend([str(c) for c in outs])
        expected_cols = sorted(set([c for c in expected_cols if c]))

        # output_col -> feature node (for repairing missing cols)
        out2node = {}
        for feat_name, feat_info in features_cfg.items():
            outs = feat_info.get("output_columns", [feat_name]) or [feat_name]
            for c in outs:
                out2node[str(c)] = str(feat_name)

        warmup_months = max(0, int(args.warmup_months))
        warmup_bars = max(0, int(args.warmup_bars))

        # Monthly cache + warmup path (avoid full contiguous compute).
        use_contiguous = False
        df_feats_full = None

        monthly_groups = df_raw.groupby(pd.Grouper(freq="M"))
        # If we loaded extra warmup months, skip writing months before the
        # requested start_date (keeps output range stable).
        write_start_ts = pd.Timestamp(args.start_date) if args.start_date else None
        for period, df_month_raw in monthly_groups:
            if df_month_raw.empty:
                continue
            month_str = period.strftime("%Y-%m")
            if write_start_ts is not None:
                month_start = df_month_raw.index.min()
                # Align timezone if needed (df index is often UTC tz-aware).
                if month_start.tzinfo is not None and write_start_ts.tzinfo is None:
                    write_start_ts = write_start_ts.tz_localize(month_start.tzinfo)
                if month_start < write_start_ts:
                    continue
            if store.has_month(spec, month_str):
                # Repair mode: if existing month parquet is missing expected columns, compute ONLY missing nodes
                # and overwrite the month file while preserving existing columns.
                try:
                    import pyarrow.parquet as pq

                    parquet_path, _ = store._file_paths(spec, month_str)  # type: ignore[attr-defined]
                    schema_cols = set(pq.ParquetFile(str(parquet_path)).schema.names)
                    missing = [c for c in expected_cols if c not in schema_cols]
                    if missing:
                        print(
                            f"🩹 Repair month (missing {len(missing)} cols): symbol={sym} timeframe={args.timeframe} month={month_str}",
                            flush=True,
                        )
                        # Load existing month (already contains base + many features)
                        df_existing = store.read_month(spec, month_str)
                        # Determine minimal set of nodes needed to produce missing cols
                        needed_nodes = sorted(
                            {
                                out2node.get(c)
                                for c in missing
                                if out2node.get(c) is not None
                            }
                        )
                        if needed_nodes:
                            df_fixed = feature_loader.load_features_from_requested(
                                df_existing, requested_features=needed_nodes, fit=True
                            )
                            # Merge back only missing columns
                            for c in missing:
                                if c in df_fixed.columns:
                                    df_existing[c] = df_fixed[c]

                            store.write_month(
                                spec,
                                month_str,
                                df_existing,
                                base_columns=base_cols,
                                feature_columns=[
                                    c
                                    for c in df_existing.columns
                                    if c not in set(base_cols)
                                ],
                                overwrite=True,
                                metadata={
                                    "config_dir": str(cfg_dir),
                                    "warmup_months": int(args.warmup_months),
                                    "warmup_bars": int(args.warmup_bars),
                                    "feature_cache_version": feature_cache_version,
                                    "repair_missing_cols": missing[:50],
                                },
                            )
                            continue
                    print(
                        f"↩️  Skip existing month: symbol={sym} timeframe={args.timeframe} month={month_str}",
                        flush=True,
                    )
                    continue
                except Exception:
                    print(
                        f"↩️  Skip existing month: symbol={sym} timeframe={args.timeframe} month={month_str}",
                        flush=True,
                    )
                    continue

            month_start = df_month_raw.index.min()
            month_end = df_month_raw.index.max()

            if use_contiguous and df_feats_full is not None:
                df_feats_month = df_feats_full.loc[
                    (df_feats_full.index >= month_start)
                    & (df_feats_full.index <= month_end)
                ]
                print(
                    f"⚙️  Compute month (contiguous): symbol={sym} timeframe={args.timeframe} month={month_str} "
                    f"rows={len(df_feats_month)}",
                    flush=True,
                )
            else:
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

                print(
                    f"⚙️  Compute month: symbol={sym} timeframe={args.timeframe} month={month_str} "
                    f"window=[{str(df_window.index.min())}..{str(df_window.index.max())}] rows={len(df_window)}",
                    flush=True,
                )
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
                # IMPORTANT:
                # - If base_columns is provided and feature_columns is None, FeatureStore will ONLY keep base columns.
                # - For nnmultihead we need the full wide feature table (base + computed features) for training/eval.
                # So we explicitly pass feature_columns = all non-base columns.
                base_columns=base_cols,
                feature_columns=[
                    c for c in df_feats_month.columns if c not in set(base_cols)
                ],
                overwrite=False,
                metadata={
                    "config_dir": str(cfg_dir),
                    "warmup_months": warmup_months,
                    "warmup_bars": warmup_bars,
                    "contiguous_features": bool(use_contiguous),
                    "contiguous_rows": int(len(df_raw)),
                    "feature_cache_version": feature_cache_version,
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
