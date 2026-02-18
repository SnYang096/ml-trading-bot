#!/usr/bin/env python3
"""
Cross-sectional factor evaluation on a multi-asset panel.

Supports:
- input panel parquet/csv with columns (timestamp, symbol, factors..., optional target)
- FeatureStore panel build (monthly partitions) then evaluate

Outputs:
- summary.csv / summary.json
- per-factor long_short_timeseries_<factor>.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cross_sectional.factor_backtest import LongShortBacktestConfig, long_short_backtest
from cross_sectional.factor_selection import (
    compute_cross_sectional_ic,
    filter_panel_by_assets,
)
from cross_sectional.feature_store_panel import (
    FeatureStorePanelConfig,
    load_feature_store_frames,
)
from cross_sectional.panel import FactorPanelBuilder, PanelConfig


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cross-sectional factor evaluation (IC + long/short spread)."
    )

    # Source
    p.add_argument(
        "--config", default=None, help="Optional YAML config (overrides CLI defaults)."
    )
    p.add_argument(
        "--input",
        default=None,
        help="Panel parquet/csv path (timestamp,symbol required if not MultiIndex).",
    )
    p.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated symbols (required for FeatureStore source).",
    )
    p.add_argument("--start-date", default=None, help="Start date (YYYY-MM-DD)")
    p.add_argument("--end-date", default=None, help="End date (YYYY-MM-DD)")

    # FeatureStore source
    p.add_argument(
        "--features-store-root", default="feature_store", help="FeatureStore root"
    )
    p.add_argument(
        "--features-store-layer", default=None, help="FeatureStore layer (features_xxx)"
    )
    p.add_argument("--timeframe", default="240T", help="Timeframe (e.g., 240T)")
    p.add_argument(
        "--columns",
        default=None,
        help="Comma-separated columns to load from FeatureStore (optional).",
    )

    # Factors/target
    p.add_argument(
        "--factors", default=None, help="Comma-separated factor columns to evaluate."
    )
    p.add_argument(
        "--factors-file",
        default=None,
        help="Text file containing factor columns (one per line).",
    )
    p.add_argument(
        "--factor-set-yaml",
        default=None,
        help="YAML containing factor_sets to reference.",
    )
    p.add_argument(
        "--factor-set", default=None, help="Factor set name in --factor-set-yaml."
    )
    p.add_argument(
        "--target",
        default=None,
        help="Target column (default: infer future_return_<horizon>).",
    )
    p.add_argument(
        "--horizon",
        type=int,
        default=12,
        help="Forward return horizon in bars for target computation.",
    )

    # Eval settings
    p.add_argument(
        "--min-assets", type=int, default=4, help="Minimum assets per timestamp."
    )
    p.add_argument("--quantiles", type=int, default=5, help="Quantiles for long/short.")
    p.add_argument(
        "--fee-bps", type=float, default=0.0, help="Fee (bps) applied to turnover."
    )

    # Output
    p.add_argument(
        "--output-dir",
        default="results/cross_sectional/factor_eval",
        help="Output directory.",
    )
    return p.parse_args()


def _read_config(path: str) -> dict:
    obj = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(obj, dict):
        raise ValueError("config YAML must be a mapping")
    return obj


def _split_csv(s: Optional[str]) -> Optional[List[str]]:
    if not s:
        return None
    return [x.strip() for x in str(s).split(",") if x.strip()]


def _load_factors_from_sources(
    *,
    factors: Optional[str],
    factors_file: Optional[str],
    factor_set_yaml: Optional[str],
    factor_set: Optional[str],
) -> List[str]:
    if factors:
        return [x.strip() for x in str(factors).split(",") if x.strip()]
    if factors_file:
        lines = Path(factors_file).read_text(encoding="utf-8").splitlines()
        return [x.strip() for x in lines if x.strip() and not x.strip().startswith("#")]
    if factor_set_yaml and factor_set:
        obj = yaml.safe_load(Path(factor_set_yaml).read_text(encoding="utf-8")) or {}
        sets = obj.get("factor_sets", {}) or {}
        # Support multiple sets: "a,b,c"
        names = [x.strip() for x in str(factor_set).split(",") if x.strip()]
        out: List[str] = []
        for name in names:
            if name not in sets:
                raise KeyError(f"factor_set '{name}' not found in {factor_set_yaml}")
            vals = sets.get(name) or []
            out.extend([str(x).strip() for x in vals if str(x).strip()])
        # stable unique order
        return list(dict.fromkeys(out))
    raise ValueError(
        "Must provide one of --factors / --factors-file / (--factor-set-yaml + --factor-set)"
    )


def _load_panel_from_input(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)
    if isinstance(df.index, pd.MultiIndex):
        panel = df
    else:
        if {"timestamp", "symbol"}.issubset(df.columns):
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.dropna(subset=["timestamp", "symbol"])
            panel = df.set_index(["timestamp", "symbol"])
        else:
            raise ValueError(
                "Input panel must have MultiIndex or columns: timestamp,symbol"
            )
    # Force UTC datetime
    ts = pd.to_datetime(panel.index.get_level_values(0), utc=True, errors="coerce")
    if ts.isna().any():
        raise ValueError("NaT detected in timestamp index")
    panel.index = pd.MultiIndex.from_arrays(
        [ts, panel.index.get_level_values(1)], names=["timestamp", "symbol"]
    )
    return panel


def _ensure_target(
    panel: pd.DataFrame, *, horizon: int, target: Optional[str]
) -> Tuple[pd.DataFrame, str]:
    if target:
        if target not in panel.columns:
            raise KeyError(f"Target column not found: {target}")
        return panel, target
    inferred = f"future_return_{int(horizon)}"
    if inferred in panel.columns:
        return panel, inferred

    # compute from close
    if "close" not in panel.columns:
        raise KeyError(
            "Cannot infer target: missing close and no future_return_* present. Provide --target."
        )
    cfg = PanelConfig(
        timestamp_col="timestamp",
        symbol_col="symbol",
        target_col=inferred,
        forward_return_horizon=int(horizon),
        min_assets_per_ts=2,
        fill_method=None,
        align_intersection_only=False,
        check_duplicates=False,
        sort_index=True,
        dropna_after_fill=False,
    )
    builder = FactorPanelBuilder(cfg)
    flat = panel.reset_index()
    panel2 = builder.from_concat_frame(flat)
    return panel2, inferred


def _build_panel_from_feature_store(
    *,
    symbols: Sequence[str],
    start_date: str,
    end_date: str,
    features_store_root: str,
    features_store_layer: str,
    timeframe: str,
    columns: Optional[Sequence[str]],
) -> pd.DataFrame:
    cfg = FeatureStorePanelConfig(
        root=str(features_store_root),
        layer=str(features_store_layer),
        timeframe=str(timeframe),
        timestamp_col="timestamp",
        symbol_col="symbol",
    )
    df = load_feature_store_frames(
        symbols=list(symbols),
        cfg=cfg,
        start_date=str(start_date),
        end_date=str(end_date),
        columns=list(columns) if columns else None,
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "symbol"])
    return df.set_index(["timestamp", "symbol"])


def main() -> None:
    args = _parse_args()
    cfg_obj: dict = {}
    if args.config:
        cfg_obj = _read_config(args.config)

    # Resolve inputs with config fallback
    input_path = args.input or cfg_obj.get("input")
    symbols_csv = args.symbols or cfg_obj.get("symbols")
    start_date = args.start_date or cfg_obj.get("start_date")
    end_date = args.end_date or cfg_obj.get("end_date")

    # Load panel
    if input_path:
        panel = _load_panel_from_input(str(input_path))
    else:
        # FeatureStore-based build
        features_store_layer = args.features_store_layer or cfg_obj.get(
            "features_store_layer"
        )
        if not features_store_layer:
            raise ValueError(
                "Need --input or --features-store-layer (FeatureStore source)."
            )
        if not symbols_csv:
            raise ValueError("FeatureStore source requires --symbols")
        if not start_date or not end_date:
            raise ValueError("FeatureStore source requires --start-date and --end-date")
        symbols = _split_csv(symbols_csv) or []
        cols = _split_csv(args.columns) or cfg_obj.get("columns")
        if isinstance(cols, str):
            cols = _split_csv(cols)
        panel = _build_panel_from_feature_store(
            symbols=symbols,
            start_date=str(start_date),
            end_date=str(end_date),
            features_store_root=str(args.features_store_root),
            features_store_layer=str(features_store_layer),
            timeframe=str(args.timeframe),
            columns=cols,
        )

    panel, target_col = _ensure_target(
        panel, horizon=int(args.horizon), target=args.target or cfg_obj.get("target")
    )
    panel = filter_panel_by_assets(panel, min_assets=int(args.min_assets))

    # Factor sources (resolved AFTER panel is loaded)
    factors = args.factors or cfg_obj.get("factors")
    factors_file = args.factors_file or cfg_obj.get("factors_file")
    factor_set_yaml = args.factor_set_yaml or cfg_obj.get("factor_set_yaml")
    factor_set = args.factor_set or cfg_obj.get("factor_set")

    if factors or factors_file:
        factor_cols = _load_factors_from_sources(
            factors=factors,
            factors_file=factors_file,
            factor_set_yaml=None,
            factor_set=None,
        )
    else:
        factor_cols = _load_factors_from_sources(
            factors=None,
            factors_file=None,
            factor_set_yaml=factor_set_yaml,
            factor_set=factor_set,
        )

    # IC metrics (fast summary)
    ic_df = compute_cross_sectional_ic(
        panel,
        factor_cols=factor_cols,
        target_col=target_col,
        min_assets=int(args.min_assets),
    )

    # Backtest per factor (long/short)
    outdir = Path(
        args.output_dir
        or cfg_obj.get("output_dir")
        or "results/cross_sectional/factor_eval"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, object]] = []
    for f in factor_cols:
        if f not in panel.columns:
            # tolerate missing factors (e.g. factor list from YAML larger than panel)
            continue
        bt_cfg = LongShortBacktestConfig(
            quantiles=int(args.quantiles),
            fee_bps=float(args.fee_bps),
            min_assets=int(args.min_assets),
        )
        ts_df, metrics = long_short_backtest(
            panel, factor_col=f, target_col=target_col, cfg=bt_cfg
        )
        if not ts_df.empty:
            ts_path = outdir / f"long_short_timeseries__{f}.csv"
            ts_df.to_csv(ts_path)

        ic_row = (
            ic_df.loc[f].to_dict() if (not ic_df.empty and f in ic_df.index) else {}
        )
        row = {"factor": f, "target": target_col, **ic_row, **metrics}
        summary_rows.append(row)

    summary = (
        pd.DataFrame(summary_rows).set_index("factor")
        if summary_rows
        else pd.DataFrame()
    )
    if not summary.empty:
        summary = summary.sort_values(by="ic_mean", ascending=False, na_position="last")

    summary_csv = outdir / "summary.csv"
    summary_json = outdir / "summary.json"
    summary.to_csv(summary_csv)
    summary_json.write_text(
        json.dumps(
            {
                "target": target_col,
                "min_assets": int(args.min_assets),
                "quantiles": int(args.quantiles),
                "fee_bps": float(args.fee_bps),
                "summary": summary.reset_index().to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"✅ Saved: {summary_csv}")
    print(f"✅ Saved: {summary_json}")


if __name__ == "__main__":
    main()
