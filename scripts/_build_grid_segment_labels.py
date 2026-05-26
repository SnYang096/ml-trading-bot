#!/usr/bin/env python3
"""Join chop_grid segments (KPI rows) with features parquet → segment-level labeled parquet.

This is the C-system bridge for ``quick_layer_scan``. The chop_grid backtest emits
``grid_segments.csv`` (one row per segment) but quick_layer_scan expects a parquet
where each row is "an event" carrying both feature columns and a label column.

We:

1. Load segment summaries (``grid_segments.csv`` or any parquet/csv with
   columns ``symbol, start, end, pnl_per_capital, max_drawdown, trades,
   forced_exits``).
2. Load the **features parquet** used to drive the backtest (same one that
   ``train_final`` writes); pick the **first feature row of each segment** by
   matching (``symbol``, ``start``).
3. Compute multi-leg KPIs (the labels for §2.2.1 C semantic-proxy R&D):

   - ``seg_pnl_per_capital``     — total per-capital PnL (= raw summary col)
   - ``seg_max_drawdown``        — segment-internal max DD (≤ 0)
   - ``seg_total_r_over_dd``     — pnl / |dd| (NaN if dd = 0)
   - ``seg_adverse_break_rate``  — forced_exits / max(trades, 1)
   - ``seg_maker_return_per_round`` — pnl / max(trades, 1)
   - ``seg_period_5_ok``         — bool: pnl > 0 AND |dd| ≤ 5pp (placeholder
     for §2.2.1 "5/5 period 稳定性"; tune in caller)
   - ``seg_segment_total_r``     — alias of seg_pnl_per_capital × 100 (R-like)

Output is a parquet with these label columns prefixed ``seg_``, joined to the
feature columns. Pass it directly to ``quick_layer_scan condition-set --label
seg_total_r_over_dd`` etc.

Exit codes:
    0  ok
    3  bad input
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def _coerce_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")


def _features_dt_col(df: pd.DataFrame) -> str:
    for c in ("datetime", "timestamp", "ts", "time"):
        if c in df.columns:
            return c
    raise KeyError("features parquet has no datetime/timestamp/ts/time column")


def build_segment_labels(
    *,
    segments: pd.DataFrame,
    features: pd.DataFrame,
    tolerance: pd.Timedelta = pd.Timedelta("5min"),
) -> pd.DataFrame:
    """Inner-join one feature row per segment start; append seg_* KPIs.

    Uses ``merge_asof`` with backward direction within ``tolerance``: the
    feature row at-or-just-before segment ``start`` represents the entry
    context.
    """
    required_seg = {"symbol", "start", "pnl_per_capital"}
    missing = required_seg - set(segments.columns)
    if missing:
        raise KeyError(f"segments missing columns: {sorted(missing)}")

    feat_dt = _features_dt_col(features)
    seg = segments.copy()
    seg["start"] = _coerce_dt(seg["start"])
    seg = seg.dropna(subset=["start"]).sort_values(["symbol", "start"])

    feat = features.copy()
    feat[feat_dt] = _coerce_dt(feat[feat_dt])
    feat = feat.dropna(subset=[feat_dt])
    if "symbol" not in feat.columns:
        raise KeyError("features parquet missing 'symbol' column")
    feat = feat.sort_values(["symbol", feat_dt])

    merged_parts = []
    for sym, seg_sym in seg.groupby("symbol", sort=False):
        feat_sym = feat[feat["symbol"] == sym]
        if feat_sym.empty:
            continue
        m = pd.merge_asof(
            seg_sym.sort_values("start"),
            feat_sym.sort_values(feat_dt),
            left_on="start",
            right_on=feat_dt,
            direction="backward",
            tolerance=tolerance,
            suffixes=("", "_feat"),
        )
        merged_parts.append(m)
    if not merged_parts:
        return pd.DataFrame()
    out = pd.concat(merged_parts, ignore_index=True)

    pnl = pd.to_numeric(out["pnl_per_capital"], errors="coerce")
    dd = pd.to_numeric(out.get("max_drawdown", np.nan), errors="coerce")
    trades = pd.to_numeric(out.get("trades", np.nan), errors="coerce")
    forced = pd.to_numeric(out.get("forced_exits", np.nan), errors="coerce")

    out["seg_pnl_per_capital"] = pnl
    out["seg_max_drawdown"] = dd
    abs_dd = dd.abs().where(dd.abs() > 0)
    out["seg_total_r_over_dd"] = pnl / abs_dd
    safe_trades = trades.where(trades > 0)
    out["seg_adverse_break_rate"] = (forced / safe_trades).clip(lower=0, upper=1)
    out["seg_maker_return_per_round"] = pnl / safe_trades
    out["seg_segment_total_r"] = pnl * 100.0
    out["seg_period_5_ok"] = ((pnl > 0) & (dd.abs() <= 0.05)).astype("int8")

    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description="Build segment-labeled parquet for C-system quick_layer_scan."
    )
    p.add_argument(
        "--segments",
        required=True,
        help="Path to grid_segments.csv (or any csv/parquet with the required cols).",
    )
    p.add_argument(
        "--features-parquet",
        required=True,
        help="Features parquet (same one fed to chop_grid_backtest / train_final).",
    )
    p.add_argument(
        "--tolerance",
        default="5min",
        help="merge_asof tolerance (e.g. '5min', '1h'); features row at-or-before seg start.",
    )
    p.add_argument(
        "--out",
        required=True,
        help="Output parquet path (will create parent dirs).",
    )
    args = p.parse_args()

    seg_path = Path(args.segments)
    if not seg_path.is_absolute():
        seg_path = (PROJECT_ROOT / seg_path).resolve()
    feat_path = Path(args.features_parquet)
    if not feat_path.is_absolute():
        feat_path = (PROJECT_ROOT / feat_path).resolve()
    if not seg_path.exists():
        print(f"ERROR: segments not found: {seg_path}", file=sys.stderr)
        return 3
    if not feat_path.exists():
        print(f"ERROR: features parquet not found: {feat_path}", file=sys.stderr)
        return 3

    segments = _read_any(seg_path)
    features = pd.read_parquet(feat_path)
    out_df = build_segment_labels(
        segments=segments,
        features=features,
        tolerance=pd.Timedelta(args.tolerance),
    )
    if out_df.empty:
        print("ERROR: empty join — check symbol overlap / tolerance.", file=sys.stderr)
        return 3

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (PROJECT_ROOT / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    seg_cols = [c for c in out_df.columns if c.startswith("seg_")]
    print(f"wrote {out_path}  rows={len(out_df)}  seg_cols={seg_cols}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
