#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_store.feature_store import FeatureStore, FeatureStoreSpec


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
    cols: List[str],
) -> pd.DataFrame:
    store = FeatureStore(root)
    spec = FeatureStoreSpec(layer=layer, symbol=symbol, timeframe=timeframe)
    months = _month_range(start_date, end_date)
    parts = []
    for m in months:
        if not store.has_month(spec, m):
            continue
        df_m = store.read_month(spec, m)
        keep = [c for c in cols if c in df_m.columns]
        if keep:
            parts.append(df_m[keep])
    if not parts:
        return pd.DataFrame(columns=cols)
    df = pd.concat(parts, axis=0).sort_index()
    return df


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build evidence quantiles from FeatureStore."
    )
    ap.add_argument("--feature-store-root", default="feature_store")
    ap.add_argument("--layer", required=True)
    ap.add_argument("--symbols", required=True, help="Comma-separated symbols")
    ap.add_argument("--timeframe", default="240T")
    ap.add_argument("--start-date", required=True)
    ap.add_argument("--end-date", required=True)
    ap.add_argument("--keys", default="", help="Comma-separated feature keys")
    ap.add_argument("--prefixes", default="", help="Comma-separated key prefixes")
    ap.add_argument(
        "--quantiles",
        default="0.1,0.5,0.9",
        help="Comma-separated quantiles (e.g., 0.1,0.5,0.9)",
    )
    ap.add_argument("--out", required=True, help="Output JSON file")
    ap.add_argument(
        "--global",
        dest="global_pool",
        action="store_true",
        help="Pool all symbols into one 'GLOBAL' entry",
    )
    args = ap.parse_args()

    root = Path(args.feature_store_root).resolve()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    keys = [s.strip() for s in args.keys.split(",") if s.strip()]
    prefixes = [s.strip() for s in args.prefixes.split(",") if s.strip()]
    qs = [float(x) for x in args.quantiles.split(",") if x.strip()]

    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    pool_frames = []

    for sym in symbols:
        df = _read_feature_store_df(
            root=root,
            layer=args.layer,
            symbol=sym,
            timeframe=args.timeframe,
            start_date=args.start_date,
            end_date=args.end_date,
            cols=keys if keys else [],
        )
        if prefixes:
            df = df[[c for c in df.columns if any(c.startswith(p) for p in prefixes)]]
        if df.empty:
            continue

        if args.global_pool:
            pool_frames.append(df)
            continue

        sym_out: Dict[str, Dict[str, float]] = {}
        for c in df.columns:
            if not pd.api.types.is_numeric_dtype(df[c]):
                continue
            vals = pd.to_numeric(df[c], errors="coerce").dropna()
            if vals.empty:
                continue
            sym_out[c] = {str(q): float(vals.quantile(q)) for q in qs}
        out[sym] = sym_out

    if args.global_pool and pool_frames:
        df = pd.concat(pool_frames, axis=0)
        global_out: Dict[str, Dict[str, float]] = {}
        for c in df.columns:
            if not pd.api.types.is_numeric_dtype(df[c]):
                continue
            vals = pd.to_numeric(df[c], errors="coerce").dropna()
            if vals.empty:
                continue
            global_out[c] = {str(q): float(vals.quantile(q)) for q in qs}
        out["GLOBAL"] = global_out

    Path(args.out).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
