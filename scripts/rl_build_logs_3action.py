#!/usr/bin/env python3
"""
Build RL/BC-ready logs for 3-action Router.

Consumes:
  - nnmultihead predict outputs (preds parquet/csv file OR directory of preds_*.parquet)
  - optional rule mode outputs (mode parquet/csv file OR directory)
  - raw OHLCV (close) from data/parquet_data to compute ret_mean/ret_trend + drawdown

Produces a single logs file with columns required by:
  - mlbot rl shadow-eval-3action
  - mlbot rl counterfactual-eval-3action
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_tools.data_utils import load_raw_data  # noqa: E402
from src.time_series_model.rl.build_logs_3action import (
    BuildLogs3ActionConfig,
    build_logs_3action,
)  # noqa: E402


def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _collect_files(p: Path, *, prefix: str) -> List[Path]:
    if p.is_dir():
        files = sorted(p.glob(f"{prefix}_*.parquet"))
        if not files:
            files = sorted(p.glob("*.parquet"))
        if not files:
            files = sorted(p.glob("*.csv"))
        return files
    return [p]


def _load_multi(p: Path, *, prefix: str) -> pd.DataFrame:
    parts = []
    for f in _collect_files(p, prefix=prefix):
        df = _read_any(f)
        if "symbol" not in df.columns:
            df = df.copy()
            df["symbol"] = f.stem.replace(f"{prefix}_", "")
        parts.append(df)
    return pd.concat(parts, axis=0, ignore_index=False) if parts else pd.DataFrame()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build logs for RL/BC 3-action Router.")
    ap.add_argument(
        "--preds",
        required=True,
        help="Preds file/dir from nnmultihead predict (preds_*.parquet)",
    )
    ap.add_argument(
        "--mode",
        default=None,
        help="Optional mode file/dir from mlbot rule mode-3action",
    )
    ap.add_argument(
        "--data-path", default="data/parquet_data", help="Raw data directory"
    )
    ap.add_argument("--timeframe", default="240T", help="Timeframe (must match preds)")
    ap.add_argument("--start-date", default=None)
    ap.add_argument("--end-date", default=None)
    ap.add_argument(
        "--symbols",
        default=None,
        help="Optional symbols filter (comma-separated). If omitted, infer from preds.",
    )
    ap.add_argument(
        "--model", default=None, help="Optional model.pt to infer preds_in_log1p"
    )
    ap.add_argument(
        "--preds-in-log1p",
        default=None,
        choices=["yes", "no"],
        help="Override preds space (yes=log1p)",
    )
    ap.add_argument(
        "--returns-source",
        default="momentum_proxy",
        choices=["momentum_proxy", "rr_execution"],
        help="How to build ret_mean/ret_trend (momentum_proxy fallback or rr_execution).",
    )
    ap.add_argument(
        "--momentum-lookback",
        type=int,
        default=5,
        help="Lookback for momentum proxy used in ret_mean/ret_trend",
    )
    ap.add_argument("--output", required=True, help="Output logs path (.parquet/.csv)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    preds_path = Path(args.preds)
    mode_path = Path(args.mode) if args.mode else None
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    preds_in_log1p = True
    if args.preds_in_log1p is not None:
        preds_in_log1p = bool(args.preds_in_log1p == "yes")
    elif args.model:
        payload = torch.load(args.model, map_location="cpu")
        meta = payload.get("meta") or {}
        ds_cfg = meta.get("dataset_cfg") or {}
        preds_in_log1p = bool(ds_cfg.get("log1p_targets", True))

    preds_df = _load_multi(preds_path, prefix="preds")
    if preds_df.empty:
        raise ValueError("No preds loaded.")

    if args.symbols:
        keep = {s.strip() for s in str(args.symbols).split(",") if s.strip()}
        preds_df = preds_df[preds_df["symbol"].isin(keep)]

    symbols = sorted({str(s) for s in preds_df["symbol"].unique().tolist()})
    if not symbols:
        raise ValueError("No symbols found in preds.")

    # Load raw data for the same symbols/timeframe
    raw = load_raw_data(
        data_path=args.data_path,
        symbol=",".join(symbols),
        start_date=args.start_date,
        end_date=args.end_date,
        timeframe=args.timeframe,
    )
    # Ensure symbol column name matches
    if "symbol" not in raw.columns and "_symbol" in raw.columns:
        raw = raw.rename(columns={"_symbol": "symbol"})
    if "symbol" not in raw.columns:
        raw = raw.copy()
        raw["symbol"] = raw.get("_symbol")
    # Ensure timestamp column exists
    if "timestamp" not in raw.columns and isinstance(raw.index, pd.DatetimeIndex):
        raw = raw.copy()
        raw["timestamp"] = raw.index

    mode_df: Optional[pd.DataFrame] = None
    if mode_path is not None:
        mode_df = _load_multi(mode_path, prefix="mode")
        if args.symbols:
            mode_df = mode_df[mode_df["symbol"].isin(set(symbols))]

    cfg = BuildLogs3ActionConfig(
        momentum_lookback=int(args.momentum_lookback),
        preds_in_log1p=bool(preds_in_log1p),
        returns_source=str(args.returns_source),
    )
    logs = build_logs_3action(preds_df, raw_df=raw, cfg=cfg, mode_df=mode_df)

    if out_path.suffix.lower() == ".parquet":
        logs.to_parquet(out_path, index=False)
    else:
        logs.to_csv(out_path, index=False)

    print("✅ Saved logs:", out_path)
    print("   symbols:", ",".join(symbols))
    print("   preds_in_log1p:", preds_in_log1p)
    print("   n_rows:", len(logs))


if __name__ == "__main__":
    main()
