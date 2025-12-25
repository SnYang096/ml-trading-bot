#!/usr/bin/env python3
"""
Pure rule Router (3-action): NO_TRADE/MEAN/TREND based only on nnmultihead heads.

Input:
  - A preds parquet/csv file OR a directory containing multiple per-symbol preds_*.parquet
    produced by `mlbot nnmultihead predict` (multi-symbol mode).

Output:
  - Single parquet/csv with columns: symbol,timestamp,mode,mode_action + diagnostics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.time_series_model.rule.router_3action import (
    Rule3ActionConfig,
    compute_mode_3action,
)  # noqa: E402


def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _collect_pred_files(preds_path: Path) -> List[Path]:
    if preds_path.is_dir():
        files = sorted(preds_path.glob("preds_*.parquet"))
        if not files:
            files = sorted(preds_path.glob("*.parquet"))
        return files
    return [preds_path]


def _ensure_timestamp_col(df: pd.DataFrame, *, col: str = "timestamp") -> pd.DataFrame:
    if col in df.columns:
        return df
    if isinstance(df.index, pd.DatetimeIndex):
        out = df.copy()
        out[col] = out.index
        return out
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rule mode 3-action router based on nnmultihead heads."
    )
    p.add_argument(
        "--preds",
        required=True,
        help="Preds file (.parquet/.csv) or directory of per-symbol parquet files",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Optional model.pt to infer whether preds are log1p targets",
    )
    p.add_argument(
        "--preds-in-log1p",
        default=None,
        choices=["yes", "no"],
        help="Override preds space (yes=log1p)",
    )
    p.add_argument("--output", required=True, help="Output path (.parquet or .csv)")

    # Threshold overrides (ATR units after inverse-transform)
    p.add_argument("--mfe-min", type=float, default=None)
    p.add_argument("--eff-min", type=float, default=None)
    p.add_argument("--dir-conf-trend-min", type=float, default=None)
    p.add_argument("--mfe-trend-min", type=float, default=None)
    p.add_argument("--ttm-trend-min", type=float, default=None)
    p.add_argument("--eff-mean-min", type=float, default=None)
    p.add_argument("--ttm-mean-max", type=float, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    preds_path = Path(args.preds)
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

    cfg0 = Rule3ActionConfig()
    overrides = {
        "mfe_min": args.mfe_min,
        "eff_min": args.eff_min,
        "dir_conf_trend_min": args.dir_conf_trend_min,
        "mfe_trend_min": args.mfe_trend_min,
        "ttm_trend_min": args.ttm_trend_min,
        "eff_mean_min": args.eff_mean_min,
        "ttm_mean_max": args.ttm_mean_max,
    }
    merged_cfg = {
        **cfg0.__dict__,
        **{k: v for k, v in overrides.items() if v is not None},
    }
    cfg = Rule3ActionConfig(**merged_cfg)

    parts = []
    for f in _collect_pred_files(preds_path):
        df = _ensure_timestamp_col(_read_any(f), col="timestamp")
        if "symbol" not in df.columns:
            # infer symbol from filename if possible
            sym = f.stem.replace("preds_", "")
            df["symbol"] = sym
        mode_df = compute_mode_3action(df, cfg=cfg, preds_in_log1p=preds_in_log1p)
        merged = df[["symbol"]].copy()
        if "timestamp" in df.columns:
            merged["timestamp"] = df["timestamp"]
        merged = merged.join(mode_df)
        parts.append(merged)

    out = pd.concat(parts, axis=0, ignore_index=True) if parts else pd.DataFrame()
    if out_path.suffix.lower() == ".parquet":
        out.to_parquet(out_path, index=False)
    else:
        out.to_csv(out_path, index=False)

    print("✅ Saved:", out_path)
    print("   preds_in_log1p:", preds_in_log1p)
    print(
        "   mode counts:",
        json.dumps(
            out["mode"].value_counts().to_dict() if "mode" in out.columns else {},
            ensure_ascii=False,
        ),
    )


if __name__ == "__main__":
    main()
