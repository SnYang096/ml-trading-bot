#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.time_series_model.strategies.backtesting.vectorbt_backtest import (
    VectorBTBacktest,
)


def _load_meta(meta_path: Path) -> Dict[str, Any]:
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _resolve_paths(
    artifacts_dir: Optional[str],
    meta: Optional[str],
    df: Optional[str],
    preds: Optional[str],
):
    if artifacts_dir:
        base = Path(artifacts_dir)
        return {
            "meta": base / "backtest_artifacts_meta.json",
            "df": base / "backtest_df_test.parquet",
            "preds": base / "backtest_preds.npy",
        }
    if not meta or not df or not preds:
        raise ValueError("Provide --artifacts-dir or all of --meta --df --preds")
    return {"meta": Path(meta), "df": Path(df), "preds": Path(preds)}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export vectorbt trades to JSON.")
    ap.add_argument("--artifacts-dir", default=None)
    ap.add_argument("--meta", default=None)
    ap.add_argument("--df", default=None)
    ap.add_argument("--preds", default=None)
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--max-trades", type=int, default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    paths = _resolve_paths(args.artifacts_dir, args.meta, args.df, args.preds)
    meta = _load_meta(paths["meta"])
    df = pd.read_parquet(paths["df"])
    preds = np.load(paths["preds"])

    if len(df) != len(preds):
        raise ValueError(f"Length mismatch: df={len(df)} preds={len(preds)}")

    task_type = str(meta.get("task_type", "binary"))
    backtest_params = meta.get("backtest_params", {}) or {}
    backtest_params = dict(backtest_params)
    backtest_params["debug"] = True

    bt = VectorBTBacktest()
    result = bt.run(df, preds, task_type=task_type, **backtest_params) or {}
    trades = []
    if isinstance(result, dict):
        debug = result.get("debug") or {}
        trades = debug.get("trades") or []
    if args.max_trades is not None and args.max_trades > 0:
        trades = trades[: int(args.max_trades)]

    payload = {
        "trades": trades,
        "meta": {
            "artifacts_dir": (
                str(Path(args.artifacts_dir)) if args.artifacts_dir else None
            ),
            "meta_path": str(paths["meta"]),
            "df_path": str(paths["df"]),
            "preds_path": str(paths["preds"]),
            "task_type": task_type,
            "n_trades": int(len(trades)),
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {len(trades)} trades to {out_path}")


if __name__ == "__main__":
    main()
