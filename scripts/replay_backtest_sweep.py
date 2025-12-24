#!/usr/bin/env python3
"""
Replay backtests from saved artifacts (df_test + predictions) with parameter overrides.

This avoids retraining and heavy feature recomputation when sweeping backtest-only params
like sr_fuse.max_dist_atr or RR breakeven stop.

Example:
  python3 scripts/replay_backtest_sweep.py \
    --artifacts-dir results/strategies_compare_allcand_ticks_2023_2025_test30_v1/sr_reversal_rr_reg_long \
    --strategy-config config/strategies/sr_reversal_rr_reg_long \
    --grid "nofuse,be0;nofuse,be1;fuse2,be0;fuse2,be1;fuse3,be0;fuse3,be1;fuse4,be0;fuse4,be1"
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from time_series_model.strategies.backtesting.vectorbt_backtest import VectorBTBacktest


@dataclass(frozen=True)
class GridPoint:
    sr_fuse_enabled: bool
    max_dist_atr: Optional[float]
    breakeven: bool

    def name(self) -> str:
        fuse = "nofuse" if not self.sr_fuse_enabled else f"fuse{self.max_dist_atr:g}"
        be = "be1" if self.breakeven else "be0"
        return f"{fuse}_{be}"


def load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def parse_token(tok: str) -> GridPoint:
    parts = [p.strip() for p in tok.split(",") if p.strip()]
    sr_fuse_enabled = True
    max_dist_atr: Optional[float] = None
    breakeven = False
    for p in parts:
        if p.startswith("nofuse"):
            sr_fuse_enabled = False
            max_dist_atr = None
        elif p.startswith("fuse"):
            sr_fuse_enabled = True
            max_dist_atr = float(p.replace("fuse", ""))
        elif p == "be1":
            breakeven = True
        elif p == "be0":
            breakeven = False
        else:
            raise ValueError(f"Unknown grid token part: {p}")
    if sr_fuse_enabled and max_dist_atr is None:
        raise ValueError("fuse requires a numeric max_dist_atr, e.g. fuse2")
    return GridPoint(
        sr_fuse_enabled=sr_fuse_enabled, max_dist_atr=max_dist_atr, breakeven=breakeven
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--artifacts-dir",
        required=True,
        help="Directory containing backtest_df_test.parquet and backtest_preds.npy",
    )
    ap.add_argument(
        "--strategy-config",
        required=True,
        help="Strategy config dir (to load base backtest params)",
    )
    ap.add_argument(
        "--grid",
        default="nofuse,be0;nofuse,be1;fuse2,be0;fuse2,be1;fuse3,be0;fuse3,be1;fuse4,be0;fuse4,be1",
    )
    ap.add_argument("--task-type", default="regression")
    args = ap.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    df_path = artifacts_dir / "backtest_df_test.parquet"
    pred_path = artifacts_dir / "backtest_preds.npy"
    if not df_path.exists() or not pred_path.exists():
        raise FileNotFoundError(
            f"Missing artifacts. Expected {df_path} and {pred_path}. "
            "Re-run training with backtest.params.save_artifacts=true."
        )

    df = pd.read_parquet(df_path)
    preds = np.load(pred_path)

    cfg_dir = Path(args.strategy_config)
    bt_yaml = load_yaml(cfg_dir / "backtest.yaml")
    params = ((bt_yaml.get("backtest") or {}).get("params") or {}).copy()

    # Ensure strategy direction/name are set, matching train_strategy_pipeline.run_backtest_with_strategy()
    # so RR-exit direction constraints behave identically during replay.
    if "strategy_name" not in params:
        params["strategy_name"] = cfg_dir.name
    if "strategy_direction" not in params:
        direction = None
        labels_path = cfg_dir / "labels.yaml"
        if labels_path.exists():
            labels_yaml = load_yaml(labels_path)
            combine_mode = (
                ((labels_yaml.get("labels") or {}).get("generator") or {}).get("params")
                or {}
            ).get("combine_mode")
            if combine_mode == "long_only":
                direction = "long_only"
            elif combine_mode == "short_only":
                direction = "short_only"
        if direction is None:
            # fall back to name heuristics used by VectorBTBacktest
            name = str(params.get("strategy_name", "")).lower()
            if "_long" in name or name.endswith("_long"):
                direction = "long_only"
            elif "_short" in name or name.endswith("_short"):
                direction = "short_only"
            else:
                direction = "both"
        params["strategy_direction"] = direction

    # Always enable backtest for replay
    params["enabled"] = True
    params.setdefault("debug", False)

    points: List[GridPoint] = [
        parse_token(t) for t in args.grid.split(";") if t.strip()
    ]

    backtester = VectorBTBacktest()
    rows: List[Dict[str, Any]] = []

    for gp in points:
        p = json.loads(json.dumps(params, default=str))  # deep-ish copy
        p.setdefault("rr", {})
        p["rr"]["use_breakeven_stop"] = bool(gp.breakeven)

        p.setdefault("sr_fuse", {})
        p["sr_fuse"]["enabled"] = bool(gp.sr_fuse_enabled)
        if gp.sr_fuse_enabled:
            p["sr_fuse"]["max_dist_atr"] = float(gp.max_dist_atr)
            p["sr_fuse"]["dist_is_pct"] = True

        res = backtester.run(df, preds, task_type=str(args.task_type), **p) or {}
        rows.append(
            {
                "run": gp.name(),
                "sr_fuse_enabled": gp.sr_fuse_enabled,
                "max_dist_atr": gp.max_dist_atr if gp.sr_fuse_enabled else None,
                "breakeven": gp.breakeven,
                "ret_pct": res.get("total_return_pct"),
                "sharpe": res.get("sharpe"),
                "dd_pct": res.get("max_drawdown_pct"),
                "win_rate": res.get("win_rate"),
                "trades": res.get("total_trades"),
            }
        )

    out_path = artifacts_dir / "backtest_replay_sweep.json"
    out_path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")

    try:
        dfr = pd.DataFrame(rows).sort_values(
            ["sr_fuse_enabled", "max_dist_atr", "breakeven"], na_position="first"
        )
        print(dfr.to_string(index=False))
    except Exception:
        print(json.dumps(rows, indent=2, default=str))

    print(f"\n✅ Saved: {out_path}")


if __name__ == "__main__":
    main()
