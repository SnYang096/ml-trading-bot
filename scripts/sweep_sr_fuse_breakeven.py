#!/usr/bin/env python3
"""
Sweep sr_fuse.max_dist_atr and RR breakeven stop for sr_reversal_rr_reg_long.

Why this script:
- sr_fuse and breakeven affect *backtesting* (trade generation/exits), not label generation.
- We want a reproducible grid search with results written to distinct directories.

Usage:
  python3 scripts/sweep_sr_fuse_breakeven.py \
    --symbol BTCUSDT --timeframe 240T \
    --start 2023-01-01 --end 2025-10-31 \
    --test-size 0.3
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass(frozen=True)
class SweepCfg:
    sr_fuse_enabled: bool
    sr_fuse_max_dist_atr: Optional[float]
    breakeven: bool

    def run_id(self) -> str:
        if not self.sr_fuse_enabled:
            fuse = "nofuse"
        else:
            fuse = f"fuse{self.sr_fuse_max_dist_atr:g}"
        be = "be1" if self.breakeven else "be0"
        return f"{fuse}_{be}"


def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _dump_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _set_nested(d: Dict[str, Any], keys: List[str], value: Any) -> None:
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def _run_one(
    *,
    base_backtest_yaml: Dict[str, Any],
    cfg: SweepCfg,
    backtest_yaml_path: Path,
    symbol: str,
    timeframe: str,
    test_size: float,
    out_root: Path,
    env: Dict[str, str],
) -> Dict[str, Any]:
    # Patch YAML for this run
    bt = copy.deepcopy(base_backtest_yaml)

    # rr.use_breakeven_stop
    _set_nested(
        bt, ["backtest", "params", "rr", "use_breakeven_stop"], bool(cfg.breakeven)
    )

    # sr_fuse
    _set_nested(
        bt, ["backtest", "params", "sr_fuse", "enabled"], bool(cfg.sr_fuse_enabled)
    )
    if cfg.sr_fuse_enabled:
        _set_nested(
            bt,
            ["backtest", "params", "sr_fuse", "max_dist_atr"],
            float(cfg.sr_fuse_max_dist_atr),
        )
        _set_nested(bt, ["backtest", "params", "sr_fuse", "dist_is_pct"], True)

    _dump_yaml(backtest_yaml_path, bt)

    run_root = out_root / cfg.run_id()
    run_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python3",
        "scripts/train_strategy_pipeline.py",
        "--config",
        "config/strategies/sr_reversal_rr_reg_long",
        "--symbol",
        symbol,
        "--timeframe",
        timeframe,
        "--test-size",
        str(test_size),
        "--output-root",
        str(run_root),
    ]

    print(f"\n=== RUN {cfg.run_id()} ===")
    subprocess.run(cmd, check=True, env=env)

    res_path = run_root / "sr_reversal_rr_reg_long" / "results.json"
    d = json.loads(res_path.read_text(encoding="utf-8"))
    btres = d.get("backtest", {}) or {}
    ev = d.get("evaluation", {}) or {}
    return {
        "run": cfg.run_id(),
        "sr_fuse_enabled": cfg.sr_fuse_enabled,
        "max_dist_atr": cfg.sr_fuse_max_dist_atr if cfg.sr_fuse_enabled else None,
        "breakeven": cfg.breakeven,
        "cv": d.get("avg_cv_metric"),
        "corr": ev.get("test_correlation") or ev.get("pearson_correlation"),
        "ret_pct": btres.get("total_return_pct"),
        "sharpe": btres.get("sharpe"),
        "dd_pct": btres.get("max_drawdown_pct"),
        "trades": btres.get("total_trades"),
        "results_path": str(res_path),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--timeframe", default="240T")
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default="2025-10-31")
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument(
        "--output-root",
        default="results/sweep_sr_fuse_breakeven",
        help="Base output root; each run writes into a subdir under this path.",
    )
    args = ap.parse_args()

    backtest_yaml_path = Path("config/strategies/sr_reversal_rr_reg_long/backtest.yaml")
    base_backtest_yaml = _load_yaml(backtest_yaml_path)
    base_backup = backtest_yaml_path.read_text(encoding="utf-8")

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Ensure same date window as backtest
    env = dict(os.environ)
    env["TRAIN_START_DATE"] = args.start
    env["TRAIN_END_DATE"] = args.end

    sweep: List[SweepCfg] = [
        SweepCfg(sr_fuse_enabled=False, sr_fuse_max_dist_atr=None, breakeven=False),
        SweepCfg(sr_fuse_enabled=False, sr_fuse_max_dist_atr=None, breakeven=True),
        SweepCfg(sr_fuse_enabled=True, sr_fuse_max_dist_atr=2.0, breakeven=False),
        SweepCfg(sr_fuse_enabled=True, sr_fuse_max_dist_atr=2.0, breakeven=True),
        SweepCfg(sr_fuse_enabled=True, sr_fuse_max_dist_atr=3.0, breakeven=False),
        SweepCfg(sr_fuse_enabled=True, sr_fuse_max_dist_atr=3.0, breakeven=True),
        SweepCfg(sr_fuse_enabled=True, sr_fuse_max_dist_atr=4.0, breakeven=False),
        SweepCfg(sr_fuse_enabled=True, sr_fuse_max_dist_atr=4.0, breakeven=True),
    ]

    rows: List[Dict[str, Any]] = []
    try:
        for cfg in sweep:
            rows.append(
                _run_one(
                    base_backtest_yaml=base_backtest_yaml,
                    cfg=cfg,
                    backtest_yaml_path=backtest_yaml_path,
                    symbol=args.symbol,
                    timeframe=args.timeframe,
                    test_size=args.test_size,
                    out_root=out_root,
                    env=env,
                )
            )
    finally:
        # Restore original backtest.yaml
        backtest_yaml_path.write_text(base_backup, encoding="utf-8")

    # Save summary
    summary_path = out_root / "sweep_summary.json"
    summary_path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print(f"\n✅ Saved sweep summary to {summary_path}")

    # Print compact table
    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        print("\n=== SUMMARY ===")
        print(
            df[
                [
                    "run",
                    "sr_fuse_enabled",
                    "max_dist_atr",
                    "breakeven",
                    "cv",
                    "corr",
                    "ret_pct",
                    "sharpe",
                    "dd_pct",
                    "trades",
                ]
            ].to_string(index=False)
        )
    except Exception:
        pass


if __name__ == "__main__":
    main()
