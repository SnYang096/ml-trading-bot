#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.time_series_model.diagnostics.extinction_replay_3action import (  # noqa: E402
    ExtinctionReplayConfig,
    run_extinction_replay_3action,
)
from src.time_series_model.diagnostics.ood_config import (
    load_ood_config_v1,
)  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extinction replay runner for 3-action logs."
    )
    p.add_argument("--logs", required=True, help="Path to logs_3action.parquet")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--ood-config", default="config/ood/ood_config_v1.yaml")
    p.add_argument(
        "--ood-score-col", default=None, help="Optional column name for ood_score"
    )
    p.add_argument(
        "--survival-prob-col",
        default=None,
        help="Optional column name for survival_prob",
    )
    p.add_argument("--survival-horizon-bars", type=int, default=50)
    p.add_argument("--equity-floor-frac", type=float, default=0.5)
    p.add_argument("--dd-floor", type=float, default=0.5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.logs)
    ood_cfg = load_ood_config_v1(args.ood_config) if args.ood_config else None

    cfg = ExtinctionReplayConfig(
        survival_horizon_bars=int(args.survival_horizon_bars),
        equity_floor_frac=float(args.equity_floor_frac),
        dd_floor=float(args.dd_floor),
    )
    report, sim, labels = run_extinction_replay_3action(
        df,
        cfg=cfg,
        ood_cfg=ood_cfg if (args.ood_score_col and args.survival_prob_col) else None,
        ood_score_col=args.ood_score_col,
        survival_prob_col=args.survival_prob_col,
    )

    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    sim.to_parquet(out_dir / "sim.parquet", index=False)
    labels.to_parquet(out_dir / "labels.parquet", index=False)
    print(f"[ok] wrote: {out_dir}/report.json, sim.parquet, labels.parquet")


if __name__ == "__main__":
    main()
