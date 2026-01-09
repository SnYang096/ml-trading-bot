from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.time_series_model.rl.counterfactual_eval_3action import (
    CounterfactualEvalConfig,
    train_and_counterfactual_eval_bc3,
)
from src.time_series_model.rl.sim_env_3action import SimEnvConfig
from src.time_series_model.rl.walk_forward import WalkForwardSplitConfig
from src.time_series_model.rule.router_3action import Rule3ActionConfig


def _read_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() in {".parquet"}:
        return pd.read_parquet(p)
    return pd.read_csv(p)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Counterfactual eval Rule vs BC(3-action) using ret_mean/ret_trend columns."
    )
    ap.add_argument(
        "--logs",
        required=True,
        help="Path to logs .csv/.parquet with mode + ret_mean/ret_trend + head_*",
    )
    ap.add_argument(
        "--out", required=True, help="Output directory for report artifacts."
    )
    ap.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Train ratio per symbol (time-ordered).",
    )
    ap.add_argument(
        "--entry_delay", type=int, default=0, help="Entry delay steps for sim."
    )
    ap.add_argument(
        "--cost_per_turnover", type=float, default=0.0, help="Cost per turnover unit."
    )
    ap.add_argument(
        "--slippage_bps", type=float, default=0.0, help="Slippage bps per abs exposure."
    )
    ap.add_argument(
        "--preds-in-log1p",
        type=int,
        default=1,
        help="Whether head_mfe/head_mae/head_t_to_mfe are in log1p space (1=yes, 0=no).",
    )
    ap.add_argument(
        "--router-mfe-min",
        type=float,
        default=None,
        help="Router threshold override: mfe_min (for report diagnostics).",
    )
    ap.add_argument(
        "--router-eff-min",
        type=float,
        default=None,
        help="Router threshold override: eff_min (for report diagnostics).",
    )
    ap.add_argument(
        "--router-dir-conf-trend-min",
        type=float,
        default=None,
        help="Router threshold override: dir_conf_trend_min (for report diagnostics).",
    )
    args = ap.parse_args()

    df = _read_any(args.logs)
    router_cfg = Rule3ActionConfig()
    if args.router_mfe_min is not None:
        router_cfg = Rule3ActionConfig(
            mfe_min=float(args.router_mfe_min),
            eff_min=(
                float(args.router_eff_min)
                if args.router_eff_min is not None
                else float(router_cfg.eff_min)
            ),
            dir_conf_trend_min=(
                float(args.router_dir_conf_trend_min)
                if args.router_dir_conf_trend_min is not None
                else float(router_cfg.dir_conf_trend_min)
            ),
        )
    cfg = CounterfactualEvalConfig(
        split_cfg=WalkForwardSplitConfig(train_ratio=float(args.train_ratio)),
        sim_cfg=SimEnvConfig(
            entry_delay=int(args.entry_delay),
            cost_per_turnover=float(args.cost_per_turnover),
            slippage_bps=float(args.slippage_bps),
        ),
        router_cfg=router_cfg,
        preds_in_log1p=bool(int(args.preds_in_log1p)),
    )

    Path(args.out).mkdir(parents=True, exist_ok=True)
    _, metrics, _ = train_and_counterfactual_eval_bc3(
        df, cfg=cfg, out_dir=str(args.out)
    )
    print("counterfactual metrics:", metrics)
    print("saved to:", args.out)


if __name__ == "__main__":
    main()
