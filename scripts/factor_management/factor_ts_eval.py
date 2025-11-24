#!/usr/bin/env python3
"""Single-factor time-series evaluation helper."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# Ensure project root on sys.path so we can reuse existing modules
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import train_strategy as strategy_runner  # noqa: E402
from src.data_tools.data_utils import load_raw_data  # noqa: E402
from src.features.loader.strategy_feature_loader import (
    StrategyFeatureLoader,
)  # noqa: E402
from src.strategy_config import StrategyConfigLoader  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate individual factors via the strategy pipeline"
    )
    parser.add_argument("--strategy-config", required=True, help="Path to strategy dir")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--factors", nargs="+", required=True, help="Factor columns")
    parser.add_argument("--data-path", default="data/parquet_data")
    parser.add_argument("--timeframe", default="15T")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument(
        "--quantile", type=float, default=0.2, help="Top/Bottom quantile"
    )
    parser.add_argument(
        "--feature-mode",
        choices=["strategy", "only", "append"],
        default="strategy",
        help="How to handle feature pipeline: use strategy defaults, only requested factors, or append requested factors.",
    )
    parser.add_argument("--output-dir", default="results/factor_ts_eval")
    return parser.parse_args()


def _compute_requested_features(
    df_raw: pd.DataFrame,
    feature_loader: StrategyFeatureLoader,
    requested: List[str],
    ensure_signal_cfg,
) -> pd.DataFrame:
    df_features = feature_loader.load_features_from_requested(
        df_raw,
        requested_features=requested,
        fit=True,
    )
    return strategy_runner.ensure_signal_column(df_features, ensure_signal_cfg)


def prepare_dataset(args: argparse.Namespace, strategy_cfg) -> pd.DataFrame:
    df_raw = load_raw_data(
        data_path=args.data_path,
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        timeframe=args.timeframe,
    )

    feature_loader = StrategyFeatureLoader()
    feature_mode = args.feature_mode
    extra_factors = args.factors or []

    if feature_mode == "strategy":
        df_features = strategy_runner.run_feature_pipeline(
            df_raw,
            feature_loader=feature_loader,
            pipeline_cfg=strategy_cfg.features,
            fit=True,
        )
    elif feature_mode == "only":
        if not extra_factors:
            raise ValueError("--feature-mode=only requires --factors to be specified")
        df_features = _compute_requested_features(
            df_raw,
            feature_loader,
            extra_factors,
            strategy_cfg.features.ensure_signal,
        )
    elif feature_mode == "append":
        if not extra_factors:
            raise ValueError("--feature-mode=append requires --factors to be specified")
        base_features = strategy_runner.run_feature_pipeline(
            df_raw,
            feature_loader=feature_loader,
            pipeline_cfg=strategy_cfg.features,
            fit=True,
        )
        requested_df = _compute_requested_features(
            df_raw,
            feature_loader,
            extra_factors,
            strategy_cfg.features.ensure_signal,
        )
        for col in requested_df.columns:
            if col in base_features.columns:
                continue
            base_features[col] = requested_df[col]
        df_features = base_features
    else:
        raise ValueError(f"Unsupported feature mode: {feature_mode}")

    label_func = strategy_runner.import_callable(
        strategy_cfg.labels.generator.module,
        strategy_cfg.labels.generator.function,
    )
    target_col = strategy_cfg.labels.target_column
    df_features[target_col] = label_func(
        df_features.copy(), **strategy_cfg.labels.generator.params
    )

    df_filtered = strategy_runner.apply_filters(
        df_features, strategy_cfg.labels.filters
    )
    df_filtered = strategy_runner.apply_post_label_filters(
        df_filtered,
        strategy_cfg.labels.post_label_filters,
        list(df_filtered.columns),
    )
    return df_filtered


def compute_factor_metrics(
    df: pd.DataFrame,
    factor: str,
    target_col: str,
    quantile: float,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if factor not in df.columns:
        metrics["error"] = "factor_missing"
        return metrics

    valid = df[[factor, target_col]].dropna()
    if len(valid) < 50:
        metrics["error"] = "insufficient_samples"
        metrics["n_samples"] = int(len(valid))
        return metrics

    factor_ranks = valid[factor].rank(pct=True)
    target_ranks = valid[target_col].rank(pct=True)
    rank_ic = np.corrcoef(factor_ranks, target_ranks)[0, 1]
    pearson = np.corrcoef(valid[factor], valid[target_col])[0, 1]

    high_cut = valid[factor].quantile(1 - quantile)
    low_cut = valid[factor].quantile(quantile)

    long_mask = valid[factor] >= high_cut
    short_mask = valid[factor] <= low_cut

    long_returns = valid.loc[long_mask, target_col]
    short_returns = valid.loc[short_mask, target_col]

    win_rate_long = float((long_returns > 0).mean()) if len(long_returns) else 0.0
    win_rate_short = float((short_returns < 0).mean()) if len(short_returns) else 0.0

    # Simple long-short backtest
    position = pd.Series(0.0, index=valid.index)
    position[long_mask] = 1.0
    position[short_mask] = -1.0
    strategy_ret = position.shift().fillna(0.0) * valid[target_col]
    equity_curve = strategy_ret.cumsum()
    total_return = float(equity_curve.iloc[-1])
    max_dd = float((equity_curve.cummax() - equity_curve).max())

    metrics.update(
        {
            "n_samples": int(len(valid)),
            "rank_ic": float(rank_ic),
            "pearson": float(pearson),
            "win_rate_long": win_rate_long,
            "win_rate_short": win_rate_short,
            "avg_return_long": float(long_returns.mean()) if len(long_returns) else 0.0,
            "avg_return_short": (
                float(short_returns.mean()) if len(short_returns) else 0.0
            ),
            "total_return": total_return,
            "max_drawdown": max_dd,
        }
    )
    return metrics


def main() -> None:
    args = parse_args()
    loader = StrategyConfigLoader(Path(args.strategy_config))
    strategy_cfg = loader.load()

    df = prepare_dataset(args, strategy_cfg)
    target_col = strategy_cfg.labels.target_column

    results = {}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for factor in args.factors:
        metrics = compute_factor_metrics(df, factor, target_col, args.quantile)
        results[factor] = metrics

    summary_path = output_dir / f"ts_eval_{strategy_cfg.name}_{args.symbol}.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(
            {"strategy": strategy_cfg.name, "symbol": args.symbol, "results": results},
            fh,
            indent=2,
        )

    print(f"✅ Saved time-series factor evaluation to {summary_path}")


if __name__ == "__main__":
    main()
