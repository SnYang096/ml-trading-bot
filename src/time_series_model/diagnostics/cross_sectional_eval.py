#!/usr/bin/env python3
"""Cross-sectional factor evaluation (multi-asset, multi-factor)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]  # src/diagnostics -> src -> project root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_tools.data_utils import load_raw_data  # noqa: E402
from src.features.loader.strategy_feature_loader import (
    StrategyFeatureLoader,
)  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-sectional factor evaluation")
    parser.add_argument(
        "--features-config", required=True, help="YAML file with requested features"
    )
    parser.add_argument("--symbols", required=True, help="Comma-separated symbol list")
    parser.add_argument("--data-path", default="data/parquet_data")
    parser.add_argument("--timeframe", default="240T")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument(
        "--horizon", type=int, default=24, help="Future return horizon (bars)"
    )
    parser.add_argument("--quantiles", type=int, default=5)
    parser.add_argument(
        "--ic-decay-lags",
        type=str,
        default="1,3,5",
        help="Comma separated lags (in bars) for IC decay calculation",
    )
    parser.add_argument("--min-cross-sectional", type=int, default=3)
    parser.add_argument("--output-dir", default="results/cross_sectional_eval")
    return parser.parse_args()


def load_requested_features(config_path: Path) -> List[str]:
    with open(config_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    pipeline = data.get("feature_pipeline", {})
    requested = pipeline.get("requested_features", []) or []
    if not requested:
        raise ValueError(f"No requested_features found in {config_path}")
    return requested


def compute_future_return(df: pd.DataFrame, horizon: int) -> pd.Series:
    close = df["close"].astype(float)
    future = close.shift(-horizon)
    return (future - close) / close


def compute_factor_metrics(
    df: pd.DataFrame,
    factor: str,
    target_col: str,
    quantiles: int,
    min_assets: int,
    ic_decay_lags: List[int],
) -> Tuple[Dict[str, float], pd.DataFrame]:
    metrics: Dict[str, float] = {}
    if factor not in df.columns:
        metrics["error"] = "factor_missing"
        return metrics, pd.DataFrame()

    valid = df[["timestamp", "_symbol", factor, target_col]].dropna()
    if valid.empty:
        metrics["error"] = "no_samples"
        return metrics, pd.DataFrame()

    grouped = valid.groupby("timestamp", sort=True)

    def per_timestamp_ic(group: pd.DataFrame, target: str) -> float:
        if len(group) < min_assets:
            return np.nan
        corr = spearmanr(group[factor], group[target], nan_policy="omit").correlation
        return corr if corr is not None else np.nan

    ic_series = grouped.apply(lambda g: per_timestamp_ic(g, target_col))
    metrics["rank_ic_mean"] = float(ic_series.mean(skipna=True))
    metrics["rank_ic_std"] = float(ic_series.std(skipna=True))
    metrics["rank_ic_ir"] = (
        metrics["rank_ic_mean"] / metrics["rank_ic_std"]
        if metrics["rank_ic_std"]
        else 0.0
    )

    for lag in ic_decay_lags:
        lag_col = f"{target_col}_lag{lag}"
        valid[lag_col] = valid.groupby("_symbol")[target_col].shift(-lag)
        lag_ic = grouped.apply(lambda g: per_timestamp_ic(g, lag_col))
        metrics[f"rank_ic_decay_lag_{lag}"] = float(lag_ic.mean(skipna=True))

    # Quantile portfolios
    quantile_returns = []
    for ts, group in grouped:
        if len(group) < quantiles:
            continue
        q_values = group[factor].rank(pct=True)
        long_mask = q_values >= (1 - 1 / quantiles)
        short_mask = q_values <= (1 / quantiles)
        long_ret = group.loc[long_mask, target_col].mean()
        short_ret = group.loc[short_mask, target_col].mean()
        quantile_returns.append(
            {
                "timestamp": ts,
                "long_return": long_ret,
                "short_return": short_ret,
            }
        )

    quantile_df = pd.DataFrame(quantile_returns)
    if quantile_df.empty:
        metrics["avg_long_return"] = 0.0
        metrics["avg_short_return"] = 0.0
        metrics["long_short_spread"] = 0.0
    else:
        quantile_df = quantile_df.set_index("timestamp")
        metrics["avg_long_return"] = float(quantile_df["long_return"].mean())
        metrics["avg_short_return"] = float(quantile_df["short_return"].mean())
        metrics["long_short_spread"] = (
            metrics["avg_long_return"] - metrics["avg_short_return"]
        )
        quantile_df["long_cum"] = quantile_df["long_return"].fillna(0).cumsum()
        quantile_df["short_cum"] = quantile_df["short_return"].fillna(0).cumsum()

    metrics["n_observations"] = int(len(valid))
    metrics["n_timestamps"] = int(grouped.ngroups)
    return metrics, quantile_df


def main() -> None:
    args = parse_args()
    features_config = Path(args.features_config)
    requested_features = load_requested_features(features_config)
    loader = StrategyFeatureLoader()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    all_frames = []
    for symbol in symbols:
        df_raw = load_raw_data(
            data_path=args.data_path,
            symbol=symbol,
            start_date=args.start_date,
            end_date=args.end_date,
            timeframe=args.timeframe,
        )
        df_features = loader.load_features_from_requested(
            df_raw,
            requested_features=requested_features,
            fit=True,
        )
        df_features["_symbol"] = symbol
        df_features["future_return"] = compute_future_return(df_features, args.horizon)
        df_features = df_features.dropna(subset=["future_return"])
        df_features = df_features.reset_index().rename(columns={"index": "timestamp"})
        all_frames.append(df_features)

    if not all_frames:
        raise ValueError("No data available for requested symbols.")

    combined = pd.concat(all_frames, axis=0, ignore_index=True)
    combined = combined.sort_values("timestamp")
    ic_decay_lags = [int(x.strip()) for x in args.ic_decay_lags.split(",") if x.strip()]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Dict[str, float]] = {}
    for factor in requested_features:
        metrics, quantile_df = compute_factor_metrics(
            combined,
            factor=factor,
            target_col="future_return",
            quantiles=args.quantiles,
            min_assets=args.min_cross_sectional,
            ic_decay_lags=ic_decay_lags,
        )
        summary[factor] = metrics

        if not quantile_df.empty:
            q_path = output_dir / f"{factor}_quantiles.csv"
            quantile_df.to_csv(q_path)

    summary_df = pd.DataFrame(summary).transpose()
    summary_csv = output_dir / "cross_sectional_summary.csv"
    summary_df.to_csv(summary_csv)

    summary_json = output_dir / "cross_sectional_summary.json"
    with open(summary_json, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "symbols": symbols,
                "timeframe": args.timeframe,
                "horizon": args.horizon,
                "quantiles": args.quantiles,
                "ic_decay_lags": ic_decay_lags,
                "summary": summary,
            },
            fh,
            indent=2,
        )

    html_path = output_dir / "cross_sectional_report.html"
    summary_df_html = summary_df.to_html(float_format=lambda x: f"{x:.4f}")
    html_content = f"""
    <html>
      <head><title>Cross-sectional Factor Evaluation</title></head>
      <body>
        <h1>Cross-sectional Factor Evaluation</h1>
        <p>Symbols: {', '.join(symbols)} | Timeframe: {args.timeframe} | Horizon: {args.horizon}</p>
        {summary_df_html}
      </body>
    </html>
    """
    html_path.write_text(html_content, encoding="utf-8")

    print(f"✅ Saved summary CSV to {summary_csv}")
    print(f"✅ Saved summary JSON to {summary_json}")
    print(f"✅ Saved HTML report to {html_path}")


if __name__ == "__main__":
    main()
