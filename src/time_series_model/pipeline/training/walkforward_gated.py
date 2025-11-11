"""Walk-forward evaluation for Regime-Gated Time-Series Experts.

Usage:
  PYTHONPATH=src python -m time_series_model.pipeline.training.walkforward_gated \
      --data-dir data/parquet_data \
      --symbol BTCUSDT \
      --timeframes 15T,60T,240T \
      --start 2024-01 \
      --end 2024-12 \
      --train-months 3 \
      --test-months 1 \
      --save-dir results/walkforward_gated
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from time_series_model.pipeline.training.train import _collect_files, _resample_single_asset
from data_tools.baseline_feature_engineering import BaselineFeatureEngineer
from time_series_model.pipeline.training.regime_gating import (
    RegimeGatedTimeSeriesModel,
    default_expert_configs,
)


def _load_parquet_files(paths: List[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for p in paths:
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue
        if "timestamp" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            df = df.set_index("timestamp")
        if isinstance(df.index, pd.DatetimeIndex):
            frames.append(df)
    if not frames:
        raise ValueError("No valid parquet files found")
    return pd.concat(frames, axis=0).sort_index()


def _engineer_for_timeframe(df_ohlcv: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    resampled = _resample_single_asset(df_ohlcv, timeframe)
    eng = BaselineFeatureEngineer()
    feat = eng.engineer_features(resampled, fit=True)
    for c in ["open", "high", "low", "close", "volume"]:
        if c not in feat.columns and c in resampled.columns:
            feat[c] = resampled[c]
    return feat.sort_index()


def _build_engineered_data(df_raw: pd.DataFrame, timeframes: List[str]) -> Mapping[str, pd.DataFrame]:
    return {tf: _engineer_for_timeframe(df_raw, tf) for tf in timeframes}


def _period_slice(df: pd.DataFrame, start_month: str, end_month: str) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        return df
    start = pd.Period(start_month, freq="M").start_time
    end = pd.Period(end_month, freq="M").end_time
    return df.loc[(df.index >= start) & (df.index <= end)]


def _make_periods(start_month: str, end_month: str, train_months: int, test_months: int) -> List[Tuple[str, str, str]]:
    """Return list of (train_start, train_end, test_end) months."""
    start = pd.Period(start_month, freq="M")
    end = pd.Period(end_month, freq="M")
    months = list(pd.period_range(start, end, freq="M"))
    splits: List[Tuple[str, str, str]] = []
    i = 0
    while True:
        if i + train_months + test_months > len(months):
            break
        train_start = months[i]
        train_end = months[i + train_months - 1]
        test_end = months[i + train_months + test_months - 1]
        splits.append((train_start.strftime("%Y-%m"), train_end.strftime("%Y-%m"), test_end.strftime("%Y-%m")))
        i += test_months
    return splits


def _future_log_return(close: pd.Series, bars: int) -> pd.Series:
    return np.log(close.shift(-bars) / close)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Walk-forward evaluation for regime-gated experts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", required=True)
    p.add_argument("--symbol", required=True)
    p.add_argument("--timeframes", default="15T,60T,240T")
    p.add_argument("--start", required=True, help="YYYY-MM")
    p.add_argument("--end", required=True, help="YYYY-MM")
    p.add_argument("--train-months", type=int, default=3)
    p.add_argument("--test-months", type=int, default=1)
    p.add_argument("--save-dir", default="results/walkforward_gated")
    p.add_argument("--eval-forward-bars", type=int, default=6, help="Evaluation horizon in bars for price_tf (default 6 for 1H=~6h)")
    p.add_argument("--price-tf", default="60T", help="Timeframe to derive evaluation label (default 60T)")
    p.add_argument("--multi-horizons", default="", help="Comma-separated forward horizons for fusion (e.g., 2,6,12)")
    p.add_argument("--metric", default="ic", choices=["ic", "mse"], help="Metric to select best horizon (per split)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    files = _collect_files(data=[], data_dir=args.data_dir, start=None, end=None, symbols=args.symbol)
    if not files:
        raise ValueError(f"No files for {args.symbol}")
    df_raw = _load_parquet_files(files)
    tfs = [s.strip() for s in args.timeframes.split(",") if s.strip()]
    splits = _make_periods(args.start, args.end, args.train_months, args.test_months)

    results: List[Dict[str, float]] = []
    for train_start, train_end, test_end in splits:
        test_start_period = pd.Period(train_end, freq="M") + 1
        test_start = test_start_period.strftime("%Y-%m")
        print(f"🔁 Split: train {train_start}..{train_end} | test {test_start}..{test_end}")

        # Build train/test engineered per timeframe
        eng_train = {}
        eng_test = {}
        for tf in tfs:
            full = _engineer_for_timeframe(df_raw, tf)
            eng_train[tf] = _period_slice(full, train_start, train_end)
            eng_test[tf] = _period_slice(full, test_start, test_end)

        # Guard minimal sizes
        if any(len(df) < 200 for df in eng_train.values()) or any(len(df) < 50 for df in eng_test.values()):
            print("  ⚠️ Skipping split due to insufficient data")
            continue

        horizon_arg = (args.multi_horizons or "").strip()
        horizon_list = [int(x) for x in horizon_arg.split(",") if x.strip().isdigit()] or [args.eval_forward_bars]
        horizon_metrics: List[Tuple[int, float, float, float]] = []
        ensemble_by_h: Dict[int, pd.Series] = {}
        for horizon in horizon_list:
            model = RegimeGatedTimeSeriesModel(forward_bars=horizon, include_regime_features=True)
            model.train(eng_train, expert_configs=default_expert_configs())
            preds_map = model.predict(eng_test, regime_probs=None)
            union_idx = pd.Index([])
            for s in preds_map.values():
                union_idx = union_idx.union(s.index)
            aligned = [s.reindex(union_idx).ffill() for s in preds_map.values() if len(s) > 0]
            if not aligned:
                continue
            ensemble_by_h[horizon] = pd.concat(aligned, axis=1).mean(axis=1)

        if not ensemble_by_h:
            print("  ⚠️ No predictions; skipping")
            continue

        # Build evaluation target from price_tf (once)
        price_tf = args.price_tf if args.price_tf in eng_test else (tfs[0] if tfs else None)
        if price_tf is None or len(eng_test[price_tf]) == 0:
            print("  ⚠️ No price_tf available for evaluation; skipping")
            continue
        close = eng_test[price_tf]["close"].astype(float)
        y_full = _future_log_return(close, bars=int(args.eval_forward_bars))

        best_metric_val = -np.inf if args.metric == "ic" else np.inf
        best_metrics: Dict[str, float] = {}
        best_n = 0
        best_h = None
        best_eval_df: Optional[pd.DataFrame] = None
        for horizon, ensemble in ensemble_by_h.items():
            y = y_full.reindex(ensemble.index)
            eval_df = pd.DataFrame({"ensemble": ensemble, "y": y}).dropna()
            if len(eval_df) < 30:
                continue
            ic, _ = spearmanr(eval_df["ensemble"], eval_df["y"])
            if np.isnan(ic):
                ic = 0.0
            acc = float((np.sign(eval_df["ensemble"]) == (eval_df["y"] > 0).astype(int).replace({0: -1})).mean())
            mse = float(np.mean((eval_df["ensemble"] - eval_df["y"]) ** 2))
            if args.metric == "ic":
                metric_val = ic
                better = metric_val > best_metric_val
            else:
                metric_val = mse
                better = metric_val < best_metric_val
            if better:
                best_metric_val = metric_val
                best_metrics = {"ic_spearman": float(ic), "mse": mse, "accuracy_sign": acc}
                best_n = len(eval_df)
                best_h = horizon
                best_eval_df = eval_df
        if best_eval_df is None or best_h is None:
            print("  ⚠️ No valid horizon metrics; skipping")
            continue
        summary = {
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "n": float(best_n),
            "best_horizon": int(best_h),
        }
        summary.update(best_metrics)
        results.append(summary)
        print(
            f"   → horizon={best_h}, n={best_n}, IC={best_metrics['ic_spearman']:.4f}, "
            f"Acc={best_metrics['accuracy_sign']:.4f}, MSE={best_metrics['mse']:.6f}"
        )

    # Save summary
    out_dir = Path(args.save_dir) / args.symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"splits": results}, f, indent=2, ensure_ascii=False)
    print(f"💾 Walk-forward summary saved to {summary_path}")


if __name__ == "__main__":
    main()


