"""Online monitoring utilities: calibration, bucket performance, and feature drift (PSI-like).

Usage (TS calibration on gated outputs):
  PYTHONPATH=src python -m time_series_model.monitoring.online_monitors \
      --data-dir data/parquet_data \
      --symbol BTCUSDT \
      --positions results/gated_positions/BTCUSDT/positions.parquet \
      --price-tf 60T \
      --forward-bars 6 \
      --save-dir results/monitoring
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from time_series_model.pipeline.training.train import _collect_files, _resample_single_asset


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


def _future_return(close: pd.Series, bars: int) -> pd.Series:
    return close.shift(-bars) / close - 1.0


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probs - labels) ** 2))


def calibration_buckets(scores: pd.Series, labels: pd.Series, n_bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"score": scores.values, "label": labels.values})
    df = df[np.isfinite(df["score"]) & np.isfinite(df["label"])]
    df["bin"] = pd.qcut(df["score"], q=n_bins, duplicates="drop")
    grouped = df.groupby("bin")
    out = grouped.agg(avg_score=("score", "mean"), hit_rate=("label", "mean"), n=("label", "size")).reset_index()
    return out


def psi(reference: pd.Series, current: pd.Series, n_bins: int = 10) -> float:
    """Population Stability Index (simple)."""
    ref = reference.replace([np.inf, -np.inf], np.nan).dropna()
    cur = current.replace([np.inf, -np.inf], np.nan).dropna()
    if ref.empty or cur.empty:
        return 0.0
    bins = np.quantile(ref, np.linspace(0.0, 1.0, n_bins + 1))
    bins[0] = -np.inf
    bins[-1] = np.inf
    ref_counts, _ = np.histogram(ref, bins=bins)
    cur_counts, _ = np.histogram(cur, bins=bins)
    ref_perc = ref_counts / max(1, ref_counts.sum())
    cur_perc = cur_counts / max(1, cur_counts.sum())
    epsilon = 1e-9
    psi_vals = (cur_perc - ref_perc) * np.log((cur_perc + epsilon) / (ref_perc + epsilon))
    return float(np.sum(psi_vals))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Online monitoring: calibration & drift checks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", required=True)
    p.add_argument("--symbol", required=True)
    p.add_argument("--positions", required=True, help="Parquet with ensemble_return and discrete_signal")
    p.add_argument("--price-tf", default="60T")
    p.add_argument("--forward-bars", type=int, default=6)
    p.add_argument("--save-dir", default="results/monitoring")
    p.add_argument("--n-bins", type=int, default=10)
    p.add_argument("--drift-window", type=int, default=2000, help="reference window size (bars)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # Load positions (ensemble_return as score)
    pos_df = pd.read_parquet(args.positions).sort_index()
    score = pos_df["ensemble_return"].astype(float).replace([np.inf, -np.inf], np.nan)

    # Load prices for label construction
    files = _collect_files(data=[], data_dir=args.data_dir, start=None, end=None, symbols=args.symbol)
    if not files:
        raise ValueError(f"No data for {args.symbol}")
    df_raw = _load_parquet_files(files)
    resampled = _resample_single_asset(df_raw, args.price_tf)
    close = resampled["close"].astype(float)

    # Align and compute labels
    y = _future_return(close, args.forward_bars).reindex(score.index).ffill()
    valid = score.notna() & y.notna()
    score_aligned = score[valid]
    y_aligned = y[valid]
    if len(y_aligned) == 0:
        raise RuntimeError("No aligned samples for calibration.")
    # For calibration, use binary label on sign
    label_bin = (y_aligned > 0).astype(int)
    # Map score to pseudo-prob via logistic transform
    # This is a proxy; if you have classification proba, pass that instead
    score_prob = 1.0 / (1.0 + np.exp(-score_aligned / (score_aligned.std() + 1e-9)))

    # Metrics
    brier = brier_score(score_prob.values, label_bin.values)
    buckets = calibration_buckets(pd.Series(score_prob.values, index=score_aligned.index), label_bin, n_bins=args.n_bins)

    # Drift (PSI) for the raw score using an initial reference window
    ref_window = max(50, min(args.drift_window, len(score_aligned) // 2))
    psi_val = 0.0
    if len(score_aligned) > ref_window * 2:
        ref = score_aligned.iloc[:ref_window]
        cur = score_aligned.iloc[-ref_window:]
        psi_val = psi(ref, cur, n_bins=args.n_bins)

    # Save report
    out_dir = Path(args.save_dir) / args.symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "online_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "brier_score": brier,
                "psi_score": psi_val,
                "n_samples": int(len(score_aligned)),
                "forward_bars": int(args.forward_bars),
                "price_tf": args.price_tf,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    buckets_path = out_dir / "calibration_buckets.parquet"
    buckets.to_parquet(buckets_path)
    print(f"💾 Monitoring saved: {summary_path}, {buckets_path}")


if __name__ == "__main__":
    main()


