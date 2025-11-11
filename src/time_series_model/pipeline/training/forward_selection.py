"""Forward horizon selection via information efficiency and ACF decay.

Usage:
  PYTHONPATH=src python -m time_series_model.pipeline.training.forward_selection \
      --data-dir data/parquet_data \
      --symbol BTCUSDT \
      --timeframes 15T,60T,240T \
      --max-forward 48 \
      --save-dir results/forward_selection
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

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


def compute_info_efficiency(close: pd.Series, max_forward: int = 48, min_bars: int = 1000) -> pd.Series:
    """Spearman IC-based information efficiency across horizons 1..max_forward."""
    prices = close.astype(float).dropna()
    ratio = prices / prices.shift(1)
    ratio = ratio.replace([np.inf, -np.inf], np.nan).clip(lower=1e-12)
    returns = np.log(ratio).replace([np.inf, -np.inf], np.nan).dropna()
    efficiencies: List[float] = []
    horizons: List[int] = []
    for h in range(1, max_forward + 1):
        if len(returns) <= h + min_bars:
            break
        future_ret = returns.rolling(h).sum().shift(-h)
        signal = returns.shift(1)
        valid = signal.notna() & future_ret.notna()
        if valid.sum() < 50:
            break
        ic, _ = spearmanr(signal[valid], future_ret[valid])
        if np.isnan(ic):
            ic = 0.0
        # Effective N adjustment
        n_eff = valid.sum() * (1.0 - abs(signal[valid].autocorr() or 0.0))
        efficiency = max(0.0, ic) * np.sqrt(max(n_eff, 1.0)) / np.sqrt(1.0 - ic**2 + 1e-9)
        efficiencies.append(float(efficiency))
        horizons.append(h)
    return pd.Series(efficiencies, index=horizons, dtype=float)


def pick_plateau(eff_series: pd.Series) -> int:
    """Pick plateau start as the first horizon where marginal gain turns non-positive."""
    if eff_series.empty:
        return 1
    diffs = eff_series.diff()
    negatives = diffs[diffs <= 0]
    if not negatives.empty:
        return int(max(1, negatives.index.min()))
    return int(eff_series.idxmax())


def analyze_timeframe(close: pd.Series, max_forward: int) -> Dict[str, float]:
    eff = compute_info_efficiency(close, max_forward=max_forward)
    plateau = pick_plateau(eff)
    return {
        "plateau_forward": float(plateau),
        "efficiency_max": float(eff.max() if not eff.empty else 0.0),
        "efficiency_len": float(len(eff)),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Forward horizon selection via information efficiency",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", required=True)
    p.add_argument("--symbol", required=True)
    p.add_argument("--timeframes", default="15T,60T,240T")
    p.add_argument("--max-forward", type=int, default=48)
    p.add_argument("--save-dir", default="results/forward_selection")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    files = _collect_files(data=[], data_dir=args.data_dir, start=None, end=None, symbols=args.symbol)
    if not files:
        raise ValueError(f"No data files found for {args.symbol}")
    df_raw = _load_parquet_files(files)

    result: Dict[str, Dict[str, float]] = {}
    for tf in [s.strip() for s in args.timeframes.split(",") if s.strip()]:
        resampled = _resample_single_asset(df_raw, tf)
        analysis = analyze_timeframe(resampled["close"], max_forward=args.max_forward)
        result[tf] = analysis

    out_path = Path(args.save_dir) / f"{args.symbol}_forward_selection.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "symbol": args.symbol,
                "timeframes": [s.strip() for s in args.timeframes.split(",") if s.strip()],
                "max_forward": int(args.max_forward),
                "analysis": result,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"💾 Forward selection saved to {out_path}")


if __name__ == "__main__":
    main()


