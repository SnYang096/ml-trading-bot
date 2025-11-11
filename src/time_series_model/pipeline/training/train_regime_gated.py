"""Train regime-gated time-series experts (Momentum@1h, MeanReversion@15m, Breakout@1h/4h).

Minimal CLI:
    PYTHONPATH=src python -m time_series_model.pipeline.training.train_regime_gated \\
        --data-dir data/parquet_data \\
        --symbol BTCUSDT \\
        --feature-type baseline \\
        --timeframes 15T,60T,240T
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Mapping

import pandas as pd

from time_series_model.pipeline.training.train import (  # reuse resampler and file collector
    _collect_files,
    _resample_single_asset,
)
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
        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
        if isinstance(df.index, pd.DatetimeIndex):
            frames.append(df)
    if not frames:
        raise ValueError("No valid parquet files found")
    return pd.concat(frames, axis=0).sort_index()


def _engineer_for_timeframe(df_ohlcv: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample and build baseline features while retaining OHLCV."""
    resampled = _resample_single_asset(df_ohlcv, timeframe)
    engineer = BaselineFeatureEngineer()
    feat_df = engineer.engineer_features(resampled, fit=True)
    # Ensure essential columns exist
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(feat_df.columns)
    if missing:
        # Join from resampled if feature engineering dropped something
        feat_df = feat_df.join(resampled[list(required & set(resampled.columns))],
                               how="left", rsuffix="_rs").sort_index()
    return feat_df


def _build_engineered_data(df_raw: pd.DataFrame, timeframes: List[str]) -> Mapping[str, pd.DataFrame]:
    data_by_tf: Dict[str, pd.DataFrame] = {}
    for tf in timeframes:
        tf = tf.strip()
        if not tf:
            continue
        data_by_tf[tf] = _engineer_for_timeframe(df_raw, tf)
    return data_by_tf


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train regime-gated time-series experts (Momentum/MeanReversion/Breakout)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", required=True, help="Directory containing parquet data")
    p.add_argument("--symbol", required=True, help="Single symbol, e.g., BTCUSDT")
    p.add_argument("--feature-type", default="baseline", choices=["baseline"], help="Feature type")
    p.add_argument("--timeframes", default="15T,60T,240T", help="Comma-separated timeframes")
    p.add_argument("--save-dir", default="results/gated_training", help="Directory to save artifacts")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    files = _collect_files(data=[], data_dir=args.data_dir, start=None, end=None, symbols=args.symbol)
    if not files:
        raise ValueError(f"No data files found for symbol {args.symbol} under {args.data_dir}")

    print(f"📦 Loading {len(files)} files for {args.symbol} ...")
    df_raw = _load_parquet_files(files)
    print(f"✅ Loaded {len(df_raw):,} rows")

    timeframes = [s.strip() for s in args.timeframes.split(",") if s.strip()]
    print(f"🧱 Building features for timeframes: {timeframes}")
    engineered_data = _build_engineered_data(df_raw, timeframes)
    for tf, df in engineered_data.items():
        print(f"   - {tf}: {len(df):,} rows, {df.shape[1]} columns")

    # Train regime-gated experts (uses per-expert forward horizons internally)
    model = RegimeGatedTimeSeriesModel(forward_bars=6)
    metrics = model.train(engineered_data, expert_configs=default_expert_configs())
    print("✅ Training completed. Metrics summary:")
    for expert, tf_map in metrics.items():
        for tf, m in tf_map.items():
            print(f"   [{expert} @ {tf}] -> {m}")

    # Save boosters per expert/timeframe if available
    out_dir = Path(args.save_dir) / f"{args.symbol}"
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for expert, models_by_tf in model.experts.items():
        for tf, lgbm in models_by_tf.items():
            booster = getattr(lgbm, "model", None)
            if booster is None:
                continue
            out_path = out_dir / f"{expert}_{tf}.txt"
            try:
                booster.save_model(str(out_path))
                saved += 1
            except Exception:
                pass
    print(f"💾 Saved {saved} boosters to {out_dir}")


if __name__ == "__main__":
    main()


