"""End-to-end: Regime-gated experts -> ensemble return -> Risk-managed positions.

Usage:
  PYTHONPATH=src python -m time_series_model.pipeline.workflows.gated_to_position \
      --data-dir data/parquet_data \
      --symbol BTCUSDT \
      --timeframes 15T,60T,240T \
      --save-dir results/gated_positions
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from time_series_model.pipeline.training.train import _collect_files, _resample_single_asset
from data_tools.baseline_feature_engineering import BaselineFeatureEngineer
from time_series_model.pipeline.training.regime_gating import (
    RegimeGatedTimeSeriesModel,
    default_expert_configs,
)
from regime_detection.detector import RuleBasedRegimeDetector
from time_series_model.pipeline.risk_management import RiskManager


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
    engineer = BaselineFeatureEngineer()
    feat_df = engineer.engineer_features(resampled, fit=True)
    # Retain essential OHLCV
    for c in ["open", "high", "low", "close", "volume"]:
        if c not in feat_df.columns and c in resampled.columns:
            feat_df[c] = resampled[c]
    return feat_df.sort_index()


def _build_engineered_data(df_raw: pd.DataFrame, timeframes: List[str]) -> Mapping[str, pd.DataFrame]:
    return {tf: _engineer_for_timeframe(df_raw, tf) for tf in timeframes}


def _detect_regime_probs(engineered_data: Mapping[str, pd.DataFrame]) -> Mapping[str, pd.DataFrame]:
    detector = RuleBasedRegimeDetector()
    prob_maps: Dict[str, pd.DataFrame] = {}
    for tf, df in engineered_data.items():
        result = detector.detect(df)
        labels = result.labels.reindex(df.index).ffill()
        # Build one-hot over full enum set to keep consistent columns
        enum_cols = [e.value for e in type(result.labels.iloc[0])]
        prob_df = pd.DataFrame(0.0, index=df.index, columns=enum_cols)
        for lab in labels.unique():
            prob_df.loc[labels == lab, lab.value] = 1.0
        # Fill any missing enum columns with zeros
        for col in enum_cols:
            if col not in prob_df.columns:
                prob_df[col] = 0.0
        prob_maps[tf] = prob_df[enum_cols]
    return prob_maps


def _fuse_predictions(weighted_preds_by_tf: Mapping[str, pd.Series]) -> pd.Series:
    # Union index, simple average of available timeframes each timestamp
    union_index = pd.Index([])
    for s in weighted_preds_by_tf.values():
        union_index = union_index.union(s.index)
    if len(union_index) == 0:
        return pd.Series(dtype=float)
    aligned = []
    for tf, s in weighted_preds_by_tf.items():
        aligned.append(s.reindex(union_index).ffill())
    if not aligned:
        return pd.Series(0.0, index=union_index)
    stacked = pd.concat(aligned, axis=1)
    return stacked.mean(axis=1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Regime-gated experts to risk-managed positions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", required=True)
    p.add_argument("--symbol", required=True)
    p.add_argument("--timeframes", default="15T,60T,240T")
    p.add_argument("--save-dir", default="results/gated_positions")
    p.add_argument("--account-value", type=float, default=100000.0)
    p.add_argument(
        "--multi-horizons",
        default="",
        help="Comma-separated forward horizons for multi-horizon fusion (e.g., 2,6,12). If empty, use single horizon (6).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    files = _collect_files(data=[], data_dir=args.data_dir, start=None, end=None, symbols=args.symbol)
    if not files:
        raise ValueError(f"No data files found for {args.symbol}")
    print(f"📦 Loading {len(files)} files for {args.symbol} ...")
    df_raw = _load_parquet_files(files)

    tfs = [s.strip() for s in args.timeframes.split(",") if s.strip()]
    print(f"🧱 Feature building for timeframes: {tfs}")
    engineered = _build_engineered_data(df_raw, tfs)

    # Train experts and predict (single or multi-horizon)
    horizons_arg = (args.multi_horizons or "").strip()
    horizon_list = [int(x) for x in horizons_arg.split(",") if x.strip().isdigit()]
    fused_by_tf: Dict[str, pd.Series] = {}
    if horizon_list:
        # Multi-horizon: train/predict per horizon and fuse equally
        preds_list: List[Mapping[str, pd.Series]] = []
        for h in horizon_list:
            print(f"🔧 Training gated experts for horizon={h} bars")
            mh_model = RegimeGatedTimeSeriesModel(forward_bars=h)
            mh_model.train(engineered, expert_configs=default_expert_configs())
            mh_preds = mh_model.predict(engineered, regime_probs=None)
            preds_list.append(mh_preds)
        # Fuse per timeframe
        for tf in tfs:
            series_to_fuse = [preds.get(tf) for preds in preds_list if tf in preds]
            if not series_to_fuse:
                continue
            union_index = pd.Index([])
            for s in series_to_fuse:
                union_index = union_index.union(s.index)
            aligned = [s.reindex(union_index).ffill() for s in series_to_fuse]
            fused_by_tf[tf] = pd.concat(aligned, axis=1).mean(axis=1)
        weighted_preds_by_tf = fused_by_tf
    else:
        model = RegimeGatedTimeSeriesModel(forward_bars=6)
        model.train(engineered, expert_configs=default_expert_configs())
        weighted_preds_by_tf = model.predict(engineered, regime_probs=None)  # one-hot internal
    # Detect regime probabilities (rule-based) for sizing/risk
    regime_maps = _detect_regime_probs(engineered)

    # Fuse per-timeframe series to an ensemble return
    ensemble_return = _fuse_predictions(weighted_preds_by_tf)
    discrete_signal = np.sign(ensemble_return).astype(int)
    ensemble_df = pd.DataFrame(
        {
            "ensemble_return": ensemble_return,
            "discrete_signal": discrete_signal,
        }
    ).dropna()

    # Risk management to positions (uses latest close path)
    # Use 1H (or first tf) as price reference for levels
    price_tf = "60T" if "60T" in engineered else (tfs[0] if tfs else None)
    if price_tf is None:
        raise ValueError("No timeframe available for price reference")
    price_data = engineered[price_tf][["close"]].dropna()
    # Align regime probabilities to ensemble index (use price_tf map for consistency)
    regime_df = None
    if price_tf in regime_maps:
        regime_df = regime_maps[price_tf].reindex(ensemble_df.index).ffill()

    rm = RiskManager()
    positions_df = rm.apply_risk_management(
        ensemble_df=ensemble_df,
        price_data=price_data,
        regime_probs=regime_df,
        account_value=100000.0,
        vol_window=50,
    )

    # Save
    out_dir = Path(args.save_dir) / args.symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    positions_path = out_dir / "positions.parquet"
    positions_df.to_parquet(positions_path)
    print(f"💾 Saved positions to {positions_path}")


if __name__ == "__main__":
    main()


