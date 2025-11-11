"""Cross-sectional end-to-end workflow: build panel -> robust processing -> neutralize -> portfolio -> regime overlay -> save weights.

Usage:
  PYTHONPATH=src python -m cross_sectional.workflow \
      --data-dir data/parquet_data \
      --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,ADAUSDT \
      --timeframe 15T \
      --horizon 12 \
      --feature-type baseline \
      --save-dir results/crosssec_portfolio
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from cross_sectional.panel_generation import PanelGenerationConfig, generate_cross_sectional_panel
from cross_sectional.processing import (
    winsorize_by_sigma,
    cross_sectional_zscore,
    filter_by_liquidity,
    neutralize_against,
    drop_correlated_factors,
)
from cross_sectional.portfolio import (
    PortfolioConstraints,
    construct_portfolio,
    overlay_regime_weights,
)


def _infer_factor_columns(panel: pd.DataFrame, target_col: str) -> List[str]:
    exclude = {"open", "high", "low", "close", "volume", target_col}
    exclude |= {c for c in panel.columns if c.lower().startswith("future_return")}
    return [c for c in panel.columns if c not in exclude]


def _compute_dollar_volume(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    if "dollar_volume" not in panel.columns:
        if "close" in panel.columns and "volume" in panel.columns:
            panel["dollar_volume"] = (panel["close"].astype(float) * panel["volume"].astype(float)).replace(
                [np.inf, -np.inf], np.nan
            ).fillna(0.0)
        else:
            panel["dollar_volume"] = 0.0
    return panel


def _regime_state_from_panel(panel: pd.DataFrame, trend_col: str = "trend_r2_20", timestamp_level: int = 0) -> str:
    # Heuristic: use average trend strength on last timestamp to infer state
    if trend_col not in panel.columns:
        return "RANGE"
    last_ts = panel.index.get_level_values(timestamp_level).max()
    cs = panel.xs(last_ts, level=timestamp_level)
    if cs.empty:
        return "RANGE"
    val = cs[trend_col].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).mean()
    if val >= 0.5:
        return "TRENDING"
    elif val <= 0.2:
        return "RANGE"
    return "PRE_BREAKOUT"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cross-sectional end-to-end workflow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", required=True)
    p.add_argument("--symbols", required=True, help="Comma-separated symbols")
    p.add_argument("--timeframe", default="15T")
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--feature-type", default="baseline", choices=["baseline", "comprehensive"])
    p.add_argument("--save-dir", default="results/crosssec_portfolio")
    p.add_argument("--winsorize-sigma", type=float, default=3.0)
    p.add_argument("--zscore-clip", type=float, default=3.0)
    p.add_argument("--liq-quantile", type=float, default=0.2)
    p.add_argument("--de-corr-threshold", type=float, default=0.9)
    p.add_argument("--control-cols", default="", help="Comma-separated control columns for neutralization (optional)")
    p.add_argument("--long-only", action="store_true", help="Use long-only weighting")
    p.add_argument("--max-weight", type=float, default=0.1)
    p.add_argument("--gross", type=float, default=1.0)
    p.add_argument("--turnover-penalty", type=float, default=0.0)
    p.add_argument("--cost-per-unit", type=float, default=0.0)
    p.add_argument("--regime-overlay", action="store_true")
    p.add_argument("--trend-gain", type=float, default=1.2)
    p.add_argument("--range-gain", type=float, default=0.9)
    p.add_argument("--collapse-gain", type=float, default=0.6)
    p.add_argument("--target-vol", type=float, default=0.0, help="Portfolio target vol (optional > 0 to enable inverse-vol scaling)")
    p.add_argument("--vol-lookback", type=int, default=96, help="Per-asset vol lookback bars")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    cfg = PanelGenerationConfig(
        symbols=symbols,
        timeframe=args.timeframe,
        horizon=args.horizon,
        data_path=args.data_dir,
        feature_type=args.feature_type,
        dropna=True,
    )
    panel, target_col = generate_cross_sectional_panel(cfg)
    panel = _compute_dollar_volume(panel)

    # Factor selection (all non-price non-target)
    factor_cols = _infer_factor_columns(panel, target_col)
    if not factor_cols:
        raise RuntimeError("No factor columns found.")

    # Robust processing: winsorize -> zscore -> liquidity filter
    panel = winsorize_by_sigma(panel, factor_cols, sigma=args.winsorize_sigma)
    panel = cross_sectional_zscore(panel, factor_cols, clip_sigma=args.zscore_clip)
    panel = filter_by_liquidity(panel, liq_col="dollar_volume", min_quantile=args.liq_quantile)
    # De-correlation (drop highly correlated factors)
    if args.de_corr_threshold and 0.0 < args.de_corr_threshold < 1.0:
        panel = drop_correlated_factors(panel, factor_cols, threshold=args.de_corr_threshold)
        # Re-infer factor cols after drop
        factor_cols = _infer_factor_columns(panel, target_col)

    # Optional neutralization against control columns
    if args.control_cols:
        controls = [c.strip() for c in args.control_cols.split(",") if c.strip()]
        available_controls = [c for c in controls if c in panel.columns]
        if available_controls:
            panel = neutralize_against(panel, factor_cols=factor_cols, control_cols=available_controls)

    # Build a composite score (equal-weighted average of selected factors)
    last_ts = panel.index.get_level_values(0).max()
    cs = panel.xs(last_ts, level=0)
    if cs.empty:
        raise RuntimeError("No cross-section available at last timestamp.")
    score = cs[factor_cols].mean(axis=1)
    cs = cs.assign(score=score)
    panel.loc[(last_ts, slice(None)), "score"] = cs["score"]

    # Portfolio construction at last timestamp
    constraints = PortfolioConstraints(
        max_weight_per_asset=args.max_weight,
        long_short=not args.long_only,
        gross_leverage=args.gross,
        turnover_penalty=args.turnover_penalty,
        cost_per_unit=args.cost_per_unit,
    )
    # Add per-asset volatility for inverse-vol scaling if requested
    if args.target_vol and args.target_vol > 0:
        # Compute per-asset rolling vol on last window
        panel = panel.copy()
        # Expect 'close' exists; compute pct returns per asset
        try:
            close = panel["close"].astype(float).unstack(level=1).sort_index()
            rets = close.pct_change().rolling(args.vol_lookup if hasattr(args, "vol_lookup") else args.vol_lookback).std().iloc[-1]
            # Map back to last timestamp slice
            last_ts = panel.index.get_level_values(0).max()
            mapping = rets.to_dict()
            idx = panel.xs(last_ts, level=0).index
            vol_series = pd.Series({sym: mapping.get(sym, np.nan) for sym in idx})
            vol_series = vol_series.replace([np.inf, -np.inf], np.nan).fillna(vol_series.median())
            panel.loc[(last_ts, slice(None)), "asset_vol"] = vol_series.values
            constraints.target_vol = float(args.target_vol)
        except Exception:
            pass
    weights = construct_portfolio(panel, score_col="score", constraints=constraints)

    # Regime overlay (heuristic from panel)
    if args.regime_overlay:
        state = _regime_state_from_panel(panel, trend_col="trend_r2_20")
        weights = overlay_regime_weights(weights, regime_state=state, trend_gain=args.trend_gain, range_gain=args.range_gain, collapse_gain=args.collapse_gain)

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "weights.parquet"
    weights.to_frame("weight").to_parquet(weights_path)
    print(f"💾 Saved cross-sectional weights to {weights_path}")


if __name__ == "__main__":
    main()


