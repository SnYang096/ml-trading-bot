"""End-to-end vectorized backtest combining time-series experts and cross-sectional portfolio."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from cross_sectional.panel_generation import PanelGenerationConfig, generate_cross_sectional_panel
from cross_sectional.portfolio import PortfolioConstraints, construct_portfolio
from cross_sectional.processing import (
    drop_correlated_factors,
    filter_by_liquidity,
    winsorize_by_sigma,
    cross_sectional_zscore,
)
from time_series_model.pipeline.training.forward_selection import analyze_timeframe
from time_series_model.pipeline.training.regime_gating import (
    RegimeGatedTimeSeriesModel,
    default_expert_configs,
)
from time_series_model.pipeline.workflows.gated_to_position import _fuse_predictions
from time_series_model.pipeline.risk_management import RiskManager
from time_series_model.pipeline.training.train import _resample_single_asset


@dataclass
class SyntheticConfig:
    symbols: Sequence[str]
    start: str = "2024-01-01"
    periods: int = 2000
    freq: str = "15min"
    seed: int = 123


def generate_synthetic_ohlcv(cfg: SyntheticConfig) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed)
    index = pd.date_range(cfg.start, periods=cfg.periods, freq=cfg.freq)
    data: Dict[str, pd.DataFrame] = {}
    for symbol in cfg.symbols:
        walk = np.cumsum(rng.normal(0, 0.3, size=cfg.periods))
        base = 100 + walk + rng.normal(0, 0.2, size=cfg.periods)
        df = pd.DataFrame(
            {
                "open": base + rng.normal(0, 0.1, size=cfg.periods),
                "high": base + rng.uniform(0.05, 0.3, size=cfg.periods),
                "low": base - rng.uniform(0.05, 0.3, size=cfg.periods),
                "close": base,
                "volume": rng.uniform(1000, 5000, size=cfg.periods),
            },
            index=index,
        )
        df.index.name = "timestamp"
        data[symbol] = df
    return data


def run_forward_selection(
        ohlcv: Dict[str, pd.DataFrame],
        max_forward: int = 24) -> Dict[str, Dict[str, float]]:
    return {
        symbol: analyze_timeframe(df["close"], max_forward=max_forward)
        for symbol, df in ohlcv.items()
    }


def engineer_by_timeframe(
        raw: Dict[str, pd.DataFrame],
        timeframes: Sequence[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
    per_symbol: Dict[str, Dict[str, pd.DataFrame]] = {}
    for symbol, df in raw.items():
        per_symbol[symbol] = {
            tf: _resample_single_asset(df, tf)
            for tf in timeframes
        }
    return per_symbol


def run_timeseries_backtest(
        engineered: Dict[str, Dict[str,
                                   pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
    positions: Dict[str, pd.DataFrame] = {}
    for symbol, tf_map in engineered.items():
        model = RegimeGatedTimeSeriesModel(forward_bars=6,
                                           include_regime_features=True)
        model.train(tf_map, expert_configs=default_expert_configs())

        preds = model.predict(tf_map, regime_probs=None)
        fused = _fuse_predictions(preds)
        ensemble_df = pd.DataFrame({
            "ensemble_return":
            fused,
            "discrete_signal":
            np.sign(fused).astype(int),
        })

        price_tf = next(iter(tf_map))
        price_data = tf_map[price_tf][["close"]].dropna()
        rm = RiskManager()
        positions[symbol] = rm.apply_risk_management(
            ensemble_df=ensemble_df,
            price_data=price_data,
            regime_probs=None,
            account_value=100_000.0,
        )
    return positions


def run_cross_sectional_workflow(
    ohlcv: Dict[str, pd.DataFrame],
    symbols: Sequence[str],
    timeframe: str = "60min",
    horizon: int = 12,
) -> Tuple[pd.DataFrame, pd.Series]:
    cfg = PanelGenerationConfig(
        symbols=symbols,
        timeframe=timeframe,
        horizon=horizon,
        feature_type="baseline",
        dropna=True,
        include_order_flow=False,
    )
    panel, target_col = generate_cross_sectional_panel(cfg)
    factor_cols = [
        c for c in panel.columns if c not in {"close", "volume", target_col}
    ]
    panel = winsorize_by_sigma(panel, factor_cols, sigma=3.0)
    panel = cross_sectional_zscore(panel, factor_cols, clip_sigma=3.0)
    panel = filter_by_liquidity(panel,
                                liq_col="dollar_volume",
                                min_quantile=0.2)
    panel = drop_correlated_factors(panel, factor_cols, threshold=0.9)

    last_ts = panel.index.get_level_values(0).max()
    last_slice = panel.xs(last_ts, level=0).copy()
    last_slice["score"] = last_slice[factor_cols].mean(axis=1)

    constraints = PortfolioConstraints(
        max_weight_per_asset=0.2,
        long_short=False,
        gross_leverage=1.0,
    )
    weights = construct_portfolio(panel=panel,
                                  score_col="score",
                                  constraints=constraints)
    return panel, weights


def aggregate_pnl(
    ts_positions: Dict[str, pd.DataFrame],
    cs_weights: pd.Series,
    ohlcv: Dict[str, pd.DataFrame],
) -> pd.Series:
    pnl_series = []
    for symbol, pos_df in ts_positions.items():
        returns = ohlcv[symbol]["close"].pct_change().reindex(
            pos_df.index).fillna(0.0)
        pnl_series.append(pos_df["position"].shift(1).fillna(0.0) * returns)
    ts_pnl_total = pd.concat(pnl_series, axis=1).sum(
        axis=1) if pnl_series else pd.Series(dtype=float)

    cs_pnl = pd.Series(0.0, index=ts_pnl_total.index)
    for symbol, weight in cs_weights.items():
        if symbol in ohlcv:
            cs_returns = ohlcv[symbol]["close"].pct_change().reindex(
                cs_pnl.index).fillna(0.0)
            cs_pnl = cs_pnl.add(weight * cs_returns, fill_value=0.0)

    combined = ts_pnl_total.add(cs_pnl, fill_value=0.0)
    return combined.cumsum()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vectorized backtest harness for TS + CS models")
    parser.add_argument("--use-synthetic",
                        action="store_true",
                        help="Use synthetic data instead of parquet.")
    parser.add_argument("--symbols",
                        nargs="+",
                        default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--panel-timeframe", default="60min")
    parser.add_argument("--panel-horizon", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.use_synthetic or not args.data_dir:
        cfg = SyntheticConfig(symbols=args.symbols)
        raw_data = generate_synthetic_ohlcv(cfg)
    else:
        raw_data: Dict[str, pd.DataFrame] = {}
        for symbol in args.symbols:
            path = Path(args.data_dir) / f"{symbol}.parquet"
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing parquet for {symbol}: {path}")
            df = pd.read_parquet(path)
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
            raw_data[symbol] = df.sort_index()

    forward_stats = run_forward_selection(raw_data)
    print("Forward horizon analyses:", forward_stats)

    engineered = engineer_by_timeframe(raw_data, timeframes=["15min", "60min"])
    ts_positions = run_timeseries_backtest(engineered)
    panel, cs_weights = run_cross_sectional_workflow(
        raw_data,
        symbols=args.symbols,
        timeframe=args.panel_timeframe,
        horizon=args.panel_horizon,
    )
    pnl_curve = aggregate_pnl(ts_positions, cs_weights, raw_data)
    print("Final cumulative PnL:",
          float(pnl_curve.iloc[-1]) if not pnl_curve.empty else 0.0)

    out_dir = Path("results") / "vectorbot"
    out_dir.mkdir(parents=True, exist_ok=True)
    pnl_curve.to_frame("cumulative_pnl").to_csv(out_dir / "pnl_curve.csv")
    print(f"Saved PnL curve to {out_dir / 'pnl_curve.csv'}")


if __name__ == "__main__":
    main()
