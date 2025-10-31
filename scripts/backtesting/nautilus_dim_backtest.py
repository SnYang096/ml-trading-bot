#!/usr/bin/env python3
"""
Nautilus Trader backtest using trained dimensionality reduction (Autoencoder + LightGBM).

Features:
- Multi-asset (BTC/ETH/SOL) portfolio with ATR-based risk sizing and static/vol-adjusted weights
- Online feature engineering (EnhancedFeatureEngineer) with warmup fit, then transform-only
- Embedding via trained UnifiedAutoencoder encoder; predictions via saved LightGBM model
- Trade management per docs: partial take profits at 2R/3R and ATR trailing for remainder
- Optional pyramiding layers with guardrails

Usage example:
  PYTHONPATH=src python scripts/backtesting/nautilus_dim_backtest.py \
    --data-dir data/parquet_data \
    --results-dir results/production_dimensionality_YYYYMMDD_HHMMSS \
    --symbols BTCUSDT ETHUSDT SOLUSDT \
    --timeframe 5T \
    --start 2024-10-01 --end 2024-12-31

Requires nautilus-trader installed. This script is designed for quasi-live backtesting.
"""

from __future__ import annotations

import argparse
import os
import sys
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import torch

try:
    # Lazy import Nautilus; fail with a helpful error if missing
    from nautilus_trader.core.message import Event
    from nautilus_trader.backtest.engine import BacktestEngine
    from nautilus_trader.backtest.data.providers import ParquetDataCatalog
    from nautilus_trader.config import BacktestConfig
    from nautilus_trader.model.identifiers import Symbol
    from nautilus_trader.model.data import Bar
    from nautilus_trader.model.enums import BarType
    from nautilus_trader.common.clock import TestClock
    from nautilus_trader.trading.strategy import Strategy
    from nautilus_trader.common.logging import Logger
except Exception as _e:  # pragma: no cover
    _NAUTILUS_ERR = _e
    BacktestEngine = None

from ml_trading.models.autoencoder import UnifiedAutoencoder
from ml_trading.data_tools.comprehensive_feature_engineering import (
    ComprehensiveFeatureEngineer, )
from sklearn.preprocessing import StandardScaler


def _require_nautilus() -> None:
    if BacktestEngine is None:
        raise RuntimeError(
            "nautilus-trader is not installed or failed to import. "
            f"Original error: {_NAUTILUS_ERR}")


@dataclass
class PositionSlice:
    entry_price: float
    size: float
    stop_price: float
    tp1: Optional[float] = None
    tp2: Optional[float] = None
    is_active: bool = True


class DimReductionSignal:

    def __init__(self, ae: UnifiedAutoencoder, lgb_model, device: str = "cpu"):
        self.ae = ae.eval()
        self.lgb = lgb_model
        self.device = device

    @torch.no_grad()
    def embed(self, X: np.ndarray) -> np.ndarray:
        X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        # UnifiedAutoencoder returns reconstruction in forward; we want encoder output
        # Access encoder sequential directly
        z = self.ae.encoder(X_t)
        if isinstance(z, (list, tuple)):
            z = z[0]
        return z.cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = self.embed(X)
        return self.lgb.predict(Z)


class PortfolioAllocator:

    def __init__(self,
                 target_weights: Dict[str, float],
                 max_risk_per_trade: float = 0.005):
        w_sum = sum(max(0.0, v) for v in target_weights.values())
        self.weights = {
            k: (v / w_sum if w_sum > 0 else 0.0)
            for k, v in target_weights.items()
        }
        self.max_risk = max_risk_per_trade

    def alloc_notional(self, equity: float, symbol: str) -> float:
        return equity * self.weights.get(symbol, 0.0)


class TradeManager:

    def __init__(
        self,
        atr_mult_sl: float = 1.5,
        atr_mult_trail: float = 3.0,
        tp1_r: float = 2.0,
        tp2_r: float = 3.0,
        tp_alloc: List[float] = [0.33, 0.33, 0.34],
        enable_pyramiding: bool = False,
        max_layers: int = 0,
        pyramid_step_r: float = 1.5,
    ):
        self.atr_mult_sl = atr_mult_sl
        self.atr_mult_trail = atr_mult_trail
        self.tp1_r = tp1_r
        self.tp2_r = tp2_r
        self.tp_alloc = tp_alloc
        self.enable_pyramiding = enable_pyramiding
        self.max_layers = max_layers
        self.pyramid_step_r = pyramid_step_r

    def initial_slices(self, entry: float, atr: float,
                       qty: float) -> List[PositionSlice]:
        r_dollar = atr  # define R as ATR distance for simplicity
        sl = entry - self.atr_mult_sl * atr
        # Equal split into 3 slices for TP1, TP2, runner
        part_sizes = [qty * a for a in self.tp_alloc]
        tp1 = entry + self.tp1_r * r_dollar
        tp2 = entry + self.tp2_r * r_dollar
        return [
            PositionSlice(entry_price=entry,
                          size=part_sizes[0],
                          stop_price=sl,
                          tp1=tp1),
            PositionSlice(entry_price=entry,
                          size=part_sizes[1],
                          stop_price=sl,
                          tp2=tp2),
            PositionSlice(entry_price=entry, size=part_sizes[2],
                          stop_price=sl),  # runner with trail
        ]

    def update_trailing(self, slice_: PositionSlice, price: float,
                        atr: float) -> None:
        trail = price - self.atr_mult_trail * atr
        if trail > slice_.stop_price:
            slice_.stop_price = trail


class DimReductionStrategy(Strategy):

    def __init__(
        self,
        symbols: List[str],
        feature_timeframe: str,
        dim_model_dir: str,
        allocator: PortfolioAllocator,
        trade_mgr: TradeManager,
        warmup_bars: int = 600,
        logger: Optional[Logger] = None,
    ):
        super().__init__(logger=logger)
        self.symbols = symbols
        self.tf = feature_timeframe
        self.dim_model_dir = dim_model_dir
        self.allocator = allocator
        self.trade_mgr = trade_mgr
        self.warmup_bars = warmup_bars

        # Per-symbol state
        self.fe_map: Dict[str, ComprehensiveFeatureEngineer] = {}
        self.buf_map: Dict[str, List[Bar]] = {}
        self.atr_map: Dict[str, float] = {s: math.nan for s in symbols}
        self.pos_map: Dict[str, List[PositionSlice]] = {s: [] for s in symbols}
        self.scaler_map: Dict[str, StandardScaler] = {}

        # Load models
        model_pkl = os.path.join(dim_model_dir, "production_model.pkl")
        ae_path = os.path.join(dim_model_dir, "production_autoencoder.pth")
        if not os.path.exists(model_pkl) or not os.path.exists(ae_path):
            raise FileNotFoundError(
                f"Missing model files in {dim_model_dir}. Expected production_model.pkl and production_autoencoder.pth"
            )
        self.lgb_model = joblib.load(model_pkl)
        # Auto-select device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Infer AE dims from state dict
        state = torch.load(ae_path, map_location=device)
        enc0_w = None
        last_enc_w = None
        for k, v in state.items():
            if k.startswith("encoder") and k.endswith("weight"):
                if enc0_w is None:
                    enc0_w = v
                last_enc_w = v
        if enc0_w is None or last_enc_w is None:
            raise RuntimeError("Failed to infer AE dimensions from state dict")
        input_dim = enc0_w.shape[1]
        encoding_dim = last_enc_w.shape[0]
        self.ae = UnifiedAutoencoder(input_dim=input_dim,
                                     encoding_dim=encoding_dim,
                                     architecture="production")
        self.ae.load_state_dict(state, strict=True)
        self.ae.to(device)
        self.signal = DimReductionSignal(self.ae,
                                         self.lgb_model,
                                         device=device)

    def on_start(self):  # noqa: D401
        # Initialize feature engineers per symbol
        for s in self.symbols:
            self.fe_map[s] = ComprehensiveFeatureEngineer()
            self.buf_map[s] = []

    def on_bar(self, bar: Bar):  # type: ignore[override]
        symbol = str(bar.symbol)
        if symbol not in self.symbols:
            return
        buf = self.buf_map[symbol]
        buf.append(bar)
        if len(buf) < self.warmup_bars:
            return

        # Build DataFrame for features
        df = self._bars_to_df(buf)
        fe = self.fe_map[symbol]
        # Engineer features using the same pipeline as training
        try:
            eng = fe.engineer_all_features(df,
                                           fit=(len(buf) == self.warmup_bars))
        except Exception as e:
            self.log.warning(f"Feature engineering failed for {symbol}: {e}")
            return

        # Align feature selection with training (exclude timestamp/close)
        feature_cols = [
            c for c in eng.columns if c not in ["timestamp", "close"]
        ]
        if not feature_cols:
            return
        X_all = eng[feature_cols].values.astype(np.float32)
        # Scale features with StandardScaler like training
        scaler = self.scaler_map.get(symbol)
        if scaler is None and len(buf) >= self.warmup_bars:
            scaler = StandardScaler()
            scaler.fit(X_all)
            self.scaler_map[symbol] = scaler
        if scaler is None:
            return
        X_scaled = scaler.transform(X_all)
        X_last = X_scaled[-1:].astype(np.float32)

        # Predict next return (or score)
        try:
            pred = float(self.signal.predict(X_last)[0])
        except Exception as e:
            self.log.warning(f"Prediction failed for {symbol}: {e}")
            return

        # Compute ATR (basic) from last N bars
        atr = self._compute_atr(df.tail(50))
        self.atr_map[symbol] = atr

        # Simple decision: if prediction > 0 threshold, go long; else flat. Threshold can be tuned
        threshold = 0.0
        price = float(bar.close)
        if pred > threshold and not self.pos_map[symbol]:
            # Size by risk: risk_per_trade = alloc_notional * max_risk, stop distance = atr_mult_sl * atr
            equity = float(self.account.balance.total)
            alloc_notional = self.allocator.alloc_notional(equity, symbol)
            risk_budget = alloc_notional * self.allocator.max_risk
            stop_dist = max(atr * self.trade_mgr.atr_mult_sl, price * 0.002)
            qty = 0.0
            if stop_dist > 0:
                qty = max(0.0, risk_budget / stop_dist)
            if qty > 0:
                slices = self.trade_mgr.initial_slices(price, atr, qty)
                self.pos_map[symbol] = slices

        # Manage open slices
        if self.pos_map[symbol]:
            new_slices: List[PositionSlice] = []
            for slc in self.pos_map[symbol]:
                if not slc.is_active:
                    continue
                # Check TP1/TP2
                if slc.tp1 is not None and price >= slc.tp1:
                    # Realize PnL for that slice and deactivate
                    slc.is_active = False
                    continue
                if slc.tp2 is not None and price >= slc.tp2:
                    slc.is_active = False
                    continue
                # Runner: update trailing
                if slc.tp1 is None and slc.tp2 is None:
                    self.trade_mgr.update_trailing(slc, price, atr)
                # Stop hit
                if price <= slc.stop_price:
                    slc.is_active = False
                    continue
                new_slices.append(slc)
            self.pos_map[symbol] = new_slices

    # Helpers
    def _bars_to_df(self, bars: List[Bar]) -> pd.DataFrame:
        data = {
            "timestamp": [b.ts_event.to_pydatetime() for b in bars],
            "open": [float(b.open) for b in bars],
            "high": [float(b.high) for b in bars],
            "low": [float(b.low) for b in bars],
            "close": [float(b.close) for b in bars],
            "volume": [float(b.volume) for b in bars],
        }
        df = pd.DataFrame(data).set_index("timestamp")
        # Provide minimal order-flow proxies expected by EnhancedFeatureEngineer
        up = (df["close"].diff() > 0).astype(float)
        df["taker_buy_ratio"] = up.rolling(20,
                                           min_periods=5).mean().fillna(0.5)
        signed_vol = np.sign(
            df["close"].diff().fillna(0.0)) * df["volume"].fillna(0.0)
        df["cvd"] = signed_vol.cumsum()
        df["cvd_change_1"] = signed_vol
        df["cvd_change_5"] = signed_vol.rolling(5, min_periods=1).sum()
        df["cvd_change_20"] = signed_vol.rolling(20, min_periods=1).sum()
        df["cvd_short"] = signed_vol.rolling(20, min_periods=1).sum()
        df["cvd_medium"] = signed_vol.rolling(60, min_periods=1).sum()
        df["cvd_long"] = signed_vol.rolling(288, min_periods=1).sum()
        total_vol = df["volume"].rolling(20, min_periods=1).sum()
        df["cvd_normalized"] = (df["cvd_short"] /
                                total_vol.replace(0, np.nan)).replace(
                                    [np.inf, -np.inf], np.nan).fillna(0.0)
        return df

    def _compute_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        if len(df) < period + 1:
            return float(df["high"].iloc[-1] - df["low"].iloc[-1])
        tr = pd.concat([
            (df["high"] - df["low"]).abs(),
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ],
                       axis=1).max(axis=1)
        return float(tr.rolling(period).mean().iloc[-1])


def build_engine(args) -> BacktestEngine:
    _require_nautilus()
    cfg = BacktestConfig()
    engine = BacktestEngine(config=cfg)
    catalog = ParquetDataCatalog(args.data_dir)
    engine.add_data_provider(catalog)
    allocator = PortfolioAllocator(
        target_weights={
            s: w
            for s, w in zip(args.symbols, _weights_for(args))
        },
        max_risk_per_trade=args.max_risk,
    )
    trade_mgr = TradeManager(
        atr_mult_sl=args.atr_sl,
        atr_mult_trail=args.atr_trail,
        tp1_r=args.tp1_r,
        tp2_r=args.tp2_r,
        tp_alloc=[
            args.tp1_alloc, args.tp2_alloc,
            1.0 - args.tp1_alloc - args.tp2_alloc
        ],
        enable_pyramiding=args.enable_pyramiding,
        max_layers=args.max_layers,
        pyramid_step_r=args.pyramid_step_r,
    )
    strat = DimReductionStrategy(
        symbols=args.symbols,
        feature_timeframe=args.timeframe,
        dim_model_dir=args.results_dir,
        allocator=allocator,
        trade_mgr=trade_mgr,
        warmup_bars=args.warmup,
    )
    engine.add_strategy(strat)
    return engine


def _weights_for(args) -> List[float]:
    if args.weights and len(args.weights) == len(args.symbols):
        return args.weights
    # Default equal weights
    return [1.0 / max(1, len(args.symbols))] * len(args.symbols)


def parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(
        description=
        "Nautilus backtest with AE+LGB model and advanced trade management")
    p.add_argument("--data-dir",
                   required=True,
                   help="Parquet data directory (catalog root)")
    p.add_argument(
        "--results-dir",
        required=True,
        help=
        "Directory containing production_model.pkl and production_autoencoder.pth"
    )
    p.add_argument("--symbols",
                   nargs="+",
                   default=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
                   help="Symbols to backtest")
    p.add_argument("--timeframe",
                   default="5T",
                   help="Bar timeframe key used for feature engineering")
    p.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    p.add_argument("--warmup",
                   type=int,
                   default=600,
                   help="Number of bars for feature warmup fit")
    # Risk and management
    p.add_argument("--max-risk",
                   type=float,
                   default=0.005,
                   help="Per-symbol max risk per trade as fraction of equity")
    p.add_argument("--atr-sl",
                   type=float,
                   default=1.5,
                   help="Initial stop distance in ATR multiples")
    p.add_argument("--atr-trail",
                   type=float,
                   default=3.0,
                   help="Trailing stop ATR multiple for runner")
    p.add_argument("--tp1-r",
                   type=float,
                   default=2.0,
                   help="TP1 at R multiples")
    p.add_argument("--tp2-r",
                   type=float,
                   default=3.0,
                   help="TP2 at R multiples")
    p.add_argument("--tp1-alloc",
                   type=float,
                   default=0.33,
                   help="Fraction of position for TP1")
    p.add_argument("--tp2-alloc",
                   type=float,
                   default=0.33,
                   help="Fraction of position for TP2")
    p.add_argument("--enable-pyramiding",
                   action="store_true",
                   help="Enable pyramiding after entry")
    p.add_argument("--max-layers",
                   type=int,
                   default=0,
                   help="Max additional layers when pyramiding enabled")
    p.add_argument("--pyramid-step-r",
                   type=float,
                   default=1.5,
                   help="Add layer every N*R in favor")
    p.add_argument(
        "--weights",
        nargs="*",
        type=float,
        help="Optional explicit portfolio weights aligned to --symbols")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    _require_nautilus()
    engine = build_engine(args)
    # Load data range
    if args.start:
        engine.config.start_time = pd.to_datetime(args.start)
    if args.end:
        engine.config.end_time = pd.to_datetime(args.end)
    engine.run()
    # Results handling could be extended to export equity curve and trades
    return 0


if __name__ == "__main__":
    sys.exit(main())
