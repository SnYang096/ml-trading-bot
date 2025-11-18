"""Nautilus Trader backtest using LightGBM dimensionality models."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

try:
    # Lazy import Nautilus; fail with a helpful error if missing
    from nautilus_trader.backtest.data.providers import ParquetDataCatalog
    from nautilus_trader.backtest.engine import BacktestEngine
    from nautilus_trader.common.clock import TestClock
    from nautilus_trader.common.logging import Logger
    from nautilus_trader.config import BacktestConfig
    from nautilus_trader.core.message import Event
    from nautilus_trader.model.data import Bar
    from nautilus_trader.model.enums import BarType
    from nautilus_trader.model.identifiers import Symbol
    from nautilus_trader.trading.strategy import Strategy
except Exception as _e:  # pragma: no cover
    BacktestEngine = None
    ParquetDataCatalog = None
    BacktestConfig = None
    Strategy = object  # type: ignore
    _NAUTILUS_ERR = _e

from data_tools.comprehensive_feature_engineering import ComprehensiveFeatureEngineer

try:
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "scikit-learn is required for Nautilus backtest (StandardScaler)."
    ) from exc


def _require_nautilus() -> None:
    if BacktestEngine is None or ParquetDataCatalog is None or BacktestConfig is None:
        raise RuntimeError(
            "nautilus-trader is not installed or failed to import. "
            f"Original error: {_NAUTILUS_ERR}"
        )


@dataclass
class PositionSlice:
    entry_price: float
    size: float
    stop_price: float
    tp1: Optional[float] = None
    tp2: Optional[float] = None
    is_active: bool = True


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

        self.fe_map: Dict[str, ComprehensiveFeatureEngineer] = {}
        self.buf_map: Dict[str, List[Bar]] = {}
        self.atr_map: Dict[str, float] = {s: math.nan for s in symbols}
        self.pos_map: Dict[str, List[PositionSlice]] = {s: [] for s in symbols}
        self.scaler_map: Dict[str, StandardScaler] = {}

        model_pkl = os.path.join(dim_model_dir, "production_model.pkl")
        if not os.path.exists(model_pkl):
            raise FileNotFoundError(
                f"Missing LightGBM model in {dim_model_dir}. Expected production_model.pkl"
            )
        self.lgb_model = joblib.load(model_pkl)

    def on_start(self):
        for symbol in self.symbols:
            self.fe_map[symbol] = ComprehensiveFeatureEngineer()
            self.buf_map[symbol] = []

    def on_bar(self, bar: Bar):  # type: ignore[override]
        symbol = str(bar.symbol)
        if symbol not in self.symbols:
            return
        buf = self.buf_map[symbol]
        buf.append(bar)
        if len(buf) < self.warmup_bars:
            return

        df = self._bars_to_df(buf)
        fe = self.fe_map[symbol]
        try:
            eng = fe.engineer_all_features(df, fit=(len(buf) == self.warmup_bars))
        except Exception as exc:
            self.log.warning(f"Feature engineering failed for {symbol}: {exc}")
            return

        feature_cols = [col for col in eng.columns if col not in ["timestamp", "close"]]
        if not feature_cols:
            return
        X_all = eng[feature_cols].values.astype(np.float32)
        scaler = self.scaler_map.get(symbol)
        if scaler is None and len(buf) >= self.warmup_bars:
            scaler = StandardScaler()
            scaler.fit(X_all)
            self.scaler_map[symbol] = scaler
        if scaler is None:
            return
        X_scaled = scaler.transform(X_all)
        X_last = X_scaled[-1:].astype(np.float32)

        try:
            pred = float(self.lgb_model.predict(X_last)[0])
        except Exception as exc:
            self.log.warning(f"Prediction failed for {symbol}: {exc}")
            return

        atr = self._compute_atr(df.tail(50))
        self.atr_map[symbol] = atr

        threshold = 0.0
        price = float(bar.close)
        if pred > threshold and not self.pos_map[symbol]:
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

        if self.pos_map[symbol]:
            new_slices: List[PositionSlice] = []
            for slc in self.pos_map[symbol]:
                if not slc.is_active:
                    continue
                if slc.tp1 is not None and price >= slc.tp1:
                    slc.is_active = False
                    continue
                if slc.tp2 is not None and price >= slc.tp2:
                    slc.is_active = False
                    continue
                if slc.tp1 is None and slc.tp2 is None:
                    self.trade_mgr.update_trailing(slc, price, atr)
                if price <= slc.stop_price:
                    slc.is_active = False
                    continue
                new_slices.append(slc)
            self.pos_map[symbol] = new_slices

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
        up = (df["close"].diff() > 0).astype(float)
        df["taker_buy_ratio"] = up.rolling(20, min_periods=5).mean().fillna(0.5)
        signed_vol = np.sign(df["close"].diff().fillna(0.0)) * df["volume"].fillna(0.0)
        df["cvd"] = signed_vol.cumsum()
        df["cvd_change_1"] = signed_vol
        df["cvd_change_5"] = signed_vol.rolling(5, min_periods=1).sum()
        df["cvd_change_20"] = signed_vol.rolling(20, min_periods=1).sum()
        df["cvd_short"] = signed_vol.rolling(20, min_periods=1).sum()
        df["cvd_medium"] = signed_vol.rolling(60, min_periods=1).sum()
        df["cvd_long"] = signed_vol.rolling(288, min_periods=1).sum()
        total_vol = df["volume"].rolling(20, min_periods=1).sum()
        df["cvd_normalized"] = (
            (df["cvd_short"] / total_vol.replace(0, np.nan))
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        return df

    def _compute_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        if len(df) < period + 1:
            return float(df["high"].iloc[-1] - df["low"].iloc[-1])
        tr = pd.concat(
            [
                (df["high"] - df["low"]).abs(),
                (df["high"] - df["close"].shift()).abs(),
                (df["low"] - df["close"].shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return float(tr.rolling(period).mean().iloc[-1])


class PortfolioAllocator:

    def __init__(
        self, target_weights: Dict[str, float], max_risk_per_trade: float = 0.005
    ):
        w_sum = sum(max(0.0, v) for v in target_weights.values())
        self.weights = {
            k: (v / w_sum if w_sum > 0 else 0.0) for k, v in target_weights.items()
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

    def initial_slices(
        self, entry: float, atr: float, qty: float
    ) -> List[PositionSlice]:
        r_dollar = atr
        sl = entry - self.atr_mult_sl * atr
        part_sizes = [qty * a for a in self.tp_alloc]
        tp1 = entry + self.tp1_r * r_dollar
        tp2 = entry + self.tp2_r * r_dollar
        return [
            PositionSlice(
                entry_price=entry, size=part_sizes[0], stop_price=sl, tp1=tp1
            ),
            PositionSlice(
                entry_price=entry, size=part_sizes[1], stop_price=sl, tp2=tp2
            ),
            PositionSlice(entry_price=entry, size=part_sizes[2], stop_price=sl),
        ]

    def update_trailing(self, slice_: PositionSlice, price: float, atr: float) -> None:
        trail = price - self.atr_mult_trail * atr
        if trail > slice_.stop_price:
            slice_.stop_price = trail


def _weights_for(args) -> List[float]:
    if args.weights and len(args.weights) == len(args.symbols):
        return args.weights
    return [1.0 / max(1, len(args.symbols))] * len(args.symbols)


def build_engine(args) -> BacktestEngine:
    _require_nautilus()
    cfg = BacktestConfig()
    engine = BacktestEngine(config=cfg)
    catalog = ParquetDataCatalog(args.data_dir)
    engine.add_data_provider(catalog)
    allocator = PortfolioAllocator(
        target_weights={
            symbol: weight for symbol, weight in zip(args.symbols, _weights_for(args))
        },
        max_risk_per_trade=args.max_risk,
    )
    trade_mgr = TradeManager(
        atr_mult_sl=args.atr_sl,
        atr_mult_trail=args.atr_trail,
        tp1_r=args.tp1_r,
        tp2_r=args.tp2_r,
        tp_alloc=[
            args.tp1_alloc,
            args.tp2_alloc,
            1.0 - args.tp1_alloc - args.tp2_alloc,
        ],
        enable_pyramiding=args.enable_pyramiding,
        max_layers=args.max_layers,
        pyramid_step_r=args.pyramid_step_r,
    )
    strategy = DimReductionStrategy(
        symbols=args.symbols,
        feature_timeframe=args.timeframe,
        dim_model_dir=args.results_dir,
        allocator=allocator,
        trade_mgr=trade_mgr,
        warmup_bars=args.warmup,
    )
    engine.add_strategy(strategy)
    return engine


def export_run_artifacts(args, engine: BacktestEngine, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "timestamp": datetime.utcnow().isoformat(),
        "symbols": args.symbols,
        "timeframe": args.timeframe,
        "start": args.start,
        "end": args.end,
        "warmup": args.warmup,
        "max_risk": args.max_risk,
        "atr_sl": args.atr_sl,
        "atr_trail": args.atr_trail,
        "tp1_r": args.tp1_r,
        "tp2_r": args.tp2_r,
        "tp1_alloc": args.tp1_alloc,
        "tp2_alloc": args.tp2_alloc,
        "enable_pyramiding": args.enable_pyramiding,
        "max_layers": args.max_layers,
        "pyramid_step_r": args.pyramid_step_r,
        "weights": args.weights,
        "dimensionality_model_dir": args.results_dir,
    }
    metadata_path = run_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    # Attempt to export performance/trade logs if available
    exported = []
    try:
        if hasattr(engine, "logs_dir") and engine.logs_dir:
            logs_src = Path(engine.logs_dir)
            if logs_src.exists():
                target = run_dir / "engine_logs"
                target.mkdir(exist_ok=True)
                for file in logs_src.glob("*"):
                    if file.is_file():
                        target_file = target / file.name
                        target_file.write_bytes(file.read_bytes())
                exported.append(str(target))
    except Exception as exc:  # pragma: no cover
        (run_dir / "export_warn.log").write_text(
            f"Failed to copy engine logs: {exc}", encoding="utf-8"
        )

    try:
        portfolio = getattr(engine, "portfolio", None)
        if portfolio is not None and hasattr(portfolio, "to_dict"):
            portfolio_path = run_dir / "portfolio_snapshot.json"
            with portfolio_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    portfolio.to_dict(), handle, indent=2
                )  # type: ignore[func-returns-value]
            exported.append(str(portfolio_path))
    except Exception as exc:  # pragma: no cover
        (run_dir / "export_warn.log").write_text(
            f"Failed to serialize portfolio snapshot: {exc}", encoding="utf-8"
        )

    summary = run_dir / "README.txt"
    summary.write_text(
        "Nautilus dimensionality backtest completed.\n"
        f"Artifacts: metadata.json{', ' + ', '.join(exported) if exported else ''}\n",
        encoding="utf-8",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Nautilus backtest with LightGBM dimensionality strategy",
    )
    parser.add_argument(
        "--data-dir", required=True, help="Parquet data directory (catalog root)"
    )
    parser.add_argument(
        "--results-dir", required=True, help="Directory containing production_model.pkl"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        help="Symbols to backtest",
    )
    parser.add_argument(
        "--timeframe",
        default="5T",
        help="Bar timeframe key used for feature engineering",
    )
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--warmup", type=int, default=600, help="Number of bars for feature warmup fit"
    )
    parser.add_argument(
        "--max-risk",
        type=float,
        default=0.005,
        help="Per-symbol max risk per trade (fraction of equity)",
    )
    parser.add_argument(
        "--atr-sl",
        type=float,
        default=1.5,
        help="Initial stop distance in ATR multiples",
    )
    parser.add_argument(
        "--atr-trail",
        type=float,
        default=3.0,
        help="Trailing stop ATR multiple for runner",
    )
    parser.add_argument(
        "--tp1-r", type=float, default=2.0, help="Take profit 1 in R multiples"
    )
    parser.add_argument(
        "--tp2-r", type=float, default=3.0, help="Take profit 2 in R multiples"
    )
    parser.add_argument(
        "--tp1-alloc", type=float, default=0.33, help="Fraction of position for TP1"
    )
    parser.add_argument(
        "--tp2-alloc", type=float, default=0.33, help="Fraction of position for TP2"
    )
    parser.add_argument(
        "--enable-pyramiding", action="store_true", help="Enable pyramiding after entry"
    )
    parser.add_argument(
        "--max-layers",
        type=int,
        default=0,
        help="Max additional layers when pyramiding enabled",
    )
    parser.add_argument(
        "--pyramid-step-r", type=float, default=1.5, help="Add layer every N*R in favor"
    )
    parser.add_argument(
        "--weights",
        nargs="*",
        type=float,
        help="Optional explicit portfolio weights aligned to --symbols",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.environ.get("NAUTILUS_RESULTS_DIR", "results/nautilus_backtests"),
        help="Directory to store backtest artifacts",
    )
    return parser


def run_backtest(args) -> Path:
    _require_nautilus()
    engine = build_engine(args)

    if args.start:
        engine.config.start_time = pd.to_datetime(args.start)
    if args.end:
        engine.config.end_time = pd.to_datetime(args.end)

    engine.run()

    symbol_tag = "-".join(args.symbols)
    start_tag = (args.start or "start").replace("-", "")
    end_tag = (args.end or "end").replace("-", "")
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / (
        f"{symbol_tag}_{args.timeframe}_{start_tag}_{end_tag}_{timestamp}"
    )

    export_run_artifacts(args, engine, run_dir)
    print(f"✅ Nautilus backtest artifacts saved to {run_dir}")
    return run_dir


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_backtest(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
