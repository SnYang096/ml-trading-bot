"""
Enhanced Nautilus Trader Strategy with ModelArtifact Integration

This module provides a production-ready Strategy class that integrates:
- ModelArtifact for unified model loading
- Real-time feature computation using FeatureStore
- RR-based exit logic (stop-loss, take-profit, trailing stop)
- Position management and risk control

Usage:
    strategy = NautilusStrategyEnhanced(
        strategy_name="sr_reversal_rr_reg_long",
        instrument_id=InstrumentId.from_str("BTCUSDT-PERP.BINANCE"),
        bar_type=bar_type,
        trade_size=0.001,
        model_artifact_path="models/sr_reversal_rr_reg_long",
    )
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from src.time_series_model.core.constitution.constitution_executor import (
    ConstitutionExecutor,
)
from src.time_series_model.core.constitution.execution_evidence import (
    compute_execution_evidence,
)
from src.time_series_model.live.enforcement import enforce_before_order
from src.time_series_model.live.execution_manager import (
    ExecutionManager,
    GuardedOrderContext,
)
from src.time_series_model.nnmultihead.strategy_profile import resolve_execution_profile

try:
    from nautilus_trader.model.data import Bar, QuoteTick, TradeTick
    from nautilus_trader.model.identifiers import InstrumentId
    from nautilus_trader.model.enums import OrderSide, PositionSide
    from nautilus_trader.model.objects import Price, Quantity
    from nautilus_trader.trading.strategy import Strategy

    NAUTILUS_AVAILABLE = True
except ImportError:
    NAUTILUS_AVAILABLE = False

    class Strategy:
        pass


# =============================================================================
# Data Structures
# =============================================================================


class ExitReason(Enum):
    """Trade exit reasons"""

    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"
    BREAKEVEN_STOP = "breakeven_stop"
    TIME_EXIT = "time_exit"
    SIGNAL_EXIT = "signal_exit"
    MANUAL = "manual"


@dataclass
class Position:
    """Position tracking"""

    entry_time: datetime
    entry_price: float
    direction: int  # 1=Long, -1=Short
    quantity: float
    stop_loss: float
    take_profit: float
    trailing_stop: Optional[float] = None
    breakeven_triggered: bool = False
    bars_held: int = 0
    highest_price: float = 0.0  # For trailing stop (long)
    lowest_price: float = float("inf")  # For trailing stop (short)
    atr_at_entry: float = 0.0


@dataclass
class TradeRecord:
    """Completed trade record"""

    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    direction: int
    quantity: float
    pnl: float
    pnl_pct: float
    exit_reason: ExitReason
    bars_held: int
    features_at_entry: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# Enhanced Feature Manager
# =============================================================================


class EnhancedFeatureManager:
    """
    Production-ready feature manager with proper streaming computation.
    """

    def __init__(
        self,
        strategy_name: str,
        config_base_path: str = "config/strategies",
        history_window: int = 500,
    ):
        self.strategy_name = strategy_name
        self.config_base_path = Path(config_base_path)
        self.history_window = history_window

        # Rolling window of OHLCV data
        self.ohlcv_history = pd.DataFrame()

        # Computed features cache
        self._feature_cache: Optional[pd.DataFrame] = None
        self._last_compute_idx: int = -1

        # Load feature configuration
        self._load_feature_config()

    def _load_feature_config(self):
        """Load feature configuration from YAML"""
        config_path = self.config_base_path / self.strategy_name
        features_yaml = config_path / "features.yaml"

        if features_yaml.exists():
            with open(features_yaml) as f:
                self.feature_config = yaml.safe_load(f)
        else:
            self.feature_config = {}

    def update(self, new_bar: pd.DataFrame) -> pd.DataFrame:
        """
        Add new bar and compute features incrementally.

        Args:
            new_bar: Single row DataFrame with OHLCV data

        Returns:
            DataFrame with computed features for latest bar
        """
        # Append new bar
        self.ohlcv_history = pd.concat(
            [self.ohlcv_history, new_bar], ignore_index=True
        ).tail(self.history_window)

        # Compute basic features for latest row
        return self._compute_features_for_latest()

    def _compute_features_for_latest(self) -> pd.DataFrame:
        """Compute features for the latest bar only (incremental)"""
        df = self.ohlcv_history.copy()

        if len(df) < 20:  # Need minimum history
            return pd.DataFrame()

        # Basic features that can be computed incrementally
        idx = len(df) - 1

        # Returns
        df["returns"] = df["close"].pct_change()

        # ATR (simplified)
        tr = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - df["close"].shift(1)).abs(),
                (df["low"] - df["close"].shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        df["atr"] = tr.rolling(14).mean()

        # Volatility
        df["volatility_20"] = df["returns"].rolling(20).std()

        # Momentum
        df["momentum_10"] = df["close"].pct_change(10)
        df["momentum_20"] = df["close"].pct_change(20)

        # RSI
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df["rsi"] = 100 - (100 / (1 + rs))

        # Volume features
        df["volume_sma_20"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / (df["volume_sma_20"] + 1e-10)

        # Trend features
        df["sma_20"] = df["close"].rolling(20).mean()
        df["sma_50"] = df["close"].rolling(50).mean()
        df["trend_strength"] = (df["sma_20"] - df["sma_50"]) / df["atr"]

        # Price structure
        df["wick_upper_ratio"] = (df["high"] - df[["open", "close"]].max(axis=1)) / (
            df["high"] - df["low"] + 1e-10
        )
        df["wick_lower_ratio"] = (df[["open", "close"]].min(axis=1) - df["low"]) / (
            df["high"] - df["low"] + 1e-10
        )

        self._feature_cache = df
        return df.tail(1)

    def get_latest_features(self) -> Optional[pd.DataFrame]:
        """Get the most recent computed features"""
        if self._feature_cache is None or len(self._feature_cache) == 0:
            return None
        return self._feature_cache.tail(1)

    def get_feature_columns(self) -> List[str]:
        """Get list of computed feature column names"""
        exclude = {
            "open",
            "high",
            "low",
            "close",
            "volume",
            "timestamp",
            "datetime",
            "symbol",
        }
        if self._feature_cache is not None:
            return [c for c in self._feature_cache.columns if c not in exclude]
        return []

    def get_atr(self) -> float:
        """Get current ATR value"""
        if self._feature_cache is not None and "atr" in self._feature_cache.columns:
            atr = self._feature_cache["atr"].iloc[-1]
            return float(atr) if not pd.isna(atr) else 0.0
        return 0.0

    def reset(self):
        """Reset history and cache"""
        self.ohlcv_history = pd.DataFrame()
        self._feature_cache = None
        self._last_compute_idx = -1


# =============================================================================
# Enhanced Nautilus Strategy
# =============================================================================


if NAUTILUS_AVAILABLE:

    class NautilusStrategyEnhanced(Strategy):
        """
        Production-ready Nautilus Strategy with ModelArtifact and RR exits.
        """

        def __init__(
            self,
            strategy_name: str,
            instrument_id: InstrumentId,
            bar_type,
            trade_size: float,
            config_base_path: str = "config/strategies",
            model_artifact_path: Optional[str] = None,
            history_window: int = 500,
            # RR exit parameters (can be overridden by backtest.yaml)
            use_rr_exit: bool = True,
            stop_loss_r: float = 1.0,
            take_profit_r: float = 2.0,
            max_holding_bars: int = 50,
            use_trailing_stop: bool = True,
            trailing_atr_mult: float = 1.0,
            use_breakeven_stop: bool = False,
            breakeven_trigger_r: float = 1.0,
            min_confidence: float = 0.3,
            constitution_yaml: Optional[str] = None,
        ):
            super().__init__()

            self.strategy_name = strategy_name
            self.instrument_id = instrument_id
            self.bar_type = bar_type
            self.trade_size = trade_size
            self.config_base_path = config_base_path
            self.model_artifact_path = model_artifact_path
            self.history_window = history_window

            # RR exit parameters
            self.use_rr_exit = use_rr_exit
            self.stop_loss_r = stop_loss_r
            self.take_profit_r = take_profit_r
            self.max_holding_bars = max_holding_bars
            self.use_trailing_stop = use_trailing_stop
            self.trailing_atr_mult = trailing_atr_mult
            self.use_breakeven_stop = use_breakeven_stop
            self.breakeven_trigger_r = breakeven_trigger_r
            self.min_confidence = min_confidence
            self.constitution_yaml = constitution_yaml or os.getenv(
                "MLBOT_CONSTITUTION_YAML", "config/constitution/constitution_v1.yaml"
            )

            # Will be initialized in on_start()
            self.feature_manager: Optional[EnhancedFeatureManager] = None
            self.model_artifact = None
            self.backtest_config: Dict = {}

            # Position tracking
            self.current_position: Optional[Position] = None
            self.trade_history: List[TradeRecord] = []
            self._constitution_executor: Optional[ConstitutionExecutor] = None
            self._constitution_runtime_state = None

        def _infer_mode_and_exec_id(self) -> tuple[str, str]:
            name = str(self.strategy_name).lower()
            if self.router_mode:
                mode = str(self.router_mode).upper()
            else:
                mode = "MEAN" if ("reversal" in name or "mean" in name) else "TREND"
            if self.execution_strategy_id:
                sid = str(self.execution_strategy_id)
            else:
                if mode == "MEAN":
                    sid = "FailedBreakoutFade"
                else:
                    sid = (
                        "BreakoutPullbackContinuation"
                        if ("breakout" in name)
                        else "MomentumExpansion"
                    )
            return mode, sid

        def _get_execution_meta(self) -> Dict[str, Any]:
            root = os.getenv(
                "MLBOT_NNMH_STRATEGY_PROFILE_ROOT", "config/nnmultihead/strategies"
            )
            reg = os.getenv(
                "MLBOT_NNMH_EXEC_ARCHETYPE_REGISTRY",
                "config/nnmultihead/execution_archetypes_v1.yaml",
            )
            ex = resolve_execution_profile(
                strategy_id=str(self.strategy_name),
                profile_root=root,
                archetype_registry_path=reg,
            )
            if ex is None:
                return {}
            return {
                "router_mode": ex.router_mode,
                "execution_strategy_id": ex.execution_strategy_id,
                "evidence_rules": ex.evidence_rules,
            }

        def on_start(self) -> None:
            """Initialize strategy components"""
            self.log.info(f"🚀 Starting Enhanced {self.strategy_name} strategy")

            try:
                # 1. Load backtest configuration
                self._load_backtest_config()

                # 2. Initialize feature manager
                self.feature_manager = EnhancedFeatureManager(
                    strategy_name=self.strategy_name,
                    config_base_path=self.config_base_path,
                    history_window=self.history_window,
                )
                self.log.info("✅ Feature manager initialized")

                # 3. Load ModelArtifact
                self._load_model_artifact()

                # 3.5 Load constitution executor (for live safety / whitelist)
                try:
                    if self.constitution_yaml and Path(self.constitution_yaml).exists():
                        self._constitution_executor = ConstitutionExecutor(
                            constitution_yaml=str(self.constitution_yaml)
                        )
                        self._constitution_runtime_state = (
                            self._constitution_executor.load_runtime_state()
                        )
                        self.log.info(
                            f"✅ Constitution loaded: {self.constitution_yaml}"
                        )
                        self._xm = ExecutionManager(
                            strategy=self,
                            executor=self._constitution_executor,
                            runtime_state=self._constitution_runtime_state,
                        )
                except Exception as e:
                    self.log.error(f"⚠️ Constitution init failed: {e}")

                # 4. Subscribe to market data
                self.subscribe_bars(self.bar_type)
                self.log.info(f"✅ Subscribed to {self.bar_type}")

                self.log.info("✅ Strategy initialization complete")

            except Exception as e:
                self.log.error(f"❌ Initialization error: {e}")
                import traceback

                self.log.error(traceback.format_exc())
                raise

        def _load_backtest_config(self):
            """Load backtest.yaml configuration"""
            config_path = (
                Path(self.config_base_path) / self.strategy_name / "backtest.yaml"
            )

            if config_path.exists():
                with open(config_path) as f:
                    config = yaml.safe_load(f)

                self.backtest_config = config.get("backtest", {}).get("params", {})

                # Override RR parameters from config
                if self.backtest_config.get("use_rr_exit") is not None:
                    self.use_rr_exit = self.backtest_config["use_rr_exit"]

                rr = self.backtest_config.get("rr", {})
                if rr:
                    self.stop_loss_r = rr.get("stop_loss_r", self.stop_loss_r)
                    self.take_profit_r = rr.get("take_profit_r", self.take_profit_r)
                    self.max_holding_bars = rr.get(
                        "max_holding_bars", self.max_holding_bars
                    )
                    self.use_trailing_stop = rr.get(
                        "use_trailing_stop", self.use_trailing_stop
                    )
                    self.trailing_atr_mult = rr.get(
                        "trailing_atr_mult", self.trailing_atr_mult
                    )
                    self.use_breakeven_stop = rr.get(
                        "use_breakeven_stop", self.use_breakeven_stop
                    )

                self.log.info(
                    f"✅ Loaded backtest config: use_rr_exit={self.use_rr_exit}"
                )

        def _load_model_artifact(self):
            """Load ModelArtifact from disk"""
            # Try explicit path first
            if self.model_artifact_path:
                artifact_path = Path(self.model_artifact_path)
            else:
                # Default path
                artifact_path = Path("models") / self.strategy_name

            if artifact_path.exists():
                try:
                    from src.time_series_model.strategies.models.model_artifact import (
                        ModelArtifact,
                    )

                    self.model_artifact = ModelArtifact.load(artifact_path)
                    self.log.info(f"✅ Loaded ModelArtifact from {artifact_path}")
                except Exception as e:
                    self.log.warning(f"⚠️ Failed to load ModelArtifact: {e}")
                    self.model_artifact = None
            else:
                self.log.warning(f"⚠️ No model found at {artifact_path}")

        def on_bar(self, bar: Bar) -> None:
            """Process new bar"""
            try:
                # 1. Convert bar to DataFrame
                new_bar_df = self._bar_to_dataframe(bar)

                # 2. Update features
                self.feature_manager.update(new_bar_df)

                # 3. Manage existing position
                if self.current_position is not None:
                    self._manage_position(bar)

                # 4. Check for new entry signals (only if no position)
                if self.current_position is None:
                    self._check_entry_signal(bar)

            except Exception as e:
                self.log.error(f"❌ Error processing bar: {e}")
                import traceback

                self.log.error(traceback.format_exc())

        def _bar_to_dataframe(self, bar: Bar) -> pd.DataFrame:
            """Convert Nautilus Bar to DataFrame"""
            return pd.DataFrame(
                {
                    "timestamp": [bar.ts_event],
                    "datetime": [pd.Timestamp.fromtimestamp(bar.ts_event / 1e9)],
                    "open": [float(bar.open)],
                    "high": [float(bar.high)],
                    "low": [float(bar.low)],
                    "close": [float(bar.close)],
                    "volume": [float(bar.volume)],
                    "symbol": [str(self.instrument_id)],
                }
            )

        def _check_entry_signal(self, bar: Bar):
            """Check for entry signal using model"""
            if self.model_artifact is None:
                return

            latest_features = self.feature_manager.get_latest_features()
            if latest_features is None or len(latest_features) == 0:
                return

            try:
                # Get prediction from ModelArtifact
                prediction = self.model_artifact.predict(latest_features)

                if len(prediction) == 0:
                    return

                pred_value = prediction[0]

                # For classification: check confidence
                if hasattr(self.model_artifact.model, "predict_proba"):
                    # Ensemble case
                    if isinstance(self.model_artifact.model, list):
                        probs = [
                            m.predict_proba(
                                latest_features[
                                    self.model_artifact.used_features
                                ].values
                            )
                            for m in self.model_artifact.model
                        ]
                        prob = np.mean(probs, axis=0)[0]
                    else:
                        prob = self.model_artifact.model.predict_proba(
                            latest_features[self.model_artifact.used_features].values
                        )[0]

                    confidence = max(prob) if len(prob) > 0 else 0.0

                    if confidence < self.min_confidence:
                        return

                # Generate entry signal
                atr = self.feature_manager.get_atr()
                close_price = float(bar.close)

                if pred_value > 0:  # Long signal
                    self._enter_position(
                        direction=1,
                        entry_price=close_price,
                        atr=atr,
                        features=latest_features.iloc[0].to_dict(),
                    )
                elif pred_value < 0:  # Short signal
                    self._enter_position(
                        direction=-1,
                        entry_price=close_price,
                        atr=atr,
                        features=latest_features.iloc[0].to_dict(),
                    )

            except Exception as e:
                self.log.error(f"❌ Error checking entry signal: {e}")

        def _enter_position(
            self,
            direction: int,
            entry_price: float,
            atr: float,
            features: Dict,
        ):
            """Enter a new position"""
            if atr <= 0:
                atr = entry_price * 0.01  # Fallback: 1% of price

            # Calculate stop-loss and take-profit
            if direction == 1:  # Long
                stop_loss = entry_price - self.stop_loss_r * atr
                take_profit = entry_price + self.take_profit_r * atr
            else:  # Short
                stop_loss = entry_price + self.stop_loss_r * atr
                take_profit = entry_price - self.take_profit_r * atr

            self.current_position = Position(
                entry_time=datetime.now(),
                entry_price=entry_price,
                direction=direction,
                quantity=self.trade_size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                trailing_stop=None,
                breakeven_triggered=False,
                bars_held=0,
                highest_price=entry_price,
                lowest_price=entry_price,
                atr_at_entry=atr,
            )

            # Submit order
            side = OrderSide.BUY if direction == 1 else OrderSide.SELL
            quantity = self.instrument.make_qty(self.trade_size)

            # Constitution enforcement (whitelist + slots) BEFORE submit_order
            if (
                self._constitution_executor is not None
                and self._constitution_runtime_state is not None
            ):
                ex_meta = self._get_execution_meta()
                mode = str(
                    ex_meta.get("router_mode") or self._infer_mode_and_exec_id()[0]
                ).upper()
                exec_id = str(
                    ex_meta.get("execution_strategy_id")
                    or self._infer_mode_and_exec_id()[1]
                )
                try:
                    evidence = compute_execution_evidence(
                        features=dict(features or {}),
                        rules=ex_meta.get("evidence_rules") or [],
                    )
                except Exception as e:
                    self.log.error(f"❌ evidence_dsl_error -> NO_TRADE: {e}")
                    return
                enforce_before_order(
                    executor=self._constitution_executor,
                    runtime_state=self._constitution_runtime_state,
                    position_id=f"{self.strategy_name}:{int(datetime.now().timestamp() * 1e9)}",
                    symbol=str(self.instrument_id),
                    mode=mode,
                    execution_strategy=exec_id,
                    execution_tags=[str(self.strategy_name)],
                    execution_evidence=evidence,
                )

            order = self.order_factory.market(
                instrument_id=self.instrument_id,
                order_side=side,
                quantity=quantity,
            )
            if (
                getattr(self, "_xm", None) is not None
                and self._constitution_executor is not None
                and self._constitution_runtime_state is not None
            ):
                self._xm.submit_order_guarded(
                    order=order,
                    ctx=GuardedOrderContext(
                        position_id=f"{self.strategy_name}:{int(datetime.now().timestamp() * 1e9)}",
                        symbol=str(self.instrument_id),
                        mode=mode,
                        execution_strategy=exec_id,
                        execution_tags=[str(self.strategy_name)],
                        execution_evidence=evidence,
                    ),
                )
            else:
                self.log.error(
                    "❌ No ExecutionManager/ConstitutionExecutor; refusing to submit order"
                )
                return

            self.log.info(
                f"📈 ENTRY: {'LONG' if direction == 1 else 'SHORT'} @ {entry_price:.2f} | "
                f"SL: {stop_loss:.2f} | TP: {take_profit:.2f} | ATR: {atr:.2f}"
            )

        def _manage_position(self, bar: Bar):
            """Manage existing position with RR exit logic"""
            pos = self.current_position
            if pos is None:
                return

            pos.bars_held += 1
            close_price = float(bar.close)
            high_price = float(bar.high)
            low_price = float(bar.low)

            # Update trailing stop reference prices
            if pos.direction == 1:  # Long
                pos.highest_price = max(pos.highest_price, high_price)
            else:  # Short
                pos.lowest_price = min(pos.lowest_price, low_price)

            # Check exit conditions
            exit_reason = None
            exit_price = close_price

            if self.use_rr_exit:
                # 1. Stop-loss check
                if pos.direction == 1:
                    if low_price <= pos.stop_loss:
                        exit_reason = ExitReason.STOP_LOSS
                        exit_price = pos.stop_loss
                else:
                    if high_price >= pos.stop_loss:
                        exit_reason = ExitReason.STOP_LOSS
                        exit_price = pos.stop_loss

                # 2. Take-profit check
                if exit_reason is None:
                    if pos.direction == 1:
                        if high_price >= pos.take_profit:
                            exit_reason = ExitReason.TAKE_PROFIT
                            exit_price = pos.take_profit
                    else:
                        if low_price <= pos.take_profit:
                            exit_reason = ExitReason.TAKE_PROFIT
                            exit_price = pos.take_profit

                # 3. Trailing stop
                if exit_reason is None and self.use_trailing_stop:
                    trailing_distance = self.trailing_atr_mult * pos.atr_at_entry

                    if pos.direction == 1:
                        trailing_stop = pos.highest_price - trailing_distance
                        if (
                            close_price <= trailing_stop
                            and trailing_stop > pos.entry_price
                        ):
                            exit_reason = ExitReason.TRAILING_STOP
                            exit_price = trailing_stop
                    else:
                        trailing_stop = pos.lowest_price + trailing_distance
                        if (
                            close_price >= trailing_stop
                            and trailing_stop < pos.entry_price
                        ):
                            exit_reason = ExitReason.TRAILING_STOP
                            exit_price = trailing_stop

                # 4. Breakeven stop
                if (
                    exit_reason is None
                    and self.use_breakeven_stop
                    and not pos.breakeven_triggered
                ):
                    breakeven_threshold = (
                        pos.entry_price
                        + pos.direction * self.breakeven_trigger_r * pos.atr_at_entry
                    )

                    if pos.direction == 1 and high_price >= breakeven_threshold:
                        pos.stop_loss = pos.entry_price
                        pos.breakeven_triggered = True
                        self.log.info(f"🔒 Breakeven triggered @ {pos.entry_price:.2f}")
                    elif pos.direction == -1 and low_price <= breakeven_threshold:
                        pos.stop_loss = pos.entry_price
                        pos.breakeven_triggered = True
                        self.log.info(f"🔒 Breakeven triggered @ {pos.entry_price:.2f}")

            # 5. Time exit
            if exit_reason is None and pos.bars_held >= self.max_holding_bars:
                exit_reason = ExitReason.TIME_EXIT
                exit_price = close_price

            # Execute exit if needed
            if exit_reason is not None:
                self._exit_position(exit_price, exit_reason)

        def _exit_position(self, exit_price: float, exit_reason: ExitReason):
            """Exit current position"""
            pos = self.current_position
            if pos is None:
                return

            # Calculate PnL
            if pos.direction == 1:
                pnl = (exit_price - pos.entry_price) * pos.quantity
                pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
            else:
                pnl = (pos.entry_price - exit_price) * pos.quantity
                pnl_pct = (pos.entry_price - exit_price) / pos.entry_price

            # Record trade
            trade = TradeRecord(
                entry_time=pos.entry_time,
                exit_time=datetime.now(),
                entry_price=pos.entry_price,
                exit_price=exit_price,
                direction=pos.direction,
                quantity=pos.quantity,
                pnl=pnl,
                pnl_pct=pnl_pct,
                exit_reason=exit_reason,
                bars_held=pos.bars_held,
            )
            self.trade_history.append(trade)

            # Submit closing order
            side = OrderSide.SELL if pos.direction == 1 else OrderSide.BUY
            quantity = self.instrument.make_qty(pos.quantity)

            order = self.order_factory.market(
                instrument_id=self.instrument_id,
                order_side=side,
                quantity=quantity,
            )
            if (
                getattr(self, "_xm", None) is not None
                and self._constitution_executor is not None
                and self._constitution_runtime_state is not None
            ):
                # Conservative: enforce before exits as well (may be relaxed later with dedicated exit hook).
                self._xm.submit_order_guarded(
                    order=order,
                    ctx=GuardedOrderContext(
                        position_id=f"{self.strategy_name}:{int(datetime.now().timestamp() * 1e9)}:exit",
                        symbol=str(self.instrument_id),
                        mode="NO_TRADE",
                        execution_strategy=str(self.strategy_name),
                        execution_tags=[str(self.strategy_name), "exit"],
                        execution_evidence={},
                    ),
                )
            else:
                self.log.error(
                    "❌ No ExecutionManager/ConstitutionExecutor; refusing to submit order"
                )
                return

            self.log.info(
                f"📉 EXIT [{exit_reason.value}]: "
                f"{'LONG' if pos.direction == 1 else 'SHORT'} @ {exit_price:.2f} | "
                f"PnL: {pnl:+.4f} ({pnl_pct:+.2%}) | Bars: {pos.bars_held}"
            )

            self.current_position = None

        def on_stop(self) -> None:
            """Cleanup on strategy stop"""
            self.log.info(f"🛑 Stopping Enhanced {self.strategy_name} strategy")

            # Close any open position
            if self.current_position is not None:
                self.log.warning("⚠️ Force closing open position")
                # Would need current price here - skip for now

            # Summary
            if self.trade_history:
                total_trades = len(self.trade_history)
                wins = sum(1 for t in self.trade_history if t.pnl > 0)
                total_pnl = sum(t.pnl for t in self.trade_history)

                self.log.info(
                    f"📊 Summary: {total_trades} trades | "
                    f"Win rate: {wins/total_trades:.1%} | "
                    f"Total PnL: {total_pnl:+.4f}"
                )

            if self.feature_manager:
                self.feature_manager.reset()

        def get_trade_history(self) -> List[Dict]:
            """Get trade history as list of dicts"""
            return [
                {
                    "entry_time": str(t.entry_time),
                    "exit_time": str(t.exit_time),
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "direction": "LONG" if t.direction == 1 else "SHORT",
                    "quantity": t.quantity,
                    "pnl": t.pnl,
                    "pnl_pct": t.pnl_pct,
                    "exit_reason": t.exit_reason.value,
                    "bars_held": t.bars_held,
                }
                for t in self.trade_history
            ]

else:

    class NautilusStrategyEnhanced:
        """Placeholder when Nautilus Trader is not installed"""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Nautilus Trader is not installed. "
                "Install with: pip install nautilus-trader"
            )
