"""
Nautilus Trader Strategy with Feature Engineering Integration

This module provides a Strategy class that integrates the YAML-based feature
loading system with Nautilus Trader's event-driven architecture.

Usage:
    from nautilus_trader.model import InstrumentId, BarType
    from nautilus_trader.model import BarSpecification, BarAggregation, PriceType
    from nautilus_trader.model import AggregationSource

    instrument_id = InstrumentId.from_str("BTCUSDT-PERP.BINANCE")
    bar_type = BarType(
        instrument_id=instrument_id,
        bar_spec=BarSpecification(15, BarAggregation.MINUTE, PriceType.LAST),
        aggregation_source=AggregationSource.EXTERNAL,
    )

    strategy = NautilusStrategyWithFeatures(
        strategy_name="sr_reversal",
        instrument_id=instrument_id,
        bar_type=bar_type,
        trade_size=0.001,
    )
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import numpy as np

try:
    from nautilus_trader.model import Bar
    from nautilus_trader.model import QuoteTick
    from nautilus_trader.model import TradeTick
    from nautilus_trader.model import InstrumentId
    from nautilus_trader.model import BarType
    from nautilus_trader.model import OrderSide
    from nautilus_trader.model import MarketOrder
    from nautilus_trader.model import LimitOrder
    from nautilus_trader.trading import Strategy

    NAUTILUS_AVAILABLE = True
except ImportError:
    NAUTILUS_AVAILABLE = False

    # 提供占位符类用于文档生成
    class Strategy:
        pass

    class Bar:
        pass

    class InstrumentId:
        pass

    class BarType:
        pass


from src.time_series_model.core.constitution.constitution_executor import (
    ConstitutionExecutor,
)
from src.time_series_model.core.constitution.execution_evidence import (
    compute_execution_evidence,
    load_evidence_quantiles,
)
from src.time_series_model.live.enforcement import enforce_before_order
from src.time_series_model.live.execution_manager import (
    ExecutionManager,
    GuardedOrderContext,
)
from src.time_series_model.nnmultihead.strategy_profile import (
    resolve_execution_profile,
    resolve_execution_profile_paths,
)
from src.time_series_model.live.live_runtime_paths import resolve_live_runtime_paths


class RealtimeFeatureManager:
    """
    Minimal real-time feature manager.
    Keeps a rolling window of bars and returns latest row as features.
    """

    def __init__(self, strategy_name: str, history_window: int, config_base_path: str):
        self.strategy_name = strategy_name
        self.history_window = history_window
        self.config_base_path = config_base_path
        self.history = pd.DataFrame()

    def compute_features(self, new_bar_df: pd.DataFrame) -> pd.DataFrame:
        # Append new bar and keep rolling window
        self.history = pd.concat([self.history, new_bar_df], ignore_index=True).tail(
            self.history_window
        )
        return self.history.copy()

    def get_latest_features(self) -> Optional[pd.DataFrame]:
        if self.history.empty:
            return None
        return self.history.tail(1)

    def get_feature_columns(self) -> list:
        return list(self.history.columns)

    def reset_history(self) -> None:
        self.history = pd.DataFrame()


if NAUTILUS_AVAILABLE:

    class NautilusStrategyWithFeatures(Strategy):
        """
        Nautilus Trader Strategy with integrated feature engineering.

        This strategy integrates the YAML-based feature loading system with
        Nautilus Trader's event-driven architecture. Features are computed
        incrementally as new bars arrive, and trading signals are generated
        based on the computed features.

        Attributes:
            strategy_name: Name of the strategy (e.g., "sr_reversal")
            instrument_id: Trading instrument identifier
            bar_type: Bar type specification for data subscription
            trade_size: Base trade size
            feature_manager: RealtimeFeatureManager instance
            strategy_config: Strategy configuration loaded from YAML
            model: Trained ML model for signal generation
        """

        def __init__(
            self,
            strategy_name: str,
            instrument_id: InstrumentId,
            bar_type: BarType,
            trade_size: float,
            config_base_path: str = "config/strategies",
            history_window: int = 1000,
            model_path: Optional[str] = None,
            constitution_yaml: Optional[str] = None,
        ):
            """
            Initialize the strategy.

            Args:
                strategy_name: Strategy name (must match config directory)
                instrument_id: Trading instrument identifier
                bar_type: Bar type for data subscription
                trade_size: Base trade size
                config_base_path: Base path for strategy configs
                history_window: Size of historical data window for features
                model_path: Path to trained model file (optional)
            """
            super().__init__()
            self.strategy_name = strategy_name
            self.instrument_id = instrument_id
            self.bar_type = bar_type
            self.trade_size = trade_size
            self.config_base_path = config_base_path
            self.history_window = history_window
            self.model_path = model_path
            live_paths = resolve_live_runtime_paths()
            self.constitution_yaml = (
                constitution_yaml or live_paths["constitution_yaml"]
            )

            # Will be initialized in on_start()
            self.feature_manager: Optional[RealtimeFeatureManager] = None
            self.strategy_config = None  # deprecated; tree configs are not used in live
            self.model = None
            self.feature_columns: Optional[list] = None
            self._constitution_executor: Optional[ConstitutionExecutor] = None
            self._constitution_runtime_state = None

        def _infer_mode_and_exec_id(self) -> tuple[str, str]:
            name = str(self.strategy_name).lower()
            mode = "MEAN" if ("reversal" in name or "mean" in name) else "TREND"
            sid = "FailedBreakoutFade" if mode == "MEAN" else "MomentumExpansion"
            return mode, sid

        def _get_execution_meta(self) -> Dict[str, Any]:
            # nnmultihead-first: read from config/nnmultihead/strategies/<strategy_id>/profile.yaml
            root, reg = resolve_execution_profile_paths()
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
            """Called when the strategy starts."""
            self.log.info(f"🚀 Starting {self.strategy_name} strategy")

            try:
                # 1. Tree strategy configs are NOT supported in live (by design).
                self.log.info(
                    "ℹ️ Live does not load config/strategies/* (tree configs are research-only)."
                )

                # 2. Initialize feature manager
                self.feature_manager = RealtimeFeatureManager(
                    strategy_name=self.strategy_name,
                    history_window=self.history_window,
                    config_base_path=self.config_base_path,
                )
                self.log.info(f"✅ Initialized feature manager")

                # 3. Load trained model (if provided)
                if self.model_path:
                    self.model = self._load_model(self.model_path)
                    self.log.info(f"✅ Loaded model from {self.model_path}")
                else:
                    # Try to find model in default location
                    default_model_path = (
                        Path("models") / self.strategy_name / "model.pkl"
                    )
                    if default_model_path.exists():
                        self.model = self._load_model(str(default_model_path))
                        self.log.info(f"✅ Loaded model from {default_model_path}")
                    else:
                        self.log.warning(
                            f"⚠️ No model found. Signal generation will be disabled."
                        )

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
                self.log.error(f"❌ Error during strategy initialization: {e}")
                import traceback

                self.log.error(traceback.format_exc())
                raise

        def on_bar(self, bar: Bar) -> None:
            """
            Called when a new bar is received.

            Args:
                bar: Nautilus Trader Bar object
            """
            try:
                # 1. Convert Bar to DataFrame
                new_bar_df = self._bar_to_dataframe(bar)

                # 2. Compute features
                features_df = self.feature_manager.compute_features(new_bar_df)

                # 3. Get latest features for signal generation
                latest_features = self.feature_manager.get_latest_features()

                if latest_features is None or len(latest_features) == 0:
                    self.log.debug("⚠️ No features available yet (need more history)")
                    return

                # 4. Generate trading signal
                signal = self._generate_signal(latest_features)

                # 5. Execute trade if signal exists
                if signal:
                    self._execute_trade(signal, bar)

            except Exception as e:
                self.log.error(f"❌ Error processing bar: {e}")
                import traceback

                self.log.error(traceback.format_exc())

        def on_tick(self, tick: QuoteTick) -> None:
            """
            Called when a new quote tick is received.

            Note: This is optional. Most strategies work with bars.
            If you need tick-level features, implement this method.

            Args:
                tick: Nautilus Trader QuoteTick object
            """
            # Optional: Implement tick-level feature computation
            pass

        def on_trade_tick(self, tick: TradeTick) -> None:
            """
            Called when a new trade tick is received.

            Note: This is optional. Most strategies work with bars.
            If you need trade-level features, implement this method.

            Args:
                tick: Nautilus Trader TradeTick object
            """
            # Optional: Implement trade-level feature computation
            pass

        def _bar_to_dataframe(self, bar: Bar) -> pd.DataFrame:
            """
            Convert Nautilus Trader Bar to DataFrame.

            Args:
                bar: Nautilus Trader Bar object

            Returns:
                DataFrame with columns: timestamp, datetime, open, high, low, close, volume, symbol
            """
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

        def _load_model(self, model_path: str) -> Any:
            """
            Load trained model from file.

            Args:
                model_path: Path to model file

            Returns:
                Loaded model object
            """
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            return model

        def _generate_signal(
            self, features_df: pd.DataFrame
        ) -> Optional[Dict[str, Any]]:
            """
            Generate trading signal from features.

            Args:
                features_df: DataFrame with features (single row)

            Returns:
                Signal dictionary with side, quantity, price, etc., or None
            """
            if self.model is None:
                # Fallback: Use rule-based logic if no model
                return self._generate_rule_based_signal(features_df)

            try:
                # 1. Get feature columns
                if self.feature_columns is None:
                    self.feature_columns = self.feature_manager.get_feature_columns()

                # 2. Extract feature values
                X = features_df[self.feature_columns].values

                # 3. Model prediction
                if hasattr(self.model, "predict_proba"):
                    prediction = self.model.predict(X)[0]
                    probability = self.model.predict_proba(X)[0]
                else:
                    prediction = self.model.predict(X)[0]
                    probability = None

                # 4. Generate signal based on prediction
                signal = {
                    "prediction": int(prediction),
                    "probability": (
                        probability.tolist() if probability is not None else None
                    ),
                    "features": features_df.iloc[0].to_dict(),
                }

                # 5. Determine trade side
                if prediction == 1:  # Buy signal
                    signal["side"] = OrderSide.BUY
                elif prediction == -1:  # Sell signal
                    signal["side"] = OrderSide.SELL
                else:
                    return None  # No signal

                return signal

            except Exception as e:
                self.log.error(f"❌ Error generating signal: {e}")
                return None

        def _generate_rule_based_signal(
            self, features_df: pd.DataFrame
        ) -> Optional[Dict[str, Any]]:
            """
            Generate rule-based signal when model is not available.

            This is a simple example. Replace with your own rule logic.

            Args:
                features_df: DataFrame with features

            Returns:
                Signal dictionary or None
            """
            # Disabled by constitutional strategy subtraction:
            # "indicator mean" (RSI/Stoch/CCI etc) is forbidden as entry logic.
            return None

        def _execute_trade(self, signal: Dict[str, Any], bar: Bar) -> None:
            """
            Execute trade based on signal.

            Args:
                signal: Trading signal dictionary
                bar: Current bar data
            """
            try:
                side = signal["side"]
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
                    feats = signal.get("features") or {}
                    try:
                        quantiles = load_evidence_quantiles(
                            os.getenv("MLBOT_EVIDENCE_QUANTILES_JSON")
                        )
                        evidence = compute_execution_evidence(
                            features=dict(feats) if isinstance(feats, dict) else {},
                            rules=ex_meta.get("evidence_rules") or [],
                            quantiles=quantiles,
                        )
                    except Exception as e:
                        self.log.error(f"❌ evidence_dsl_error -> NO_TRADE: {e}")
                        return
                    enforce_before_order(
                        executor=self._constitution_executor,
                        runtime_state=self._constitution_runtime_state,
                        position_id=f"{self.strategy_name}:{int(bar.ts_event)}",
                        symbol=str(self.instrument_id),
                        mode=mode,
                        execution_strategy=exec_id,
                        execution_tags=[str(self.strategy_name)],
                        execution_evidence=evidence,
                    )

                # Create market order
                order = self.order_factory.market(
                    instrument_id=self.instrument_id,
                    order_side=side,
                    quantity=quantity,
                )

                # Submit order
                if (
                    getattr(self, "_xm", None) is not None
                    and self._constitution_executor is not None
                    and self._constitution_runtime_state is not None
                ):
                    self._xm.submit_order_guarded(
                        order=order,
                        ctx=GuardedOrderContext(
                            position_id=f"{self.strategy_name}:{int(bar.ts_event)}",
                            symbol=str(self.instrument_id),
                            mode=mode,
                            execution_strategy=exec_id,
                            execution_tags=[str(self.strategy_name)],
                            execution_evidence=evidence,
                        ),
                    )
                else:
                    # Very conservative fallback: if executor not loaded, do not submit.
                    self.log.error(
                        "❌ No ExecutionManager/ConstitutionExecutor; refusing to submit order"
                    )
                    return

                self.log.info(
                    f"📊 Signal: {side} {quantity} @ {bar.close} "
                    f"(prediction: {signal.get('prediction')}, "
                    f"prob: {signal.get('probability')})"
                )

            except Exception as e:
                self.log.error(f"❌ Error executing trade: {e}")
                import traceback

                self.log.error(traceback.format_exc())

        def on_stop(self) -> None:
            """Called when the strategy stops."""
            self.log.info(f"🛑 Stopping {self.strategy_name} strategy")
            if self.feature_manager:
                self.feature_manager.reset_history()
                self.log.info("✅ Feature manager cleaned up")

else:
    # Fallback when Nautilus Trader is not available
    class NautilusStrategyWithFeatures:
        """Placeholder class when Nautilus Trader is not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Nautilus Trader is not installed. "
                "Install it with: pip install nautilus-trader"
            )
