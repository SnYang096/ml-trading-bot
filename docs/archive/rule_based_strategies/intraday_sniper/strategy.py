"""
Intraday Sniper Strategy - Main Class (Updated for MDC Indicator)
"""
import logging
from typing import Dict, Optional, Deque
from datetime import datetime, time
from collections import deque
import pandas as pd
import numpy as np
from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.correctness import PyCondition
from nautilus_trader.core.data import Data
from nautilus_trader.core.message import Event
from nautilus_trader.model.objects import Quantity, Price
from nautilus_trader.core.rust.model import OrderSide, OrderType, TimeInForce
from nautilus_trader.model.data import TradeTick, QuoteTick, Bar
from nautilus_trader.model.events.order import OrderAccepted, OrderFilled
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments.base import Instrument
from nautilus_trader.model.orders.base import Order
from nautilus_trader.model.position import Position
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.indicators import MovingAverageType
from nautilus_trader.indicators import BollingerBands, AverageTrueRange
from nautilus_trader.indicators import VolumeWeightedAveragePrice, MovingAverage, AverageTrueRange
from nautilus_trader.model.data import BarType
from nautilus_trader.adapters.binance import BINANCE_VENUE

from yin_bot.intraday_sniper.config import IntradaySniperConfig
from yin_bot.intraday_sniper.indicators.cvd import CVD
from yin_bot.intraday_sniper.indicators.scorer import BreakoutQualityScorer
from yin_bot.intraday_sniper.indicators.compression import AdaptiveMultiDimCompressionIndicator, IndicatorConfig, ThresholdConfig
from yin_bot.intraday_sniper.risk.position_sizer import PositionSizer
from yin_bot.intraday_sniper.filters.time_filter import TimeFilter
from yin_bot.intraday_sniper.filters.event_filter import EventFilter
from yin_bot.intraday_sniper.compression_cycle import CompressionCycle


class IntradaySniperStrategy(Strategy):
    """
    Intraday Sniper: Order-Flow Enhanced Bollinger Breakout Strategy (Updated for MDC)
    """

    def __init__(self, config: IntradaySniperConfig):
        super().__init__(config)
        self.instrument_id = config.instrument_id
        self.bar_type = BarType.from_str(config.bar_type)
        
        # Extract technical indicator parameters
        indicators = config.indicators
        self.bb_period = indicators["bollinger_bands"]["period"]
        self.bb_stddev = indicators["bollinger_bands"]["stddev"]
        self.atr_period = config.stop_loss["atr_period"]
        mdc_indicator_config = indicators["adaptive_multi_dim_compression"]["indicator_config"]
        self.compression_window = mdc_indicator_config["compression_window"]
        self.volume_ma_window = mdc_indicator_config["volume_window"]
        self.delta_ma_window = indicators["order_flow"]["delta_ma_window"]
        self.trailing_stop_atr_mult = config.stop_loss["trailing_stop_atr_mult"]
        self.cvd_window_size = indicators["cvd"]["window_size"]
        
        # Extract other parameters
        self.cooloff_after_event = config.event_filter["cooloff_after_event"]
        # Fix: Get min_breakout_score from config instead of hardcoding
        self.min_breakout_score = config.risk_management.get("min_breakout_score", 5.0)
        self.risk_per_trade = config.risk_management["risk_per_trade"]
        self.target_r_ratio = config.risk_management["target_r_ratio"]
        self.breakout_quality_weights = config.breakout_quality_scorer["weights"]
        self.session_start = time.fromisoformat(config.session["start"])
        self.session_end = time.fromisoformat(config.session["end"])
        self.initial_capital = config.risk_management["initial_capital"]

        # Set log level
        numeric_level = getattr(logging, config.logging["level"].upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f'Invalid log level: {config.logging["level"]}')

        self.log.info(f"IntradaySniper策略初始化完成")
        self.log.info(f"交易品种: {self.instrument_id}")
        self.log.info(f"Bar类型: {self.bar_type}")
        self.log.info(f"布林带周期: {self.bb_period}")
        self.log.info(f"突破质量权重: {self.breakout_quality_weights}")
        self.log.info(f"会话时间: {self.session_start} - {self.session_end}")
        self.log.info(f"初始资金: {self.initial_capital}")

        # Validate config parameters
        self._validate_config()

        # State variables
        self.instrument: Optional[Instrument] = None
        self.position: Optional[Position] = None
        self.entry_price: Optional[float] = None
        self.stop_loss: Optional[float] = None
        self.take_profit: Optional[float] = None
        self.trail_stop: Optional[float] = None
        self.pending_order_id: Optional[str] = None
        self.atr: Optional[AverageTrueRange] = None

        # Compression cycle management
        self.current_cycle: Optional[CompressionCycle] = None
        self.last_compression_state = False
        self.bar_index = 0  # Track current bar index for cycle management

        # Extract compression indicator config
        mdc_config = indicators["adaptive_multi_dim_compression"]
        mdc_indicator_config = IndicatorConfig(**mdc_config["indicator_config"])
        mdc_threshold_config = ThresholdConfig(**mdc_config["threshold_config"])
         
        self.compression_indicator = AdaptiveMultiDimCompressionIndicator(
            indicator_config=mdc_indicator_config,
            threshold_config=mdc_threshold_config
        )
        self.cvd_tracker = CVD(window_size=self.cvd_window_size)
        self.scorer = BreakoutQualityScorer(
            weights=self.breakout_quality_weights)
        self.position_sizer = PositionSizer(risk_per_trade=self.risk_per_trade,
                                            target_r_ratio=self.target_r_ratio,
                                            leverage=100.0)  # 100x leverage
        self.time_filter = TimeFilter(self.session_start, self.session_end)
        self.event_filter = EventFilter(
            cooloff_period=self.cooloff_after_event)

        # Order tracking
        self.order_fills = {}

        # Historical bar buffer for scorer (if needed)
        self.bar_buffer: Deque[Bar] = deque(maxlen=100)
        # Volume history for scorer
        self.volume_history: Deque[float] = deque(maxlen=20)

    def _validate_config(self):
        """Validate config parameters"""
        if self.min_breakout_score <= 0:
            raise ValueError("min_breakout_score must be positive")
        if not (0 < self.risk_per_trade <= 0.1):
            raise ValueError("risk_per_trade should be between 0 and 0.1")
        if self.target_r_ratio <= 0:
            raise ValueError("target_r_ratio must be positive")
        if self.atr_period <= 0:
            raise ValueError("atr_period must be positive")
        if self.compression_window <= 0:
            raise ValueError("compression_window must be positive")
        if self.volume_ma_window <= 0:
            raise ValueError("volume_ma_window must be positive")
        if self.delta_ma_window <= 0:
            raise ValueError("delta_ma_window must be positive")
        if self.cvd_window_size <= 0:
            raise ValueError("cvd_window_size must be positive")
        if self.cooloff_after_event < 0:
            raise ValueError("cooloff_after_event cannot be negative")
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")

    def on_start(self):
        self.instrument = self.cache.instrument(self.instrument_id)
        if self.instrument is None:
            self.log.error(
                f"Instrument {self.instrument_id} not found in cache.")
            self.stop()
            return

        # Initialize ATR indicator
        self.atr = AverageTrueRange(self.atr_period)

        # Register indicators
        # self.register_indicator_for_bars(self.bar_type, self.bollinger)
        self.register_indicator_for_bars(self.bar_type, self.atr)

        self.subscribe_trade_ticks(self.instrument_id)
        self.subscribe_bars(self.bar_type)

        self.log.info("Intraday Sniper Strategy started.")

    def on_stop(self):
        # Close any open positions
        # Get current open position from cache
        open_positions = self.cache.positions_open()
        current_position = open_positions[0] if open_positions else None
        if current_position:
            self.close_position(current_position)
        
        # Generate compression zone and breakout signal chart
        if len(self.bar_buffer) > 50:  # Ensure there is enough data
            bars_list = list(self.bar_buffer)
            self.plot_compression_and_breakout(bars_list, "compression_breakout_analysis.png")
        
        self.log.info("Intraday Sniper Strategy stopped.")

    def on_reset(self):
        self.position = None
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.trail_stop = None
        self.pending_order_id = None
        # self.bollinger.reset()
        self.compression_indicator.reset_compression_state()
        self.cvd_tracker.reset()
        self.bar_buffer.clear()
        self.volume_history.clear()
        
        # Reset compression cycle state
        self.current_cycle = None
        self.last_compression_state = False
        self.bar_index = 0
        
        self.log.info("Strategy state reset.")

    def on_tick(self, tick: Data):
        if isinstance(tick, TradeTick):
            self.cvd_tracker.on_trade(tick)

    def on_bar(self, bar: Bar):
        # Update bar index
        self.bar_index += 1
        
        # Update compression cycle state
        self._update_compression_cycle(bar)
        
        # Update indicators
        # self.bollinger.handle_bar(bar)
        self.compression_indicator.handle_bar(bar)
        self.bar_buffer.append(bar)
        self.volume_history.append(bar.volume.as_double())

        # Add diagnostic logging
        self.log.debug(f"[STRATEGY] Processing bar {self.bar_index} at {bar.ts_event}")
        self.log.debug(f"[STRATEGY] Bar OHLC: Open={bar.open.as_double():.5f}, High={bar.high.as_double():.5f}, Low={bar.low.as_double():.5f}, Close={bar.close.as_double():.5f}")
        self.log.debug(f"[STRATEGY] Bar Volume: {bar.volume.as_double():.2f}")
        
        # Log compression cycle state
        if self.current_cycle:
            self.log.debug(f"[STRATEGY] Compression Cycle - Start: {self.current_cycle.start_idx}, End: {self.current_cycle.end_idx}, Breakout: {self.current_cycle.breakout_idx}, Entry Triggered: {self.current_cycle.entry_triggered}")
        
        # Check if compression indicator is initialized
        if not self.compression_indicator.initialized:
            self.log.debug(f"[STRATEGY] Compression indicator not initialized. Buffer size: {len(self.compression_indicator._bars)}")
            return

        # Check compression state
        in_compression = self.compression_indicator.is_compression()
        state_summary = self.compression_indicator.get_state_summary()
        self.log.debug(f"[STRATEGY] Compression state: {state_summary}")

        # Check trading session
        if not self.time_filter.is_trading_time(bar.ts_event):
            self.log.debug(f"[STRATEGY] Outside trading hours - Time: {bar.ts_event}")
            if self.time_filter.is_close_time(bar.ts_event):
                # Get current open position from cache
                open_positions = self.cache.positions_open()
                if open_positions:
                    self.log.debug("[STRATEGY] Closing position at end of day")
                    self.close_position(open_positions[0], "EOD_CLOSE")
            return

        # Check event filter
        if self.event_filter.is_cooling_off(bar.ts_event):
            self.log.debug(f"[STRATEGY] In cool-off period - Time: {bar.ts_event}")
            return

        # Get current open position from cache
        open_positions = self.cache.positions_open()
        current_position = open_positions[0] if open_positions else None

        # Check for entry signal
        if not current_position:
            self.log.debug("[STRATEGY] Checking for entry signal - No open position")
            
            # Use time-dependent entry conditions based on compression cycle
            entry_signal, entry_reason = self._check_time_dependent_entry_conditions(bar)
            
            if entry_signal:
                self.log.debug(f"[STRATEGY] {entry_reason}, executing entry")
                # Extract signal name from entry reason for position execution
                signal_name = self._extract_signal_name(entry_reason)
                # Use a fixed score for simplified entry logic
                self._execute_entry(bar, 5.0, signal_name)
            else:
                self.log.debug(f"[STRATEGY] Conditions not met for entry")
            
        # Manage existing position
        elif current_position:
            self.log.debug("[STRATEGY] Managing existing position")
            self._manage_position(bar)

    def _update_compression_cycle(self, bar: Bar):
        """Update compression cycle state based on current compression indicator state"""
        is_compressing = self.compression_indicator.is_compression()
        state_summary = self.compression_indicator.get_state_summary()
        
        # Entering compression: record start
        if is_compressing and not self.last_compression_state:
            self.log.debug(f"[COMPRESSION_CYCLE] Compression zone starting at bar {self.bar_index}")
            # We don't create a cycle yet, just record that compression started
            
        # Exiting compression: create cycle and mark end
        elif not is_compressing and self.last_compression_state:
            self.log.debug(f"[COMPRESSION_CYCLE] Compression zone ending at bar {self.bar_index}")
            
            # Get compression extremes when compression ends
            extremes = state_summary.get('extremes', {})
            compression_high = extremes.get('high')
            compression_low = extremes.get('low')
            
            if compression_high is not None and compression_low is not None:
                try:
                    comp_high = compression_high.as_double()
                    comp_low = compression_low.as_double()
                    
                    # Create new compression cycle when compression ends
                    self.current_cycle = CompressionCycle(
                        start_idx=self.bar_index,  # Start tracking from when compression ends
                        end_idx=self.bar_index,    # Compression ended at this bar
                        high=comp_high,
                        low=comp_low,
                        duration=0,  # Will be updated when we detect breakout timing
                        volume_compression_ratio=self.compression_indicator._last_vol_density or 0.0,
                        atr_compression_ratio=self.compression_indicator._last_entropy or 0.0,
                        breakout_idx=None,
                        breakout_direction=None
                    )
                    self.log.info(f"[COMPRESSION_CYCLE] Created cycle for ended compression: High={comp_high:.2f}, Low={comp_low:.2f}")
                except Exception as e:
                    self.log.error(f"[COMPRESSION_CYCLE] Error creating compression cycle: {e}")
            else:
                # No valid compression data, clear any existing cycle
                self.current_cycle = None
        
        # Update breakout information if we have a cycle
        elif not is_compressing and self.current_cycle:
            # Check if breakout occurred (only if not already recorded)
            if self.current_cycle.breakout_idx is None:
                current_close = bar.close.as_double()
                comp_high = self.current_cycle.high
                comp_low = self.current_cycle.low
                
                # Record breakout if it happened
                if current_close > comp_high:
                    self.current_cycle.set_breakout(self.bar_index, "up")
                    self.log.info(f"[COMPRESSION_CYCLE] Breakout UP detected at bar {self.bar_index}")
                elif current_close < comp_low:
                    self.current_cycle.set_breakout(self.bar_index, "down")
                    self.log.info(f"[COMPRESSION_CYCLE] Breakout DOWN detected at bar {self.bar_index}")
        
        # Update last compression state
        self.last_compression_state = is_compressing

    def _extract_signal_name(self, entry_reason: str) -> str:
        """
        Extract the signal name from the entry reason string
        """
        # Extract the signal name from the reason string
        if "Spring Load" in entry_reason:
            return "Spring Load"
        elif "Silent Accumulation" in entry_reason:
            return "Silent Accumulation"
        elif "Breakout Ignition" in entry_reason:
            return "Breakout Ignition"
        elif "ATR Squeeze + POC Escape" in entry_reason:
            return "ATR Squeeze + POC Escape"
        elif "Volatility Vacuum" in entry_reason:
            return "Volatility Vacuum"
        elif "Momentum Fusion" in entry_reason:
            return "Momentum Fusion"
        elif "Condition 1" in entry_reason:
            return "Volume Compression + Pre-breakout Silence"
        elif "Condition 2" in entry_reason:
            return "ATR Compression + POC Breakout"
        else:
            return "Unknown Signal"

    def _check_time_dependent_entry_conditions(self, bar: Bar) -> tuple[bool, str]:
        """
        Check entry conditions based on time-dependent compression cycle logic
        """
        current_close = bar.close.as_double()
        
        # Must have an active compression cycle (created when compression ends)
        if not self.current_cycle:
            return False, "No compression cycle available"
        
        # Must have valid compression data
        comp_high = self.current_cycle.high
        comp_low = self.current_cycle.low
        
        if comp_high is None or comp_low is None:
            return False, "No valid compression data in cycle"
        
        # Check if we're still in compression (should not be checking entry conditions)
        is_compressing = self.compression_indicator.is_compression()
        if is_compressing:
            return False, "Still in compression zone"
        
        # Check if breakout has occurred
        if self.current_cycle.breakout_idx is None:
            # Check for breakout now
            if current_close > comp_high:
                self.current_cycle.set_breakout(self.bar_index, "up")
            elif current_close < comp_low:
                self.current_cycle.set_breakout(self.bar_index, "down")
            
            # If still no breakout, exit
            if self.current_cycle.breakout_idx is None:
                return False, "No breakout detected in current cycle"
        
        # Check if we're within the valid breakout window
        if not self.current_cycle.is_within_breakout_window(self.bar_index):
            bars_since = self.current_cycle.bars_since_breakout(self.bar_index)
            return False, f"Breakout too old ({bars_since} bars ago)"
        
        # Check if cycle is still active
        if not self.current_cycle.is_active(self.bar_index):
            return False, "Compression cycle expired"
        
        # Get signal weight based on time since breakout
        signal_weight = self.current_cycle.get_signal_weight(self.bar_index)
        self.log.debug(f"[STRATEGY] Signal weight: {signal_weight:.3f}")
        
        # Confirm breakout with simplified scorer
        breakout_confirmed, breakout_details = self.scorer.confirm_breakout(
            bar=bar,
            cvd_tracker=self.cvd_tracker,
            mdc_indicator=self.compression_indicator,
            volume_history=list(self.volume_history),
            recent_bars=list(self.bar_buffer)
        )
        
        if not breakout_confirmed:
            return False, f"Breakout not confirmed: {breakout_details.get('reason', 'unknown')}"
        
        # Check time-dependent conditions in priority order
        bars_since_breakout = self.current_cycle.bars_since_breakout(self.bar_index)
        
        # 1. Breakout Ignition - must be first bar after breakout
        if self.current_cycle.is_first_breakout_bar(self.bar_index):
            if self._check_breakout_ignition(bar, comp_high, comp_low):
                # Check if signal weight is sufficient
                if signal_weight > 0.5:  # Only trigger if weight is significant
                    self.current_cycle.entry_triggered = True
                    return True, "Breakout Ignition signal: First breakout + volume spike + CVD momentum"
                else:
                    self.log.debug(f"[STRATEGY] Breakout Ignition detected but weight too low: {signal_weight:.3f}")
        
        # 2. Silent Accumulation - must be second bar after breakout
        elif self.current_cycle.is_second_breakout_bar(self.bar_index):
            if self._check_silent_accumulation(bar):
                # Check if signal weight is sufficient
                if signal_weight > 0.3:  # Slightly lower threshold for pullback
                    self.current_cycle.entry_triggered = True
                    return True, "Silent Accumulation signal: Pre-breakout silence + price density + order absorption"
                else:
                    self.log.debug(f"[STRATEGY] Silent Accumulation detected but weight too low: {signal_weight:.3f}")
        
        # 3. Other conditions - can trigger within breakout window if not already triggered
        elif self.current_cycle.can_trigger_entry(self.bar_index):
            # ATR Squeeze + POC Escape
            state_summary = self.compression_indicator.get_state_summary()
            poc = state_summary.get('volume_profile_poc')
            if self._check_atr_squeeze_poc_escape(bar, comp_high, comp_low, poc):
                self.current_cycle.entry_triggered = True
                return True, "ATR Squeeze + POC Escape signal: ATR compression + POC breakout + momentum"
            
            # Momentum Fusion
            if self._check_momentum_fusion(bar, comp_high, comp_low):
                self.current_cycle.entry_triggered = True
                return True, "Momentum Fusion signal: Multi-momentum convergence + compression release + CVD confirmation"
            
            # Condition 1: Volume compression + pre-breakout silence + price density + volume spike
            if self._check_condition_1(bar, comp_high, comp_low):
                self.current_cycle.entry_triggered = True
                return True, "Condition 1 met: Volume compression + pre-breakout silence + price density + volume spike"
                
            # Condition 2: ATR compression + POC breakout + momentum
            if self._check_condition_2(bar, comp_high, comp_low, poc):
                self.current_cycle.entry_triggered = True
                return True, "Condition 2 met: ATR compression + POC breakout + momentum"
                
            # Spring Load (warning signal - should have triggered before breakout)
            if self._check_spring_load(bar):
                self.current_cycle.entry_triggered = True
                return True, "Spring Load signal: ATR + Volume compression + structural tightness"
                
            # Volatility Vacuum
            if self._check_volatility_vacuum(bar):
                self.current_cycle.entry_triggered = True
                return True, "Volatility Vacuum signal: Ultra low volatility + high price density"
        
        return False, "Conditions not met in current cycle"

    def _check_condition_1(self, bar: Bar, comp_high: float, comp_low: float) -> bool:
        """
        Check Condition 1: Volume compression + pre-breakout silence + price density + volume spike
        """
        current_close = bar.close.as_double()
        current_volume = bar.volume.as_double()
        
        # 1. Volume compression - check if recent volume is lower than historical average
        if len(self.volume_history) >= 20:
            recent_volume_avg = sum(list(self.volume_history)[-10:]) / 10
            historical_volume_avg = sum(list(self.volume_history)[-20:]) / 20
            volume_compression = recent_volume_avg < historical_volume_avg * 0.8
        else:
            volume_compression = False
            
        # 2. Pre-breakout silence - check if recent ATR is lower than historical average
        if len(self.compression_indicator._recent_atrs) >= 5:
            recent_atr_avg = sum(list(self.compression_indicator._recent_atrs)[-5:]) / 5
            if self.compression_indicator._t_silence_atr_estimator.count > 10:
                silence_threshold = self.compression_indicator._t_silence_atr_estimator.quantile(0.3)
                pre_breakout_silence = recent_atr_avg < silence_threshold
            else:
                pre_breakout_silence = False
        else:
            pre_breakout_silence = False
            
        # 3. Price density - check internal price density score
        price_density_score = self.compression_indicator._last_internal_price_density
        high_price_density = price_density_score > 0.6
        
        # 4. Volume spike - check if current volume is significantly higher than recent average
        if len(self.volume_history) >= 5:
            recent_volume_avg = sum(list(self.volume_history)[-5:]) / 5
            volume_spike = current_volume > recent_volume_avg * 2.0  # 2x spike
        else:
            volume_spike = False
            
        # All conditions must be met
        return volume_compression and pre_breakout_silence and high_price_density and volume_spike

    def _check_condition_2(self, bar: Bar, comp_high: float, comp_low: float, poc) -> bool:
        """
        Check Condition 2: ATR compression + POC breakout + momentum
        """
        current_close = bar.close.as_double()
        
        # 1. ATR compression - check if recent ATR is lower than historical average
        if len(self.compression_indicator._atrs) >= 20:
            recent_atr_avg = sum(list(self.compression_indicator._atrs)[-10:]) / 10
            historical_atr_avg = sum(list(self.compression_indicator._atrs)[-20:]) / 20
            atr_compression = recent_atr_avg < historical_atr_avg * 0.8
        else:
            atr_compression = False
            
        # 2. POC breakout - check if price has broken out past the Point of Control
        if poc is not None:
            poc_val = float(poc) if hasattr(poc, 'as_double') else float(poc)
            poc_breakout_up = current_close > poc_val and current_close > comp_high
            poc_breakout_down = current_close < poc_val and current_close < comp_low
            poc_breakout = poc_breakout_up or poc_breakout_down
        else:
            # Fallback to regular breakout if no POC
            poc_breakout = True
            
        # 3. Momentum - check CVD slope direction
        if self.cvd_tracker.initialized:
            cvd_slope = self.cvd_tracker.get_slope()
            # For upward breakout, we want positive momentum
            # For downward breakout, we want negative momentum
            is_breakout_up = current_close > comp_high
            momentum_confirmed = (cvd_slope > 0 and is_breakout_up) or (cvd_slope < 0 and not is_breakout_up)
        else:
            momentum_confirmed = True  # If CVD not initialized, don't block entry
            
        # All conditions must be met
        return atr_compression and poc_breakout and momentum_confirmed

    def _check_spring_load(self, bar: Bar) -> bool:
        """
        Spring Load signal: ATR + Volume 压缩 + 结构紧致，等待爆发
        """
        # Check compression indicator components
        # Get threshold values from compression indicator
        threshold_config = self.compression_indicator.threshold_config
        
        # Check ATR compression (quantile < 0.2)
        atr_quantile = self.compression_indicator._t_atr_estimator.quantile(0.2) if self.compression_indicator._t_atr_estimator.count > 10 else 1.0
        bw_atr = self.compression_indicator._compute_bandwidth_atr(self.compression_indicator._atrs[-1]) if self.compression_indicator._atrs else 1.0
        strong_atr_compression = bw_atr < atr_quantile
        
        # Check Volume compression (quantile < 0.2)
        vol_quantile = self.compression_indicator._t_vol_estimator.quantile(0.2) if self.compression_indicator._t_vol_estimator.count > 10 else 1.0
        bw_vol = self.compression_indicator._compute_bandwidth_volume(self.compression_indicator._vol_ma_indicator.value) if self.compression_indicator._vol_ma_indicator.value else 1.0
        strong_volume_compression = bw_vol < vol_quantile
        
        # Check structural compression (body ratio < 0.3 and ATR ratio < 0.4)
        structural_body_check = self.compression_indicator._last_body_ratio < 0.3
        bw_atr_ratio = self.compression_indicator._t_atr_estimator.quantile(0.5) if self.compression_indicator._t_atr_estimator.count > 10 else 1.0
        structural_atr_check = bw_atr_ratio < 0.4
        
        # Check high compression confidence (> 0.7)
        compression_confidence = self.compression_indicator._last_score
        high_confidence = compression_confidence > 0.7
        
        # Check duration bonus and price density加分项
        duration_bonus = self.compression_indicator._calculate_duration_modifier()
        good_duration = duration_bonus > 0.1
        price_density = self.compression_indicator._last_internal_price_density
        high_density = price_density > 0.8
        
        # All core conditions must be met
        return (strong_atr_compression and strong_volume_compression and 
                structural_body_check and structural_atr_check and high_confidence)

    def _check_silent_accumulation(self, bar: Bar) -> bool:
        """
        Silent Accumulation signal: 突破前的低波动静默 + 内部价格密集 + 订单吸收迹象
        """
        # Check pre-breakout silence (quantile < 0.15)
        if len(self.compression_indicator._recent_atrs) >= 5:
            recent_atr_avg = sum(list(self.compression_indicator._recent_atrs)[-5:]) / 5
            if self.compression_indicator._t_silence_atr_estimator.count > 10:
                silence_threshold = self.compression_indicator._t_silence_atr_estimator.quantile(0.15)
                pre_breakout_silence = recent_atr_avg < silence_threshold
            else:
                pre_breakout_silence = False
        else:
            pre_breakout_silence = False
            
        # Check internal price density (> 0.75)
        price_density = self.compression_indicator._last_internal_price_density
        high_density = price_density > 0.75
        
        # Check order absorption (simplified check - look for small body candles with high volume)
        body_size = abs(bar.close.as_double() - bar.open.as_double())
        avg_body = np.mean([abs(b.close.as_double() - b.open.as_double()) for b in list(self.bar_buffer)[-10:]]) if len(self.bar_buffer) >= 10 else body_size
        order_absorption = body_size < avg_body * 0.5 and bar.volume.as_double() > np.mean(list(self.volume_history)[-10:]) if len(self.volume_history) >= 10 else False
        
        # Check direction ordered (entropy < 0.3)
        direction_ordered = self.compression_indicator._last_entropy < 0.3
        
        # All core conditions must be met
        return pre_breakout_silence and high_density and order_absorption and direction_ordered

    def _check_breakout_ignition(self, bar: Bar, comp_high: float, comp_low: float) -> bool:
        """
        Breakout Ignition signal: 压缩后首次爆发 + 成交量 spike + CVD 动能确认
        """
        current_close = bar.close.as_double()
        current_volume = bar.volume.as_double()
        
        # Check if this is the first breakout bar
        is_first_breakout = False
        if len(self.bar_buffer) >= 2:
            prev_bar = list(self.bar_buffer)[-2]
            prev_close = prev_bar.close.as_double()
            # Check if previous bar was in compression zone
            if comp_low <= prev_close <= comp_high:
                is_first_breakout = True
        
        # Check price breakout strength
        price_breakout_strength = abs(current_close - (comp_high if current_close > comp_high else comp_low))
        avg_range = np.mean([b.high.as_double() - b.low.as_double() for b in list(self.bar_buffer)[-10:]]) if len(self.bar_buffer) >= 10 else 1.0
        strong_price_breakout = price_breakout_strength > avg_range * 2.0
        
        # Check volume spike
        if len(self.volume_history) >= 5:
            recent_volume_avg = sum(list(self.volume_history)[-5:]) / 5
            volume_spike = current_volume > recent_volume_avg * 2.0  # 2x spike
        else:
            volume_spike = False
            
        # Check CVD momentum alignment
        if self.cvd_tracker.initialized:
            cvd_slope = self.cvd_tracker.get_slope()
            is_breakout_up = current_close > comp_high
            momentum_confirmed = (cvd_slope > 0 and is_breakout_up) or (cvd_slope < 0 and not is_breakout_up)
        else:
            momentum_confirmed = True
            
        # Check volume and CVD alignment
        volume_cvd_aligned = volume_spike and momentum_confirmed
        
        # Check if it's a dominant volume bar
        if len(self.volume_history) >= 10:
            volume_percentile = np.percentile(list(self.volume_history)[-10:], 80)
            volume_dominant = current_volume > volume_percentile
        else:
            volume_dominant = False
            
        # All core conditions must be met
        return is_first_breakout and strong_price_breakout and volume_spike and momentum_confirmed

    def _check_atr_squeeze_poc_escape(self, bar: Bar, comp_high: float, comp_low: float, poc) -> bool:
        """
        ATR Squeeze + POC Escape signal: 波动率压缩 + 突破关键价值节点（POC）+ 动量跟进
        """
        current_close = bar.close.as_double()
        
        # Check ATR compression (using compression indicator weight)
        atr_compression_active = True  # Simplified - assume ATR compression is active if we're here
        
        # Check POC breakout
        if poc is not None:
            poc_val = float(poc) if hasattr(poc, 'as_double') else float(poc)
            poc_breakout_up = current_close > poc_val and current_close > comp_high
            poc_breakout_down = current_close < poc_val and current_close < comp_low
            poc_breakout = poc_breakout_up or poc_breakout_down
        else:
            poc_breakout = False
            
        # Check momentum convergence (price threshold triggered)
        momentum_convergence = self.compression_indicator._check_momentum_convergence()
        
        # Check price breakout strength
        price_breakout_strength = abs(current_close - (comp_high if current_close > comp_high else comp_low))
        avg_range = np.mean([b.high.as_double() - b.low.as_double() for b in list(self.bar_buffer)[-10:]]) if len(self.bar_buffer) >= 10 else 1.0
        decent_price_breakout = price_breakout_strength > avg_range * 1.5
        
        # Check proximity to POC
        if poc is not None:
            poc_val = float(poc)
            range_size = comp_high - comp_low
            distance_to_poc = abs(current_close - poc_val)
            pre_break_proximity = distance_to_poc < range_size * 0.2
        else:
            pre_break_proximity = False
            
        # Check strong CVD momentum
        if self.cvd_tracker.initialized:
            cvd_slope = self.cvd_tracker.get_slope()
            strong_cvd_momentum = abs(cvd_slope) > np.mean([abs(self.cvd_tracker.get_slope()) for _ in range(5)]) if hasattr(self.cvd_tracker, 'get_slope') else 0
        else:
            strong_cvd_momentum = False
            
        # All core conditions must be met
        return atr_compression_active and poc_breakout and momentum_convergence and decent_price_breakout

    def _check_volatility_vacuum(self, bar: Bar) -> bool:
        """
        Volatility Vacuum signal: 极低波动 + 极低成交量 + 高价格密度 → 预示剧烈波动即将来临
        """
        # Check volatility density (high density > 0.8)
        volatility_dense = self.compression_indicator._last_vol_density > 0.8
        
        # Check volume compression (quantile < 0.1)
        if len(self.volume_history) >= 20:
            recent_volume_avg = sum(list(self.volume_history)[-10:]) / 10
            historical_volume_avg = sum(list(self.volume_history)[-20:]) / 20
            extreme_volume_compression = recent_volume_avg < historical_volume_avg * 0.5  # Even more extreme
        else:
            extreme_volume_compression = False
            
        # Check pre-breakout silence (quantile < 0.1)
        if len(self.compression_indicator._recent_atrs) >= 5:
            recent_atr_avg = sum(list(self.compression_indicator._recent_atrs)[-5:]) / 5
            if self.compression_indicator._t_silence_atr_estimator.count > 10:
                silence_threshold = self.compression_indicator._t_silence_atr_estimator.quantile(0.1)
                extreme_silence = recent_atr_avg < silence_threshold
            else:
                extreme_silence = False
        else:
            extreme_silence = False
            
        # Check internal price density (very high > 0.9)
        price_density = self.compression_indicator._last_internal_price_density
        very_high_density = price_density > 0.9
        
        # Check direction ordered (entropy < 0.2)
        direction_ordered = self.compression_indicator._last_entropy < 0.2
        
        # All core conditions must be met
        return volatility_dense and extreme_volume_compression and extreme_silence and very_high_density and direction_ordered

    def _check_momentum_fusion(self, bar: Bar, comp_high: float, comp_low: float) -> bool:
        """
        Momentum Fusion signal: 多种动量信号汇聚 + 压缩释放 + CVD确认
        """
        current_close = bar.close.as_double()
        
        # Check momentum convergence (price threshold triggered)
        momentum_convergence = self.compression_indicator._check_momentum_convergence()
        
        # Check strong CVD momentum
        if self.cvd_tracker.initialized:
            cvd_slope = self.cvd_tracker.get_slope()
            strong_cvd_momentum = abs(cvd_slope) > 0.5  # Strong momentum threshold
        else:
            strong_cvd_momentum = False
            
        # Check volume spike
        current_volume = bar.volume.as_double()
        if len(self.volume_history) >= 5:
            recent_volume_avg = sum(list(self.volume_history)[-5:]) / 5
            volume_spike = current_volume > recent_volume_avg * 1.5  # 1.5x spike for fusion
        else:
            volume_spike = False
            
        # Check expansion confidence (> 0.6)
        expansion_confidence = 1.0 - self.compression_indicator._last_score  # Inverse of compression score
        good_expansion = expansion_confidence > 0.6
        
        # Check POC direction alignment with trend
        # (simplified - assume breakout direction is trend direction)
        is_breakout_up = current_close > comp_high
        poc_direction_aligned = True  # Simplified for now
        
        # Check duration bonus (long compression time)
        duration_bonus = self.compression_indicator._calculate_duration_modifier()
        long_compression = duration_bonus < -0.1  # Negative modifier means long duration
        
        # All core conditions must be met
        return momentum_convergence and strong_cvd_momentum and volume_spike and good_expansion

    def plot_compression_and_breakout(self, bars: list, save_path: str = "compression_breakout_chart.png"):
        """
        Plot candlestick chart showing compression zones and breakout signals
        
        Parameters:
        bars: List of candlestick data
        save_path: Path to save the image
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            import pandas as pd
            import numpy as np
            from matplotlib.dates import DateFormatter
            import matplotlib.dates as mdates
            
            # Prepare data
            timestamps = [pd.Timestamp(bar.ts_event).to_pydatetime() for bar in bars]
            opens = [float(bar.open.as_double()) for bar in bars]
            highs = [float(bar.high.as_double()) for bar in bars]
            lows = [float(bar.low.as_double()) for bar in bars]
            closes = [float(bar.close.as_double()) for bar in bars]
            
            # Create chart
            fig, ax = plt.subplots(figsize=(15, 8))
            
            # Convert timestamps to matplotlib date format
            mpl_dates = mdates.date2num(timestamps)
            
            # Plot candlesticks
            for i in range(len(bars)):
                color = 'green' if closes[i] >= opens[i] else 'red'
                # Plot wicks
                ax.plot([mpl_dates[i], mpl_dates[i]], [lows[i], highs[i]], color='black', linewidth=0.5)
                # Plot body
                ax.add_patch(patches.Rectangle(
                    (float(mpl_dates[i]), opens[i]), 
                    1/24/60,  # 1 minute in matplotlib date format
                    closes[i] - opens[i],
                    facecolor=color,
                    edgecolor='black',
                    linewidth=0.3
                ))
            
            # Find compression zones and breakout signals
            compression_zones = []
            breakout_signals = []
            
            # To avoid modifying original indicator state, create a temporary indicator instance
            temp_indicator = AdaptiveMultiDimCompressionIndicator(
                indicator_config=self.compression_indicator.indicator_config,
                threshold_config=self.compression_indicator.threshold_config
            )
            
            for i, bar in enumerate(bars):
                # Get compression state
                temp_indicator.handle_bar(bar)
                in_compression = temp_indicator.is_compression()
                state_summary = temp_indicator.get_state_summary()
                extremes = state_summary.get('extremes', {})
                compression_high = extremes.get('high')
                compression_low = extremes.get('low')
                
                # Record compression zone
                if in_compression and compression_high is not None and compression_low is not None:
                    compression_zones.append({
                        'index': i,
                        'high': float(compression_high.as_double()),
                        'low': float(compression_low.as_double()),
                        'timestamp': float(mpl_dates[i])
                    })
                
                # Check for breakout signal (only in compression state)
                if in_compression and compression_high is not None and compression_low is not None:
                    current_close = float(bar.close.as_double())
                    compression_high_val = float(compression_high.as_double())
                    compression_low_val = float(compression_low.as_double())
                    
                    is_long_breakout = current_close > compression_high_val
                    is_short_breakout = current_close < compression_low_val
                    
                    if is_long_breakout or is_short_breakout:
                        breakout_signals.append({
                            'index': i,
                            'timestamp': float(mpl_dates[i]),
                            'price': current_close,
                            'type': 'long' if is_long_breakout else 'short'
                        })
            
            # Plot compression zones
            if compression_zones:
                # Merge consecutive compression zones
                merged_zones = []
                if compression_zones:
                    current_zone = compression_zones[0].copy()
                    
                    for zone in compression_zones[1:]:
                        # If it's a consecutive compression zone, merge
                        if zone['index'] == current_zone['index'] + 1:
                            current_zone['high'] = max(current_zone['high'], zone['high'])
                            current_zone['low'] = min(current_zone['low'], zone['low'])
                            current_zone['index'] = zone['index']
                            # Update timestamp to the latest
                            current_zone['start_timestamp'] = current_zone.get('start_timestamp', current_zone['timestamp'])
                            current_zone['end_timestamp'] = zone['timestamp']
                        else:
                            # Save current zone, start a new zone
                            # Add start and end timestamps
                            if 'start_timestamp' not in current_zone:
                                current_zone['start_timestamp'] = current_zone['timestamp']
                                current_zone['end_timestamp'] = current_zone['timestamp']
                            merged_zones.append(current_zone)
                            current_zone = zone.copy()
                    
                    # Add the last zone
                    if 'start_timestamp' not in current_zone:
                        current_zone['start_timestamp'] = current_zone['timestamp']
                        current_zone['end_timestamp'] = current_zone['timestamp']
                    merged_zones.append(current_zone)
                
                # Plot compression zones (limited to x-axis range)
                for zone in merged_zones:
                    # Use Rectangle instead of axhspan to limit x-axis range
                    start_time = zone.get('start_timestamp', zone['timestamp'])
                    end_time = zone.get('end_timestamp', zone['timestamp'])
                    # Slightly extend time range to ensure full coverage of candlesticks
                    time_width = end_time - start_time
                    if time_width == 0:
                        # If it's a single candlestick, extend time range a bit
                        time_extension = 1/24/60/2  # 30 seconds
                        start_time -= time_extension
                        end_time += time_extension
                    
                    rect = patches.Rectangle(
                        (start_time, zone['low']),
                        end_time - start_time,
                        zone['high'] - zone['low'],
                        linewidth=1,
                        edgecolor='yellow',
                        facecolor='yellow',
                        alpha=0.2,
                        label='Compression Zone' if merged_zones.index(zone) == 0 else ""
                    )
                    ax.add_patch(rect)
            
            # Plot breakout signals
            for signal in breakout_signals:
                marker = '^' if signal['type'] == 'long' else 'v'
                color = 'blue' if signal['type'] == 'long' else 'orange'
                ax.scatter(signal['timestamp'], signal['price'], 
                          marker=marker, color=color, s=100, 
                          label=f'{signal["type"].capitalize()} Breakout' if breakout_signals.index(signal) < 2 else "",
                          zorder=5)
            
            # Set chart properties
            ax.set_title('Intraday Sniper - Compression Zones and Breakout Signals')
            ax.set_xlabel('Time')
            ax.set_ylabel('Price')
            ax.legend()
            
            # Format x-axis time display
            ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            fig.autofmt_xdate()
            
            # Save chart
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.log.info(f"Compression and breakout chart saved to {save_path}")
            
        except ImportError as e:
            self.log.warning(f"Matplotlib not available for plotting: {e}")
        except Exception as e:
            self.log.error(f"Error plotting compression and breakout: {e}")

    def _execute_entry(self, bar: Bar, score: float, signal_name: str = "Unknown"):
        self.log.debug(f"[STRATEGY] _execute_entry called with score: {score:.2f}, signal: {signal_name}")
        
        if self.pending_order_id:
            self.log.debug("[STRATEGY] Pending order exists, not executing entry")
            return

        current_close = bar.close.as_double()
        # 获取压缩区极值和POC用于计算方向和止损
        state_summary = self.compression_indicator.get_state_summary()
        extremes = state_summary.get('extremes', {})
        compression_high = extremes.get('high')
        compression_low = extremes.get('low')
        poc = state_summary.get('volume_profile_poc')
        
        self.log.debug(f"[STRATEGY] Compression extremes - High: {compression_high}, Low: {compression_low}, POC: {poc}")
        
        if compression_high is None or compression_low is None or poc is None:
            self.log.warning(
                "Compression extremes or POC not valid, cannot execute entry.")
            self.log.debug("[STRATEGY] Compression extremes or POC not valid, cannot execute entry")
            return

        compression_high_val = compression_high.as_double()
        compression_low_val = compression_low.as_double()
        poc_val = poc
        
        # 使用POC和当前价格比较来确定交易方向
        direction = 1 if current_close > poc_val else -1
        side = OrderSide.BUY if direction == 1 else OrderSide.SELL
        
        self.log.debug(f"[STRATEGY] Executing entry - Direction: {direction}, Side: {side}")

        # 使用压缩区边界作为初始止损参考
        initial_stop = compression_low_val if direction == 1 else compression_high_val
        # 确保止损价与方向一致
        if direction == 1:  # Long
            initial_stop = max(
                initial_stop,
                bar.low.as_double()) - (self.instrument.price_increment.as_double() if self.instrument else 0)
        else:  # Short
            initial_stop = min(
                initial_stop,
                bar.high.as_double()) + (self.instrument.price_increment.as_double() if self.instrument else 0)

        # Get ATR value if available
        atr_value = None
        if self.atr and self.atr.initialized:
            atr_value = self.atr.value
            self.log.debug(f"[STRATEGY] ATR value: {atr_value}")

        if atr_value is None or atr_value <= 0:
            self.log.warning("[STRATEGY] ATR not initialized or available, cannot calculate position size")
            return

        # Calculate position size using ATR and score
        size = 0.0
        try:
            size = self.position_sizer.calculate(
                equity=10000.0,
                entry_price=current_close,
                atr=atr_value,
                score=score,
                atr_multiplier=self.trailing_stop_atr_mult)  # Use the ATR multiplier from config
            self.log.debug(f"[STRATEGY] Position size calculation - Equity: 10000.0, Entry price: {current_close}, ATR: {atr_value}, Score: {score}, ATR Multiplier: {self.trailing_stop_atr_mult}, Size: {size}")
        except Exception as e:
            self.log.error(f"[STRATEGY] Error calculating position size: {e}")
            return

        if size <= 0:
            self.log.warning("Calculated position size is zero or negative.")
            self.log.debug("[STRATEGY] Calculated position size is zero or negative")
            return

        order = self.order_factory.market(instrument_id=self.instrument_id,
                                          order_side=side,
                                          quantity=Quantity.from_str(f"{size:.3f}"))

        self.submit_order(order)
        self.pending_order_id = order.client_order_id.value
        self.log.info(
            f"Submitted {side} order for {size:.3f} units at {current_close}, score: {score:.2f}, signal: {signal_name}"
        )
        self.log.debug(f"[STRATEGY] Submitted {side} order for {size:.3f} units at {current_close}, score: {score:.2f}, signal: {signal_name}")

    def _calculate_initial_stop(self, bar: Bar, direction: int) -> float:
        """
        This method is used in on_order_filled to initialize stop loss.
        Adjusted to use MDC compression zone, but kept here for compatibility with old logic, actually calculated in _execute_entry.
        """
        # In fact, initial stop has already been calculated based on MDC in _execute_entry, this can be deprecated or kept for consistency.
        # To keep compatibility with old code, still return calculation based on bar.
        if direction == 1:  # Long
            return float(bar.low - (self.instrument.price_increment if self.instrument else 0))
        else:  # Short
            return float(bar.high + (self.instrument.price_increment if self.instrument else 0))

    def _manage_position(self, bar: Bar):
        # Get current open position from cache
        open_positions = self.cache.positions_open()
        current_position = open_positions[0] if open_positions else None
        
        if not current_position:
            return

        current_price = bar.close.as_double()

        # Stop loss
        if self.stop_loss:
            if (current_position.is_long and current_price <= self.stop_loss) or \
               (current_position.is_short and current_price >= self.stop_loss):
                self.close_position(current_position)
                return

        # Take profit
        if self.take_profit:
            if (current_position.is_long and current_price >= self.take_profit) or \
               (current_position.is_short and current_price <= self.take_profit):
                self.close_position(current_position)
                return

        # Trailing stop logic can be enhanced later
        # For now, we rely on fixed SL/TP

    def on_order_accepted(self, event: OrderAccepted):
        if event.client_order_id.value == self.pending_order_id:
            self.log.info(f"Order {event.client_order_id} accepted.")

    def on_order_filled(self, event: OrderFilled):
        self.log.debug(f"[STRATEGY] Order filled - Client order ID: {event.client_order_id.value}, Pending order ID: {self.pending_order_id}")
        if event.client_order_id.value == self.pending_order_id:
            self.pending_order_id = None
            self.log.info(f"Entry order filled at {event.last_px}")
            self.log.debug(f"[STRATEGY] Entry order filled at {event.last_px}")

            # Get current open position from cache
            open_positions = self.cache.positions_open()
            pos = open_positions[0] if open_positions else None
            if pos:
                self.position = pos
                self.entry_price = float(event.last_px)

                direction = 1 if event.order_side == OrderSide.BUY else -1
                # In on_order_filled, use _calculate_initial_stop method to calculate stop loss
                initial_stop = self._calculate_initial_stop(
                    Bar(bar_type=self.bar_type,
                        open=Price.from_str(str(self.entry_price)),
                        high=Price.from_str(str(self.entry_price)),
                        low=Price.from_str(str(self.entry_price)),
                        close=Price.from_str(str(self.entry_price)),
                        volume=Quantity.from_int(1),
                        ts_event=event.ts_event,
                        ts_init=event.ts_init), direction)
                self.stop_loss = initial_stop
                self.take_profit = self.entry_price + (
                    self.entry_price - initial_stop) * self.target_r_ratio * (
                        1 if direction == 1 else -1)

                self.log.debug(f"[STRATEGY] Position opened: {pos.side} {pos.quantity} at {self.entry_price}, SL: {self.stop_loss:.5f}, TP: {self.take_profit:.5f}")
                self.log.info(
                    f"Position opened: {pos.side} {pos.quantity} at {self.entry_price}, "
                    f"SL: {self.stop_loss:.5f}, TP: {self.take_profit:.5f}")
        else:
            # Get current open position from cache
            open_positions = self.cache.positions_open()
            pos = open_positions[0] if open_positions else None
            if pos:
                self.position = pos
                if pos.is_closed:
                    self.entry_price = None
                    self.stop_loss = None
                    self.take_profit = None
                    self.trail_stop = None
                    self.log.debug("[STRATEGY] Position closed")
                    self.log.info("Position closed.")

    def on_position_changed(self, position: Position):
        self.position = position
        # Note: PositionChanged event doesn't have is_closed or is_open attributes
        # We need to check if the position is closed by looking at the quantity
        if hasattr(position, 'quantity') and position.quantity == 0:
            self.entry_price = None
            self.stop_loss = None
            self.take_profit = None
            self.trail_stop = None
            self.log.debug("[STRATEGY] Position closed")
            self.log.info("Position closed.")

    def on_data(self, data: Data):
        pass

    def get_compression_and_breakout_data(self, bars: list):
        """
        Collect compression zone and breakout signal data for plotting
        
        Parameters:
        bars: List of candlestick data
        
        Returns:
        dict: Data containing compression zones and breakout signals
        """
        # Find compression zones and breakout signals
        compression_zones = []
        breakout_signals = []
        
        # To avoid modifying original indicator state, create a temporary indicator instance
        temp_indicator = AdaptiveMultiDimCompressionIndicator(
            indicator_config=self.compression_indicator.indicator_config,
            threshold_config=self.compression_indicator.threshold_config
        )
        
        for i, bar in enumerate(bars):
            # Get compression state
            temp_indicator.handle_bar(bar)
            in_compression = temp_indicator.is_compression()
            state_summary = temp_indicator.get_state_summary()
            extremes = state_summary.get('extremes', {})
            compression_high = extremes.get('high')
            compression_low = extremes.get('low')
            
            # Record compression zone
            if in_compression and compression_high is not None and compression_low is not None:
                compression_zones.append({
                    'index': i,
                    'high': float(compression_high.as_double()),
                    'low': float(compression_low.as_double()),
                    'timestamp': pd.Timestamp(bar.ts_event).to_pydatetime()
                })
            
            # Check for breakout signal (only in compression state)
            if in_compression and compression_high is not None and compression_low is not None:
                current_close = float(bar.close.as_double())
                compression_high_val = float(compression_high.as_double())
                compression_low_val = float(compression_low.as_double())
                
                is_long_breakout = current_close > compression_high_val
                is_short_breakout = current_close < compression_low_val
                
                if is_long_breakout or is_short_breakout:
                    breakout_signals.append({
                        'index': i,
                        'timestamp': pd.Timestamp(bar.ts_event).to_pydatetime(),
                        'price': current_close,
                        'type': 'long' if is_long_breakout else 'short'
                    })
        
        return {
            'compression_zones': compression_zones,
            'breakout_signals': breakout_signals
        }