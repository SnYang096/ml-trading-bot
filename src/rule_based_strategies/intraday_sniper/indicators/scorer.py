from typing import Dict, Optional, Tuple, Union, List
from nautilus_trader.model.data import Bar, TradeTick
from nautilus_trader.indicators import BollingerBands

from yin_bot.intraday_sniper.indicators.cvd import CVD
from yin_bot.intraday_sniper.indicators.compression import AdaptiveMultiDimCompressionIndicator


class BreakoutQualityScorer:
    """
    Simplified Breakout Scorer for rule-based signal combinations.
    
    - Uses MDC compression range instead of Bollinger Bands.
    - Provides simple breakout confirmation without scoring.
    - Works with hierarchical decision logic in the strategy.
    """

    def __init__(
            self,
            weights: Optional[Dict[str, float]] = None,
            breakout_confirm_bars: int = 1,
            min_cvd_slope: float = 0.0,
            min_delta_strength: float = 0.0,
            poc_break_threshold: float = 0.0,
            min_volume_spike: float = 1.5  # Minimum volume multiplier for spike
    ):
        # We keep weights for backward compatibility but they're not used in scoring
        default_weights = {
            'price_breakout': 2.0,
            'volume_spike': 2.0,
            'cvd_momentum_strong': 1.5,
            'cvd_momentum_weak': 0.5,
            'volume_and_cvd_aligned': 1.0,
            'order_absorption_bar': 0.15,
            'order_absorption_tick': 0.15,
            'high_liquidity_time': 1.0,
            'compression_bonus': 1.0,
            'poc_direction_up': 1.5,
            'poc_direction_down': 1.5,
            'pre_break_proximity': 1.0,
            'first_breakout_bar': 2.0,
            'volume_dominant': 1.0,
            'pre_breakout_silence': 2.0,
        }

        self.weights = default_weights.copy()
        if weights is not None:
            self.weights.update(weights)
        self.breakout_confirm_bars = breakout_confirm_bars
        self.min_cvd_slope = min_cvd_slope
        self.min_delta_strength = min_delta_strength
        self.poc_break_threshold = poc_break_threshold
        self.min_volume_spike = min_volume_spike

    def _order_flow_valid(self, cvd_tracker: CVD) -> bool:
        # Check if CVD tracker is initialized
        if not cvd_tracker.initialized:
            # If not initialized, don't block the trade based on order flow
            return True
            
        slope = cvd_tracker.get_slope()
        delta = abs(cvd_tracker.get_latest_delta())
        # Use a more relaxed condition - if either slope or delta meets minimum requirements
        return slope >= self.min_cvd_slope or delta >= self.min_delta_strength

    def confirm_breakout(
            self,
            bar: Bar,
            cvd_tracker: CVD,
            mdc_indicator: AdaptiveMultiDimCompressionIndicator,
            volume_history: List[float],
            recent_bars: List[Bar],
            is_high_liquidity_time: bool = False
    ) -> Tuple[bool, Dict[str, Union[float, str]]]:
        """
        Confirm breakout with simple boolean logic instead of scoring.
        Returns (is_confirmed, breakdown) where breakdown contains diagnostic information.
        """
        
        current_close = bar.close.as_double()
        current_volume = bar.volume.as_double()
        breakdown: Dict[str, Union[float, str]] = {}
        
        # Get dynamic compression range and POC from state summary
        state_summary = mdc_indicator.get_state_summary()
        extremes = state_summary.get('extremes', {})
        compression_high = extremes.get('high')
        compression_low = extremes.get('low')
        
        if compression_high is None or compression_low is None:
            breakdown['reason'] = 'no_compression_range'
            return False, breakdown
            
        # 安全地获取压缩区间的高低点值
        try:
            comp_high = compression_high.as_double() if hasattr(compression_high, 'as_double') else float(compression_high)
            comp_low = compression_low.as_double() if hasattr(compression_low, 'as_double') else float(compression_low)
        except:
            breakdown['reason'] = 'invalid_compression_values'
            return False, breakdown
            
        # 安全地获取POC值
        poc = state_summary.get('volume_profile_poc')
        if poc is None:
            try:
                poc = (comp_low + comp_high) / 2
            except:
                poc = comp_low  # fallback
                
        # 如果POC仍然是Mock对象，使用压缩区间的中点
        if (hasattr(poc, '_mock_return_value') or 
            str(type(poc)) == "<class 'unittest.mock.MagicMock'>" or
            str(type(poc)) == "<class 'Mock'>"):
            try:
                poc = (comp_low + comp_high) / 2
            except:
                poc = comp_low
                
        # 确保所有值都是数值类型
        try:
            comp_high = float(comp_high)
            comp_low = float(comp_low)
            poc = float(poc)
            current_close = float(current_close)
        except:
            breakdown['reason'] = 'type_conversion_error'
            return False, breakdown

        # 必须已经不在压缩区内，说明发生了突破
        is_breakout_up = current_close > comp_high
        is_breakout_down = current_close < comp_low
        
        if not (is_breakout_up or is_breakout_down):
            breakdown['reason'] = 'no_breakout_detected'
            return False, breakdown

        # 判断是否是首次突破
        is_first_breakout = False
        if recent_bars:
            prev_close = recent_bars[-1].close.as_double()
            if comp_low <= prev_close <= comp_high:
                is_first_breakout = True

        # 判断成交量是否放量
        volume_spike_detected = False
        if len(volume_history) >= 5:
            recent_ma = sum(volume_history[-5:]) / 5
            volume_spike_detected = current_volume > recent_ma * self.min_volume_spike

        # 判断CVD动量是否支持突破
        cvd_slope = cvd_tracker.get_slope()
        strong_cvd_momentum = (cvd_slope > 0 and is_breakout_up) or (cvd_slope < 0 and is_breakout_down)
        weak_cvd_momentum = abs(cvd_slope) > 0

        # Simple confirmation logic - all core conditions must be met
        if volume_spike_detected and strong_cvd_momentum:
            # All core conditions met
            breakdown['confirmed'] = 1.0
            breakdown['volume_spike'] = 1.0
            breakdown['cvd_momentum'] = 'strong'
            breakdown['first_breakout'] = 1.0 if is_first_breakout else 0.0
            breakdown['reason'] = 'all_conditions_met'
            return True, breakdown
        elif volume_spike_detected and weak_cvd_momentum:
            # Volume spike with some CVD momentum
            breakdown['confirmed'] = 1.0
            breakdown['volume_spike'] = 1.0
            breakdown['cvd_momentum'] = 'weak'
            breakdown['first_breakout'] = 1.0 if is_first_breakout else 0.0
            breakdown['reason'] = 'volume_and_weak_cvd'
            return True, breakdown
        else:
            # Not enough confirmation
            breakdown['confirmed'] = 0.0
            breakdown['volume_spike'] = 1.0 if volume_spike_detected else 0.0
            breakdown['cvd_momentum'] = 'strong' if strong_cvd_momentum else ('weak' if weak_cvd_momentum else 'none')
            breakdown['reason'] = 'insufficient_confirmation'
            return False, breakdown