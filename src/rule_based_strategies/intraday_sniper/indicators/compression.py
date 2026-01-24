from logging import Logger


from collections import deque, defaultdict
from typing import Dict, Optional, List, Any
import math
import numpy as np
import logging
from decimal import Decimal

from nautilus_trader.indicators import SimpleMovingAverage, AverageTrueRange, ExponentialMovingAverage
from nautilus_trader.model.data import Bar
from yin_bot.common.online_quantile_estimator import OnlineQuantileEstimator
from yin_bot.common.logger_config import setup_logger
from yin_bot.common.tdigest_volatility import TDigestVolatilityBuffer

log: Logger = setup_logger('indicator.compress', level=logging.DEBUG)


class IndicatorConfig:
    """
    指标使用的配置参数
    """

    def __init__(
            self,
            compression_window: int = 200,
            volatility_window: int = 20,
            volume_window: int = 20,
            entropy_window: int = 20,
            compression_tdigest: float = 100.0,
            minimum_tdigest_warmup: int = 50,
            duration_bonus_window: int = 200,  # 持续时间计算窗口
            pre_breakout_silence_window: int = 10,
            internal_price_density_window: int = 20):
        self.entropy_window = entropy_window  # 压缩检测窗口长度
        self.compression_tdigest = compression_tdigest  # 压缩检测窗口长度
        self.compression_window = compression_window  # 压缩检测窗口长度
        self.volatility_window = volatility_window  # 波动率计算窗口长度
        self.volume_window = volume_window  # 成交量移动平均窗口长度
        self.minimum_tdigest_warmup = minimum_tdigest_warmup  # tdigest预热所需的最小数据量
        self.duration_bonus_window = duration_bonus_window  # 计算持续时间加分的窗口长度
        self.pre_breakout_silence_window = pre_breakout_silence_window  # 检测突破前静默的窗口长度
        self.internal_price_density_window = internal_price_density_window  # 价格密度计算窗口


class ThresholdConfig:
    """
    规则类指标判断的阈值配置
    """

    def __init__(
            self,
            compression_atr_threshold_quantile: float = 0.3,
            compression_volume_threshold_quantile: float = 0.4,
            structural_compression_body_ratio: float = 0.5,
            structural_compression_atr_ratio: float = 0.8,
            momentum_convergence_price_threshold: float = 1e-5,
            direction_ordered_entropy_threshold: float = 0.7,
            volatility_dense_threshold: float = 0.3,
            compression_confidence_threshold: float = 0.6,  # 压缩信心阈值
            expansion_confidence_threshold: float = 0.3,  # 扩张信心阈值
            volatility_buffer_base_threshold: float = 0.1,
            pre_breakout_silence_atr_threshold_quantile: float = 0.2,
            internal_price_density_threshold: float = 0.6):

        # 压缩判断阈值配置 - 用于各维度的判断标准
        self.compression_atr_threshold_quantile = compression_atr_threshold_quantile  # ATR分位数阈值
        self.compression_volume_threshold_quantile = compression_volume_threshold_quantile  # 成交量分位数阈值
        self.structural_compression_body_ratio = structural_compression_body_ratio  # 结构性压缩-实体比例阈值
        self.structural_compression_atr_ratio = structural_compression_atr_ratio  # 结构性压缩-ATR比例阈值
        self.momentum_convergence_price_threshold = momentum_convergence_price_threshold  # 动量收敛价格变化阈值
        self.direction_ordered_entropy_threshold = direction_ordered_entropy_threshold  # 方向有序性熵阈值
        self.volatility_dense_threshold = volatility_dense_threshold  # 波动密集度阈值
        self.pre_breakout_silence_atr_threshold_quantile = pre_breakout_silence_atr_threshold_quantile  # 突破前静默ATR阈值
        self.internal_price_density_threshold = internal_price_density_threshold  # 内部价格密度阈值
        self.compression_confidence_threshold = compression_confidence_threshold  # 进入压缩状态的最低信心阈值
        self.expansion_confidence_threshold = expansion_confidence_threshold  # 退出压缩状态的信心阈值
        # 波动缓冲机制 - 用于平滑状态转换，避免频繁切换
        self.volatility_buffer_base_threshold = volatility_buffer_base_threshold


class AdaptiveMultiDimCompressionIndicator:
    """
    自适应多维度压缩区指标
    用于检测市场进入窄幅震荡或高概率突破前的压缩状态
    通过分层决策结构评估当前市场是否处于压缩状态
    """

    # 合法窗口档位（离散化用）
    VALID_WINDOW_SIZES = [50, 75, 100, 150, 200]

    def __init__(self,
                 indicator_config: Optional[IndicatorConfig] = None,
                 threshold_config: Optional[ThresholdConfig] = None):

        # 初始化配置对象
        self.indicator_config = indicator_config or IndicatorConfig()
        self.threshold_config = threshold_config or ThresholdConfig()

        # 设置MDC压缩区历史最高/最低点
        self.mdc_high = None  # MDC压缩区历史最高点
        self.mdc_low = None  # MDC压缩区历史最低点

        # 智能压缩判断参数
        self.adaptive_state_weights = True  # 是否使用自适应状态权重（移除滞回机制，使用波动缓冲）

        # 波动缓冲机制 - 用于平滑状态转换，避免频繁切换
        self.volatility_buffer = TDigestVolatilityBuffer(
            self.threshold_config.volatility_buffer_base_threshold)

        # 当前窗口大小（始终使用动态窗口）
        self.current_window = self.indicator_config.compression_window

        # 缓冲区长度 - 存储历史数据的队列，限制最大长度以节省内存
        maxlen = max(self.indicator_config.compression_window,
                     int(self.indicator_config.compression_window *
                         2))  # 确保缓冲区足够大
        self._bars = deque(maxlen=maxlen)  # K线数据缓存
        self._atrs = deque(maxlen=maxlen)  # ATR值缓存
        self._volumes = deque(maxlen=maxlen)  # 成交量缓存
        self._direction_entropy_history = deque(maxlen=maxlen)  # 方向熵历史
        self._body_ratios = deque(
            maxlen=maxlen)  # Cache body ratio for structural check
        self._highs = deque(maxlen=maxlen)  # 最高价缓存
        self._lows = deque(maxlen=maxlen)  # 最低价缓存
        self._recent_atrs = deque(
            maxlen=self.indicator_config.pre_breakout_silence_window)
        self._compression_range_prices = deque(
            maxlen=self.indicator_config.internal_price_density_window)

        # 压缩持续时间跟踪 - 用于计算持续时间加分
        self._compression_start_bar = None  # 压缩开始时的bar索引
        self._compression_duration = 0  # 当前压缩持续时间
        self._current_state = 'expansion'  # 当前状态：'compression' 或 'expansion'
        self._compression_high = None  # 当前压缩区最高点
        self._compression_low = None  # 当前压缩区最低点
        self._compression_highs = deque(maxlen=self.indicator_config.
                                        duration_bonus_window)  # 用于持续时间计算的高点历史
        self._compression_lows = deque(maxlen=self.indicator_config.
                                       duration_bonus_window)  # 用于持续时间计算的低点历史

        # 技术指标 - 使用nautilus_trader内置指标
        self._atr_indicator = AverageTrueRange(
            self.indicator_config.volatility_window)  # ATR指标
        self._vol_ma_indicator = SimpleMovingAverage(
            self.indicator_config.volume_window)  # 成交量移动平均
        self._entropy_ema = ExponentialMovingAverage(
            self.indicator_config.entropy_window)  # 熵的EMA平滑

        # 在线分布估计器 (t-digest based) - 用于计算历史数据的分位数值
        self._t_atr_estimator = OnlineQuantileEstimator(
            compression=self.indicator_config.compression_tdigest)  # ATR历史分布估计
        self._t_vol_estimator = OnlineQuantileEstimator(
            compression=self.indicator_config.compression_tdigest)  # 成交量历史分布估计
        self._t_body_ratio_estimator = OnlineQuantileEstimator(
            compression=self.indicator_config.compression_tdigest
        )  # 实体比例历史分布估计
        # 新增：用于突破前静默ATR分位数估计
        self._t_silence_atr_estimator = OnlineQuantileEstimator(
            compression=self.indicator_config.compression_tdigest)

        # 状态标志
        self._compression_state_history = deque(maxlen=200)  # 压缩状态历史记录
        self._in_compression_zone = False  # 当前是否在压缩区
        self.initialized = False  # 指标是否已初始化
        self.debug_mode = False  # 是否开启调试模式

        # 诊断值 - 用于调试和分析
        self._last_body_ratio = 0.0  # 上次计算的实体比例
        self._last_vol_density = 0.0  # 上次计算的波动密度
        self._last_entropy = 0.0  # 上次计算的方向熵
        # 新增诊断值
        self._last_internal_price_density = 0.0  # 上次内部价格密度
        self._last_pre_breakout_silence_score = 0.0  # 上次突破前静默得分
        self._last_score = 0.0  # 上次压缩分数 (初始化为0.0)

        # 波动率历史，用于波动缓冲
        self._volatility_history = deque(maxlen=100)

        # 用于跟踪状态转换
        self._state_scores = deque(maxlen=200)  # 状态分数历史
        self._transition_scores = deque(maxlen=200)  # 转换分数历史
        self._state_weights_history = deque(maxlen=200)  # 状态权重历史

    def calculate_dynamic_window(
        self,
        base_window: int,
        price_series: List[float],
        lookback: int = 20,
        min_window: int = 50,
        max_window: int = 0,
        min_scale: float = 0.6,
        max_scale: float = 1.8,
    ) -> int:
        """
        改进版动态窗口计算：
        1. 使用 ATR 带宽在历史分布中的分位数（CDF）直接映射 scale
        2. 输出窗口离散化到 [50, 75, 100, 150, 200]
        """
        if max_window is None:
            max_window = int(self.indicator_config.compression_window * 2)

        # 保护 - 确保有足够的数据进行计算
        n = min(len(price_series), lookback)
        if n < 5:
            return self._discretize_window(base_window)

        prices = np.asarray(price_series[-n:], dtype=np.float64)

        # 相对ATR：使用 self._atrs 的最后值
        rel_atr = None
        try:
            if len(self._atrs) > 0:
                recent_atr = float(np.mean(list(self._atrs)[-n:]))
                mean_price = np.mean(prices)
                rel_atr = recent_atr / (mean_price + 1e-9)
        except Exception:
            rel_atr = None

        # 使用 ATR 带宽的分位数（CDF）来决定 scale
        current_atr_bandwidth = rel_atr  # 可替换为更复杂的 bandwidth 计算
        if current_atr_bandwidth is not None and self._t_atr_estimator.count >= self.indicator_config.minimum_tdigest_warmup:
            # 获取当前波动率在历史中的分位数（0~1）
            quantile = self._t_atr_estimator.cdf(float(current_atr_bandwidth))
            # 映射到 scale：低分位 → 小窗口，高分位 → 大窗口
            scale = min_scale + (1.0 - quantile) * (max_scale - min_scale)
        else:
            scale = 1.0  # 未充分预热时使用默认

        # 计算新窗口
        new_window = int(round(base_window * scale))
        new_window = max(min_window, min(new_window, max_window))

        # ✅ 改进1：离散化窗口
        return self._discretize_window(new_window)

    def _discretize_window(self, window: int) -> int:
        """将窗口大小离散化到预定义的合法档位"""
        return min(self.VALID_WINDOW_SIZES, key=lambda x: abs(x - window))

    def _calculate_poc(self, comp_low: float, comp_high: float) -> float:
        """
        在压缩区间内计算 Volume Profile POC (Point of Control)
        即成交量最高的价格水平
        """
        if len(self._bars) < 5 or comp_high <= comp_low:
            return (comp_low + comp_high) / 2  # fallback

        price_volume = []
        for bar in self._bars:
            price = bar.close.as_double()
            vol = bar.volume.as_double()
            if comp_low <= price <= comp_high:
                price_volume.append((price, vol))

        if not price_volume:
            return (comp_low + comp_high) / 2

        bin_volumes = defaultdict(float)
        for price, vol in price_volume:
            bin_key = round(price / 0.01) * 0.01
            bin_volumes[bin_key] += vol

        if not bin_volumes:
            return (comp_low + comp_high) / 2

        # Convert to regular dict to avoid typing issues
        bin_volumes_dict = dict(bin_volumes)
        if bin_volumes_dict:
            poc_bin = max(bin_volumes_dict.keys(), key=lambda x: bin_volumes_dict[x])
            return float(poc_bin)
        else:
            return (comp_low + comp_high) / 2

    # -------------------------
    # 公共接口
    # -------------------------
    def handle_bar(self, bar: Bar) -> bool:
        """
        处理新的K线数据，更新压缩状态
        使用分层决策结构替代简单的阈值评分方法
        """
        log.debug("Processing new bar in handle_bar")

        # 将新的K线数据加入缓存
        self._bars.append(bar)
        self._highs.append(bar.high)
        self._lows.append(bar.low)

        # 更新技术指标
        self._atr_indicator.handle_bar(bar)
        self._vol_ma_indicator.handle_bar(bar)

        # 计算动态窗口（已改进）
        if len(self._bars) >= 20:
            price_series = [b.close.as_double() for b in self._bars]
            self.current_window = self.calculate_dynamic_window(
                self.indicator_config.compression_window, price_series)
        else:
            self.current_window = self.indicator_config.compression_window

        log.debug(
            f"[AdaptiveMDC] Buffer size: {len(self._bars)}, Current window: {self.current_window}, "
            f"Min required: {max(self.indicator_config.compression_window, self.indicator_config.volatility_window, self.indicator_config.volume_window, self.indicator_config.minimum_tdigest_warmup)}"
        )

        # 检查初始化 - 确保有足够的数据进行计算
        self.initialized = len(self._bars) >= max(
            self.current_window, self.indicator_config.volatility_window,
            self.indicator_config.volume_window,
            self.indicator_config.minimum_tdigest_warmup)

        if not self.initialized:
            log.debug("Indicator not initialized yet")
            log.debug(f"[COMPRESSION] Indicator not initialized - Buffer size: {len(self._bars)}, Min required: {max(self.current_window, self.indicator_config.volatility_window, self.indicator_config.volume_window, self.indicator_config.minimum_tdigest_warmup)}")
            return self._in_compression_zone

        # warmup 检查 - 确保tdigest有足够的数据
        if len(self._bars) < self.indicator_config.minimum_tdigest_warmup:
            log.debug(f"[COMPRESSION] Not enough warmup data - Buffer size: {len(self._bars)}, Min required: {self.indicator_config.minimum_tdigest_warmup}")
            return self._in_compression_zone

        # 获取ATR和成交量移动平均值
        atr_val = self._atr_indicator.value
        vol_ma = self._vol_ma_indicator.value

        if atr_val is None or atr_val <= 0:
            log.debug("ATR is invalid or zero")
            log.debug(f"[COMPRESSION] ATR is invalid or zero: {atr_val}")
            return self._in_compression_zone
        self._atrs.append(atr_val)

        if vol_ma is None or vol_ma <= 0:
            log.debug("Volume MA is invalid or zero")
            log.debug(f"[COMPRESSION] Volume MA is invalid or zero: {vol_ma}")
            return self._in_compression_zone
        self._volumes.append(bar.volume)

        # 计算波动率历史，用于波动缓冲
        if len(self._atrs) > 1:
            current_volatility = abs(self._atrs[-1] -
                                     self._atrs[-2]) / (self._atrs[-2] + 1e-6)
            self._volatility_history.append(current_volatility)
            self.volatility_buffer.update(current_volatility)

        # 1) 计算带宽ATR：平均范围 / 当前ATR
        bw_atr = self._compute_bandwidth_atr(atr_val)

        # 2) 计算带宽成交量：平均成交量 / 当前成交量移动平均
        bw_vol = self._compute_bandwidth_volume(vol_ma)

        # 更新分布估计器 - 用于后续的分位数比较
        self._t_atr_estimator.add(bw_atr)
        self._t_vol_estimator.add(bw_vol)
        # 新增：更新静默ATR估计器
        self._t_silence_atr_estimator.add(atr_val)
        self._recent_atrs.append(atr_val)

        # 3) 计算实体比例：|收盘价 - 开盘价| / ATR
        body = abs(bar.close - bar.open)
        body_ratio = float(body) / atr_val
        self._t_body_ratio_estimator.add(body_ratio)
        self._body_ratios.append(body_ratio)
        self._last_body_ratio = body_ratio

        # 4) 计算波动密度：平均实体 / 平均范围
        vol_density = self._compute_volatility_density()
        self._last_vol_density = vol_density

        # 5) 计算方向熵：价格方向的随机性
        entropy = self._compute_direction_entropy()
        self._direction_entropy_history.append(entropy)
        self._last_entropy = entropy

        # 7) 维度检查 - 使用分层决策结构
        is_atr_compression = bw_atr < self._t_atr_estimator.quantile(
            self.threshold_config.compression_atr_threshold_quantile
        )  # ATR是否低于分位数阈值
        is_volume_compression = bw_vol < self._t_vol_estimator.quantile(
            self.threshold_config.compression_volume_threshold_quantile
        )  # 成交量是否低于分位数阈值
        is_structural_compression = self._check_structural_compression(
        )  # 是否为结构性压缩
        has_momentum_convergence = self._check_momentum_convergence()  # 是否动量收敛
        is_direction_ordered = entropy < self.threshold_config.direction_ordered_entropy_threshold  # 方向是否有序
        is_volatility_dense = vol_density < self.threshold_config.volatility_dense_threshold  # 波动是否密集

        # 新增：突破前静默程度检查
        is_pre_breakout_silent = self._check_pre_breakout_silence()
        # 新增：内部价格密度检查
        internal_price_density_score = self._check_internal_price_density(bar)

        # Diagnostic logging
        log.debug(f"[COMPRESSION] ATR Compression: {is_atr_compression} (bw_atr: {bw_atr:.6f}, threshold: {self._t_atr_estimator.quantile(self.threshold_config.compression_atr_threshold_quantile):.6f})")
        log.debug(f"[COMPRESSION] Volume Compression: {is_volume_compression} (bw_vol: {bw_vol:.6f}, threshold: {self._t_vol_estimator.quantile(self.threshold_config.compression_volume_threshold_quantile):.6f})")
        log.debug(f"[COMPRESSION] Structural Compression: {is_structural_compression}")
        log.debug(f"[COMPRESSION] Momentum Convergence: {has_momentum_convergence}")
        log.debug(f"[COMPRESSION] Direction Ordered: {is_direction_ordered} (entropy: {entropy:.6f}, threshold: {self.threshold_config.direction_ordered_entropy_threshold})")
        log.debug(f"[COMPRESSION] Volatility Dense: {is_volatility_dense} (density: {vol_density:.6f}, threshold: {self.threshold_config.volatility_dense_threshold})")
        log.debug(f"[COMPRESSION] Pre-breakout Silent: {is_pre_breakout_silent}")
        log.debug(f"[COMPRESSION] Internal Price Density: {internal_price_density_score:.6f}")

        # 使用分层决策结构判断是否进入压缩状态
        # 判断1：核心压缩条件 - ATR压缩和成交量压缩必须同时满足
        core_compression = is_atr_compression and is_volume_compression
        
        # 判断2：辅助压缩条件 - 至少满足其一
        auxiliary_compression = (
            is_structural_compression or 
            has_momentum_convergence or 
            is_direction_ordered or 
            is_volatility_dense
        )
        
        # 判断3：加强信号 - 加分项
        enhancement_signals = (
            float(is_pre_breakout_silent) * 0.5 +
            internal_price_density_score * 0.3 +
            float(is_direction_ordered) * 0.2
        )
        
        # 分层决策逻辑
        if core_compression and auxiliary_compression:
            # 核心条件和辅助条件都满足，进入压缩状态
            compression_confidence = 0.7 + enhancement_signals
        elif core_compression:
            # 只满足核心条件，根据加强信号判断
            compression_confidence = 0.5 + enhancement_signals
        else:
            # 不满足核心条件，不进入压缩状态
            compression_confidence = 0.0

        # 9) 确定状态 - 使用波动缓冲机制确定当前状态
        is_currently_in_compression = compression_confidence >= self.threshold_config.compression_confidence_threshold

        # 10) 使用波动缓冲机制确定状态转换
        current_flag = self._determine_compression_state(
            compression_confidence, is_currently_in_compression)

        # 11) 更新状态 - 根据当前情况更新压缩状态
        self._update_compression_state(
            current_flag, bar, compression_confidence
        )
        
        # 12) 计算持续时间调节
        duration_modifier = self._calculate_duration_modifier()

        # 13) 最终压缩分数 (用于诊断和调试)
        self._last_score = compression_confidence + duration_modifier

        self._compression_state_history.append(current_flag)
        self._in_compression_zone = current_flag

        log.debug(f"[COMPRESSION] Compression Confidence: {compression_confidence:.6f}, Duration Modifier: {duration_modifier:.6f}, Final Score: {self._last_score:.6f}")
        log.debug(f"[COMPRESSION] Current State: {self._current_state}, Duration: {self._compression_duration}")

        if self.debug_mode:
            state_change = "→" if len(
                self._compression_state_history
            ) >= 2 and self._compression_state_history[
                -2] != self._compression_state_history[-1] else " "
            log.debug(
                f"[AdaptiveMDC]{state_change} ts={bar.ts_event} state={self._current_state} score={self._last_score:.3f} dur={self._compression_duration} win={self.current_window}"
            )

        return self._in_compression_zone

    def _check_pre_breakout_silence(self) -> bool:
        """
        检查突破前静默程度：在最近N根K线内，ATR是否显著低于历史分位数
        """
        if len(self._recent_atrs
               ) < self.indicator_config.pre_breakout_silence_window:
            return False

        recent_avg_atr = np.mean(list(self._recent_atrs))
        silence_atr_threshold = self._t_silence_atr_estimator.quantile(
            self.threshold_config.pre_breakout_silence_atr_threshold_quantile)

        is_silent = recent_avg_atr < silence_atr_threshold
        self._last_pre_breakout_silence_score = float(is_silent)
        return bool(is_silent)

    def _check_internal_price_density(self, bar: Bar) -> float:
        """
        检查压缩区间内部价格密度：价格在窄幅区间内反复震荡的程度
        返回0-1之间的分数，越高表示密度越高
        """
        if self._current_state != 'compression' or self._compression_high is None or self._compression_low is None:
            self._last_internal_price_density = 0.0
            return 0.0

        # 只在压缩区间内记录价格
        current_price = bar.close.as_double()
        comp_high_val = self._compression_high.as_double()
        comp_low_val = self._compression_low.as_double()

        if comp_low_val <= current_price <= comp_high_val:
            self._compression_range_prices.append(current_price)
        elif len(self._compression_range_prices) == 0:
            # 如果当前价格不在区间内且缓冲区为空，无法计算，返回0
            self._last_internal_price_density = 0.0
            return 0.0

        if len(self._compression_range_prices) < 5:  # 至少需要5个点来计算
            self._last_internal_price_density = 0.0
            return 0.0

        prices = np.array(list(self._compression_range_prices))
        range_size = comp_high_val - comp_low_val
        if range_size < 1e-8:  # 压缩区间极小，认为密度很高
            score = 1.0
        else:
            # 计算价格分布的方差（越小表示越密集）
            price_variance = np.var(prices)
            normalized_variance = price_variance / (range_size**2)
            # 反比关系：方差越小，密度得分越高，
            score = max(0.0, min(
                1.0, 1.0 - normalized_variance * 5))  # 乘以5来调整敏感度，注意这个5是合理的

        self._last_internal_price_density = score
        return score

    def _determine_compression_state(
            self, score: float, is_currently_in_compression: bool) -> bool:
        """
        Centralize state transition logic using volatility buffer
        Use buffer during state transitions to avoid frequent switching due to short-term market fluctuations
        """
        # Fix: Determine state transition based on current state and score
        if self._current_state == 'expansion':
            # In expansion state: if score is high, enter compression
            threshold = self.threshold_config.compression_confidence_threshold
            if self.volatility_buffer:
                threshold = self.volatility_buffer.get_threshold(
                    "enter_compression")
            log.debug(
                f"Enter compression threshold: score-{score}, threshold-{threshold}"
            )
            should_enter_compression = bool(score >= threshold)
            return should_enter_compression
        else:
            # In compression state: if score is low, exit compression
            threshold = self.threshold_config.expansion_confidence_threshold
            if self.volatility_buffer:
                threshold = self.volatility_buffer.get_threshold(
                    "exit_compression")
            log.debug(
                f"Exit compression threshold: score-{score}, threshold-{threshold}"
            )
            should_exit_compression = bool(score < threshold)
            return bool(not should_exit_compression)  # Return True to continue compression, False to exit

    def _update_compression_state(self, is_currently_in_compression: bool,
                                  bar: Bar,
                                  compression_intensity_score: float):
        """
        Merge update compression state and determine state transition logic
        """
        previous_state = self._current_state

        self._state_scores.append(compression_intensity_score)
        self._transition_scores.append(
            1.0 if is_currently_in_compression else 0.0)

        if is_currently_in_compression:
            if previous_state == 'expansion':
                # From expansion state to compression state
                self._current_state = 'compression'
                self._compression_start_bar = len(self._bars)
                self._compression_duration = 1
                self._compression_high = bar.high
                self._compression_low = bar.low
                self._compression_highs.clear()
                self._compression_lows.clear()

                if self.mdc_high is None or bar.high > self.mdc_high:
                    self.mdc_high = bar.high
                if self.mdc_low is None or bar.low < self.mdc_low:
                    self.mdc_low = bar.low
            else:
                # In compression state, continue
                self._compression_duration += 1
                if bar.high > self._compression_high:
                    self._compression_high = bar.high
                if bar.low < self._compression_low:
                    self._compression_low = bar.low

            self._compression_highs.append(bar.high)
            self._compression_lows.append(bar.low)
        else:
            if previous_state == 'compression':
                # From compression state to expansion state
                self._current_state = 'expansion'
                self._compression_duration = 0
                self._compression_start_bar = None
                self._compression_high = None
                self._compression_low = None
                # Clear compression range price cache
                self._compression_range_prices.clear()
            else:
                pass

    def _calculate_duration_modifier(self) -> float:
        """
        计算持续时间调节器，用于动态调整进入/退出阈值
        持续时间越长，越容易退出压缩状态（防止久盘必跌）
        """
        if self._current_state != 'compression' or self._compression_duration <= 0:
            return 0.0

        # 持续时间越长，调节值越负（降低退出阈值）
        normalized_duration = min(
            self._compression_duration /
            self.indicator_config.duration_bonus_window, 1.0)
        duration_modifier = -min(normalized_duration * 0.3, 0.3)  # 最大降低30%的阈值

        return duration_modifier

    def _compute_bandwidth_atr(self, current_atr: float) -> float:
        """
        Calculate bandwidth ATR: average range / current ATR
        Measure the size of current ATR relative to historical average range
        """
        if len(self._bars) < self.current_window or current_atr <= 0:
            return 1.0
        recent = list(self._bars)[-self.current_window:]

        # Fix: Convert Decimal objects to float, not using non-existent as_double method
        avg_range = float(np.median([float(b.high - b.low) for b in recent]))

        return avg_range / current_atr

    def _compute_bandwidth_volume(self, vol_ma: float) -> float:
        """
        Calculate bandwidth volume: average volume / current volume moving average
        Measure the size of current volume relative to historical average volume
        """
        if len(self._bars) < self.current_window or vol_ma <= 0:
            return 1.0
        recent = list(self._bars)[-self.current_window:]
        avg_vol = float(np.median([b.volume.as_double() for b in recent]))
        return avg_vol / vol_ma

    def _compute_volatility_density(self) -> float:
        """
        Calculate volatility density: average body / average range
        Measure the density of price fluctuations within the range, smaller value indicates smaller volatility
        """
        if len(self._bars) < self.current_window:
            return 1.0
        recent = list(self._bars)[-self.current_window:]

        # Fix: Convert Decimal objects to float, not using non-existent as_double method
        avg_body = float(
            np.median([abs(float(b.close - b.open)) for b in recent]))
        avg_range = float(np.median([float(b.high - b.low) for b in recent]))

        return (avg_body / avg_range) if avg_range > 0 else 0.0

    def _compute_direction_entropy(self) -> float:
        """
        Calculate direction entropy: randomness of price direction
        Smaller value indicates more ordered direction, more likely to be in compression state
        """
        n = min(10, max(1, len(self._bars) - 1))
        if n <= 1:
            return 1.0

        directions = []
        for i in range(-n, 0):
            prev = self._bars[i - 1].close
            curr = self._bars[i].close
            if curr > prev:
                directions.append(1)
            elif curr < prev:
                directions.append(-1)
            else:
                directions.append(0)

        unique_dirs, counts = np.unique(directions, return_counts=True)
        probs = counts / counts.sum()
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)

        max_entropy = math.log2(
            len(unique_dirs)) if len(unique_dirs) > 1 else 0.0
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        try:
            if hasattr(self._entropy_ema, 'update_raw'):
                self._entropy_ema.update_raw(norm_entropy)
                return self._entropy_ema.value if self._entropy_ema.value is None else norm_entropy
            else:
                print(
                    f"[AdaptiveMDC] EMA does not support update_raw method, using raw entropy value"
                )
                return norm_entropy
        except AttributeError:
            print(
                f"[AdaptiveMDC] EMA update_raw method not available, using raw entropy value"
            )
            return norm_entropy

    def _check_structural_compression(self) -> bool:
        """
        Check structural compression: body ratio and ATR ratio are both low
        Indicates small candle bodies and low volatility, fitting compression characteristics
        """
        if len(self._body_ratios) < 5:
            return False
        avg_body = float(np.median(list(self._body_ratios)[-5:]))
        bw_atr = self._t_atr_estimator.quantile(0.5)
        return (
            avg_body < self.threshold_config.structural_compression_body_ratio
        ) and (bw_atr < self.threshold_config.structural_compression_atr_ratio)

    def _check_momentum_convergence(self) -> bool:
        """
        Check momentum convergence: volume decreasing and price change small
        Indicates weakening market momentum, possibly entering compression state
        """
        n = min(10, len(self._bars))
        if n <= 2:
            return False

        recent = list(self._bars)[-n:]
        x = np.arange(n, dtype=np.float64)
        recent_vol = np.array([float(b.volume.as_double()) for b in recent],
                              dtype=np.float64)

        try:
            vol_slope = np.polyfit(x, recent_vol, 1)[0]
            price_change = float(
                (self._bars[-1].close - self._bars[-n].close).as_double())
            return vol_slope < 0 and abs(
                price_change
            ) < self.threshold_config.momentum_convergence_price_threshold
        except Exception:
            return False

    def is_compression(self) -> bool:
        """Return whether currently in compression state"""
        return self._in_compression_zone

    def get_current_state(self) -> str:
        """
        Get current state
        Returns 'compression', 'expansion', 'enter_compression', 'enter_expansion', or 'unknown'
        """
        if not self._compression_state_history:
            return "unknown"
        if len(self._compression_state_history) >= 2:
            prev, curr = self._compression_state_history[
                -2], self._compression_state_history[-1]
            if not prev and curr:
                return "enter_compression"
            if prev and not curr:
                return "enter_expansion"
        return "compression" if self._in_compression_zone else "expansion"

    def set_debug_mode(self, enabled: bool):
        """Set debug mode"""
        self.debug_mode = enabled

    def reset_compression_state(self):
        """
        Reset compression state
        Reset current state to expansion state, clear all related data
        """
        self._current_state = 'expansion'
        self._compression_duration = 0
        self._compression_start_bar = None
        self._compression_high = None
        self._compression_low = None
        self._compression_highs.clear()
        self._compression_lows.clear()
        self._compression_range_prices.clear()  # 新增：重置压缩区间价格缓存
        self._compression_state_history.clear()
        self._in_compression_zone = False
        self._state_scores.clear()
        self._transition_scores.clear()
        self._state_weights_history.clear()

    def get_state_summary(self) -> Dict[str, Any]:
        """Unified output of all key data of current compression state"""
        # Get compression extremes values
        comp_high_val = self._compression_high.as_double() if self._compression_high else 0
        comp_low_val = self._compression_low.as_double() if self._compression_low else 0
        
        return {
            "state":
            self._current_state,  # 当前状态 compression / expansion
            "score":
            getattr(self, '_last_score', 0.0),  # 最新压缩分数
            "in_zone":
            self._in_compression_zone,  # 是否处于压缩区
            "window":
            self.current_window,  # 当前动态窗口
            "duration":
            self._compression_duration,  # 压缩持续时间
            "extremes": {
                "high": self._compression_high,
                "low": self._compression_low,
            },
            "mdc_extremes": {
                "high": self.mdc_high,
                "low": self.mdc_low,
            },
            "volume_profile_poc":
            self._calculate_poc(comp_low_val, comp_high_val),
            "last_values": {
                "atr": self._atrs[-1] if len(self._atrs) > 0 else None,
                "body_ratio": self._last_body_ratio,
                "vol_density": self._last_vol_density,
                "entropy": self._last_entropy,
                "internal_price_density": self._last_internal_price_density,
            }
        }
