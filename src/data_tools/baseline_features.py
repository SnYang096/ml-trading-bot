"""统一的基础指标和特征工程模块

合并了 base_indicators.py 和 baseline_feature_engineering.py 的功能，
并添加了无量纲特征和优化的依赖关系管理。

主要改进：
1. 合并基础指标和 baseline 特征到一个模块
2. 添加 ZigZag、POC、HAL、Swing High/Low 的无量纲特征
3. 添加基础价格与量能相对变化特征
4. 优化依赖关系管理，支持按需计算
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict
from sklearn.linear_model import LinearRegression

import talib

# ============================================================================
# BaselineFeatureEngineer 类
# ============================================================================


class BaselineFeatureEngineer:
    """Baseline SR and compression features (合并后的版本)"""

    def __init__(
        self,
        percentile_window: int = 288,
        compression_threshold_pct: float = 0.2,
        feature_shift: int = 0,
        feature_clip_bound: float = 10.0,
        vwap_window: int = 160,
        rolling_zscore_windows: List[
            int
        ] = None,  # NEW: Multiple windows for rolling z-score (Feature Stacking)
        enable_diagnostics: bool = False,
    ) -> None:
        self.percentile_window = percentile_window
        self.compression_threshold_pct = compression_threshold_pct
        self.feature_shift = feature_shift
        self.feature_clip_bound = float(feature_clip_bound)
        self.vwap_window = vwap_window
        # Default windows: [50 (short-term), 288 (24h), 500 (long-term)]
        # For 5-minute K-line: 50≈4h, 288=24h, 500≈2 days
        if rolling_zscore_windows is None:
            self.rolling_zscore_windows = [50, 288, 500]
        else:
            self.rolling_zscore_windows = rolling_zscore_windows
        self.enable_diagnostics = enable_diagnostics
        self.diagnostic_report: Dict[str, Dict[str, float]] = {}

        if self.feature_clip_bound <= 0:
            raise ValueError("feature_clip_bound must be positive")
        if not self.rolling_zscore_windows or not all(
            w > 0 for w in self.rolling_zscore_windows
        ):
            raise ValueError(
                "rolling_zscore_windows must be a non-empty list of positive integers"
            )

    # ========================================================================
    # 基础指标计算静态方法
    # ========================================================================

    @staticmethod
    def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """计算相对强弱指数 (RSI)."""
        series = pd.to_numeric(series, errors="coerce").astype(float)
        values = talib.RSI(series.values, timeperiod=period)
        return pd.Series(values, index=series.index)

    @staticmethod
    def compute_macd(
        series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """计算MACD指标."""
        series = pd.to_numeric(series, errors="coerce").astype(float)
        macd_line, signal_line, histogram = talib.MACD(
            series.values, fastperiod=fast, slowperiod=slow, signalperiod=signal
        )
        index = series.index
        return (
            pd.Series(macd_line, index=index),
            pd.Series(signal_line, index=index),
            pd.Series(histogram, index=index),
        )

    @staticmethod
    def compute_bollinger_bands(
        series: pd.Series, period: int = 20, std_dev: int = 2
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """计算布林带."""
        series = pd.to_numeric(series, errors="coerce").astype(float)
        upper, middle, lower = talib.BBANDS(
            series.values, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev, matype=0
        )
        index = series.index
        return (
            pd.Series(upper, index=index),
            pd.Series(middle, index=index),
            pd.Series(lower, index=index),
        )

    @staticmethod
    def compute_atr(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """计算平均真实波幅 (ATR)."""
        high = pd.to_numeric(high, errors="coerce").astype(float)
        low = pd.to_numeric(low, errors="coerce").astype(float)
        close = pd.to_numeric(close, errors="coerce").astype(float)
        atr_values = talib.ATR(high.values, low.values, close.values, timeperiod=period)
        return pd.Series(atr_values, index=high.index)

    @staticmethod
    def compute_zigzag(
        high: pd.Series, low: pd.Series, threshold: float = 0.05
    ) -> pd.Series:
        """计算ZigZag指标."""
        high = pd.to_numeric(high, errors="coerce")
        low = pd.to_numeric(low, errors="coerce")
        if len(high) < 2:
            return pd.Series(index=high.index, dtype=float)
        zigzag = pd.Series(index=high.index, dtype=float)
        last_pivot = high.iloc[0]
        trend = None
        try:
            for i in range(1, len(high)):
                if trend is None:
                    if high.iloc[i] >= last_pivot * (1 + threshold):
                        trend = "up"
                        last_pivot = high.iloc[i]
                        zigzag.iloc[i] = high.iloc[i]
                    elif low.iloc[i] <= last_pivot * (1 - threshold):
                        trend = "down"
                        last_pivot = low.iloc[i]
                        zigzag.iloc[i] = low.iloc[i]
                elif trend == "up":
                    if low.iloc[i] <= last_pivot * (1 - threshold):
                        trend = "down"
                        last_pivot = low.iloc[i]
                        zigzag.iloc[i] = low.iloc[i]
                    elif high.iloc[i] >= last_pivot:
                        last_pivot = high.iloc[i]
                        zigzag.iloc[i] = high.iloc[i]
                else:  # trend == 'down'
                    if high.iloc[i] >= last_pivot * (1 + threshold):
                        trend = "up"
                        last_pivot = high.iloc[i]
                        zigzag.iloc[i] = high.iloc[i]
                    elif low.iloc[i] <= last_pivot:
                        last_pivot = low.iloc[i]
                        zigzag.iloc[i] = low.iloc[i]
            zigzag = zigzag.ffill()
        except Exception:
            zigzag = pd.Series(0, index=high.index, dtype=float)
        return zigzag

    @staticmethod
    def compute_zigzag_high_low(zigzag: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        从 ZigZag 序列中提取高点和低点

        Returns:
            (zz_high, zz_low): ZigZag 高点和低点序列
        """
        zz_high = pd.Series(index=zigzag.index, dtype=float)
        zz_low = pd.Series(index=zigzag.index, dtype=float)

        # 找到转折点
        zigzag_diff = zigzag.diff()
        turn_points = (zigzag_diff * zigzag_diff.shift(1) < 0) | (zigzag_diff != 0) & (
            zigzag_diff.shift(1) == 0
        )

        # 分离高点和低点
        for i in range(len(zigzag)):
            if turn_points.iloc[i]:
                if i > 0:
                    if zigzag.iloc[i] > zigzag.iloc[i - 1]:
                        zz_high.iloc[i] = zigzag.iloc[i]
                    else:
                        zz_low.iloc[i] = zigzag.iloc[i]

        # 前向填充
        zz_high = zz_high.ffill()
        zz_low = zz_low.ffill()

        return zz_high, zz_low

    @staticmethod
    def compute_poc(
        high: pd.Series,
        low: pd.Series,
        volume: pd.Series,
        window: int = 160,
        bins: int = 50,
        value_area_ratio: float = 0.7,
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        计算 POC (Point of Control) 和 HAL (Value Area 70% 价格区间的上下界)

        Args:
            high: 最高价序列
            low: 最低价序列
            volume: 成交量序列
            window: 滚动窗口大小
            bins: 价格分档数量
            value_area_ratio: Value Area 的成交量占比（默认 0.7，即 70%）

        Returns:
            (poc, poc_volume_ratio, hal_high, hal_low):
            - poc: POC 价格序列
            - poc_volume_ratio: POC 价格档的成交量占比序列
            - hal_high: HAL 高点（Value Area 上界）
            - hal_low: HAL 低点（Value Area 下界）
        """
        poc = pd.Series(index=high.index, dtype=float)
        poc_volume_ratio = pd.Series(index=high.index, dtype=float)
        hal_high = pd.Series(index=high.index, dtype=float)
        hal_low = pd.Series(index=high.index, dtype=float)

        for i in range(window, len(high)):
            window_high = high.iloc[i - window : i].max()
            window_low = low.iloc[i - window : i].min()

            if window_high <= window_low:
                poc.iloc[i] = (high.iloc[i] + low.iloc[i]) / 2
                hal_high.iloc[i] = window_high
                hal_low.iloc[i] = window_low
                # 无法计算成交量占比，保持 NaN
                continue

            # 创建价格分档
            price_bins = np.linspace(window_low, window_high, bins + 1)
            bin_volumes = np.zeros(bins)

            # 计算每个价格档的成交量
            for j in range(i - window, i):
                price = (high.iloc[j] + low.iloc[j]) / 2
                vol = volume.iloc[j]

                # 找到价格所在的分档
                bin_idx = np.digitize(price, price_bins) - 1
                bin_idx = max(0, min(bins - 1, bin_idx))
                bin_volumes[bin_idx] += vol

            # 找到成交量最大的分档（POC）
            max_vol_idx = np.argmax(bin_volumes)
            poc.iloc[i] = (price_bins[max_vol_idx] + price_bins[max_vol_idx + 1]) / 2

            # 计算 POC 价格档的成交量占比
            total_volume = bin_volumes.sum()
            if total_volume > 0:
                poc_volume_ratio.iloc[i] = bin_volumes[max_vol_idx] / total_volume

                # 计算 Value Area (70% 价格区间) - HAL 上下界
                target_volume = total_volume * value_area_ratio
                accumulated_volume = bin_volumes[max_vol_idx]

                # 从 POC 开始，向上下扩展，累积成交量直到达到 70%
                upper_idx = max_vol_idx
                lower_idx = max_vol_idx

                while accumulated_volume < target_volume:
                    # 决定向上还是向下扩展
                    upper_vol = (
                        bin_volumes[upper_idx + 1] if upper_idx + 1 < bins else 0
                    )
                    lower_vol = bin_volumes[lower_idx - 1] if lower_idx - 1 >= 0 else 0

                    if upper_vol >= lower_vol and upper_idx + 1 < bins:
                        upper_idx += 1
                        accumulated_volume += bin_volumes[upper_idx]
                    elif lower_idx - 1 >= 0:
                        lower_idx -= 1
                        accumulated_volume += bin_volumes[lower_idx]
                    else:
                        # 无法再扩展
                        break

                # HAL 上下界对应价格档的边界
                hal_high.iloc[i] = price_bins[upper_idx + 1]
                hal_low.iloc[i] = price_bins[lower_idx]
            # 否则保持 NaN

        poc = poc.ffill()
        hal_high = hal_high.ffill()
        hal_low = hal_low.ffill()
        return poc, poc_volume_ratio, hal_high, hal_low

    @staticmethod
    def _compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
        """计算 ATR（内部方法，使用类的静态方法）"""
        return BaselineFeatureEngineer.compute_atr(
            df["high"], df["low"], df["close"], period=window
        )

    # ========================================================================
    # 特征添加静态方法
    # ========================================================================

    @staticmethod
    def add_basic_indicators(
        df: pd.DataFrame, required_features: Optional[set] = None
    ) -> pd.DataFrame:
        """
        添加基础技术指标到DataFrame（优化版：支持按需计算）

        Args:
            df: 包含OHLCV数据的DataFrame
            required_features: 需要计算的指标集合，None 表示计算所有
        """
        if df.empty:
            return df

        result = df.copy()

        # 确保所有列都是数值类型
        for col in ["open", "high", "low", "close", "volume"]:
            if col in result.columns:
                result[col] = pd.to_numeric(result[col], errors="coerce")

        result = result.dropna(subset=["open", "high", "low", "close", "volume"])
        if result.empty:
            return result

        # 按需计算 RSI
        if required_features is None or "rsi" in required_features:
            if "rsi" not in result.columns:
                result["rsi"] = BaselineFeatureEngineer.compute_rsi(result["close"])

        # 按需计算 MACD
        need_macd = required_features is None or any(
            f in required_features for f in ["macd", "macd_signal", "macd_histogram"]
        )
        if need_macd and "macd" not in result.columns:
            try:
                macd_line, signal_line, histogram = (
                    BaselineFeatureEngineer.compute_macd(result["close"])
                )
                result["macd"] = macd_line
                result["macd_signal"] = signal_line
                result["macd_histogram"] = histogram
            except Exception as e:
                print(f"Warning: Error computing MACD: {e}")
                result["macd"] = 0
                result["macd_signal"] = 0
                result["macd_histogram"] = 0

        # 按需计算 Bollinger Bands
        need_bb = required_features is None or any(
            f in required_features for f in ["bb_upper", "bb_middle", "bb_lower"]
        )
        if need_bb and "bb_upper" not in result.columns:
            try:
                upper_band, middle_band, lower_band = (
                    BaselineFeatureEngineer.compute_bollinger_bands(result["close"])
                )
                result["bb_upper"] = upper_band
                result["bb_middle"] = middle_band
                result["bb_lower"] = lower_band
            except Exception as e:
                print(f"Warning: Error computing Bollinger Bands: {e}")
                result["bb_upper"] = result["close"]
                result["bb_middle"] = result["close"]
                result["bb_lower"] = result["close"]

        # 按需计算 ATR
        if required_features is None or "atr" in required_features:
            if "atr" not in result.columns:
                try:
                    result["atr"] = BaselineFeatureEngineer.compute_atr(
                        result["high"], result["low"], result["close"]
                    )
                except Exception as e:
                    print(f"Warning: Error computing ATR: {e}")
                    result["atr"] = 0

        # 按需计算 ZigZag
        if required_features is None or "zigzag" in required_features:
            if "zigzag" not in result.columns:
                try:
                    result["zigzag"] = BaselineFeatureEngineer.compute_zigzag(
                        result["high"], result["low"]
                    )
                except Exception as e:
                    print(f"Warning: Error computing ZigZag: {e}")
                    result["zigzag"] = 0

        # 按需计算价格变化和波动率
        if required_features is None or "price_change" in required_features:
            if "price_change" not in result.columns:
                result["price_change"] = result["close"].pct_change()

        if required_features is None or "volatility" in required_features:
            if "volatility" not in result.columns:
                price_change_numeric = pd.to_numeric(
                    result["price_change"], errors="coerce"
                ).astype(float)
                values = talib.STDDEV(
                    price_change_numeric.values, timeperiod=14, nbdev=1
                )
                result["volatility"] = pd.Series(
                    values, index=price_change_numeric.index
                )

        # 按需计算成交量特征
        if (
            required_features is None
            or "volume_sma" in required_features
            or "volume_ratio" in required_features
        ):
            if "volume_sma" not in result.columns:
                volume_numeric = pd.to_numeric(
                    result["volume"], errors="coerce"
                ).astype(float)
                values = talib.SMA(volume_numeric.values, timeperiod=20)
                result["volume_sma"] = pd.Series(values, index=volume_numeric.index)
            if "volume_ratio" not in result.columns:
                result["volume_ratio"] = result["volume"] / result[
                    "volume_sma"
                ].replace(0, np.nan)

        return result

    @staticmethod
    def ensure_basic_indicators(
        df: pd.DataFrame, required_features: Optional[set] = None
    ) -> pd.DataFrame:
        """确保基础指标存在（优化版：支持按需计算）"""
        if df.empty:
            return df

        # 检查需要的指标是否都已存在
        if required_features:
            missing = required_features - set(df.columns)
            if not missing:
                return df

        return BaselineFeatureEngineer.add_basic_indicators(df, required_features)

    @staticmethod
    def add_zigzag_dimensionless_features(
        df: pd.DataFrame, required_features: Optional[set] = None
    ) -> pd.DataFrame:
        """
        添加 ZigZag 相关的无量纲特征

        新增特征：
        - price_to_zz_high_pct: 当前价格到最近 ZigZag 高点的相对距离
        - price_to_zz_low_pct: 当前价格到最近 ZigZag 低点的相对距离
        - zz_amplitude_pct: ZigZag 波幅（相对）
        - zz_duration: ZigZag 持续时间（bar 数，无量纲）
        - zz_slope: ZigZag 斜率（归一化）
        """
        if df.empty:
            return df

        result = df.copy()

        # 确保 zigzag 存在
        if "zigzag" not in result.columns:
            if required_features and any(
                "zz_" in f or "zigzag" in f for f in required_features
            ):
                result = BaselineFeatureEngineer.ensure_basic_indicators(
                    result, {"zigzag"}
                )
            else:
                return result

        # 确保 atr 存在（zz_slope 需要 atr）
        if "atr" not in result.columns:
            if required_features and "zz_slope" in required_features:
                result = BaselineFeatureEngineer.ensure_basic_indicators(
                    result, {"atr"}
                )
            elif required_features is None:
                # 如果没有指定 required_features，也确保 atr 存在
                result = BaselineFeatureEngineer.ensure_basic_indicators(
                    result, {"atr"}
                )

        close = result["close"].replace(0, np.nan)
        zigzag = result["zigzag"]

        # 提取 ZigZag 高点和低点
        zz_high, zz_low = BaselineFeatureEngineer.compute_zigzag_high_low(zigzag)

        # 1. 当前价格距离最近 ZigZag 高/低点的相对距离
        if required_features is None or "price_to_zz_high_pct" in required_features:
            if "price_to_zz_high_pct" not in result.columns:
                result["price_to_zz_high_pct"] = ((zz_high - close) / close).replace(
                    [np.inf, -np.inf], np.nan
                )

        if required_features is None or "price_to_zz_low_pct" in required_features:
            if "price_to_zz_low_pct" not in result.columns:
                result["price_to_zz_low_pct"] = ((close - zz_low) / close).replace(
                    [np.inf, -np.inf], np.nan
                )

        # 2. ZigZag 波幅（相对）
        if required_features is None or "zz_amplitude_pct" in required_features:
            if "zz_amplitude_pct" not in result.columns:
                zz_low_safe = zz_low.replace(0, np.nan)
                result["zz_amplitude_pct"] = ((zz_high - zz_low) / zz_low_safe).replace(
                    [np.inf, -np.inf], np.nan
                )

        # 3. ZigZag 持续时间（从上一个转折点至今的 bar 数）
        if required_features is None or "zz_duration" in required_features:
            if "zz_duration" not in result.columns:
                zigzag_diff = zigzag.diff()
                turn_points = (zigzag_diff * zigzag_diff.shift(1) < 0) | (
                    (zigzag_diff != 0) & (zigzag_diff.shift(1) == 0)
                )

                duration = pd.Series(index=zigzag.index, dtype=float)
                last_turn_idx = 0
                for i in range(len(zigzag)):
                    if turn_points.iloc[i]:
                        last_turn_idx = i
                    duration.iloc[i] = i - last_turn_idx
                result["zz_duration"] = duration.fillna(0.0)

        # 4. ZigZag 斜率（归一化）
        if required_features is None or "zz_slope" in required_features:
            if "zz_slope" not in result.columns:
                if "atr" not in result.columns:
                    raise ValueError(
                        "ATR is required for computing zz_slope. "
                        "Please ensure 'atr' is computed before calling this function."
                    )

                window = 5
                zz_slope_raw = zigzag.diff(window) / window

                # 归一化：使用 ATR
                atr_safe = result["atr"].replace(0, np.nan)
                result["zz_slope"] = (zz_slope_raw / atr_safe).replace(
                    [np.inf, -np.inf], np.nan
                )

        return result

    @staticmethod
    def add_poc_hal_dimensionless_features(
        df: pd.DataFrame, required_features: Optional[set] = None, poc_window: int = 160
    ) -> pd.DataFrame:
        """
        添加 POC (Point of Control) 和 HAL (Value Area 70% 价格区间的上下界) 相关的无量纲特征

        注意：POC 和 HAL 的计算合并在一起，因为它们都基于相同的 volume profile 计算，
        避免重复计算浪费性能。

        POC 相关特征：
        - price_to_poc_pct: 当前价格到 POC 的相对距离
        - poc_position_ratio: POC 在价格区间中的位置（0-1）
        - poc_volume_ratio: POC 位置的成交量占比

        HAL 相关特征：
        - price_to_hal_high_pct: 当前价格到 HAL 高点的相对距离
        - price_to_hal_low_pct: 当前价格到 HAL 低点的相对距离
        - price_to_hal_mid_pct: 当前价格到 HAL 中点的相对距离
        - hal_bandwidth_pct: HAL 带宽（相对）
        """
        if df.empty:
            return df

        result = df.copy()

        # 检查是否需要计算 POC 或 HAL
        need_poc = required_features is None or any(
            "poc" in f for f in required_features
        )
        need_hal = required_features is None or any(
            "hal" in f for f in required_features
        )

        if not (need_poc or need_hal):
            return result

        # 计算 POC 和 HAL（一次性计算，避免重复）
        need_compute = False
        if need_poc and (
            "poc" not in result.columns or "poc_volume_ratio" not in result.columns
        ):
            need_compute = True
        if need_hal and (
            "hal_high" not in result.columns or "hal_low" not in result.columns
        ):
            need_compute = True

        if need_compute:
            poc, poc_volume_ratio, hal_high, hal_low = (
                BaselineFeatureEngineer.compute_poc(
                    result["high"], result["low"], result["volume"], window=poc_window
                )
            )
            result["poc"] = poc
            result["poc_volume_ratio"] = poc_volume_ratio
            result["hal_high"] = hal_high
            result["hal_low"] = hal_low
            result["hal_mid"] = (hal_high + hal_low) / 2.0

        close = result["close"].replace(0, np.nan)
        poc = result["poc"]
        hal_high = result["hal_high"]
        hal_low = result["hal_low"]
        hal_mid = result["hal_mid"]
        high = result["high"]
        low = result["low"]

        # ========== POC 相关特征 ==========
        # 1. 当前价格到 POC 的相对距离
        if required_features is None or "price_to_poc_pct" in required_features:
            if "price_to_poc_pct" not in result.columns:
                result["price_to_poc_pct"] = ((poc - close) / close).replace(
                    [np.inf, -np.inf], np.nan
                )

        # 2. POC 在价格区间中的位置（0-1）
        if required_features is None or "poc_position_ratio" in required_features:
            if "poc_position_ratio" not in result.columns:
                price_range = (high - low).replace(0, np.nan)
                result["poc_position_ratio"] = (
                    ((poc - low) / price_range)
                    .replace([np.inf, -np.inf], np.nan)
                    .clip(0.0, 1.0)
                )

        # ========== HAL 相关特征 ==========
        # 1. 当前价格到 HAL 的相对距离
        if required_features is None or "price_to_hal_high_pct" in required_features:
            if "price_to_hal_high_pct" not in result.columns:
                result["price_to_hal_high_pct"] = ((hal_high - close) / close).replace(
                    [np.inf, -np.inf], np.nan
                )

        if required_features is None or "price_to_hal_low_pct" in required_features:
            if "price_to_hal_low_pct" not in result.columns:
                result["price_to_hal_low_pct"] = ((close - hal_low) / close).replace(
                    [np.inf, -np.inf], np.nan
                )

        if required_features is None or "price_to_hal_mid_pct" in required_features:
            if "price_to_hal_mid_pct" not in result.columns:
                result["price_to_hal_mid_pct"] = ((hal_mid - close) / close).replace(
                    [np.inf, -np.inf], np.nan
                )

        # 2. HAL 带宽（相对）
        if required_features is None or "hal_bandwidth_pct" in required_features:
            if "hal_bandwidth_pct" not in result.columns:
                hal_mid_safe = hal_mid.replace(0, np.nan)
                result["hal_bandwidth_pct"] = (
                    (hal_high - hal_low) / hal_mid_safe
                ).replace([np.inf, -np.inf], np.nan)

        return result

    @staticmethod
    def add_swing_dimensionless_features(
        df: pd.DataFrame,
        required_features: Optional[set] = None,
        swing_win_short: int = 20,
        swing_win_long: int = 60,
    ) -> pd.DataFrame:
        """
        添加 Swing High/Low 相关的无量纲特征

        新增特征：
        - swing_high_pct_close: Swing High 相对收盘价的比率
        - swing_low_pct_close: Swing Low 相对收盘价的比率
        - swing_amplitude_pct: Swing 波幅（相对）
        """
        if df.empty:
            return df

        result = df.copy()

        close = result["close"].replace(0, np.nan)

        # 计算 Swing High/Low（如果不存在）
        if "roll_high_s" not in result.columns:
            if required_features and any("swing" in f for f in required_features):
                result["roll_high_s"] = (
                    result["high"].rolling(swing_win_short, min_periods=1).max()
                )
                result["roll_low_s"] = (
                    result["low"].rolling(swing_win_short, min_periods=1).min()
                )
                result["roll_high_l"] = (
                    result["high"].rolling(swing_win_long, min_periods=1).max()
                )
                result["roll_low_l"] = (
                    result["low"].rolling(swing_win_long, min_periods=1).min()
                )
            else:
                return result

        # 1. Swing High/Low 相对收盘价的比率
        if required_features is None or "swing_high_pct_close" in required_features:
            if "swing_high_pct_close" not in result.columns:
                result["swing_high_pct_close"] = (
                    (result["roll_high_s"] - close) / close.replace(0, np.nan)
                ).replace([np.inf, -np.inf], np.nan)

        if required_features is None or "swing_low_pct_close" in required_features:
            if "swing_low_pct_close" not in result.columns:
                result["swing_low_pct_close"] = (
                    (close - result["roll_low_s"]) / close.replace(0, np.nan)
                ).replace([np.inf, -np.inf], np.nan)

        # 2. Swing 波幅（相对）
        if required_features is None or "swing_amplitude_pct" in required_features:
            if "swing_amplitude_pct" not in result.columns:
                roll_low_s_safe = result["roll_low_s"].replace(0, np.nan)
                result["swing_amplitude_pct"] = (
                    (result["roll_high_s"] - result["roll_low_s"]) / roll_low_s_safe
                ).replace([np.inf, -np.inf], np.nan)

        return result

    @staticmethod
    def add_price_volume_relative_features(
        df: pd.DataFrame, required_features: Optional[set] = None
    ) -> pd.DataFrame:
        """
        添加基础价格与量能相对变化特征

        新增特征：
        - ret_1h, ret_4h, ret_24h: 对数收益率（1小时、4小时、24小时）
        - rv_4h, rv_24h: 已实现波动率
        - vol_ma_ratio: 成交量移动平均比率
        - vol_zscore: 成交量 Z-score
        """
        if df.empty:
            return df

        result = df.copy()
        close = result["close"].replace(0, np.nan)
        volume = result["volume"]

        # 1. 对数收益率（常用）
        # 注意：这里假设数据是 5 分钟 K 线，1h=12根，4h=48根，24h=288根
        # 实际应该根据时间框架动态计算
        periods_1h = 12  # 假设 5 分钟 K 线
        periods_4h = 48
        periods_24h = 288

        if required_features is None or "ret_1h" in required_features:
            if "ret_1h" not in result.columns:
                result["ret_1h"] = np.log(close / close.shift(periods_1h))

        if required_features is None or "ret_4h" in required_features:
            if "ret_4h" not in result.columns:
                result["ret_4h"] = np.log(close / close.shift(periods_4h))

        if required_features is None or "ret_24h" in required_features:
            if "ret_24h" not in result.columns:
                result["ret_24h"] = np.log(close / close.shift(periods_24h))

        # 2. 已实现波动率（基于 ret_1h）
        if required_features is None or "rv_4h" in required_features:
            if "rv_4h" not in result.columns and "ret_1h" in result.columns:
                result["rv_4h"] = (
                    result["ret_1h"]
                    .rolling(window=periods_4h // periods_1h, min_periods=1)
                    .std()
                )

        if required_features is None or "rv_24h" in required_features:
            if "rv_24h" not in result.columns and "ret_1h" in result.columns:
                result["rv_24h"] = (
                    result["ret_1h"]
                    .rolling(window=periods_24h // periods_1h, min_periods=1)
                    .std()
                )

        # 3. 成交量异常度
        if required_features is None or "vol_ma_ratio" in required_features:
            if "vol_ma_ratio" not in result.columns:
                vol_ma = volume.rolling(
                    window=periods_24h, min_periods=periods_24h
                ).mean()
                vol_ma_ratio = (
                    (volume / vol_ma.replace(0, np.nan))
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(1.0)
                )
                # 滚动统计衍生特征统一 shift(1)，确保在 t 时刻仅使用 t-1 及之前的数据
                result["vol_ma_ratio"] = vol_ma_ratio.shift(1)

        if required_features is None or "vol_zscore" in required_features:
            if "vol_zscore" not in result.columns:
                result["vol_zscore"] = BaselineFeatureEngineer._rolling_zscore(
                    volume.astype(float),
                    window=periods_24h,
                    min_periods=periods_24h,
                )

        return result

    @staticmethod
    def add_common_derived_features(
        df: pd.DataFrame,
        required_features: Optional[set] = None,
        rolling_zscore_windows: List[int] = None,
    ) -> pd.DataFrame:
        """
        添加常用衍生特征（优化版：支持按需计算，不强制计算所有基础指标）
        """
        if df.empty:
            return df

        result = df.copy()
        close = result["close"].replace(0, np.nan)

        # 解析依赖关系：确定需要哪些基础指标
        needed_basic = set()
        if required_features:
            # 分析需要哪些基础指标
            if any("rsi" in f for f in required_features):
                needed_basic.add("rsi")
            if any("macd" in f for f in required_features):
                needed_basic.update(["macd", "macd_signal", "macd_histogram"])
            if any("bb_" in f for f in required_features):
                needed_basic.update(["bb_upper", "bb_lower", "bb_middle"])
            if any("atr" in f for f in required_features):
                needed_basic.add("atr")
        else:
            # 如果没有指定，只确保必要的基础指标
            needed_basic = {"rsi", "atr"}  # 最小集合

        # 按需计算基础指标
        if needed_basic:
            result = BaselineFeatureEngineer.ensure_basic_indicators(
                result, needed_basic
            )

        # 只在需要时计算特征
        if not required_features or "returns" in required_features:
            if "returns" not in result.columns:
                result["returns"] = close.pct_change()

        if not required_features or "log_returns" in required_features:
            if "log_returns" not in result.columns:
                shifted = close.shift(1).replace(0, np.nan)
                result["log_returns"] = np.log(close / shifted)

        if not required_features or "price_change" in required_features:
            if "price_change" not in result.columns:
                result["price_change"] = close.diff()

        if not required_features or "volatility" in required_features:
            if "volatility" not in result.columns:
                if "returns" in result.columns:
                    returns_numeric = pd.to_numeric(
                        result["returns"], errors="coerce"
                    ).astype(float)
                    values = talib.STDDEV(
                        returns_numeric.values, timeperiod=20, nbdev=1
                    )
                    result["volatility"] = pd.Series(
                        values, index=returns_numeric.index
                    )
                else:
                    price_change = close.pct_change()
                    price_change_numeric = pd.to_numeric(
                        price_change, errors="coerce"
                    ).astype(float)
                    values = talib.STDDEV(
                        price_change_numeric.values, timeperiod=20, nbdev=1
                    )
                    result["volatility"] = pd.Series(
                        values, index=price_change_numeric.index
                    )

        # BB 相关特征
        if {"bb_upper", "bb_lower"}.issubset(result.columns):
            if not required_features or "bb_position" in required_features:
                if "bb_position" not in result.columns:
                    denom = (result["bb_upper"] - result["bb_lower"]).replace(0, np.nan)
                    result["bb_position"] = (
                        ((close - result["bb_lower"]) / denom)
                        .replace([np.inf, -np.inf], np.nan)
                        .fillna(0.5)
                    )

            if not required_features or "bb_width" in required_features:
                if "bb_width" not in result.columns:
                    # bb_width 归一化：除以 bb_middle（布林带中线）或 close
                    # 这样消除量纲，使其成为相对值
                    bb_width_raw = (result["bb_upper"] - result["bb_lower"]).abs()
                    if "bb_middle" in result.columns:
                        bb_middle_safe = result["bb_middle"].replace(0, np.nan)
                        result["bb_width"] = (bb_width_raw / bb_middle_safe).replace(
                            [np.inf, -np.inf], np.nan
                        )
                    else:
                        # 如果没有 bb_middle，使用 close
                        close_safe = close.replace(0, np.nan)
                        result["bb_width"] = (bb_width_raw / close_safe).replace(
                            [np.inf, -np.inf], np.nan
                        )

        # 归一化特征（价格归一化，保留用于向后兼容）
        # 注意：RSI 本身就是 0~100 的标准范围，不需要归一化
        if not required_features or "macd_normalized" in required_features:
            if "macd_normalized" not in result.columns and "macd" in result.columns:
                result["macd_normalized"] = (
                    result["macd"] / close.replace(0, np.nan)
                ).replace([np.inf, -np.inf], np.nan)

        if not required_features or "atr_normalized" in required_features:
            if "atr_normalized" not in result.columns and "atr" in result.columns:
                result["atr_normalized"] = (
                    result["atr"] / close.replace(0, np.nan)
                ).replace([np.inf, -np.inf], np.nan)

        # ========== 滚动 Z-score 特征（推荐：比价格归一化更优）==========
        # 使用多个滚动窗口标准化（Feature Stacking），让模型学习不同时间尺度的信息
        # 默认窗口：[50 (短期), 288 (24h), 500 (长期)]
        # 对于 5 分钟 K 线：50≈4h, 288=24h, 500≈2天
        if rolling_zscore_windows is None:
            rolling_zscore_windows = [50, 288, 500]

        # Helper function to generate z-score features for multiple windows
        def add_multi_window_zscore(
            base_col: str,
            feature_prefix: str,
            windows: List[int],
            required_features: Optional[set],
        ) -> None:
            """为某个基础指标生成多个窗口的 z-score 特征"""
            if base_col not in result.columns:
                return

            for window in windows:
                zscore_col = f"{feature_prefix}_zscore_w{window}"
                # Check if this specific feature is required
                if required_features is not None:
                    # Check if any zscore variant is requested or this specific one
                    if not any(
                        f"{feature_prefix}_zscore" in f for f in required_features
                    ):
                        continue
                    if zscore_col not in required_features and not any(
                        f.startswith(f"{feature_prefix}_zscore")
                        and f.endswith(f"_w{window}")
                        for f in required_features
                    ):
                        # If specific windows are requested, only generate those
                        if any(
                            f"{feature_prefix}_zscore_w" in f for f in required_features
                        ):
                            continue

                if zscore_col not in result.columns:
                    # 【关键修复】：强制 min_periods=window，确保满窗才输出，减少早期噪声
                    result[zscore_col] = BaselineFeatureEngineer._rolling_zscore(
                        result[base_col], window=window, min_periods=window
                    )

        # 1. RSI 滚动 Z-score（虽然 RSI 本身是 0-100，但 Z-score 能突出极端值）
        # 生成多个窗口：rsi_zscore_w50, rsi_zscore_w288, rsi_zscore_w500
        if required_features is None or any(
            "rsi_zscore" in f for f in required_features or [""]
        ):
            add_multi_window_zscore(
                "rsi", "rsi", rolling_zscore_windows, required_features
            )

        # 2. MACD 滚动 Z-score（MACD 绝对值随价格变化，Z-score 标准化更优）
        if required_features is None or any(
            "macd_zscore" in f for f in required_features or [""]
        ):
            add_multi_window_zscore(
                "macd", "macd", rolling_zscore_windows, required_features
            )

        # 3. MACD Histogram 滚动 Z-score（波动更大，Z-score 能突出极端动量变化）
        if required_features is None or any(
            "macd_histogram_zscore" in f for f in required_features or [""]
        ):
            add_multi_window_zscore(
                "macd_histogram",
                "macd_histogram",
                rolling_zscore_windows,
                required_features,
            )

        # 4. Momentum 滚动 Z-score（不同资产的 ROC 量级差异巨大，Z-score 必须）
        for period in [5, 10, 20]:
            momentum_col = f"momentum_{period}"
            if required_features is None or any(
                f"momentum_{period}_zscore" in f for f in required_features or [""]
            ):
                add_multi_window_zscore(
                    momentum_col,
                    f"momentum_{period}",
                    rolling_zscore_windows,
                    required_features,
                )

        # 5. ATR 滚动 Z-score（ATR 绝对值与价格成正比，Z-score 可判断波动率异常）
        if required_features is None or any(
            "atr_zscore" in f for f in required_features or [""]
        ):
            add_multi_window_zscore(
                "atr", "atr", rolling_zscore_windows, required_features
            )

        # 6. Volume 滚动 Z-score（交易量绝对值与流动性相关，Z-score 捕捉相对变化）
        if required_features is None or any(
            "volume_zscore" in f for f in required_features or [""]
        ):
            add_multi_window_zscore(
                "volume", "volume", rolling_zscore_windows, required_features
            )

        # 7. BB Width 滚动 Z-score（布林带宽度反映波动性，Z-score 标准化）
        if required_features is None or any(
            "bb_width_zscore" in f for f in required_features or [""]
        ):
            add_multi_window_zscore(
                "bb_width", "bb_width", rolling_zscore_windows, required_features
            )

        # 8. Volatility 滚动 Z-score（波动率指标的 Z-score）
        if required_features is None or any(
            "volatility_zscore" in f for f in required_features or [""]
        ):
            add_multi_window_zscore(
                "volatility", "volatility", rolling_zscore_windows, required_features
            )

        # Momentum features
        for period in [5, 10, 20]:
            momentum_col = f"momentum_{period}"
            if not required_features or momentum_col in required_features:
                if momentum_col not in result.columns:
                    result[momentum_col] = close.pct_change(period)

        # SMA features
        sma_map = {5: "sma_5", 10: "sma_10", 20: "sma_20"}
        for window, col_name in sma_map.items():
            if not required_features or col_name in required_features:
                if col_name not in result.columns:
                    close_numeric = pd.to_numeric(close, errors="coerce").astype(float)
                    values = talib.SMA(close_numeric.values, timeperiod=window)
                    result[col_name] = pd.Series(
                        values, index=close_numeric.index
                    ).fillna(close)

        # SMA/EMA 相对 close 的百分比
        close_safe = close.replace(0, np.nan)
        for col_name in [
            "sma_5",
            "sma_10",
            "sma_20",
            "ema_5",
            "ema_10",
            "ema_20",
            "ema_50",
            "wma_20",
        ]:
            pct_col = f"{col_name}_pct_close"
            if not required_features or pct_col in required_features:
                if col_name in result.columns and pct_col not in result.columns:
                    result[pct_col] = ((result[col_name] / close_safe - 1.0)).replace(
                        [np.inf, -np.inf], np.nan
                    )

        # SMA ratios
        if not required_features or "sma_ratio_5_20" in required_features:
            if {"sma_5", "sma_20"}.issubset(
                result.columns
            ) and "sma_ratio_5_20" not in result.columns:
                result["sma_ratio_5_20"] = (
                    (result["sma_5"] / result["sma_20"].replace(0, np.nan))
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(1.0)
                )

        if not required_features or "sma_ratio_10_20" in required_features:
            if {"sma_10", "sma_20"}.issubset(
                result.columns
            ) and "sma_ratio_10_20" not in result.columns:
                result["sma_ratio_10_20"] = (
                    (result["sma_10"] / result["sma_20"].replace(0, np.nan))
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(1.0)
                )

        # Volume features
        if not required_features or "volume_sma_20" in required_features:
            if "volume_sma_20" not in result.columns:
                volume_numeric = pd.to_numeric(
                    result["volume"], errors="coerce"
                ).astype(float)
                values = talib.SMA(volume_numeric.values, timeperiod=20)
                result["volume_sma_20"] = pd.Series(
                    values, index=volume_numeric.index
                ).fillna(result["volume"])

        if not required_features or "volume_ratio" in required_features:
            if "volume_ratio" not in result.columns:
                if "volume_sma_20" in result.columns:
                    denom = result["volume_sma_20"].replace(0, np.nan)
                    result["volume_ratio"] = (
                        (result["volume"] / denom)
                        .replace([np.inf, -np.inf], np.nan)
                        .fillna(1.0)
                    )

        # Final cleanup: 处理所有数值列的 inf，保留 NaN 到预处理阶段
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in result.columns:
                result[col] = result[col].replace([np.inf, -np.inf], np.nan)

        return result

    @staticmethod
    def _rolling_percentile(
        series: pd.Series, window: int, min_periods: int = None, shift: bool = True
    ) -> pd.Series:
        """
        安全滚动百分位排名（严格因果，无未来信息）

        【核心原则：滚动窗口统计特征需要 shift(1)】
        - 虽然我们在 t 时刻可以使用 close[t]，但对于"滚动窗口统计特征"仍需 shift(1)
        - 原因：避免将当前值包含在历史分布中，即使我们已经排除了当前值
        - 更严格的做法：在 t 时刻使用的特征基于 t-1 及之前的数据计算

        【关键区分】
        - 基础滚动指标（sma, ema, atr, bb_width）：不需要 shift(1)
        - 滚动窗口统计特征（zscore, percentile, entropy）：需要 shift(1)

        Args:
            series: 输入序列
            window: 滚动窗口长度
            min_periods: 最小观测数才开始输出（默认=window，最稳健）
            shift: 是否 shift(1) 以确保完全因果（默认=True，推荐）

        Returns:
            滚动百分位排名序列（0~1，早期不足窗口处为 NaN）
        """
        if min_periods is None:
            min_periods = window  # 默认：必须满窗才输出，最稳健

        min_periods = min(min_periods, window)

        def _percentile(x: np.ndarray) -> float:
            """
            计算当前值在历史窗口中的百分位排名（严格因果，无自我参照偏差）

            【实现说明】
            - current = x[-1]：当前值（如 close[t]），作为"新来的考生"
            - history = x[:-1]：历史窗口（如 [t-N, t-1]），作为"老考生的成绩分数线"
            - percentile = (history <= current).sum() / len(history)
              表示：当前值在历史中的相对位置，完全基于历史评估当前状态
            """
            if len(x) < 2 or not np.isfinite(x[-1]):
                return np.nan
            current = x[-1]  # 当前值（如 close[t]），作为"新来的考生"
            history = x[
                :-1
            ]  # ← 关键：只用历史（如 [t-N, t-1]），作为"老考生的成绩分数线"
            history = history[np.isfinite(history)]
            if len(history) == 0:
                return np.nan
            # 当前值在历史中的分位：(历史中 ≤ 当前值的数量) / 历史总数量
            # 这表示：当前值相对于历史的位置，完全基于历史评估当前状态
            return (history <= current).sum() / float(len(history))

        percentile_series = series.rolling(
            window=window, min_periods=min_periods
        ).apply(_percentile, raw=True)

        # 【关键修复】：对滚动窗口统计特征强制 shift(1)，确保完全因果
        # 在 t 时刻使用的特征基于 t-1 及之前的数据计算
        if shift:
            percentile_series = percentile_series.shift(1)

        return percentile_series

    @staticmethod
    def _rolling_zscore(
        series: pd.Series,
        window: int,
        min_periods: int = None,
        return_quality: bool = False,
        shift: bool = True,
    ):
        """
        安全滚动 Z-score（严格因果，无未来信息）

        【核心原则：滚动窗口统计特征需要 shift(1)】
        - 虽然我们在 t 时刻可以使用 close[t]，但对于"滚动窗口统计特征"仍需 shift(1)
        - 原因：避免将当前值包含在历史分布中，即使 rolling 本身是因果的
        - 更严格的做法：在 t 时刻使用的特征基于 t-1 及之前的数据计算

        【关键区分】
        - 基础滚动指标（sma, ema, atr, bb_width）：不需要 shift(1)
        - 滚动窗口统计特征（zscore, percentile, entropy）：需要 shift(1)

        【说明】
        - min_periods 小会导致早期统计量不稳定（小样本噪声）
        - 但 rolling() 只使用历史和当前数据，绝不包含未来
        - 提高 min_periods 能降低虚假相关，因为剔除了高噪声样本

        Args:
            series: 输入时间序列（如 ATR、volatility）
            window: 滚动窗口长度（如 288）
            min_periods: 最小观测数才开始输出（默认=window，最稳健）
            return_quality: 是否同时返回质量分数（0~1，1=完整窗口）
            shift: 是否 shift(1) 以确保完全因果（默认=True，推荐）

        Returns:
            zscore: 标准化后的序列（早期不足窗口处为 NaN）
            quality (可选): 每个点的统计质量 = 实际样本数 / window
        """
        if min_periods is None:
            min_periods = window  # 默认：必须满窗才输出，最稳健

        min_periods = min(min_periods, window)

        # 滚动统计量（只依赖历史和当前，无未来信息）
        rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
        rolling_std = series.rolling(window=window, min_periods=min_periods).std()

        # 计算 Z-score: (x - mean) / std
        zscore = (series - rolling_mean) / rolling_std.replace(0, np.nan)

        # 【关键修复】：对滚动窗口统计特征强制 shift(1)，确保完全因果
        # 在 t 时刻使用的特征基于 t-1 及之前的数据计算
        if shift:
            zscore = zscore.shift(1)

        if return_quality:
            # 计算质量分数：实际样本数 / window（0~1，1表示使用了完整窗口）
            count = series.rolling(window=window, min_periods=1).count()
            quality = count / window
            # Quality 也需要 shift(1) 以保持对齐
            if shift:
                quality = quality.shift(1)
            return zscore, quality

        # 处理 inf 和 NaN：将 inf 替换为 NaN，然后保留 NaN（不填充，让下游处理）
        zscore = zscore.replace([np.inf, -np.inf], np.nan)

        return zscore

    @staticmethod
    def _trend_r2(prices: pd.Series, window: int = 20, *, lag: int = 0) -> pd.Series:
        """计算趋势R²特征（基于对数价格序列）"""
        log_price = np.log(prices.replace(0, np.nan)).ffill()

        def _compute_r2(series):
            if len(series) < 3:
                return 0.0
            try:
                x = np.arange(len(series))
                y = series.values
                slope, intercept = np.polyfit(x, y, 1)
                y_pred = slope * x + intercept
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
                return max(0.0, min(1.0, r2))
            except Exception:
                return 0.0

        r2_series = log_price.rolling(window=window, min_periods=3).apply(
            _compute_r2, raw=False
        )
        if lag == 0:
            return r2_series
        return r2_series.shift(lag)

    @staticmethod
    def _price_entropy(close: pd.Series, window: int = 50) -> pd.Series:
        """价格方向熵"""
        ret = close.pct_change().fillna(0.0)
        sign = np.sign(ret).replace(0, 1)

        def _entropy(x: np.ndarray) -> float:
            if len(x) == 0:
                return np.nan
            p_up = (x > 0).mean()
            p_dn = 1.0 - p_up
            eps = 1e-9
            return -(p_up * np.log2(p_up + eps) + p_dn * np.log2(p_dn + eps)) / 1.0

        return sign.rolling(window=window, min_periods=1).apply(_entropy, raw=True)

    @staticmethod
    def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Wilder-style RSI."""
        return BaselineFeatureEngineer.compute_rsi(series, period)

    @staticmethod
    def _rolling_skew(series: pd.Series, window: int) -> pd.Series:
        """Rolling skewness"""
        return series.rolling(window=window, min_periods=window).skew()

    @staticmethod
    def _ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    def _shift_feature(self, series: pd.Series, *, offset: int = 0) -> pd.Series:
        """Apply configurable lag to feature series"""
        total_shift = self.feature_shift + offset
        if total_shift == 0:
            return series
        return series.shift(total_shift)

    def _run_diagnostics(self, df: pd.DataFrame, feature_cols: List[str]) -> None:
        """收集诊断信息"""
        report: Dict[str, Dict[str, float]] = {}
        tol_zero = 1e-9
        tol_clip = 1e-3
        clip_bound = float(self.feature_clip_bound)
        near_clip = 0.9 * clip_bound

        for col in feature_cols:
            series = df[col].replace([np.inf, -np.inf], np.nan)
            total = len(series)
            if total == 0:
                continue

            metrics: Dict[str, float] = {}
            metrics["nan_ratio"] = float(series.isna().mean())
            valid = series.dropna()
            if valid.empty:
                report[col] = metrics
                continue

            metrics["zero_ratio"] = float(np.isclose(valid, 0.0, atol=tol_zero).mean())
            metrics["abs_ge_90pct_ratio"] = float((valid.abs() >= near_clip).mean())
            metrics["mean"] = float(valid.mean())
            metrics["std"] = float(valid.std())
            report[col] = metrics

        self.diagnostic_report = report

    def _add_advanced_derived_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """添加高级衍生特征"""
        df = data.copy()
        try:
            # 需要的基础特征
            if "bb_upper" not in df.columns or "atr" not in df.columns:
                return df

            # 1. BB Width相关
            if "bb_width" not in df.columns:
                df["bb_width"] = (df["bb_upper"] - df["bb_lower"]).abs()
            if "bb_width_normalized" not in df.columns:
                df["bb_width_normalized"] = df["bb_width"] / df["atr"].replace(
                    0, np.nan
                )
                df["bb_width_normalized"] = (
                    df["bb_width_normalized"]
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0)
                )

            # 2. Range ratio
            if "range_ratio_5bar" not in df.columns:
                if "hl" not in df.columns:
                    df["hl"] = df["high"] - df["low"]
                range_ratio_raw = df["hl"].rolling(5).mean() / df["hl"].rolling(
                    20
                ).mean().replace(0, np.nan)
                range_ratio_raw = range_ratio_raw.fillna(1)
                range_ratio_log = np.log1p(range_ratio_raw)
                range_ratio_mean = range_ratio_log.rolling(50, min_periods=5).mean()
                range_ratio_std = range_ratio_log.rolling(50, min_periods=5).std()
                df["range_ratio_5bar"] = (
                    (
                        (range_ratio_log - range_ratio_mean)
                        / range_ratio_std.replace(0, np.nan)
                    )
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                )

            # 3. Volatility reversal score
            if "volatility_reversal_score" not in df.columns:
                atr_mean = df["atr"].rolling(50).mean()
                atr_std = df["atr"].rolling(50).std()
                df["volatility_reversal_score"] = (
                    (df["atr"] - atr_mean) / atr_std.replace(0, np.nan)
                ).fillna(0)

            # 4. Price range symmetry
            if "price_range_symmetry" not in df.columns:
                price_range_symmetry_raw = (
                    self._shift_feature(df["high"]) - self._shift_feature(df["close"])
                ) / (
                    (
                        self._shift_feature(df["close"])
                        - self._shift_feature(df["low"])
                    ).replace(0, np.nan)
                )
                price_range_symmetry_raw = price_range_symmetry_raw.replace(
                    [np.inf, -np.inf], np.nan
                ).fillna(1)
                price_range_symmetry_log = np.log1p(
                    np.abs(price_range_symmetry_raw)
                ) * np.sign(price_range_symmetry_raw)
                price_range_symmetry_mean = price_range_symmetry_log.rolling(
                    50, min_periods=5
                ).mean()
                price_range_symmetry_std = price_range_symmetry_log.rolling(
                    50, min_periods=5
                ).std()
                df["price_range_symmetry"] = (
                    (
                        (price_range_symmetry_log - price_range_symmetry_mean)
                        / price_range_symmetry_std.replace(0, np.nan)
                    )
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                )

            # 5. Volume anomaly
            if "volume_anomaly" not in df.columns:
                volume_anomaly_raw = df["volume"] / df["volume"].ewm(
                    span=20, min_periods=10
                ).mean().replace(0, np.nan)
                volume_anomaly_raw = volume_anomaly_raw.fillna(1)
                volume_anomaly_log = np.log1p(volume_anomaly_raw)
                volume_anomaly_mean = volume_anomaly_log.rolling(
                    50, min_periods=10
                ).mean()
                volume_anomaly_std = volume_anomaly_log.rolling(
                    50, min_periods=10
                ).std()
                df["volume_anomaly"] = (
                    (
                        (volume_anomaly_log - volume_anomaly_mean)
                        / volume_anomaly_std.replace(0, np.nan)
                    )
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                )

            # 6. ROC and acceleration
            if "roc_5" not in df.columns:
                roc_raw = df["close"].pct_change(5)
                roc_mean = roc_raw.rolling(window=50, min_periods=5).mean()
                roc_std = roc_raw.rolling(window=50, min_periods=5).std()
                roc_std = roc_std.clip(lower=roc_raw.abs().quantile(0.01))
                df["roc_5"] = (
                    ((roc_raw - roc_mean) / roc_std.replace(0, np.nan))
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                )

            if "acceleration_3" not in df.columns:
                roc_3 = df["close"].pct_change(3)
                roc_3_mean = roc_3.rolling(window=50, min_periods=5).mean()
                roc_3_std = roc_3.rolling(window=50, min_periods=5).std()
                roc_3_std = roc_3_std.clip(lower=roc_3.abs().quantile(0.01))
                roc_3_norm = (
                    ((roc_3 - roc_3_mean) / roc_3_std.replace(0, np.nan))
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                )
                current = self._shift_feature(roc_3_norm)
                prev = self._shift_feature(roc_3_norm, offset=1)
                df["acceleration_3"] = current - prev

            # 7. Trend R²
            if "trend_r2_20" not in df.columns:
                df["trend_r2_20"] = self._trend_r2(
                    df["close"], window=20, lag=self.feature_shift
                )
            if "trend_r2_50" not in df.columns:
                df["trend_r2_50"] = self._trend_r2(
                    df["close"], window=50, lag=self.feature_shift
                )

            # 8. Slope consistency
            if "slope_consistency_score" not in df.columns:
                ema10 = df["close"].ewm(span=10).mean()
                ema20 = df["close"].ewm(span=20).mean()
                ema50 = df["close"].ewm(span=50).mean()
                slope10 = np.sign(ema10.diff())
                slope20 = np.sign(ema20.diff())
                slope50 = np.sign(ema50.diff())
                df["slope_consistency_score"] = (
                    (slope10 == slope20).astype(int)
                    + (slope20 == slope50).astype(int)
                    + (slope10 == slope50).astype(int)
                )

            # 9. Trend volatility alignment
            if "trend_volatility_alignment" not in df.columns:
                if "atr_percentile" in df.columns and "roc_5" in df.columns:
                    df["trend_volatility_alignment"] = np.sign(df["roc_5"]).fillna(
                        0
                    ) * df["atr_percentile"].fillna(0)

            # 10. Compression to breakout probability
            if "compression_to_breakout_prob" not in df.columns:
                if "compression_duration" in df.columns and "roc_5" in df.columns:
                    df["compression_to_breakout_prob"] = df[
                        "compression_duration"
                    ].fillna(0) * df["roc_5"].fillna(0)

        except Exception as e:
            print(f"      Warning: 高级衍生特征计算失败: {e}")
        return df

    def engineer_features(
        self,
        df: pd.DataFrame,
        *,
        fit: bool = True,
        required_features: Optional[set] = None,
    ) -> pd.DataFrame:
        """工程特征（合并后的版本，包含所有新特征）

        对于多资产数据，rolling 操作会按 symbol 分组进行，避免跨资产数据泄露。
        """
        if not {"open", "high", "low", "close", "volume"}.issubset(df.columns):
            raise ValueError(
                "DataFrame must contain open, high, low, close, volume columns"
            )

        # Check if this is multi-asset data (has _symbol or symbol column)
        has_symbol_col = "_symbol" in df.columns or "symbol" in df.columns
        symbol_col = (
            "_symbol"
            if "_symbol" in df.columns
            else ("symbol" if "symbol" in df.columns else None)
        )
        is_multi_asset = (
            has_symbol_col and df[symbol_col].nunique() > 1 if symbol_col else False
        )

        if is_multi_asset:
            # Multi-asset: process each symbol separately to avoid cross-asset leakage
            print(
                f"   🔒 Multi-asset detected ({df[symbol_col].nunique()} symbols): processing each symbol separately"
            )
            processed_groups = []
            for symbol in df[symbol_col].unique():
                symbol_mask = df[symbol_col] == symbol
                symbol_df = df[symbol_mask].copy()
                # Process this symbol's data
                symbol_processed = self._engineer_features_single_asset(
                    symbol_df, fit=fit, required_features=required_features
                )
                processed_groups.append(symbol_processed)
            # Combine all symbols
            data = pd.concat(processed_groups, axis=0).sort_index()
        else:
            # Single asset: process directly
            data = self._engineer_features_single_asset(
                df.copy(), fit=fit, required_features=required_features
            )

        return data

    def _engineer_features_single_asset(
        self,
        df: pd.DataFrame,
        *,
        fit: bool = True,
        required_features: Optional[set] = None,
    ) -> pd.DataFrame:
        """工程特征（单资产版本，内部方法）"""
        data = df.copy()

        # Core ATR（按需计算）
        if required_features is None or any(
            "atr" in f for f in required_features or [""]
        ):
            if "atr" not in data.columns:
                data["atr"] = self._compute_atr(data, window=14)

        # SR proximity using rolling swing proxies
        swing_win_short = 20
        swing_win_long = 60

        need_swing = required_features is None or any(
            "swing" in f or "sr_dist" in f for f in required_features or [""]
        )

        if need_swing:
            data["roll_high_s"] = (
                data["high"].rolling(swing_win_short, min_periods=1).max()
            )
            data["roll_low_s"] = (
                data["low"].rolling(swing_win_short, min_periods=1).min()
            )
            data["roll_high_l"] = (
                data["high"].rolling(swing_win_long, min_periods=1).max()
            )
            data["roll_low_l"] = (
                data["low"].rolling(swing_win_long, min_periods=1).min()
            )

            eps = 1e-9
            data["sr_dist_high_s"] = (data["close"] - data["roll_high_s"]) / (
                data["atr"] + eps
            )
            data["sr_dist_low_s"] = (data["close"] - data["roll_low_s"]) / (
                data["atr"] + eps
            )
            data["sr_dist_high_l"] = (data["close"] - data["roll_high_l"]) / (
                data["atr"] + eps
            )
            data["sr_dist_low_l"] = (data["close"] - data["roll_low_l"]) / (
                data["atr"] + eps
            )

        # VWAP (Volume Weighted Average Price) 价格偏离特征
        # 当价格通常"贴着 VWAP 走"时，直接使用"价格对 VWAP 的偏离度"作为特征更简洁、有效
        need_vwap = required_features is None or any(
            "vwap" in f for f in required_features or [""]
        )

        if need_vwap:
            # 计算滚动 VWAP（滚动窗口成交量加权平均价格）
            # VWAP = Σ(price × volume) / Σ(volume) over rolling window
            # 使用典型价格 (high + low + close) / 3 作为价格基准
            typical_price = (data["high"] + data["low"] + data["close"]) / 3.0

            # 计算价格×成交量
            price_volume = typical_price * data["volume"]

            # 滚动求和：价格×成交量的滚动和
            rolling_pv = price_volume.rolling(
                window=self.vwap_window, min_periods=1
            ).sum()
            # 滚动求和：成交量的滚动和
            rolling_vol = (
                data["volume"].rolling(window=self.vwap_window, min_periods=1).sum()
            )

            # 滚动 VWAP = 滚动价格×成交量总和 / 滚动成交量总和
            vwap = (rolling_pv / rolling_vol.replace(0, np.nan)).fillna(typical_price)

            # 保存原始值（用于计算归一化特征，但会被排除）
            data["vwap"] = vwap  # VWAP 原始值（有量纲，会被排除）

            eps = 1e-9
            # VWAP 价格偏离程度特征（无量纲）
            vwap_safe = vwap.replace(0, np.nan)
            # 价格相对 VWAP 的百分比偏离
            data["price_to_vwap_pct"] = (
                (data["close"] - vwap_safe) / vwap_safe
            ).replace([np.inf, -np.inf], np.nan)
            # 价格相对 VWAP 的 ATR 归一化偏离
            data["price_to_vwap_atr"] = (
                (data["close"] - vwap_safe) / (data["atr"] + eps)
            ).replace([np.inf, -np.inf], np.nan)
            # VWAP 偏离的 Z-score（相对于历史分布）
            vwap_deviation = (data["close"] - vwap_safe) / vwap_safe
            vwap_dev_mean = vwap_deviation.rolling(window=50, min_periods=10).mean()
            vwap_dev_std = vwap_deviation.rolling(window=50, min_periods=10).std()
            price_to_vwap_zscore = (
                (vwap_deviation - vwap_dev_mean) / vwap_dev_std.replace(0, np.nan)
            ).replace([np.inf, -np.inf], np.nan)
            data["price_to_vwap_zscore"] = price_to_vwap_zscore.shift(1)

            # VWAP 趋势（VWAP 的变化率，无量纲）
            # 使用 6 根 K 线的 VWAP 变化率作为趋势指标
            # 如果数据不足无法计算，保留 NaN 表示"无法计算"
            data["vwap_trend_6h"] = vwap.pct_change(periods=6)

        # Price-volume divergence signals
        if required_features is None or any(
            "divergence" in f for f in required_features or [""]
        ):
            # 使用 rsi（如果不存在则计算，默认 period=14）
            if "rsi" not in data.columns:
                data["rsi"] = self._compute_rsi(data["close"], period=14)

            # RSI 背离检测
            # 顶背离：价格创新高但 RSI 没有创新高（看跌信号）
            recent_high = data["close"].rolling(20, min_periods=5).max().ffill()
            recent_rsi_high = data["rsi"].rolling(20, min_periods=5).max().ffill()
            tol = 1e-8
            top_divergence_mask = (
                recent_high.notna()
                & recent_rsi_high.notna()
                & (data["close"] >= (recent_high - tol))
                & (data["rsi"] < (recent_rsi_high - tol))
            )

            # 底背离：价格创新低但 RSI 没有创新低（看涨信号）
            recent_low = data["close"].rolling(20, min_periods=5).min().ffill()
            recent_rsi_low = data["rsi"].rolling(20, min_periods=5).min().ffill()
            bottom_divergence_mask = (
                recent_low.notna()
                & recent_rsi_low.notna()
                & (data["close"] <= (recent_low + tol))
                & (data["rsi"] > (recent_rsi_low + tol))
            )

            # 合并背离信号：顶背离=-1（看跌），底背离=+1（看涨），无背离=0
            data["rsi_divergence"] = bottom_divergence_mask.astype(
                float
            ) - top_divergence_mask.astype(float)

            # 成交量背离检测
            # 顶背离：价格上涨但成交量下降（看跌信号）
            price_vs_past_up = (data["close"] > data["close"].shift(5)).fillna(False)
            avg_volume_20 = data["volume"].rolling(20, min_periods=5).mean()
            low_volume_mask = (data["volume"] < avg_volume_20).fillna(False)
            volume_top_div_mask = price_vs_past_up & low_volume_mask

            # 底背离：价格下跌但成交量上升（看涨信号）
            price_vs_past_down = (data["close"] < data["close"].shift(5)).fillna(False)
            high_volume_mask = (data["volume"] > avg_volume_20).fillna(False)
            volume_bottom_div_mask = price_vs_past_down & high_volume_mask

            # 合并背离信号：底背离=+1（看涨），顶背离=-1（看跌），无背离=0
            data["volume_divergence"] = volume_bottom_div_mask.astype(
                float
            ) - volume_top_div_mask.astype(float)

        # Compression features
        need_compression = required_features is None or any(
            "compression" in f or "atr_percentile" in f
            for f in required_features or [""]
        )

        if need_compression:
            atr_pct = self._rolling_percentile(
                data["atr"], window=self.percentile_window
            )
            data["atr_percentile"] = atr_pct

            volatility_regime_window = 200
            volatility_regime_threshold = 0.7
            atr_quantile_70 = (
                data["atr"]
                .rolling(window=volatility_regime_window, min_periods=1)
                .quantile(volatility_regime_threshold)
            )
            data["volatility_regime"] = (
                (data["atr"] > atr_quantile_70).astype(int).fillna(0)
            )

            returns = data["close"].pct_change()
            realized_skew = self._rolling_skew(returns, window=20)
            data["realized_skew"] = realized_skew

            vol5 = returns.rolling(5, min_periods=1).std()
            vol60 = returns.rolling(60, min_periods=1).std()
            volatility_ratio = (vol5 / vol60.replace(0, np.nan)).replace(
                [np.inf, -np.inf], np.nan
            )
            data["volatility_ratio"] = volatility_ratio

            # 【说明】：这不是数据泄漏问题，而是统计可靠性问题。
            # - min_periods 小会导致早期统计量不稳定（小样本噪声）
            # - 但 rolling() 只使用历史和当前数据，绝不包含未来
            # - 提高 min_periods 能降低虚假相关，因为剔除了高噪声样本
            # 使用 min_periods=window（最稳健），确保早期数据使用完整的统计量
            atr_mean_hist = (
                data["atr"]
                .rolling(self.percentile_window, min_periods=self.percentile_window)
                .mean()
            )
            eps = 1e-9
            # 【关键修复】：atr_compression_ratio 是比率特征，也需要 shift(1) 确保完全因果
            # 在 t 时刻使用的特征基于 t-1 及之前的数据计算
            atr_compression_ratio_raw = (atr_mean_hist / (data["atr"] + eps)).replace(
                [np.inf, -np.inf], np.nan
            )
            data["atr_compression_ratio"] = atr_compression_ratio_raw.shift(1)

            vol_pct = self._rolling_percentile(
                data["volume"].astype(float), window=self.percentile_window
            )
            data["volume_percentile"] = vol_pct

            data["price_entropy"] = self._price_entropy(data["close"], window=50)

            threshold = self.compression_threshold_pct
            # 如果百分位是 NaN，使用 0.5（中位数）作为默认值
            below = (data["atr_percentile"].fillna(0.5) <= threshold).astype(int)
            run = np.zeros(len(below), dtype=int)
            cnt = 0
            for i, v in enumerate(below.values):
                if v == 1:
                    cnt += 1
                else:
                    cnt = 0
                run[i] = cnt
            data["compression_duration"] = run

            short_window = 30
            data["pre_break_silence"] = (
                data["atr_percentile"].rolling(short_window, min_periods=1).mean()
                <= threshold
            ).astype(float)

            small = 20
            large = 100
            var_small = data["close"].rolling(small, min_periods=1).var()
            var_large = data["close"].rolling(large, min_periods=1).var()
            density = 1.0 - (var_small / (var_large + eps))
            data["internal_price_density"] = density.clip(0.0, 1.0)

            # 如果百分位是 NaN，使用 0.5（中位数）作为默认值
            # 【说明】：compression_confidence 使用了已 shift(1) 的 atr_percentile 和 volume_percentile
            # 所以它本身已经是因果的，不需要再次 shift
            atr_norm = data["atr_percentile"].fillna(0.5)
            vol_norm = data["volume_percentile"].fillna(0.5)
            dens_norm = data["internal_price_density"].fillna(0.0)
            data["compression_confidence"] = (
                0.5 * (1 - atr_norm) + 0.3 * (1 - vol_norm) + 0.2 * dens_norm
            )

        # Advanced derived features
        data = self._add_advanced_derived_features(data)

        # Time factors
        try:
            idx = data.index
            if hasattr(idx, "hour") and hasattr(idx, "dayofweek"):
                try:
                    if getattr(idx, "tz", None) is not None:
                        utc_idx = idx.tz_convert("UTC")
                        hour = utc_idx.hour.astype(int)
                        midnight_delta = (
                            utc_idx - utc_idx.normalize()
                        ).total_seconds() / 60.0
                    else:
                        hour = idx.hour.astype(int)
                        midnight_delta = (idx - idx.normalize()).total_seconds() / 60.0

                    data["hour_sin"] = np.sin(2 * np.pi * hour / 24)
                    data["hour_cos"] = np.cos(2 * np.pi * hour / 24)
                    data["Hour_of_Day"] = hour
                    data["minutes_since_reset"] = pd.Series(
                        midnight_delta, index=data.index
                    ).fillna(0.0)
                    data["Is_Weekend"] = (idx.dayofweek >= 5).astype(int)
                except Exception:
                    data["hour_sin"] = 0.0
                    data["hour_cos"] = 1.0
                    data["Hour_of_Day"] = 0
                    data["minutes_since_reset"] = 0.0
                    data["Is_Weekend"] = 0
            else:
                data["hour_sin"] = 0.0
                data["hour_cos"] = 1.0
                data["Hour_of_Day"] = 0
                data["minutes_since_reset"] = 0.0
                data["Is_Weekend"] = 0
        except Exception:
            data["hour_sin"] = 0.0
            data["hour_cos"] = 1.0
            data["Hour_of_Day"] = 0
            data["minutes_since_reset"] = 0.0
            data["Is_Weekend"] = 0

        # 添加新的无量纲特征
        # ZigZag 无量纲特征
        if required_features is None or any(
            "zz_" in f or "zigzag" in f for f in required_features or [""]
        ):
            data = BaselineFeatureEngineer.add_zigzag_dimensionless_features(
                data, required_features
            )

        # POC 和 HAL 无量纲特征（合并计算，避免重复计算 volume profile）
        if required_features is None or any(
            "poc" in f or "hal" in f for f in required_features or [""]
        ):
            data = BaselineFeatureEngineer.add_poc_hal_dimensionless_features(
                data, required_features
            )

        # Swing 无量纲特征
        if required_features is None or any(
            "swing" in f for f in required_features or [""]
        ):
            data = BaselineFeatureEngineer.add_swing_dimensionless_features(
                data, required_features
            )

        # 基础价格与量能相对变化特征
        if required_features is None or any(
            f in required_features or ""
            for f in [
                "ret_1h",
                "ret_4h",
                "ret_24h",
                "rv_4h",
                "rv_24h",
                "vol_ma_ratio",
                "vol_zscore",
            ]
        ):
            data = BaselineFeatureEngineer.add_price_volume_relative_features(
                data, required_features
            )

        # 常用衍生特征（包括多个窗口的滚动 Z-score）
        data = BaselineFeatureEngineer.add_common_derived_features(
            data, required_features, rolling_zscore_windows=self.rolling_zscore_windows
        )

        # 如果指定了required_features，只保留需要的特征
        if required_features is not None:
            data_cols = {
                "open",
                "high",
                "low",
                "close",
                "volume",
                "timestamp",
                "datetime",
            }
            cols_to_keep = [
                c
                for c in data.columns
                if c in data_cols
                or c in required_features
                or not pd.api.types.is_numeric_dtype(data[c])
            ]
            data = data[cols_to_keep]

        if self.enable_diagnostics:
            feature_cols = [
                c
                for c in data.columns
                if c
                not in ["open", "high", "low", "close", "volume", "timestamp", "symbol"]
            ]
            self._run_diagnostics(data, feature_cols)

        return data

    def save_scalers(self, path: str) -> None:
        """保存标准化器（仅保存配置参数，percentile 是实时计算的）"""
        import pickle

        scalers_data = {
            "percentile_window": self.percentile_window,
            "compression_threshold_pct": self.compression_threshold_pct,
            "vwap_window": self.vwap_window,
            "feature_clip_bound": self.feature_clip_bound,
            "rolling_zscore_windows": self.rolling_zscore_windows,
        }
        with open(path, "wb") as f:
            pickle.dump(scalers_data, f)
        print(f"✅ Baseline scalers saved to: {path}")

    def load_scalers(self, path: str) -> None:
        """加载标准化器（仅加载配置参数）"""
        import pickle

        with open(path, "rb") as f:
            scalers_data = pickle.load(f)
        # 向后兼容：如果存在旧的 quantiles 字段，忽略它们
        self.percentile_window = scalers_data.get("percentile_window", 288)
        self.compression_threshold_pct = scalers_data.get(
            "compression_threshold_pct", 0.2
        )
        self.vwap_window = scalers_data.get("vwap_window", 160)
        self.feature_clip_bound = scalers_data.get("feature_clip_bound", 10.0)
        # Backward compatibility: support both old single window and new multiple windows
        if "rolling_zscore_windows" in scalers_data:
            self.rolling_zscore_windows = scalers_data["rolling_zscore_windows"]
        elif "rolling_zscore_window" in scalers_data:
            # Migrate from old single window to new multiple windows
            old_window = scalers_data["rolling_zscore_window"]
            self.rolling_zscore_windows = [
                50,
                old_window,
                500,
            ]  # Add short and long term
        else:
            self.rolling_zscore_windows = [50, 288, 500]  # Default
        print(f"✅ Baseline scalers loaded from: {path}")


# ============================================================================
# 便捷函数（保持向后兼容）
# ============================================================================


def engineer_baseline_features(
    df: pd.DataFrame,
    engineer: Optional[BaselineFeatureEngineer] = None,
    *,
    fit: bool = True,
    required_features: Optional[set] = None,
) -> Tuple[pd.DataFrame, BaselineFeatureEngineer]:
    """工程 baseline 特征"""
    if engineer is None:
        engineer = BaselineFeatureEngineer()
    out = engineer.engineer_features(df, fit=fit, required_features=required_features)
    return out, engineer


def create_binary_labels_baseline(
    df: pd.DataFrame, *, forward_bars: int = 3, threshold: float = 0.005
) -> pd.DataFrame:
    """创建二分类标签"""
    df = df.copy()
    df["future_return"] = df["close"].shift(-forward_bars) / df["close"] - 1
    df["binary_signal"] = (df["future_return"] > threshold).astype(int)
    df["signal"] = df["binary_signal"]
    return df


def get_baseline_feature_columns(df: pd.DataFrame) -> List[str]:
    """获取 baseline 特征列"""
    exclude = {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "signal",
        "binary_signal",
        "future_return",
        # Note: _symbol is now included as a categorical feature (not excluded)
        # This allows the model to learn both shared patterns and asset-specific behavior
        # 排除原始布林带值（有量纲），保留归一化的 bb_position 和 bb_width
        "bb_upper",
        "bb_lower",
        "bb_middle",
        # 排除原始 ATR（有量纲），保留归一化的 atr_normalized
        "atr",
        # 排除原始 zigzag（有量纲），保留归一化的 zigzag 特征
        "zigzag",
        # 排除 returns 和 log_returns（虽然无量纲，但不同资产分布差异大）
        # 它们主要用于计算其他特征（如 volatility），不作为最终特征
        "returns",
        "log_returns",
        # 排除 price_change（有量纲），保留归一化的特征
        "price_change",
        # 排除原始均线值（有量纲），保留归一化的 _pct_close 和 _ratio 特征
        "sma_5",
        "sma_10",
        "sma_20",
        "ema_5",
        "ema_10",
        "ema_20",
        "ema_50",
        "wma_20",
        # 注意：momentum_5/10/20 是百分比（pct_change），已经是无量纲的，保留
        # 排除原始成交量值（有量纲），保留归一化的 volume_ratio, vol_ma_ratio, vol_zscore 等
        "volume_sma_20",
        "volume_sma",  # 如果存在的话
        # 排除原始 MACD 值（有量纲），保留归一化的 macd_normalized
        "macd",
        "macd_signal",
        "macd_histogram",
        # 排除原始 POC/HAL 价格值（有量纲），保留归一化的 price_to_poc_pct 等
        "poc",
        "hal_high",
        "hal_low",
        "hal_mid",
        # 排除原始 VWAP 值（有量纲），保留归一化的 price_to_vwap_* 特征
        "vwap",
        # 排除滚动高低点（有量纲），保留归一化的 swing_*_pct_close 和 sr_dist_* 特征
        "roll_high_s",
        "roll_low_s",
        "roll_high_l",
        "roll_low_l",
        # 排除 volatility（虽然基于收益率，但不同资产分布差异大）
        "volatility",
    }
    exclude.update(
        [
            col
            for col in df.columns
            if (
                col.startswith("signal_")
                or col.startswith("binary_signal_")
                or col.startswith("future_return_")
            )
        ]
    )
    return [c for c in df.columns if c not in exclude]


# ============================================================================
# 导出（保持向后兼容）
# ============================================================================

__all__ = [
    # BaselineFeatureEngineer 类（包含所有静态方法和特征添加方法）
    "BaselineFeatureEngineer",
    # 便捷函数
    "engineer_baseline_features",
    "get_baseline_feature_columns",
    "create_binary_labels_baseline",
]
