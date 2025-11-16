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

try:
    import talib
except ImportError:
    talib = None

# ============================================================================
# 基础指标计算函数（从 base_indicators.py）
# ============================================================================


def _maybe_talib_sma(series: pd.Series, period: int) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce").astype(float)
    if talib is not None:
        values = talib.SMA(series.values, timeperiod=period)
        return pd.Series(values, index=series.index)
    return series.rolling(window=period, min_periods=period).mean()


def _maybe_talib_std(series: pd.Series, period: int) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce").astype(float)
    if talib is not None:
        values = talib.STDDEV(series.values, timeperiod=period, nbdev=1)
        return pd.Series(values, index=series.index)
    return series.rolling(window=period, min_periods=period).std()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """计算相对强弱指数 (RSI)."""
    series = pd.to_numeric(series, errors="coerce").astype(float)
    if talib is not None:
        values = talib.RSI(series.values, timeperiod=period)
        return pd.Series(values, index=series.index)
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(series: pd.Series,
                 fast: int = 12,
                 slow: int = 26,
                 signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """计算MACD指标."""
    series = pd.to_numeric(series, errors="coerce").astype(float)
    if talib is not None:
        macd_line, signal_line, histogram = talib.MACD(series.values,
                                                       fastperiod=fast,
                                                       slowperiod=slow,
                                                       signalperiod=signal)
        index = series.index
        return (
            pd.Series(macd_line, index=index),
            pd.Series(signal_line, index=index),
            pd.Series(histogram, index=index),
        )
    ema_fast = series.ewm(span=fast).mean()
    ema_slow = series.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger_bands(
        series: pd.Series,
        period: int = 20,
        std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """计算布林带."""
    series = pd.to_numeric(series, errors="coerce").astype(float)
    if talib is not None:
        upper, middle, lower = talib.BBANDS(series.values,
                                            timeperiod=period,
                                            nbdevup=std_dev,
                                            nbdevdn=std_dev,
                                            matype=0)
        index = series.index
        return (
            pd.Series(upper, index=index),
            pd.Series(middle, index=index),
            pd.Series(lower, index=index),
        )
    middle_band = series.rolling(window=period).mean()
    std_series = series.rolling(window=period).std()
    upper_band = middle_band + (std_dev * std_series)
    lower_band = middle_band - (std_dev * std_series)
    return upper_band, middle_band, lower_band


def compute_atr(high: pd.Series,
                low: pd.Series,
                close: pd.Series,
                period: int = 14) -> pd.Series:
    """计算平均真实波幅 (ATR)."""
    high = pd.to_numeric(high, errors="coerce").astype(float)
    low = pd.to_numeric(low, errors="coerce").astype(float)
    close = pd.to_numeric(close, errors="coerce").astype(float)
    if talib is not None:
        atr_values = talib.ATR(high.values,
                               low.values,
                               close.values,
                               timeperiod=period)
        return pd.Series(atr_values, index=high.index)
    tr0 = abs(high - low)
    tr1 = abs(high - close.shift())
    tr2 = abs(low - close.shift())
    tr = pd.DataFrame({"tr0": tr0, "tr1": tr1, "tr2": tr2}).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def compute_zigzag(high: pd.Series,
                   low: pd.Series,
                   threshold: float = 0.05) -> pd.Series:
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
    turn_points = (zigzag_diff * zigzag_diff.shift(1)
                   < 0) | (zigzag_diff != 0) & (zigzag_diff.shift(1) == 0)

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


def compute_poc(high: pd.Series,
                low: pd.Series,
                volume: pd.Series,
                window: int = 20,
                bins: int = 50) -> pd.Series:
    """
    计算 POC (Point of Control) - 成交量最大对应的价格
    
    Args:
        high: 最高价序列
        low: 最低价序列
        volume: 成交量序列
        window: 滚动窗口大小
        bins: 价格分档数量
    
    Returns:
        POC 价格序列
    """
    poc = pd.Series(index=high.index, dtype=float)

    for i in range(window, len(high)):
        window_high = high.iloc[i - window:i].max()
        window_low = low.iloc[i - window:i].min()

        if window_high <= window_low:
            poc.iloc[i] = (high.iloc[i] + low.iloc[i]) / 2
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

        # 找到成交量最大的分档
        max_vol_idx = np.argmax(bin_volumes)
        poc.iloc[i] = (price_bins[max_vol_idx] +
                       price_bins[max_vol_idx + 1]) / 2

    poc = poc.ffill()
    return poc


def compute_hal(high: pd.Series,
                low: pd.Series,
                window: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    计算 HAL (High Average Low)
    
    Returns:
        (hal_high, hal_low, hal_mid): HAL 高点、低点、中点
    """
    hal_high = high.rolling(window=window, min_periods=1).mean()
    hal_low = low.rolling(window=window, min_periods=1).mean()
    hal_mid = (hal_high + hal_low) / 2.0
    return hal_high, hal_low, hal_mid


# ============================================================================
# 优化的基础指标添加函数（支持按需计算）
# ============================================================================

BASIC_INDICATOR_COLUMNS = {
    "rsi",
    "macd",
    "macd_signal",
    "macd_histogram",
    "bb_upper",
    "bb_middle",
    "bb_lower",
    "atr",
    "zigzag",
    "price_change",
    "volatility",
    "volume_sma",
    "volume_ratio",
}


def add_basic_indicators(
        df: pd.DataFrame,
        required_features: Optional[set] = None) -> pd.DataFrame:
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
            result["rsi"] = compute_rsi(result["close"])

    # 按需计算 MACD
    need_macd = required_features is None or any(
        f in required_features
        for f in ["macd", "macd_signal", "macd_histogram"])
    if need_macd and "macd" not in result.columns:
        try:
            macd_line, signal_line, histogram = compute_macd(result["close"])
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
        f in required_features for f in ["bb_upper", "bb_middle", "bb_lower"])
    if need_bb and "bb_upper" not in result.columns:
        try:
            upper_band, middle_band, lower_band = compute_bollinger_bands(
                result["close"])
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
                result["atr"] = compute_atr(result["high"], result["low"],
                                            result["close"])
            except Exception as e:
                print(f"Warning: Error computing ATR: {e}")
                result["atr"] = 0

    # 按需计算 ZigZag
    if required_features is None or "zigzag" in required_features:
        if "zigzag" not in result.columns:
            try:
                result["zigzag"] = compute_zigzag(result["high"],
                                                  result["low"])
            except Exception as e:
                print(f"Warning: Error computing ZigZag: {e}")
                result["zigzag"] = 0

    # 按需计算价格变化和波动率
    if required_features is None or "price_change" in required_features:
        if "price_change" not in result.columns:
            result["price_change"] = result["close"].pct_change()

    if required_features is None or "volatility" in required_features:
        if "volatility" not in result.columns:
            result["volatility"] = _maybe_talib_std(result["price_change"], 14)

    # 按需计算成交量特征
    if required_features is None or "volume_sma" in required_features or "volume_ratio" in required_features:
        if "volume_sma" not in result.columns:
            result["volume_sma"] = _maybe_talib_sma(result["volume"], 20)
        if "volume_ratio" not in result.columns:
            result["volume_ratio"] = result["volume"] / result[
                "volume_sma"].replace(0, np.nan)

    # 填充NaN值
    feature_cols = [
        col for col in result.columns
        if col not in ["open", "high", "low", "close", "volume"]
    ]
    for col in feature_cols:
        result[col] = result[col].fillna(0)

    return result


def ensure_basic_indicators(
        df: pd.DataFrame,
        required_features: Optional[set] = None) -> pd.DataFrame:
    """确保基础指标存在（优化版：支持按需计算）"""
    if df.empty:
        return df

    # 检查需要的指标是否都已存在
    if required_features:
        missing = required_features - set(df.columns)
        if not missing:
            return df
    elif BASIC_INDICATOR_COLUMNS.issubset(df.columns):
        return df

    return add_basic_indicators(df, required_features)


# ============================================================================
# 新增：ZigZag 无量纲特征
# ============================================================================


def add_zigzag_dimensionless_features(
        df: pd.DataFrame,
        required_features: Optional[set] = None) -> pd.DataFrame:
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
        if required_features and any("zz_" in f or "zigzag" in f
                                     for f in required_features):
            result = ensure_basic_indicators(result, {"zigzag"})
        else:
            return result

    close = result["close"].replace(0, np.nan)
    zigzag = result["zigzag"]

    # 提取 ZigZag 高点和低点
    zz_high, zz_low = compute_zigzag_high_low(zigzag)

    # 1. 当前价格距离最近 ZigZag 高/低点的相对距离
    if required_features is None or "price_to_zz_high_pct" in required_features:
        if "price_to_zz_high_pct" not in result.columns:
            result["price_to_zz_high_pct"] = ((zz_high - close) /
                                              close).replace(
                                                  [np.inf, -np.inf],
                                                  np.nan).fillna(0.0)

    if required_features is None or "price_to_zz_low_pct" in required_features:
        if "price_to_zz_low_pct" not in result.columns:
            result["price_to_zz_low_pct"] = ((close - zz_low) / close).replace(
                [np.inf, -np.inf], np.nan).fillna(0.0)

    # 2. ZigZag 波幅（相对）
    if required_features is None or "zz_amplitude_pct" in required_features:
        if "zz_amplitude_pct" not in result.columns:
            zz_low_safe = zz_low.replace(0, np.nan)
            result["zz_amplitude_pct"] = ((zz_high - zz_low) /
                                          zz_low_safe).replace(
                                              [np.inf, -np.inf],
                                              np.nan).fillna(0.0)

    # 3. ZigZag 持续时间（从上一个转折点至今的 bar 数）
    if required_features is None or "zz_duration" in required_features:
        if "zz_duration" not in result.columns:
            zigzag_diff = zigzag.diff()
            turn_points = (zigzag_diff * zigzag_diff.shift(1) < 0) | (
                (zigzag_diff != 0) & (zigzag_diff.shift(1) == 0))

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
            window = 5
            zz_slope_raw = zigzag.diff(window) / window

            # 归一化：如果有 ATR，用 ATR；否则用 close
            if "atr" in result.columns:
                atr_safe = result["atr"].replace(0, np.nan)
                result["zz_slope"] = (zz_slope_raw / atr_safe).replace(
                    [np.inf, -np.inf], np.nan).fillna(0.0)
            else:
                result["zz_slope"] = (zz_slope_raw / close).replace(
                    [np.inf, -np.inf], np.nan).fillna(0.0)

    return result


# ============================================================================
# 新增：POC 无量纲特征
# ============================================================================


def add_poc_dimensionless_features(df: pd.DataFrame,
                                   required_features: Optional[set] = None,
                                   poc_window: int = 20) -> pd.DataFrame:
    """
    添加 POC (Point of Control) 相关的无量纲特征
    
    新增特征：
    - price_to_poc_pct: 当前价格到 POC 的相对距离
    - poc_position_ratio: POC 在价格区间中的位置（0-1）
    - poc_volume_ratio: POC 位置的成交量占比
    """
    if df.empty:
        return df

    result = df.copy()

    # 计算 POC
    if "poc" not in result.columns:
        if required_features and any("poc" in f for f in required_features):
            result["poc"] = compute_poc(result["high"],
                                        result["low"],
                                        result["volume"],
                                        window=poc_window)
        else:
            return result

    close = result["close"].replace(0, np.nan)
    poc = result["poc"]
    high = result["high"]
    low = result["low"]

    # 1. 当前价格到 POC 的相对距离
    if required_features is None or "price_to_poc_pct" in required_features:
        if "price_to_poc_pct" not in result.columns:
            result["price_to_poc_pct"] = ((poc - close) / close).replace(
                [np.inf, -np.inf], np.nan).fillna(0.0)

    # 2. POC 在价格区间中的位置（0-1）
    if required_features is None or "poc_position_ratio" in required_features:
        if "poc_position_ratio" not in result.columns:
            price_range = (high - low).replace(0, np.nan)
            result["poc_position_ratio"] = ((poc - low) / price_range).replace(
                [np.inf, -np.inf], np.nan).fillna(0.5).clip(0.0, 1.0)

    # 3. POC 成交量占比（简化版：使用滚动窗口内的成交量分布）
    if required_features is None or "poc_volume_ratio" in required_features:
        if "poc_volume_ratio" not in result.columns:
            # 简化计算：POC 附近的成交量占比
            volume_window = result["volume"].rolling(window=poc_window,
                                                     min_periods=1).sum()
            # 这里简化处理，实际应该计算 POC 价格档的成交量
            result["poc_volume_ratio"] = (
                result["volume"] / volume_window.replace(0, np.nan)).replace(
                    [np.inf, -np.inf], np.nan).fillna(0.0)

    return result


# ============================================================================
# 新增：HAL 无量纲特征
# ============================================================================


def add_hal_dimensionless_features(df: pd.DataFrame,
                                   required_features: Optional[set] = None,
                                   hal_window: int = 20) -> pd.DataFrame:
    """
    添加 HAL (High Average Low) 相关的无量纲特征
    
    新增特征：
    - price_to_hal_high_pct: 当前价格到 HAL 高点的相对距离
    - price_to_hal_low_pct: 当前价格到 HAL 低点的相对距离
    - price_to_hal_mid_pct: 当前价格到 HAL 中点的相对距离
    - hal_bandwidth_pct: HAL 带宽（相对）
    """
    if df.empty:
        return df

    result = df.copy()

    # 计算 HAL
    if "hal_high" not in result.columns:
        if required_features and any("hal" in f for f in required_features):
            hal_high, hal_low, hal_mid = compute_hal(result["high"],
                                                     result["low"],
                                                     window=hal_window)
            result["hal_high"] = hal_high
            result["hal_low"] = hal_low
            result["hal_mid"] = hal_mid
        else:
            return result

    close = result["close"].replace(0, np.nan)
    hal_high = result["hal_high"]
    hal_low = result["hal_low"]
    hal_mid = result["hal_mid"]

    # 1. 当前价格到 HAL 的相对距离
    if required_features is None or "price_to_hal_high_pct" in required_features:
        if "price_to_hal_high_pct" not in result.columns:
            result["price_to_hal_high_pct"] = ((hal_high - close) /
                                               close).replace(
                                                   [np.inf, -np.inf],
                                                   np.nan).fillna(0.0)

    if required_features is None or "price_to_hal_low_pct" in required_features:
        if "price_to_hal_low_pct" not in result.columns:
            result["price_to_hal_low_pct"] = ((close - hal_low) /
                                              close).replace(
                                                  [np.inf, -np.inf],
                                                  np.nan).fillna(0.0)

    if required_features is None or "price_to_hal_mid_pct" in required_features:
        if "price_to_hal_mid_pct" not in result.columns:
            result["price_to_hal_mid_pct"] = ((hal_mid - close) /
                                              close).replace(
                                                  [np.inf, -np.inf],
                                                  np.nan).fillna(0.0)

    # 2. HAL 带宽（相对）
    if required_features is None or "hal_bandwidth_pct" in required_features:
        if "hal_bandwidth_pct" not in result.columns:
            hal_mid_safe = hal_mid.replace(0, np.nan)
            result["hal_bandwidth_pct"] = ((hal_high - hal_low) /
                                           hal_mid_safe).replace(
                                               [np.inf, -np.inf],
                                               np.nan).fillna(0.0)

    return result


# ============================================================================
# 新增：Swing High/Low 无量纲特征
# ============================================================================


def add_swing_dimensionless_features(df: pd.DataFrame,
                                     required_features: Optional[set] = None,
                                     swing_win_short: int = 20,
                                     swing_win_long: int = 60) -> pd.DataFrame:
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
            result["roll_high_s"] = result["high"].rolling(
                swing_win_short, min_periods=1).max()
            result["roll_low_s"] = result["low"].rolling(swing_win_short,
                                                         min_periods=1).min()
            result["roll_high_l"] = result["high"].rolling(
                swing_win_long, min_periods=1).max()
            result["roll_low_l"] = result["low"].rolling(swing_win_long,
                                                         min_periods=1).min()
        else:
            return result

    # 1. Swing High/Low 相对收盘价的比率
    if required_features is None or "swing_high_pct_close" in required_features:
        if "swing_high_pct_close" not in result.columns:
            result["swing_high_pct_close"] = ((result["roll_high_s"] - close) /
                                              close).replace(
                                                  [np.inf, -np.inf],
                                                  np.nan).fillna(0.0)

    if required_features is None or "swing_low_pct_close" in required_features:
        if "swing_low_pct_close" not in result.columns:
            result["swing_low_pct_close"] = ((close - result["roll_low_s"]) /
                                             close).replace([np.inf, -np.inf],
                                                            np.nan).fillna(0.0)

    # 2. Swing 波幅（相对）
    if required_features is None or "swing_amplitude_pct" in required_features:
        if "swing_amplitude_pct" not in result.columns:
            roll_low_s_safe = result["roll_low_s"].replace(0, np.nan)
            result["swing_amplitude_pct"] = (
                (result["roll_high_s"] - result["roll_low_s"]) /
                roll_low_s_safe).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return result


# ============================================================================
# 新增：基础价格与量能相对变化特征
# ============================================================================


def add_price_volume_relative_features(
        df: pd.DataFrame,
        required_features: Optional[set] = None) -> pd.DataFrame:
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
            result["ret_1h"] = np.log(close /
                                      close.shift(periods_1h)).fillna(0.0)

    if required_features is None or "ret_4h" in required_features:
        if "ret_4h" not in result.columns:
            result["ret_4h"] = np.log(close /
                                      close.shift(periods_4h)).fillna(0.0)

    if required_features is None or "ret_24h" in required_features:
        if "ret_24h" not in result.columns:
            result["ret_24h"] = np.log(close /
                                       close.shift(periods_24h)).fillna(0.0)

    # 2. 已实现波动率（基于 ret_1h）
    if required_features is None or "rv_4h" in required_features:
        if "rv_4h" not in result.columns and "ret_1h" in result.columns:
            result["rv_4h"] = result["ret_1h"].rolling(
                window=periods_4h // periods_1h,
                min_periods=1).std().fillna(0.0)

    if required_features is None or "rv_24h" in required_features:
        if "rv_24h" not in result.columns and "ret_1h" in result.columns:
            result["rv_24h"] = result["ret_1h"].rolling(
                window=periods_24h // periods_1h,
                min_periods=1).std().fillna(0.0)

    # 3. 成交量异常度
    if required_features is None or "vol_ma_ratio" in required_features:
        if "vol_ma_ratio" not in result.columns:
            vol_ma = volume.rolling(window=periods_24h, min_periods=1).mean()
            result["vol_ma_ratio"] = (volume /
                                      vol_ma.replace(0, np.nan)).replace(
                                          [np.inf, -np.inf],
                                          np.nan).fillna(1.0)

    if required_features is None or "vol_zscore" in required_features:
        if "vol_zscore" not in result.columns:
            vol_mean = volume.rolling(window=periods_24h, min_periods=1).mean()
            vol_std = volume.rolling(window=periods_24h, min_periods=1).std()
            result["vol_zscore"] = ((volume - vol_mean) /
                                    vol_std.replace(0, np.nan)).replace(
                                        [np.inf, -np.inf], np.nan).fillna(0.0)

    return result


# ============================================================================
# 优化的衍生特征函数（支持按需计算）
# ============================================================================

COMMON_DERIVED_COLUMNS = {
    "returns",
    "log_returns",
    "price_change",
    "volatility",
    "bb_position",
    "bb_width",
    "rsi_normalized",
    "macd_normalized",
    "atr_normalized",
    "momentum_5",
    "momentum_10",
    "momentum_20",
    "sma_5",
    "sma_10",
    "sma_20",
    "sma_5_pct_close",
    "sma_10_pct_close",
    "sma_20_pct_close",
    "ema_5_pct_close",
    "ema_10_pct_close",
    "ema_20_pct_close",
    "ema_50_pct_close",
    "wma_20_pct_close",
    "sma_ratio_5_20",
    "sma_ratio_10_20",
    "volume_sma_20",
}


def add_common_derived_features(
        df: pd.DataFrame,
        required_features: Optional[set] = None) -> pd.DataFrame:
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
        result = ensure_basic_indicators(result, needed_basic)

    # 只在需要时计算特征
    if not required_features or "returns" in required_features:
        if "returns" not in result.columns:
            result["returns"] = close.pct_change()

    if not required_features or "log_returns" in required_features:
        if "log_returns" not in result.columns:
            shifted = close.shift(1).replace(0, np.nan)
            result["log_returns"] = np.log(close / shifted).fillna(0.0)

    if not required_features or "price_change" in required_features:
        if "price_change" not in result.columns:
            result["price_change"] = close.diff()

    if not required_features or "volatility" in required_features:
        if "volatility" not in result.columns:
            if "returns" in result.columns:
                result["volatility"] = _maybe_talib_std(result["returns"], 20)
            else:
                result["volatility"] = _maybe_talib_std(close.pct_change(), 20)

    # BB 相关特征
    if {"bb_upper", "bb_lower"}.issubset(result.columns):
        if not required_features or "bb_position" in required_features:
            if "bb_position" not in result.columns:
                denom = (result["bb_upper"] - result["bb_lower"]).replace(
                    0, np.nan)
                result["bb_position"] = ((close - result["bb_lower"]) /
                                         denom).replace([np.inf, -np.inf],
                                                        np.nan).fillna(0.5)

        if not required_features or "bb_width" in required_features:
            if "bb_width" not in result.columns:
                result["bb_width"] = ((result["bb_upper"] -
                                       result["bb_lower"]).abs()).fillna(0.0)

    # 归一化特征
    if not required_features or "rsi_normalized" in required_features:
        if "rsi_normalized" not in result.columns and "rsi" in result.columns:
            result["rsi_normalized"] = ((result["rsi"] - 50) / 50).fillna(0.0)

    if not required_features or "macd_normalized" in required_features:
        if "macd_normalized" not in result.columns and "macd" in result.columns:
            result["macd_normalized"] = (result["macd"] / close).replace(
                [np.inf, -np.inf], np.nan).fillna(0.0)

    if not required_features or "atr_normalized" in required_features:
        if "atr_normalized" not in result.columns and "atr" in result.columns:
            result["atr_normalized"] = (result["atr"] / close).replace(
                [np.inf, -np.inf], np.nan).fillna(0.0)

    # Momentum features
    for period in [5, 10, 20]:
        momentum_col = f"momentum_{period}"
        if not required_features or momentum_col in required_features:
            if momentum_col not in result.columns:
                result[momentum_col] = close.pct_change(period).fillna(0.0)

    # SMA features
    sma_map = {5: "sma_5", 10: "sma_10", 20: "sma_20"}
    for window, col_name in sma_map.items():
        if not required_features or col_name in required_features:
            if col_name not in result.columns:
                result[col_name] = _maybe_talib_sma(close,
                                                    window).fillna(close)

    # SMA/EMA 相对 close 的百分比
    close_safe = close.replace(0, np.nan)
    for col_name in [
            "sma_5", "sma_10", "sma_20", "ema_5", "ema_10", "ema_20", "ema_50",
            "wma_20"
    ]:
        pct_col = f"{col_name}_pct_close"
        if not required_features or pct_col in required_features:
            if col_name in result.columns and pct_col not in result.columns:
                result[pct_col] = ((result[col_name] / close_safe -
                                    1.0)).replace([np.inf, -np.inf],
                                                  np.nan).fillna(0.0)

    # SMA ratios
    if not required_features or "sma_ratio_5_20" in required_features:
        if {"sma_5", "sma_20"}.issubset(
                result.columns) and "sma_ratio_5_20" not in result.columns:
            result["sma_ratio_5_20"] = (
                result["sma_5"] / result["sma_20"].replace(0, np.nan)).replace(
                    [np.inf, -np.inf], np.nan).fillna(1.0)

    if not required_features or "sma_ratio_10_20" in required_features:
        if {"sma_10", "sma_20"}.issubset(
                result.columns) and "sma_ratio_10_20" not in result.columns:
            result["sma_ratio_10_20"] = (
                result["sma_10"] /
                result["sma_20"].replace(0, np.nan)).replace(
                    [np.inf, -np.inf], np.nan).fillna(1.0)

    # Volume features
    if not required_features or "volume_sma_20" in required_features:
        if "volume_sma_20" not in result.columns:
            result["volume_sma_20"] = _maybe_talib_sma(
                result["volume"], 20).fillna(result["volume"])

    if not required_features or "volume_ratio" in required_features:
        if "volume_ratio" not in result.columns:
            if "volume_sma_20" in result.columns:
                denom = result["volume_sma_20"].replace(0, np.nan)
                result["volume_ratio"] = (result["volume"] / denom).replace(
                    [np.inf, -np.inf], np.nan).fillna(1.0)

    # Final cleanup
    for col in COMMON_DERIVED_COLUMNS:
        if col in result.columns:
            result[col] = result[col].replace([np.inf, -np.inf],
                                              np.nan).fillna(0)

    return result


# ============================================================================
# BaselineFeatureEngineer 类（从 baseline_feature_engineering.py 合并）
# ============================================================================


class BaselineFeatureEngineer:
    """Baseline SR and compression features (合并后的版本)"""

    def __init__(self,
                 percentile_window: int = 288,
                 compression_threshold_pct: float = 0.2,
                 feature_shift: int = 0,
                 feature_clip_bound: float = 10.0,
                 enable_diagnostics: bool = False) -> None:
        self.percentile_window = percentile_window
        self.compression_threshold_pct = compression_threshold_pct
        self.feature_shift = feature_shift
        self.feature_clip_bound = float(feature_clip_bound)
        self.enable_diagnostics = enable_diagnostics
        self.diagnostic_report: Dict[str, Dict[str, float]] = {}
        self._fitted_atr_quantiles: Optional[np.ndarray] = None
        self._fitted_vol_quantiles: Optional[np.ndarray] = None

        if self.feature_clip_bound <= 0:
            raise ValueError("feature_clip_bound must be positive")

    @staticmethod
    def _compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
        """计算 ATR（与 compute_atr 函数保持一致）"""
        return compute_atr(df["high"], df["low"], df["close"], window=window)

    @staticmethod
    def _rolling_percentile(series: pd.Series, window: int) -> pd.Series:
        """滚动百分位排名"""

        def _rank(x: np.ndarray) -> float:
            if len(x) <= 1 or not np.isfinite(x[-1]):
                return np.nan
            last = x[-1]
            arr = x[np.isfinite(x)]
            if len(arr) == 0:
                return np.nan
            return (arr <= last).sum() / float(len(arr))

        return series.rolling(window=window, min_periods=1).apply(_rank,
                                                                  raw=True)

    @staticmethod
    def _trend_r2(prices: pd.Series,
                  window: int = 20,
                  *,
                  lag: int = 0) -> pd.Series:
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
                ss_res = np.sum((y - y_pred)**2)
                ss_tot = np.sum((y - np.mean(y))**2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
                return max(0.0, min(1.0, r2))
            except Exception:
                return 0.0

        r2_series = log_price.rolling(window=window,
                                      min_periods=3).apply(_compute_r2,
                                                           raw=False)
        if lag == 0:
            return r2_series.fillna(0.0)
        return r2_series.shift(lag).fillna(0.0)

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
            return -(p_up * np.log2(p_up + eps) +
                     p_dn * np.log2(p_dn + eps)) / 1.0

        return sign.rolling(window=window, min_periods=1).apply(_entropy,
                                                                raw=True)

    @staticmethod
    def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Wilder-style RSI."""
        return compute_rsi(series, period)

    @staticmethod
    def _rolling_skew(series: pd.Series, window: int) -> pd.Series:
        """Rolling skewness"""
        return series.rolling(window=window, min_periods=window).skew()

    @staticmethod
    def _ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    def _shift_feature(self,
                       series: pd.Series,
                       *,
                       offset: int = 0) -> pd.Series:
        """Apply configurable lag to feature series"""
        total_shift = self.feature_shift + offset
        if total_shift == 0:
            return series
        return series.shift(total_shift)

    def _run_diagnostics(self, df: pd.DataFrame,
                         feature_cols: List[str]) -> None:
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

            metrics["zero_ratio"] = float(
                np.isclose(valid, 0.0, atol=tol_zero).mean())
            metrics["abs_ge_90pct_ratio"] = float((valid.abs()
                                                   >= near_clip).mean())
            metrics["mean"] = float(valid.mean())
            metrics["std"] = float(valid.std())
            report[col] = metrics

        self.diagnostic_report = report

    def _add_advanced_derived_features(self,
                                       data: pd.DataFrame) -> pd.DataFrame:
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
                    0, np.nan)
                df["bb_width_normalized"] = (df["bb_width_normalized"].replace(
                    [np.inf, -np.inf], np.nan).fillna(0))

            # 2. Range ratio
            if "range_ratio_5bar" not in df.columns:
                if "hl" not in df.columns:
                    df["hl"] = df["high"] - df["low"]
                range_ratio_raw = df["hl"].rolling(
                    5).mean() / df["hl"].rolling(20).mean().replace(0, np.nan)
                range_ratio_raw = range_ratio_raw.fillna(1)
                range_ratio_log = np.log1p(range_ratio_raw)
                range_ratio_mean = range_ratio_log.rolling(
                    50, min_periods=5).mean()
                range_ratio_std = range_ratio_log.rolling(50,
                                                          min_periods=5).std()
                df["range_ratio_5bar"] = (
                    (range_ratio_log - range_ratio_mean) /
                    range_ratio_std.replace(0, np.nan)).replace(
                        [np.inf, -np.inf],
                        np.nan).fillna(0).clip(-self.feature_clip_bound,
                                               self.feature_clip_bound)

            # 3. Volatility reversal score
            if "volatility_reversal_score" not in df.columns:
                atr_mean = df["atr"].rolling(50).mean()
                atr_std = df["atr"].rolling(50).std()
                df["volatility_reversal_score"] = (
                    (df["atr"] - atr_mean) /
                    atr_std.replace(0, np.nan)).fillna(0)

            # 4. Price range symmetry
            if "price_range_symmetry" not in df.columns:
                price_range_symmetry_raw = (
                    (self._shift_feature(df["high"]) -
                     self._shift_feature(df["close"])) /
                    ((self._shift_feature(df["close"]) -
                      self._shift_feature(df["low"])).replace(0, np.nan)))
                price_range_symmetry_raw = price_range_symmetry_raw.replace(
                    [np.inf, -np.inf], np.nan).fillna(1)
                price_range_symmetry_log = np.log1p(
                    np.abs(price_range_symmetry_raw)) * np.sign(
                        price_range_symmetry_raw)
                price_range_symmetry_mean = price_range_symmetry_log.rolling(
                    50, min_periods=5).mean()
                price_range_symmetry_std = price_range_symmetry_log.rolling(
                    50, min_periods=5).std()
                df["price_range_symmetry"] = (
                    (price_range_symmetry_log - price_range_symmetry_mean) /
                    price_range_symmetry_std.replace(0, np.nan)).replace(
                        [np.inf, -np.inf],
                        np.nan).fillna(0).clip(-self.feature_clip_bound,
                                               self.feature_clip_bound)

            # 5. Volume anomaly
            if "volume_anomaly" not in df.columns:
                volume_anomaly_raw = df["volume"] / df["volume"].ewm(
                    span=20, min_periods=10).mean().replace(0, np.nan)
                volume_anomaly_raw = volume_anomaly_raw.fillna(1)
                volume_anomaly_log = np.log1p(volume_anomaly_raw)
                volume_anomaly_mean = volume_anomaly_log.rolling(
                    50, min_periods=10).mean()
                volume_anomaly_std = volume_anomaly_log.rolling(
                    50, min_periods=10).std()
                df["volume_anomaly"] = (
                    (volume_anomaly_log - volume_anomaly_mean) /
                    volume_anomaly_std.replace(0, np.nan)).replace(
                        [np.inf, -np.inf],
                        np.nan).fillna(0).clip(-self.feature_clip_bound,
                                               self.feature_clip_bound)

            # 6. ROC and acceleration
            if "roc_5" not in df.columns:
                roc_raw = df["close"].pct_change(5)
                roc_mean = roc_raw.rolling(window=50, min_periods=5).mean()
                roc_std = roc_raw.rolling(window=50, min_periods=5).std()
                roc_std = roc_std.clip(lower=roc_raw.abs().quantile(0.01))
                df["roc_5"] = ((roc_raw - roc_mean) /
                               roc_std.replace(0, np.nan)).replace(
                                   [np.inf, -np.inf], np.nan).fillna(0).clip(
                                       -self.feature_clip_bound,
                                       self.feature_clip_bound)

            if "acceleration_3" not in df.columns:
                roc_3 = df["close"].pct_change(3)
                roc_3_mean = roc_3.rolling(window=50, min_periods=5).mean()
                roc_3_std = roc_3.rolling(window=50, min_periods=5).std()
                roc_3_std = roc_3_std.clip(lower=roc_3.abs().quantile(0.01))
                roc_3_norm = ((roc_3 - roc_3_mean) /
                              roc_3_std.replace(0, np.nan)).replace(
                                  [np.inf, -np.inf], np.nan).fillna(0).clip(
                                      -self.feature_clip_bound,
                                      self.feature_clip_bound)
                current = self._shift_feature(roc_3_norm)
                prev = self._shift_feature(roc_3_norm, offset=1)
                df["acceleration_3"] = current - prev

            # 7. Trend R²
            if "trend_r2_20" not in df.columns:
                df["trend_r2_20"] = self._trend_r2(df["close"],
                                                   window=20,
                                                   lag=self.feature_shift)
            if "trend_r2_50" not in df.columns:
                df["trend_r2_50"] = self._trend_r2(df["close"],
                                                   window=50,
                                                   lag=self.feature_shift)

            # 8. Slope consistency
            if "slope_consistency_score" not in df.columns:
                ema10 = df["close"].ewm(span=10).mean()
                ema20 = df["close"].ewm(span=20).mean()
                ema50 = df["close"].ewm(span=50).mean()
                slope10 = np.sign(ema10.diff())
                slope20 = np.sign(ema20.diff())
                slope50 = np.sign(ema50.diff())
                df["slope_consistency_score"] = (
                    (slope10 == slope20).astype(int) +
                    (slope20 == slope50).astype(int) +
                    (slope10 == slope50).astype(int))

            # 9. Trend volatility alignment
            if "trend_volatility_alignment" not in df.columns:
                if "atr_percentile" in df.columns and "roc_5" in df.columns:
                    df["trend_volatility_alignment"] = np.sign(
                        df["roc_5"]).fillna(0) * df["atr_percentile"].fillna(0)

            # 10. Compression to breakout probability
            if "compression_to_breakout_prob" not in df.columns:
                if "compression_duration" in df.columns and "roc_5" in df.columns:
                    df["compression_to_breakout_prob"] = df[
                        "compression_duration"].fillna(0) * df["roc_5"].fillna(
                            0)

        except Exception as e:
            print(f"      Warning: 高级衍生特征计算失败: {e}")
        return df

    def engineer_features(
            self,
            df: pd.DataFrame,
            *,
            fit: bool = True,
            required_features: Optional[set] = None) -> pd.DataFrame:
        """工程特征（合并后的版本，包含所有新特征）"""
        if not {"open", "high", "low", "close", "volume"}.issubset(df.columns):
            raise ValueError(
                "DataFrame must contain open, high, low, close, volume columns"
            )

        data = df.copy()

        # Core ATR（按需计算）
        if required_features is None or any(
                "atr" in f for f in required_features or [""]):
            if "atr" not in data.columns:
                data["atr"] = self._compute_atr(data, window=14)

        # SR proximity using rolling swing proxies
        swing_win_short = 20
        swing_win_long = 60

        need_swing = required_features is None or any(
            "swing" in f or "sr_dist" in f for f in required_features or [""])

        if need_swing:
            data["roll_high_s"] = data["high"].rolling(swing_win_short,
                                                       min_periods=1).max()
            data["roll_low_s"] = data["low"].rolling(swing_win_short,
                                                     min_periods=1).min()
            data["roll_high_l"] = data["high"].rolling(swing_win_long,
                                                       min_periods=1).max()
            data["roll_low_l"] = data["low"].rolling(swing_win_long,
                                                     min_periods=1).min()

            eps = 1e-9
            data["sr_dist_high_s"] = (data["close"] - data["roll_high_s"]) / (
                data["atr"] + eps)
            data["sr_dist_low_s"] = (data["close"] -
                                     data["roll_low_s"]) / (data["atr"] + eps)
            data["sr_dist_high_l"] = (data["close"] - data["roll_high_l"]) / (
                data["atr"] + eps)
            data["sr_dist_low_l"] = (data["close"] -
                                     data["roll_low_l"]) / (data["atr"] + eps)

        # Simple channel via EMAs
        need_channel = required_features is None or any(
            "channel" in f for f in required_features or [""])

        if need_channel:
            ema_fast = self._ema(data["close"], span=20)
            ema_slow = self._ema(data["close"], span=60)
            mid = (ema_fast + ema_slow) / 2.0
            band_half = (data["high"].rolling(20, min_periods=1).max() -
                         data["low"].rolling(20, min_periods=1).min()) / 4.0
            upper = mid + band_half
            lower = mid - band_half
            data["channel_upper"] = upper
            data["channel_lower"] = lower
            data["channel_mid"] = mid

            eps = 1e-9
            data["channel_bandwidth"] = (upper - lower) / (data["atr"] + eps)
            data["channel_upper_distance"] = (upper - data["close"]) / (
                data["atr"] + eps)
            data["channel_lower_distance"] = (data["close"] -
                                              lower) / (data["atr"] + eps)

            close_safe = data["close"].replace(0, np.nan)
            data["channel_upper_pct_close"] = (upper / close_safe -
                                               1.0).replace([np.inf, -np.inf],
                                                            np.nan).fillna(0.0)
            data["channel_lower_pct_close"] = (lower / close_safe -
                                               1.0).replace([np.inf, -np.inf],
                                                            np.nan).fillna(0.0)
            data["channel_mid_pct_close"] = (mid / close_safe - 1.0).replace(
                [np.inf, -np.inf], np.nan).fillna(0.0)

        # Price-volume divergence signals
        if required_features is None or any(
                "divergence" in f for f in required_features or [""]):
            if "rsi_14" not in data.columns:
                data["rsi_14"] = self._compute_rsi(data["close"], period=14)

            recent_high = data["close"].rolling(20,
                                                min_periods=5).max().ffill()
            recent_rsi_high = data["rsi_14"].rolling(
                20, min_periods=5).max().ffill()
            tol = 1e-8
            divergence_mask = (recent_high.notna() & recent_rsi_high.notna() &
                               (data["close"] >= (recent_high - tol)) &
                               (data["rsi_14"] < (recent_rsi_high - tol)))
            data["rsi_divergence"] = divergence_mask.astype(float) * -1.0

            price_vs_past = (data["close"]
                             > data["close"].shift(5)).fillna(False)
            avg_volume_20 = data["volume"].rolling(20, min_periods=5).mean()
            low_volume_mask = (data["volume"] < avg_volume_20).fillna(False)
            volume_div_mask = price_vs_past & low_volume_mask
            data["volume_divergence"] = volume_div_mask.astype(float) * -1.0

        # Compression features
        need_compression = required_features is None or any(
            "compression" in f or "atr_percentile" in f
            for f in required_features or [""])

        if need_compression:
            atr_pct = self._rolling_percentile(data["atr"],
                                               window=self.percentile_window)
            data["atr_percentile"] = atr_pct

            volatility_regime_window = 200
            volatility_regime_threshold = 0.7
            atr_quantile_70 = data["atr"].rolling(
                window=volatility_regime_window,
                min_periods=1).quantile(volatility_regime_threshold)
            data["volatility_regime"] = (
                data["atr"] > atr_quantile_70).astype(int).fillna(0)

            returns = data["close"].pct_change().fillna(0.0)
            realized_skew = self._rolling_skew(returns, window=20)
            data["realized_skew"] = realized_skew.fillna(0.0).clip(
                -self.feature_clip_bound, self.feature_clip_bound)

            vol5 = returns.rolling(5, min_periods=1).std()
            vol60 = returns.rolling(60, min_periods=1).std()
            volatility_ratio = (vol5 / vol60.replace(0, np.nan)).replace(
                [np.inf, -np.inf], np.nan).fillna(0.0)
            data["volatility_ratio"] = volatility_ratio.clip(
                -self.feature_clip_bound, self.feature_clip_bound)

            atr_mean_hist = data["atr"].rolling(self.percentile_window,
                                                min_periods=1).mean()
            eps = 1e-9
            data["atr_compression_ratio"] = (atr_mean_hist /
                                             (data["atr"] + eps)).replace(
                                                 [np.inf, -np.inf], np.nan)

            vol_pct = self._rolling_percentile(data["volume"].astype(float),
                                               window=self.percentile_window)
            data["volume_percentile"] = vol_pct

            data["price_entropy"] = self._price_entropy(data["close"],
                                                        window=50)

            threshold = self.compression_threshold_pct
            below = (data["atr_percentile"].fillna(0.0)
                     <= threshold).astype(int)
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
            data["pre_break_silence"] = (data["atr_percentile"].rolling(
                short_window, min_periods=1).mean() <= threshold).astype(float)

            small = 20
            large = 100
            var_small = data["close"].rolling(small, min_periods=1).var()
            var_large = data["close"].rolling(large, min_periods=1).var()
            density = 1.0 - (var_small / (var_large + eps))
            data["internal_price_density"] = density.clip(0.0, 1.0)

            atr_norm = (data["atr_percentile"].fillna(0.0))
            vol_norm = (data["volume_percentile"].fillna(0.0))
            dens_norm = data["internal_price_density"].fillna(0.0)
            data["compression_confidence"] = 0.5 * (1 - atr_norm) + 0.3 * (
                1 - vol_norm) + 0.2 * dens_norm

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
                        midnight_delta = (utc_idx - utc_idx.normalize()
                                          ).total_seconds() / 60.0
                    else:
                        hour = idx.hour.astype(int)
                        midnight_delta = (
                            idx - idx.normalize()).total_seconds() / 60.0

                    data["hour_sin"] = np.sin(2 * np.pi * hour / 24)
                    data["hour_cos"] = np.cos(2 * np.pi * hour / 24)
                    data["Hour_of_Day"] = hour
                    data["minutes_since_reset"] = pd.Series(
                        midnight_delta, index=data.index).fillna(0.0)
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
                "zz_" in f or "zigzag" in f
                for f in required_features or [""]):
            data = add_zigzag_dimensionless_features(data, required_features)

        # POC 无量纲特征
        if required_features is None or any(
                "poc" in f for f in required_features or [""]):
            data = add_poc_dimensionless_features(data, required_features)

        # HAL 无量纲特征
        if required_features is None or any(
                "hal" in f for f in required_features or [""]):
            data = add_hal_dimensionless_features(data, required_features)

        # Swing 无量纲特征
        if required_features is None or any(
                "swing" in f for f in required_features or [""]):
            data = add_swing_dimensionless_features(data, required_features)

        # 基础价格与量能相对变化特征
        if required_features is None or any(
                f in required_features or "" for f in [
                    "ret_1h", "ret_4h", "ret_24h", "rv_4h", "rv_24h",
                    "vol_ma_ratio", "vol_zscore"
                ]):
            data = add_price_volume_relative_features(data, required_features)

        # 如果指定了required_features，只保留需要的特征
        if required_features is not None:
            data_cols = {
                'open', 'high', 'low', 'close', 'volume', 'timestamp',
                'datetime'
            }
            cols_to_keep = [
                c for c in data.columns
                if c in data_cols or c in required_features
                or not pd.api.types.is_numeric_dtype(data[c])
            ]
            data = data[cols_to_keep]

        if self.enable_diagnostics:
            feature_cols = [
                c for c in data.columns if c not in [
                    "open", "high", "low", "close", "volume", "timestamp",
                    "symbol"
                ]
            ]
            self._run_diagnostics(data, feature_cols)

        return data

    def save_scalers(self, path: str) -> None:
        """保存标准化器"""
        import pickle
        scalers_data = {
            "fitted_atr_quantiles": self._fitted_atr_quantiles,
            "fitted_vol_quantiles": self._fitted_vol_quantiles,
            "percentile_window": self.percentile_window,
            "compression_threshold_pct": self.compression_threshold_pct,
        }
        with open(path, "wb") as f:
            pickle.dump(scalers_data, f)
        print(f"✅ Baseline scalers saved to: {path}")

    def load_scalers(self, path: str) -> None:
        """加载标准化器"""
        import pickle
        with open(path, "rb") as f:
            scalers_data = pickle.load(f)
        self._fitted_atr_quantiles = scalers_data.get("fitted_atr_quantiles",
                                                      None)
        self._fitted_vol_quantiles = scalers_data.get("fitted_vol_quantiles",
                                                      None)
        self.percentile_window = scalers_data.get("percentile_window", 288)
        self.compression_threshold_pct = scalers_data.get(
            "compression_threshold_pct", 0.2)
        print(f"✅ Baseline scalers loaded from: {path}")


# ============================================================================
# 便捷函数（保持向后兼容）
# ============================================================================


def engineer_baseline_features(
    df: pd.DataFrame,
    engineer: Optional[BaselineFeatureEngineer] = None,
    *,
    fit: bool = True,
    required_features: Optional[set] = None
) -> Tuple[pd.DataFrame, BaselineFeatureEngineer]:
    """工程 baseline 特征"""
    if engineer is None:
        engineer = BaselineFeatureEngineer()
    out = engineer.engineer_features(df,
                                     fit=fit,
                                     required_features=required_features)
    return out, engineer


def create_binary_labels_baseline(df: pd.DataFrame,
                                  *,
                                  forward_bars: int = 3,
                                  threshold: float = 0.005) -> pd.DataFrame:
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
    }
    exclude.update([
        col for col in df.columns
        if (col.startswith("signal_") or col.startswith("binary_signal_")
            or col.startswith("future_return_"))
    ])
    return [c for c in df.columns if c not in exclude]


# ============================================================================
# 导出（保持向后兼容）
# ============================================================================

__all__ = [
    # 基础指标计算函数
    "compute_rsi",
    "compute_macd",
    "compute_bollinger_bands",
    "compute_atr",
    "compute_zigzag",
    "compute_poc",
    "compute_hal",
    "compute_zigzag_high_low",

    # 基础指标添加函数（优化版）
    "add_basic_indicators",
    "ensure_basic_indicators",

    # 新增无量纲特征函数
    "add_zigzag_dimensionless_features",
    "add_poc_dimensionless_features",
    "add_hal_dimensionless_features",
    "add_swing_dimensionless_features",
    "add_price_volume_relative_features",

    # 衍生特征函数（优化版）
    "add_common_derived_features",

    # BaselineFeatureEngineer 类
    "BaselineFeatureEngineer",
    "engineer_baseline_features",
    "get_baseline_feature_columns",
    "create_binary_labels_baseline",

    # 常量
    "BASIC_INDICATOR_COLUMNS",
    "COMMON_DERIVED_COLUMNS",
]
