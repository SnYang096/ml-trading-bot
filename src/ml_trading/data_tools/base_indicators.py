"""基础技术指标计算函数 - 共享模块，避免代码重复.

这个模块包含所有特征工程模块共享的基础指标计算函数。
所有其他特征工程模块应该从这里导入，而不是重复定义。
"""

import pandas as pd
import numpy as np
from typing import Tuple

try:
    import talib
except ImportError:  # pragma: no cover - safety fallback when TA-Lib missing
    talib = None


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
    """
    计算相对强弱指数 (RSI).

    Args:
        series: 价格序列
        period: RSI周期

    Returns:
        RSI值序列
    """
    series = pd.to_numeric(series, errors="coerce").astype(float)

    if talib is not None:
        values = talib.RSI(series.values, timeperiod=period)
        return pd.Series(values, index=series.index)

    # Fallback实现
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
    """
    计算MACD指标.

    Args:
        series: 价格序列
        fast: 快速EMA周期
        slow: 慢速EMA周期
        signal: 信号线周期

    Returns:
        (MACD线, 信号线, 柱状图)
    """
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

    # Fallback实现
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
    """
    计算布林带.

    Args:
        series: 价格序列
        period: 移动平均周期
        std_dev: 标准差倍数

    Returns:
        (上轨, 中轨, 下轨)
    """
    series = pd.to_numeric(series, errors="coerce").astype(float)

    if talib is not None:
        upper, middle, lower = talib.BBANDS(
            series.values,
            timeperiod=period,
            nbdevup=std_dev,
            nbdevdn=std_dev,
            matype=0,
        )
        index = series.index
        return (
            pd.Series(upper, index=index),
            pd.Series(middle, index=index),
            pd.Series(lower, index=index),
        )

    # Fallback实现
    middle_band = series.rolling(window=period).mean()
    std_series = series.rolling(window=period).std()
    upper_band = middle_band + (std_dev * std_series)
    lower_band = middle_band - (std_dev * std_series)
    return upper_band, middle_band, lower_band


def compute_atr(high: pd.Series,
                low: pd.Series,
                close: pd.Series,
                period: int = 14) -> pd.Series:
    """
    计算平均真实波幅 (ATR).

    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        period: ATR周期

    Returns:
        ATR值序列
    """
    high = pd.to_numeric(high, errors="coerce").astype(float)
    low = pd.to_numeric(low, errors="coerce").astype(float)
    close = pd.to_numeric(close, errors="coerce").astype(float)

    if talib is not None:
        atr_values = talib.ATR(high.values,
                               low.values,
                               close.values,
                               timeperiod=period)
        return pd.Series(atr_values, index=high.index)

    # Fallback实现
    tr0 = abs(high - low)
    tr1 = abs(high - close.shift())
    tr2 = abs(low - close.shift())
    tr = pd.DataFrame({"tr0": tr0, "tr1": tr1, "tr2": tr2}).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def compute_zigzag(high: pd.Series,
                   low: pd.Series,
                   threshold: float = 0.05) -> pd.Series:
    """
    计算ZigZag指标.

    Args:
        high: 最高价序列
        low: 最低价序列
        threshold: 反转阈值（百分比）

    Returns:
        ZigZag值序列
    """
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


def add_basic_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    添加基础技术指标到DataFrame.

    这是一个便利函数，可以一次性添加所有基础指标。

    Args:
        df: 包含OHLCV数据的DataFrame

    Returns:
        添加了技术指标的DataFrame
    """
    if df.empty:
        return df

    result = df.copy()

    # 确保所有列都是数值类型
    for col in ["open", "high", "low", "close", "volume"]:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    # 移除OHLCV列中有NaN的行
    result = result.dropna(subset=["open", "high", "low", "close", "volume"])

    if result.empty:
        return result

    # RSI
    result["rsi"] = compute_rsi(result["close"])

    # MACD
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

    # Bollinger Bands
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

    # ATR
    try:
        result["atr"] = compute_atr(result["high"], result["low"],
                                    result["close"])
    except Exception as e:
        print(f"Warning: Error computing ATR: {e}")
        result["atr"] = 0

    # ZigZag
    try:
        result["zigzag"] = compute_zigzag(result["high"], result["low"])
    except Exception as e:
        print(f"Warning: Error computing ZigZag: {e}")
        result["zigzag"] = 0

    # Price change and volatility features
    result["price_change"] = result["close"].pct_change()
    result["volatility"] = _maybe_talib_std(result["price_change"], 14)

    # Volume features
    result["volume_sma"] = _maybe_talib_sma(result["volume"], 20)
    result["volume_ratio"] = result["volume"] / result["volume_sma"]

    # 填充NaN值
    feature_cols = [
        col for col in result.columns
        if col not in ["open", "high", "low", "close", "volume"]
    ]
    for col in feature_cols:
        result[col] = result[col].fillna(0)

    return result


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


def ensure_basic_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the core basic indicators from :func:`add_basic_indicators` exist."""

    if df.empty:
        return df

    if BASIC_INDICATOR_COLUMNS.issubset(df.columns):
        return df

    return add_basic_indicators(df)


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
    "sma_ratio_5_20",
    "sma_ratio_10_20",
    "volume_sma_20",
}


def add_common_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach commonly used derived features built on top of basic indicators."""

    if df.empty:
        return df

    result = ensure_basic_indicators(df.copy())

    close = result["close"]

    if "returns" not in result.columns:
        result["returns"] = close.pct_change()

    if "log_returns" not in result.columns:
        shifted = close.shift(1).replace(0, np.nan)
        result["log_returns"] = np.log(close.replace(0, np.nan) / shifted)

    if "price_change" not in result.columns:
        result["price_change"] = close.diff()

    if "volatility" not in result.columns:
        result["volatility"] = _maybe_talib_std(result["returns"], 20)

    if "bb_position" not in result.columns and {"bb_upper", "bb_lower"
                                                }.issubset(result.columns):
        denom = (result["bb_upper"] - result["bb_lower"]).replace(0, np.nan)
        result["bb_position"] = (close - result["bb_lower"]) / denom

    if "bb_width" not in result.columns and {"bb_upper", "bb_lower"}.issubset(
            result.columns):
        result["bb_width"] = (result["bb_upper"] - result["bb_lower"]).abs()

    if "rsi_normalized" not in result.columns and "rsi" in result.columns:
        result["rsi_normalized"] = (result["rsi"] - 50) / 50

    if "macd_normalized" not in result.columns and "macd" in result.columns:
        result["macd_normalized"] = result["macd"] / close.replace(0, np.nan)

    if "atr_normalized" not in result.columns and "atr" in result.columns:
        result["atr_normalized"] = result["atr"] / close.replace(0, np.nan)

    # Momentum and SMA features
    for period in [5, 10, 20]:
        momentum_col = f"momentum_{period}"
        if momentum_col not in result.columns:
            result[momentum_col] = close.pct_change(period)

    sma_map = {
        5: "sma_5",
        10: "sma_10",
        20: "sma_20",
    }
    for window, col_name in sma_map.items():
        if col_name not in result.columns:
            result[col_name] = _maybe_talib_sma(close, window)

    if {"sma_5", "sma_20"}.issubset(
            result.columns) and "sma_ratio_5_20" not in result.columns:
        result["sma_ratio_5_20"] = result["sma_5"] / result["sma_20"].replace(
            0, np.nan)

    if {"sma_10", "sma_20"}.issubset(
            result.columns) and "sma_ratio_10_20" not in result.columns:
        result[
            "sma_ratio_10_20"] = result["sma_10"] / result["sma_20"].replace(
                0, np.nan)

    if "volume_sma_20" not in result.columns:
        result["volume_sma_20"] = _maybe_talib_sma(result["volume"], 20)

    if "volume_ratio" in result.columns:
        result["volume_ratio"] = (result["volume_ratio"].replace(
            [np.inf, -np.inf], np.nan))
    else:
        denom = result["volume_sma_20"].replace(0, np.nan)
        result["volume_ratio"] = result["volume"] / denom

    # Final cleanup
    for col in COMMON_DERIVED_COLUMNS:
        if col in result.columns:
            result[col] = result[col].replace([np.inf, -np.inf],
                                              np.nan).fillna(0)

    return result
