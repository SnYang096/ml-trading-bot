"""基础技术指标计算函数 - 共享模块，避免代码重复.

这个模块包含所有特征工程模块共享的基础指标计算函数。
所有其他特征工程模块应该从这里导入，而不是重复定义。
"""

import pandas as pd
import numpy as np
from typing import Tuple


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    计算相对强弱指数 (RSI).

    Args:
        series: 价格序列
        period: RSI周期

    Returns:
        RSI值序列
    """
    series = pd.to_numeric(series, errors="coerce")
    delta = series.diff()
    delta = pd.to_numeric(delta, errors="coerce")
    gain = (delta.where(delta > 0, 0.0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
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
    series = pd.to_numeric(series, errors="coerce")
    ema_fast = series.ewm(span=fast).mean()
    ema_slow = series.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger_bands(
    series: pd.Series, period: int = 20, std_dev: int = 2
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    计算布林带.

    Args:
        series: 价格序列
        period: 移动平均周期
        std_dev: 标准差倍数

    Returns:
        (上轨, 中轨, 下轨)
    """
    series = pd.to_numeric(series, errors="coerce")
    middle_band = series.rolling(window=period).mean()
    std_series = series.rolling(window=period).std()
    upper_band = middle_band + (std_dev * std_series)
    lower_band = middle_band - (std_dev * std_series)
    return upper_band, middle_band, lower_band


def compute_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
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
    high = pd.to_numeric(high, errors="coerce")
    low = pd.to_numeric(low, errors="coerce")
    close = pd.to_numeric(close, errors="coerce")

    tr0 = abs(high - low)
    tr1 = abs(high - close.shift())
    tr2 = abs(low - close.shift())
    tr = pd.DataFrame({"tr0": tr0, "tr1": tr1, "tr2": tr2}).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def compute_zigzag(
    high: pd.Series, low: pd.Series, threshold: float = 0.05
) -> pd.Series:
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
        upper_band, middle_band, lower_band = compute_bollinger_bands(result["close"])
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
        result["atr"] = compute_atr(result["high"], result["low"], result["close"])
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
    result["volatility"] = result["price_change"].rolling(window=14).std()

    # Volume features
    result["volume_sma"] = result["volume"].rolling(window=20).mean()
    result["volume_ratio"] = result["volume"] / result["volume_sma"]

    # 填充NaN值
    feature_cols = [
        col
        for col in result.columns
        if col not in ["open", "high", "low", "close", "volume"]
    ]
    for col in feature_cols:
        result[col] = result[col].fillna(0)

    return result
