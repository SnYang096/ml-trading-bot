"""
扩展波动率特征提取器
添加历史波动率、滞后特征、趋势特征等，提高波动率预测准确性
"""

import numpy as np
import pandas as pd
from typing import Optional


def extract_extended_volatility_features(
    df: pd.DataFrame,
    price_col: str = "close",
    atr_col: str = "atr",
    window: int = 20,
    lag_periods: list = [1, 2, 3, 5, 10],
) -> pd.DataFrame:
    """
    提取扩展的波动率特征
    
    Args:
        df: DataFrame with price data
        price_col: Column name for price
        atr_col: Column name for ATR
        window: Rolling window size for volatility calculation
        lag_periods: List of lag periods for lag features
    
    Returns:
        DataFrame with extended volatility features:
        - vol_historical_*: Historical volatility (rolling std of returns)
        - vol_atr_ratio_*: ATR ratio features
        - vol_lag_*: Lag features of volatility
        - vol_trend_*: Trend features of volatility
        - vol_ma_*: Moving average of volatility
    """
    df = df.copy()
    
    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found")
    
    # Calculate returns
    returns = df[price_col].pct_change()
    
    # Initialize result DataFrame
    result = pd.DataFrame(index=df.index)
    
    # 1. Historical volatility (rolling std of returns)
    for w in [5, 10, 20, 60]:
        vol_hist = returns.rolling(window=w, min_periods=w//2).std()
        result[f"vol_historical_{w}"] = vol_hist
    
    # 2. ATR ratio features
    if atr_col in df.columns:
        atr = df[atr_col]
        price = df[price_col]
        
        # ATR / Price ratio (normalized volatility)
        result["vol_atr_price_ratio"] = atr / (price + 1e-8)
        
        # ATR rolling statistics
        for w in [5, 10, 20]:
            result[f"vol_atr_ma_{w}"] = atr.rolling(window=w, min_periods=w//2).mean()
            result[f"vol_atr_std_{w}"] = atr.rolling(window=w, min_periods=w//2).std()
            result[f"vol_atr_max_{w}"] = atr.rolling(window=w, min_periods=w//2).max()
            result[f"vol_atr_min_{w}"] = atr.rolling(window=w, min_periods=w//2).min()
        
        # ATR ratio (current / historical mean)
        atr_ma_20 = atr.rolling(window=20, min_periods=10).mean()
        result["vol_atr_ratio_ma20"] = atr / (atr_ma_20 + 1e-8)
        
        # ATR change rate
        result["vol_atr_change"] = atr.pct_change()
        result["vol_atr_change_abs"] = atr.diff().abs()
    
    # 3. Lag features of volatility
    vol_base = returns.rolling(window=window, min_periods=window//2).std()
    for lag in lag_periods:
        result[f"vol_lag_{lag}"] = vol_base.shift(lag)
    
    # 4. Trend features of volatility
    vol_base = returns.rolling(window=window, min_periods=window//2).std()
    
    # Volatility slope (linear trend)
    for w in [5, 10, 20]:
        vol_window = vol_base.rolling(window=w, min_periods=w//2)
        # Simple linear regression slope
        def calc_slope(x):
            if len(x) < 2:
                return 0.0
            y = x.values
            x_vals = np.arange(len(y))
            if np.std(x_vals) < 1e-8:
                return 0.0
            slope = np.polyfit(x_vals, y, 1)[0]
            return slope
        
        result[f"vol_trend_slope_{w}"] = vol_window.apply(calc_slope, raw=False)
    
    # Volatility acceleration (second derivative)
    vol_slope_5 = result["vol_trend_slope_5"] if "vol_trend_slope_5" in result.columns else pd.Series(0, index=df.index)
    result["vol_acceleration"] = vol_slope_5.diff()
    
    # 5. Moving average of volatility
    vol_base = returns.rolling(window=window, min_periods=window//2).std()
    for w in [5, 10, 20]:
        result[f"vol_ma_{w}"] = vol_base.rolling(window=w, min_periods=w//2).mean()
        result[f"vol_ema_{w}"] = vol_base.ewm(span=w, min_periods=w//2).mean()
    
    # 6. Volatility regime features
    vol_base = returns.rolling(window=window, min_periods=window//2).std()
    vol_ma_20 = vol_base.rolling(window=20, min_periods=10).mean()
    vol_std_20 = vol_base.rolling(window=20, min_periods=10).std()
    
    # Z-score of volatility (重命名避免与baseline_features中的vol_zscore冲突)
    result["vol_volatility_zscore"] = (vol_base - vol_ma_20) / (vol_std_20 + 1e-8)
    
    # Volatility percentile rank
    result["vol_percentile_rank"] = vol_base.rolling(window=60, min_periods=30).apply(
        lambda x: (x.iloc[-1] > x).sum() / len(x) if len(x) > 0 else 0.5,
        raw=False
    )
    
    # 7. Volatility range features
    vol_base = returns.rolling(window=window, min_periods=window//2).std()
    for w in [10, 20]:
        vol_max = vol_base.rolling(window=w, min_periods=w//2).max()
        vol_min = vol_base.rolling(window=w, min_periods=w//2).min()
        result[f"vol_range_{w}"] = vol_max - vol_min
        result[f"vol_range_ratio_{w}"] = (vol_base - vol_min) / (vol_max - vol_min + 1e-8)
    
    # 8. Volatility momentum features
    vol_base = returns.rolling(window=window, min_periods=window//2).std()
    for w in [3, 5, 10]:
        result[f"vol_momentum_{w}"] = vol_base / (vol_base.shift(w) + 1e-8) - 1.0
    
    # Fill NaN values
    result = result.fillna(method="ffill").fillna(0.0)
    
    return result

