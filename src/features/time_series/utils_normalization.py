"""
特征归一化工具函数（公共模块）

用于多资产训练的特征归一化，确保不同价格水平的资产可以一起训练
"""

import numpy as np
import pandas as pd
from typing import Union


def normalize_series(x: Union[np.ndarray, pd.Series]) -> np.ndarray:
    """
    Z-score归一化，保留形状
    
    Args:
        x: 输入序列（numpy array 或 pandas Series）
    
    Returns:
        归一化后的序列（均值=0，标准差=1）
    
    Note:
        这个函数用于将不同量级的序列归一化到相同尺度，
        使得多资产训练时不同价格水平的资产可以一起使用。
    """
    x = np.array(x)
    mean = np.mean(x)
    std = np.std(x)
    if std < 1e-8:
        return np.zeros_like(x)
    return (x - mean) / (std + 1e-8)


def normalize_by_group(
    df: pd.DataFrame,
    value_col: str,
    group_col: str = "_symbol",
    method: str = "zscore",
) -> pd.Series:
    """
    按组归一化（用于多资产数据）
    
    Args:
        df: DataFrame with data
        value_col: Column to normalize
        group_col: Group column (e.g., "_symbol" for multi-asset)
        method: Normalization method ("zscore" or "minmax")
    
    Returns:
        Normalized Series
    """
    if group_col not in df.columns:
        # Single asset, normalize globally
        if method == "zscore":
            return pd.Series(
                normalize_series(df[value_col].values), index=df.index
            )
        else:  # minmax
            values = df[value_col].values
            min_val = values.min()
            max_val = values.max()
            if max_val - min_val < 1e-8:
                return pd.Series(0.0, index=df.index)
            return pd.Series(
                (values - min_val) / (max_val - min_val + 1e-8), index=df.index
            )
    
    # Multi-asset: normalize per group
    normalized = pd.Series(index=df.index, dtype=float)
    
    for group in df[group_col].unique():
        mask = df[group_col] == group
        group_values = df.loc[mask, value_col].values
        
        if method == "zscore":
            normalized.loc[mask] = normalize_series(group_values)
        else:  # minmax
            min_val = group_values.min()
            max_val = group_values.max()
            if max_val - min_val < 1e-8:
                normalized.loc[mask] = 0.0
            else:
                normalized.loc[mask] = (group_values - min_val) / (
                    max_val - min_val + 1e-8
                )
    
    return normalized

