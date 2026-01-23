"""
反身性监测特征（Reflexivity Monitoring Features）

实现反身性监测指标，用于识别市场反身性风险：
- OFCI (Order Flow Consensus Index): 订单流一致性指数
- SHD (Strategy Homogeneity Detector): 策略同质化探测器

注意：LFI (Liquidity Fragility Index) 需要订单簿数据，暂不实现。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from src.features.registry import register_feature
from src.features.time_series.baseline_features import compute_percentile_rank_from_series


@register_feature("compute_ofci_from_trades", category="reflexivity")
def compute_ofci_from_trades(
    trades: pd.DataFrame,
    window: int = 100,
) -> pd.DataFrame:
    """
    计算订单流一致性指数（Order Flow Consensus Index, OFCI）
    
    定义：衡量全市场"方向性共识强度"——越高越危险
    
    Args:
        trades: 逐笔成交数据，必须包含'side'列（+1=buy, -1=sell）
                如果只有OHLCV数据，可以从volume和price变化推断方向
        window: 滚动窗口大小（建议100，对应10-30秒）
    
    Returns:
        DataFrame with column 'ofci': [-1, 1] 的对称指标
        - |OFCI| > 0.7 → 高度一致（警惕反身性踩踏或追涨）
        - |OFCI| < 0.3 → 方向分散（相对安全）
    """
    if len(trades) == 0:
        return pd.DataFrame(columns=["ofci"], dtype=float)
    
    # 检查是否有side列
    if "side" in trades.columns:
        # 从tick数据计算
        directions = trades["side"].copy()
        # 标准化side值（确保是+1/-1）
        if directions.dtype == "object":
            directions = directions.map({"buy": 1, "sell": -1, "BUY": 1, "SELL": -1, 1: 1, -1: -1})
        else:
            directions = directions.astype(float)
        
        # 过滤无效值
        valid_mask = directions.isin([1, -1])
        if not valid_mask.any():
            return pd.DataFrame(index=trades.index, columns=["ofci"], dtype=float).fillna(0.0)
        
        # 计算buy_ratio（滚动窗口）
        buy_count = (directions == 1).rolling(window=window, min_periods=window).sum()
        total_count = valid_mask.rolling(window=window, min_periods=window).sum()
        buy_ratio = buy_count / total_count.replace(0, np.nan)
        
        # 转换为[-1, 1]对称指标
        ofci = 2 * buy_ratio - 1
        ofci = ofci.fillna(0.0).clip(-1.0, 1.0)
        
        return pd.DataFrame({"ofci": ofci}, index=trades.index)
    
    else:
        # 如果没有side列，返回0（需要tick数据）
        return pd.DataFrame(index=trades.index, columns=["ofci"], dtype=float).fillna(0.0)


@register_feature("compute_ofci_pct_from_series", category="reflexivity")
def compute_ofci_pct_from_series(
    *,
    ofci: pd.Series,
    window: int = 288,
    shift: int = 1,
) -> pd.DataFrame:
    """
    计算OFCI的percentile rank（用于跨symbol稳定性）
    
    注意：OFCI是[-1, 1]的对称指标，为了正确反映极端一致性（无论是正还是负），
    我们对abs(ofci)计算percentile rank。
    
    Args:
        ofci: OFCI序列（[-1, 1]）
        window: 滚动窗口大小
        shift: 滞后shift
    
    Returns:
        DataFrame with column 'ofci_pct': [0, 1] 的percentile值
        - ofci_pct > 0.9 表示极端一致性（无论是正还是负）
    """
    # 使用绝对值计算percentile rank，以反映极端一致性
    ofci_abs = ofci.abs()
    return compute_percentile_rank_from_series(
        series=ofci_abs,
        window=window,
        shift=shift,
        output_name="ofci_pct",
    )


@register_feature("compute_shd_from_series", category="reflexivity")
def compute_shd_from_series(
    *,
    cvd_series: pd.Series,
    price_returns: pd.Series,
    window: int = 60,
) -> pd.DataFrame:
    """
    计算策略同质化探测器（Strategy Homogeneity Detector, SHD）
    
    定义：检测"是否太多人在用类似逻辑交易"
    原理：如果CVD和价格变动高度同步，说明大量人用订单流策略
    
    Args:
        cvd_series: 累计成交量差额序列（CVD）
        price_returns: 价格收益率序列（log return或pct_change）
        window: 滚动窗口大小（建议60，对应1-5分钟）
    
    Returns:
        DataFrame with column 'shd': [0, 1] 的指标
        - SHD > 0.6 → 多数量化在用相似信号 → 反身性风险高
        - SHD接近0 → 多种策略在博弈（健康）
        - SHD接近1 → "同一类人推动价格"（危险）
    """
    if len(cvd_series) == 0 or len(price_returns) == 0:
        return pd.DataFrame(columns=["shd"], dtype=float)
    
    # 确保索引对齐
    common_index = cvd_series.index.intersection(price_returns.index)
    if len(common_index) == 0:
        return pd.DataFrame(columns=["shd"], dtype=float)
    
    cvd = cvd_series.loc[common_index].copy()
    ret = price_returns.loc[common_index].copy()
    
    # 计算CVD的变化（ΔCVD）
    # 根据文档：纯成交流版本使用 rolling_corr(ΔCVD, return)
    d_cvd = cvd.diff().fillna(0.0)
    
    # 使用pandas的rolling.corr计算滚动相关系数（更高效）
    # 根据文档：SHD = abs(rolling_corr(d_cvd, ret))
    rolling_corr = d_cvd.rolling(window=window, min_periods=max(window // 2, 10)).corr(ret)
    
    # 取绝对值作为SHD（符合文档要求）
    shd_series = rolling_corr.abs().fillna(0.0).clip(0.0, 1.0)
    
    return pd.DataFrame({"shd": shd_series}, index=common_index)


@register_feature("compute_shd_pct_from_series", category="reflexivity")
def compute_shd_pct_from_series(
    *,
    shd: pd.Series,
    window: int = 288,
    shift: int = 1,
) -> pd.DataFrame:
    """
    计算SHD的percentile rank（用于跨symbol稳定性）
    
    Args:
        shd: SHD序列
        window: 滚动窗口大小
        shift: 滞后shift
    
    Returns:
        DataFrame with column 'shd_pct': [0, 1] 的percentile值
    """
    return compute_percentile_rank_from_series(
        series=shd,
        window=window,
        shift=shift,
        output_name="shd_pct",
    )


@register_feature("compute_shd_from_ohlcv", category="reflexivity")
def compute_shd_from_ohlcv(
    *,
    close: pd.Series,
    cvd: pd.Series,
    window: int = 60,
) -> pd.DataFrame:
    """
    从OHLCV数据计算SHD（不需要tick数据）
    
    这是SHD的简化版本，使用价格收益率和CVD序列计算。
    
    Args:
        close: 收盘价序列
        cvd: CVD序列（从orderflow features中获取）
        window: 滚动窗口大小
    
    Returns:
        DataFrame with column 'shd': [0, 1] 的指标
    """
    # 计算价格收益率
    price_returns = np.log(close / close.shift(1)).fillna(0.0)
    
    # 使用标准SHD计算
    return compute_shd_from_series(
        cvd_series=cvd,
        price_returns=price_returns,
        window=window,
    )
