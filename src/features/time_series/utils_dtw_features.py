"""
DTW (Dynamic Time Warping) 特征提取器
用于形态匹配和模式识别
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
import warnings

warnings.filterwarnings("ignore")

try:
    from dtaidistance import dtw
    DTW_AVAILABLE = True
except ImportError:
    DTW_AVAILABLE = False
    print("⚠️ dtaidistance package not available. DTW features will be disabled.")

# 使用公共归一化模块
from src.features.time_series.utils_normalization import normalize_series


def smooth_template(template: np.ndarray, window: int = 3) -> np.ndarray:
    """Simple moving average smoothing to avoid jagged edges"""
    if len(template) < window:
        return template
    return np.convolve(template, np.ones(window) / window, mode="same")


def create_dtw_templates() -> Dict[str, np.ndarray]:
    """
    创建DTW模板库（看涨、看跌、中继形态）
    
    Returns:
        Dictionary of template names and normalized arrays
    """
    templates = {}
    
    # === 看涨形态 ===
    # 1. Hammer (锤子线)
    hammer = np.concatenate([
        np.linspace(1.0, 0.4, 15),
        [0.2, 0.35, 0.5, 0.6, 0.7]
    ])
    templates["hammer"] = smooth_template(hammer)
    
    # 2. 头肩底
    head_shoulders_bottom = np.array([
        0.8, 0.4, 0.6,  # 左肩
        0.3, 0.1, 0.4, 0.6,  # 头部
        0.5, 0.3, 0.5,  # 右肩
        0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0  # 突破
    ])
    templates["head_shoulder_bottom"] = smooth_template(head_shoulders_bottom)
    
    # 3. 双底 (W底)
    double_bottom = np.concatenate([
        np.linspace(1.0, 0.3, 6),  # 第一次下跌
        np.linspace(0.3, 0.7, 4),  # 反弹
        np.linspace(0.7, 0.25, 5),  # 第二次下跌
        np.linspace(0.25, 1.0, 5)  # 突破
    ])
    templates["double_bottom"] = smooth_template(double_bottom)
    
    # 4. 看涨吞没
    bullish_engulfing = np.concatenate([
        np.full(16, 0.8),
        [0.7, 0.6, 0.9, 1.0]
    ])
    templates["bullish_engulfing"] = bullish_engulfing
    
    # === 看跌形态 ===
    # 1. Shooting Star (射击之星)
    shooting_star = np.concatenate([
        np.linspace(0.2, 0.8, 15),
        [1.0, 0.85, 0.7, 0.6, 0.5]
    ])
    templates["shooting_star"] = smooth_template(shooting_star)
    
    # 2. 头肩顶
    head_shoulders_top = np.array([
        0.2, 0.6, 0.4,  # 左肩
        0.7, 1.0, 0.6, 0.4,  # 头部
        0.5, 0.7, 0.5,  # 右肩
        0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # 跌破
    ])
    templates["head_shoulder_top"] = smooth_template(head_shoulders_top)
    
    # 3. 双顶 (M顶)
    double_top = np.concatenate([
        np.linspace(0.2, 0.9, 6),  # 第一次上涨
        np.linspace(0.9, 0.5, 4),  # 回调
        np.linspace(0.5, 0.95, 5),  # 第二次上涨
        np.linspace(0.95, 0.0, 5)  # 跌破
    ])
    templates["double_top"] = smooth_template(double_top)
    
    # 4. 看跌吞没
    bearish_engulfing = np.concatenate([
        np.full(16, 0.3),
        [0.4, 0.5, 0.2, 0.1]
    ])
    templates["bearish_engulfing"] = bearish_engulfing
    
    # === 中继形态 ===
    # 1. 上升旗形
    bull_flag = np.concatenate([
        np.linspace(0.0, 1.0, 5),  # 旗杆
        np.linspace(0.95, 0.8, 15)  # 旗面
    ])
    templates["bull_flag"] = bull_flag
    
    # 2. 下降旗形
    bear_flag = np.concatenate([
        np.linspace(1.0, 0.0, 5),  # 旗杆
        np.linspace(0.05, 0.2, 15)  # 旗面
    ])
    templates["bear_flag"] = bear_flag
    
    # 3. 对称三角收敛
    triangle = np.concatenate([
        np.linspace(0.8, 0.2, 10),
        np.linspace(0.2, 0.8, 10)
    ])
    templates["triangle"] = triangle
    
    # 4. 下跌后横盘（压缩起点）
    decline_consolidation = np.concatenate([
        np.linspace(1.0, 0.3, 8),  # 快速下跌
        np.full(12, 0.3)  # 横盘
    ])
    templates["decline_consolidation"] = decline_consolidation
    
    return templates


def extract_dtw_features(
    df: pd.DataFrame,
    price_col: str = "close",
    window: int = 20,
    templates: Optional[Dict[str, np.ndarray]] = None,
    compute_only_near_sr: bool = False,
    sr_dist_col: Optional[str] = None,
    sr_threshold: float = 1.0,
) -> pd.DataFrame:
    """
    提取DTW形态匹配特征
    
    Args:
        df: DataFrame with price data
        price_col: Column name for price
        window: Window size for DTW matching
        templates: Custom templates dict (if None, uses default templates)
        compute_only_near_sr: Only compute DTW near SR levels (for efficiency)
        sr_dist_col: Column name for distance to nearest SR (if compute_only_near_sr=True)
        sr_threshold: Threshold for "near SR" (in ATR units)
    
    Returns:
        DataFrame with DTW distance features:
        - dtw_{template_name}_dist: DTW distance to each template
        - dtw_min_dist: Minimum distance across all templates
        - dtw_best_match: Name of best matching template
    """
    if not DTW_AVAILABLE:
        # Return empty features if dtaidistance is not available
        default_templates = create_dtw_templates()
        return pd.DataFrame(
            index=df.index,
            columns=[f"dtw_{name}_dist" for name in default_templates.keys()] + 
                    ["dtw_min_dist", "dtw_best_match"],
        ).fillna(1.0)
    
    df = df.copy()
    
    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found")
    
    if templates is None:
        templates = create_dtw_templates()
    
    # Initialize result DataFrame
    result_cols = [f"dtw_{name}_dist" for name in templates.keys()]
    result_cols.extend(["dtw_min_dist", "dtw_best_match"])
    result = pd.DataFrame(
        index=df.index,
        columns=result_cols,
    )
    
    prices = df[price_col].values
    n = len(df)
    
    # Determine which indices to compute
    if compute_only_near_sr and sr_dist_col and sr_dist_col in df.columns:
        # Only compute near SR levels
        # sr_dist_col is in price units, need to normalize by ATR
        sr_dist = df[sr_dist_col].abs()
        
        # If ATR column exists, normalize by ATR
        if "atr" in df.columns:
            atr = df["atr"].fillna(df["atr"].median())
            sr_dist_normalized = sr_dist / (atr + 1e-8)
            compute_mask = sr_dist_normalized <= sr_threshold
        else:
            # Fallback: use absolute distance (less accurate)
            compute_mask = sr_dist <= sr_threshold
        
        compute_indices = np.where(compute_mask)[0]
        if len(compute_indices) == 0:
            # If no indices match, compute for all (fallback)
            compute_indices = np.arange(window, n)
    else:
        # Compute for all indices
        compute_indices = np.arange(window, n)
    
    # Compute DTW distances
    for i in compute_indices:
        if i < window:
            continue
        
        # Get price window
        price_window = prices[i - window : i]
        
        # Skip if too many NaN or constant values
        if np.sum(np.isnan(price_window)) > window * 0.1:
            continue
        if np.std(price_window) < 1e-8:
            continue
        
        # Normalize price window
        norm_price = normalize_series(price_window)
        
        # Compute DTW distance to each template
        min_dist = np.inf
        best_match = None
        
        for template_name, template in templates.items():
            try:
                # Normalize template
                norm_template = normalize_series(template)
                
                # Compute DTW distance
                distance = dtw.distance(norm_price, norm_template)
                
                # Store distance
                col_name = f"dtw_{template_name}_dist"
                if col_name in result.columns:
                    result.loc[df.index[i], col_name] = distance
                
                # Track minimum
                if distance < min_dist:
                    min_dist = distance
                    best_match = template_name
                    
            except Exception:
                # Skip failed DTW computations
                continue
        
        # Store minimum distance and best match
        if min_dist < np.inf:
            result.loc[df.index[i], "dtw_min_dist"] = min_dist
            result.loc[df.index[i], "dtw_best_match"] = best_match
    
    # Forward fill NaN values
    result = result.fillna(method="ffill")
    
    # Fill remaining NaN with large values (no match)
    for col in result.columns:
        if col != "dtw_best_match":
            result[col] = result[col].fillna(1.0)
        else:
            result[col] = result[col].fillna("none")
    
    return result

