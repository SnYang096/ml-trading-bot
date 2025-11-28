"""
EVT (Extreme Value Theory) 特征提取器
用于尾部风险预警和黑天鹅检测

目的：通过历史极端下跌事件，估计未来发生"黑天鹅"的概率和损失程度，
      并将 ξ、VaR、ES 作为机器学习特征输入 LightGBM。

聚焦左尾风险（暴跌风险），基于 Peak Over Threshold (POT) 方法拟合广义帕累托分布（GPD）

核心特征（用于LightGBM）：
- evt_tail_shape_left (ξ): 尾部形状参数，反映极端事件的概率分布特征
  * ξ > 0: 重尾分布（金融数据通常如此）
  * ξ > 0.3: 极高尾部风险（黑天鹅风险高）
  * 0.1 < ξ ≤ 0.3: 典型金融重尾（常见范围）
  * ξ ≤ 0.1: 相对较轻尾部
  * ξ > 0.5: 方差无限（极端罕见，理论极限）
  * ξ ≈ 0: 指数分布（中等尾部风险）
  * ξ < 0: 薄尾分布（极端事件概率低，金融中罕见）
- evt_var_99_left: 99% VaR，估计未来1%概率下的最大损失（负值）
- evt_es_99_left: 99% ES，估计超过VaR条件下的平均损失（负值）
"""

import numpy as np
import pandas as pd
from typing import Optional
import warnings

warnings.filterwarnings("ignore")

try:
    from scipy.stats import genpareto
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️ scipy package not available. EVT features will be disabled.")


def extract_evt_features(
    df: pd.DataFrame,
    price_col: str = "close",
    window: int = 120,
    threshold_quantile: float = 0.1,
    min_excesses: int = 10,
    separate_tails: bool = True,
    var_confidence: float = 0.99,
) -> pd.DataFrame:
    """
    提取EVT尾部风险特征（聚焦左尾暴跌风险），用于LightGBM模型输入
    
    目的：通过历史极端下跌事件，估计未来发生"黑天鹅"的概率和损失程度
    
    Args:
        df: DataFrame with price data
        price_col: Column name for price
        window: Rolling window size for EVT fitting (default 120)
        threshold_quantile: Quantile for left tail threshold (default 0.1 for 10th percentile)
                          Lower values capture more extreme events
        min_excesses: Minimum number of excesses required for fitting (default 10)
        separate_tails: Whether to model left and right tails separately (default True)
                       If True, also outputs right tail features for bubble detection
                       If False, only outputs left tail features (focus on crash risk)
        var_confidence: Confidence level for VaR/ES (default 0.99 for 99% VaR)
    
    Returns:
        DataFrame with EVT features (suitable for LightGBM):
        
        核心特征（左尾 - 暴跌风险，重点）：
        - evt_tail_shape_left (ξ): 尾部形状参数，反映极端事件概率分布
          * ξ > 0: 重尾分布（金融数据通常如此）
          * ξ > 0.3: 极高尾部风险（黑天鹅风险高）
          * 0.1 < ξ ≤ 0.3: 典型金融重尾（常见范围，金融收益率通常在此区间）
          * ξ ≤ 0.1: 相对较轻尾部
          * ξ > 0.5: 方差无限（极端罕见，理论极限）
          * ξ ≈ 0: 指数分布（中等尾部风险）
          * ξ < 0: 薄尾分布（极端事件概率低，金融中罕见）
        - evt_scale_left (σ): 尺度参数，反映尾部损失的尺度
        - evt_var_99_left: 99% VaR，估计未来1%概率下的最大损失（负值，绝对值越大风险越高）
        - evt_es_99_left: 99% ES，估计超过VaR条件下的平均损失（负值，绝对值越大风险越高）
        
        向后兼容列名（映射到左尾）：
        - evt_tail_shape: 等同于 evt_tail_shape_left
        - evt_scale: 等同于 evt_scale_left
        - evt_var_99: 等同于 evt_var_99_left
        - evt_es_99: 等同于 evt_es_99_left
        
        可选特征（右尾 - 泡沫风险，仅当separate_tails=True时）：
        - evt_tail_shape_right: 右尾形状参数
        - evt_scale_right: 右尾尺度参数
        - evt_var_99_right: 右尾99% VaR（正值，表示预期最大涨幅）
        - evt_es_99_right: 右尾99% ES（正值）
        
    Note:
        - LightGBM可以处理NaN值，但建议在特征工程阶段处理缺失值
        - VaR和ES为负值表示损失，绝对值越大表示风险越高
        - 特征已使用ffill()向前填充，保留最后有效估计
    """
    if not SCIPY_AVAILABLE:
        # Return empty features if scipy is not available
        cols = [
            "evt_tail_shape_left",
            "evt_scale_left",
            "evt_var_99_left",
            "evt_es_99_left",
        ]
        if separate_tails:
            cols.extend([
                "evt_tail_shape_right",
                "evt_scale_right",
                "evt_var_99_right",
                "evt_es_99_right",
            ])
        return pd.DataFrame(index=df.index, columns=cols)
    
    df = df.copy()
    
    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found")
    
    # Calculate returns
    returns = df[price_col].pct_change().dropna()
    
    if len(returns) < window + 10:
        # Not enough data
        cols = [
            "evt_tail_shape_left",
            "evt_scale_left",
            "evt_var_99_left",
            "evt_es_99_left",
        ]
        if separate_tails:
            cols.extend([
                "evt_tail_shape_right",
                "evt_scale_right",
                "evt_var_99_right",
                "evt_es_99_right",
            ])
        return pd.DataFrame(index=df.index, columns=cols)
    
    # Initialize result arrays
    n = len(df)
    xi_left = np.full(n, np.nan)
    scale_left = np.full(n, np.nan)
    var_99_left = np.full(n, np.nan)
    es_99_left = np.full(n, np.nan)
    
    if separate_tails:
        xi_right = np.full(n, np.nan)
        scale_right = np.full(n, np.nan)
        var_99_right = np.full(n, np.nan)
        es_99_right = np.full(n, np.nan)
    
    # Rolling EVT fitting
    returns_arr = returns.values
    returns_index = returns.index
    
    # Tail probability: proportion of returns below threshold
    tail_prob_left = threshold_quantile
    tail_prob_right = 1 - threshold_quantile if separate_tails else None
    
    # Probability level for VaR/ES (e.g., 0.01 for 99% VaR)
    p_level = 1 - var_confidence
    
    for i in range(window, len(returns)):
        try:
            # Get window data
            window_returns = returns_arr[i - window : i]
            
            # Skip if too many NaN
            if np.sum(np.isnan(window_returns)) > window * 0.1:
                continue
            if np.std(window_returns) < 1e-8:
                continue
            
            df_idx = df.index.get_loc(returns_index[i])
            if df_idx >= n:
                continue
            
            # ========== LEFT TAIL (CRASH RISK) - 核心特征 ==========
            # 目标：通过历史极端下跌事件，估计未来"黑天鹅"的概率和损失程度
            # Threshold: lower quantile (e.g., 10th percentile)
            u_left = np.quantile(window_returns, threshold_quantile)
            
            # Extract excesses: positive values for GPD fitting
            # 提取超过阈值的超额损失（转换为正数，符合GPD要求）
            excesses_left = u_left - window_returns[window_returns < u_left]
            
            if len(excesses_left) >= min_excesses:
                try:
                    # Fit GPD to left tail excesses
                    xi_l, loc, sigma_l = genpareto.fit(excesses_left, floc=0)
                    
                    # Store shape and scale (核心特征用于LightGBM)
                    # ξ (xi): 尾部形状参数，反映极端事件的概率分布特征
                    #   - ξ > 0: 重尾分布（金融数据通常如此）
                    #   - ξ > 0.3: 极高尾部风险（黑天鹅风险高）
                    #   - 0.1 < ξ ≤ 0.3: 典型金融重尾（常见范围，金融收益率通常在此区间）
                    #   - ξ ≤ 0.1: 相对较轻尾部
                    #   - ξ > 0.5: 方差无限（极端罕见，理论极限）
                    #   - ξ ≈ 0: 指数分布（中等尾部风险）
                    #   - ξ < 0: 薄尾分布（极端事件概率低，金融中罕见）
                    xi_left[df_idx] = xi_l
                    scale_left[df_idx] = sigma_l
                    
                    # Calculate 99% VaR (left tail) - 核心风险指标
                    # VaR: 估计未来1%概率下的最大损失（负值）
                    # Formula: VaR_p = u - (σ/ξ) * [(p / ζ_u)^(-ξ) - 1]
                    # where ζ_u = P(X < u) = tail_prob_left
                    if xi_l > 0:
                        # Standard GPD formula
                        var_99_left[df_idx] = u_left - (sigma_l / xi_l) * (
                            (p_level / tail_prob_left) ** (-xi_l) - 1
                        )
                    elif xi_l == 0:
                        # Exponential case (ξ = 0)
                        var_99_left[df_idx] = u_left - sigma_l * np.log(p_level / tail_prob_left)
                    else:
                        # ξ < 0: bounded distribution
                        var_99_left[df_idx] = u_left - (sigma_l / abs(xi_l)) * (
                            (p_level / tail_prob_left) ** (-xi_l) - 1
                        )
                    
                    # Calculate 99% Expected Shortfall (Conditional VaR) - 核心风险指标
                    # ES: 估计超过VaR条件下的平均损失（负值，绝对值越大风险越高）
                    # ES_p = VaR_p - (σ + ξ * (VaR_p - u)) / (1 - ξ) for ξ < 1
                    if xi_l < 1:
                        if xi_l == 0:
                            # Exponential case
                            es_99_left[df_idx] = var_99_left[df_idx] - sigma_l
                        else:
                            es_99_left[df_idx] = var_99_left[df_idx] - (
                                (sigma_l + xi_l * (var_99_left[df_idx] - u_left)) / (1 - xi_l)
                            )
                    else:
                        # ξ >= 1: ES is infinite or undefined
                        es_99_left[df_idx] = np.nan
                        
                except Exception:
                    pass
            
            # ========== RIGHT TAIL (BUBBLE RISK, 可选) ==========
            # 注意：右尾特征是可选的，主要用于泡沫检测
            # 如果只关注暴跌风险，可以设置 separate_tails=False
            if separate_tails:
                # Threshold: upper quantile (e.g., 90th percentile)
                u_right = np.quantile(window_returns, 1 - threshold_quantile)
                
                # Extract excesses: positive values for GPD fitting
                excesses_right = window_returns[window_returns > u_right] - u_right
                
                if len(excesses_right) >= min_excesses:
                    try:
                        # Fit GPD to right tail excesses
                        xi_r, loc, sigma_r = genpareto.fit(excesses_right, floc=0)
                        
                        # Store shape and scale
                        xi_right[df_idx] = xi_r
                        scale_right[df_idx] = sigma_r
                        
                        # Calculate 99% VaR (right tail, positive value)
                        # For right tail: VaR_p = u + (σ/ξ) * [((1-p) / ζ_u)^(-ξ) - 1]
                        # where ζ_u = P(X > u) = tail_prob_right
                        if xi_r > 0:
                            var_99_right[df_idx] = u_right + (sigma_r / xi_r) * (
                                ((1 - p_level) / tail_prob_right) ** (-xi_r) - 1
                            )
                        elif xi_r == 0:
                            var_99_right[df_idx] = u_right + sigma_r * np.log((1 - p_level) / tail_prob_right)
                        else:
                            var_99_right[df_idx] = u_right + (sigma_r / abs(xi_r)) * (
                                ((1 - p_level) / tail_prob_right) ** (-xi_r) - 1
                            )
                        
                        # Calculate 99% ES (right tail)
                        if xi_r < 1:
                            if xi_r == 0:
                                es_99_right[df_idx] = var_99_right[df_idx] + sigma_r
                            else:
                                es_99_right[df_idx] = var_99_right[df_idx] + (
                                    (sigma_r + xi_r * (var_99_right[df_idx] - u_right)) / (1 - xi_r)
                                )
                        else:
                            es_99_right[df_idx] = np.nan
                            
                    except Exception:
                        pass
                        
        except Exception:
            # Skip failed fits
            continue
    
    # Create result DataFrame
    # New explicit column names (recommended)
    result_dict = {
        "evt_tail_shape_left": xi_left,
        "evt_scale_left": scale_left,
        "evt_var_99_left": var_99_left,
        "evt_es_99_left": es_99_left,
    }
    
    # Backward compatibility: old column names map to left tail (risk focus)
    result_dict.update({
        "evt_tail_shape": xi_left,  # Left tail shape (risk focus)
        "evt_scale": scale_left,     # Left tail scale
        "evt_var_99": var_99_left,   # Left tail VaR (risk focus)
        "evt_es_99": es_99_left,     # Left tail ES (risk focus)
    })
    
    if separate_tails:
        result_dict.update({
            "evt_tail_shape_right": xi_right,
            "evt_scale_right": scale_right,
            "evt_var_99_right": var_99_right,
            "evt_es_99_right": es_99_right,
        })
    
    result = pd.DataFrame(result_dict, index=df.index)
    
    # Forward fill NaN values (carry forward last valid estimate)
    result = result.ffill()
    
    # Fill remaining NaN with NaN (preserve missing data rather than arbitrary defaults)
    # This allows downstream processing to handle missing values appropriately
    # If needed, users can fill with historical means or other strategies
    
    return result

