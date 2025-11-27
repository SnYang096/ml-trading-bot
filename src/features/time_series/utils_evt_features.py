"""
EVT (Extreme Value Theory) 特征提取器
用于尾部风险预警和黑天鹅检测
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
    threshold_quantile: float = 0.95,
    min_excesses: int = 10,
    separate_tails: bool = True,
) -> pd.DataFrame:
    """
    提取EVT尾部风险特征
    
    Args:
        df: DataFrame with price data
        price_col: Column name for price
        window: Rolling window size for EVT fitting
        threshold_quantile: Quantile for threshold (e.g., 0.95 for 95th percentile)
        min_excesses: Minimum number of excesses required for fitting
        separate_tails: Whether to model left and right tails separately
    
    Returns:
        DataFrame with EVT features:
        - evt_tail_shape: Shape parameter ξ (xi) for overall tail
        - evt_tail_shape_left: ξ for left tail (crashes)
        - evt_tail_shape_right: ξ for right tail (bubbles)
        - evt_scale: Scale parameter σ
        - evt_var_99: 99% VaR estimate
        - evt_es_99: 99% Expected Shortfall estimate
    """
    if not SCIPY_AVAILABLE:
        # Return empty features if scipy is not available
        return pd.DataFrame(
            index=df.index,
            columns=[
                "evt_tail_shape",
                "evt_tail_shape_left",
                "evt_tail_shape_right",
                "evt_scale",
                "evt_var_99",
                "evt_es_99",
            ],
        ).fillna(0.3)  # Default safe value
    
    df = df.copy()
    
    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found")
    
    # Calculate returns
    returns = df[price_col].pct_change().dropna()
    
    if len(returns) < window + 10:
        # Not enough data
        return pd.DataFrame(
            index=df.index,
            columns=[
                "evt_tail_shape",
                "evt_tail_shape_left",
                "evt_tail_shape_right",
                "evt_scale",
                "evt_var_99",
                "evt_es_99",
            ],
        ).fillna(0.3)
    
    # Initialize result arrays
    n = len(df)
    xi_overall = np.full(n, np.nan)
    xi_left = np.full(n, np.nan)
    xi_right = np.full(n, np.nan)
    scale = np.full(n, np.nan)
    var_99 = np.full(n, np.nan)
    es_99 = np.full(n, np.nan)
    
    # Rolling EVT fitting
    returns_arr = returns.values
    returns_index = returns.index
    
    for i in range(window, len(returns)):
        try:
            # Get window data
            window_returns = returns_arr[i - window : i]
            
            # Skip if too many NaN
            if np.sum(np.isnan(window_returns)) > window * 0.1:
                continue
            if np.std(window_returns) < 1e-8:
                continue
            
            # Set threshold
            u = np.quantile(window_returns, threshold_quantile)
            
            # Extract excesses (only positive excesses for right tail)
            excesses = window_returns[window_returns > u] - u
            
            if len(excesses) < min_excesses:
                continue
            
            # Fit GPD
            try:
                xi, loc, sigma = genpareto.fit(excesses, floc=0)
                
                # Store results
                df_idx = df.index.get_loc(returns_index[i])
                if df_idx < n:
                    xi_overall[df_idx] = xi
                    scale[df_idx] = sigma
                    
                    # Calculate VaR and ES (simplified)
                    # VaR_99 = u + σ/ξ * ((1-0.99)^(-ξ) - 1)
                    if xi > 0:
                        var_99[df_idx] = u + (sigma / xi) * ((0.01) ** (-xi) - 1)
                        # ES_99 ≈ VaR_99 * (1 + ξ) / (1 - ξ) for ξ < 1
                        if xi < 1:
                            es_99[df_idx] = var_99[df_idx] * (1 + xi) / (1 - xi)
                    else:
                        # For ξ <= 0, use exponential approximation
                        var_99[df_idx] = u + sigma * np.log(0.01)
                        es_99[df_idx] = var_99[df_idx] + sigma
            except Exception:
                pass
            
            # Separate left and right tails
            if separate_tails:
                # Left tail (negative returns, absolute value)
                u_left = np.quantile(window_returns, 1 - threshold_quantile)
                excesses_left = -(window_returns[window_returns < u_left] - u_left)
                
                if len(excesses_left) >= min_excesses:
                    try:
                        xi_l, _, _ = genpareto.fit(excesses_left, floc=0)
                        df_idx = df.index.get_loc(returns_index[i])
                        if df_idx < n:
                            xi_left[df_idx] = xi_l
                    except Exception:
                        pass
                
                # Right tail (positive returns)
                u_right = np.quantile(window_returns, threshold_quantile)
                excesses_right = window_returns[window_returns > u_right] - u_right
                
                if len(excesses_right) >= min_excesses:
                    try:
                        xi_r, _, _ = genpareto.fit(excesses_right, floc=0)
                        df_idx = df.index.get_loc(returns_index[i])
                        if df_idx < n:
                            xi_right[df_idx] = xi_r
                    except Exception:
                        pass
                        
        except Exception:
            # Skip failed fits
            continue
    
    # Create result DataFrame
    result = pd.DataFrame(
        {
            "evt_tail_shape": xi_overall,
            "evt_tail_shape_left": xi_left,
            "evt_tail_shape_right": xi_right,
            "evt_scale": scale,
            "evt_var_99": var_99,
            "evt_es_99": es_99,
        },
        index=df.index,
    )
    
    # Forward fill NaN values
    result = result.fillna(method="ffill")
    
    # Fill remaining NaN with safe defaults
    result["evt_tail_shape"] = result["evt_tail_shape"].fillna(0.3)
    result["evt_tail_shape_left"] = result["evt_tail_shape_left"].fillna(0.3)
    result["evt_tail_shape_right"] = result["evt_tail_shape_right"].fillna(0.3)
    result["evt_scale"] = result["evt_scale"].fillna(0.01)
    result["evt_var_99"] = result["evt_var_99"].fillna(-0.05)  # Default 5% VaR
    result["evt_es_99"] = result["evt_es_99"].fillna(-0.08)  # Default 8% ES
    
    return result

