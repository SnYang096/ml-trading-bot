"""
GARCH / GJR-GARCH / FIGARCH 特征提取器
用于捕捉波动聚集性和杠杆效应
"""

import numpy as np
import pandas as pd
from typing import Optional
import warnings

warnings.filterwarnings("ignore")

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    print("⚠️ arch package not available. GARCH features will be disabled.")


def extract_garch_features(
    df: pd.DataFrame,
    price_col: str = "close",
    window: int = 60,
    garch_p: int = 1,
    garch_q: int = 1,
    use_gjr: bool = True,
    use_figarch: bool = False,
) -> pd.DataFrame:
    """
    提取GARCH相关特征
    
    Args:
        df: DataFrame with price data
        price_col: Column name for price
        window: Rolling window size for GARCH fitting
        garch_p: GARCH p parameter
        garch_q: GARCH q parameter
        use_gjr: Whether to use GJR-GARCH (leverage effect)
        use_figarch: Whether to use FIGARCH (long memory)
    
    Returns:
        DataFrame with GARCH features:
        - garch_volatility: Predicted next-period volatility
        - garch_persistence: α + β (volatility clustering strength)
        - garch_leverage_gamma: Leverage effect coefficient (if use_gjr=True)
        - garch_alpha: GARCH α parameter
        - garch_beta: GARCH β parameter
    """
    if not ARCH_AVAILABLE:
        # Return empty features if arch is not available
        return pd.DataFrame(
            index=df.index,
            columns=[
                "garch_volatility",
                "garch_persistence",
                "garch_leverage_gamma",
                "garch_alpha",
                "garch_beta",
            ],
        ).fillna(0.0)
    
    df = df.copy()
    
    # Calculate returns
    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found")
    
    returns = df[price_col].pct_change().dropna()
    
    if len(returns) < window + 10:
        # Not enough data
        return pd.DataFrame(
            index=df.index,
            columns=[
                "garch_volatility",
                "garch_persistence",
                "garch_leverage_gamma",
                "garch_alpha",
                "garch_beta",
            ],
        ).fillna(0.0)
    
    # Initialize result arrays
    n = len(df)
    garch_vol = np.full(n, np.nan)
    persistence = np.full(n, np.nan)
    leverage_gamma = np.full(n, np.nan)
    garch_alpha = np.full(n, np.nan)
    garch_beta = np.full(n, np.nan)
    
    # Rolling GARCH fitting
    returns_arr = returns.values
    returns_index = returns.index
    
    for i in range(window, len(returns)):
        try:
            # Get window data
            window_returns = returns_arr[i - window : i]
            
            # Skip if too many NaN or zero returns
            if np.sum(np.isnan(window_returns)) > window * 0.1:
                continue
            if np.std(window_returns) < 1e-8:
                continue
            
            # Fit GARCH(1,1)
            model = arch_model(
                window_returns,
                vol="Garch",
                p=garch_p,
                q=garch_q,
                dist="Normal",
            )
            res = model.fit(disp="off", show_warning=False)
            
            # Get forecast
            forecast = res.forecast(horizon=1, reindex=False)
            garch_vol_idx = returns_index[i]
            df_idx = df.index.get_loc(garch_vol_idx)
            
            if df_idx < n:
                garch_vol[df_idx] = np.sqrt(forecast.variance.values[-1, 0])
                
                # Get parameters
                alpha = res.params.get("alpha[1]", 0.0)
                beta = res.params.get("beta[1]", 0.0)
                garch_alpha[df_idx] = alpha
                garch_beta[df_idx] = beta
                persistence[df_idx] = alpha + beta
            
            # Fit GJR-GARCH for leverage effect
            if use_gjr:
                try:
                    model_gjr = arch_model(
                        window_returns,
                        vol="Garch",
                        p=garch_p,
                        o=1,  # Leverage term
                        q=garch_q,
                        dist="Normal",
                    )
                    res_gjr = model_gjr.fit(disp="off", show_warning=False)
                    gamma = res_gjr.params.get("gamma[1]", 0.0)
                    
                    if df_idx < n:
                        leverage_gamma[df_idx] = gamma
                except Exception:
                    # GJR fitting failed, skip
                    pass
            
            # Fit FIGARCH (optional, slower)
            if use_figarch:
                try:
                    model_figarch = arch_model(
                        window_returns,
                        vol="FIGARCH",
                        p=1,
                        q=1,
                        dist="Normal",
                    )
                    res_figarch = model_figarch.fit(disp="off", show_warning=False)
                    # FIGARCH has different parameter structure
                    # For now, we skip detailed FIGARCH features
                except Exception:
                    pass
                    
        except Exception as e:
            # Skip failed fits
            continue
    
    # Create result DataFrame
    result = pd.DataFrame(
        {
            "garch_volatility": garch_vol,
            "garch_persistence": persistence,
            "garch_leverage_gamma": leverage_gamma,
            "garch_alpha": garch_alpha,
            "garch_beta": garch_beta,
        },
        index=df.index,
    )
    
    # Forward fill NaN values (use last valid value)
    result = result.fillna(method="ffill").fillna(0.0)
    
    return result

