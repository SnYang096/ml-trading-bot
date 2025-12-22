"""
GARCH / GJR-GARCH / FIGARCH 特征提取器
用于捕捉波动聚集性和杠杆效应
"""

import numpy as np
import pandas as pd
import warnings

from src.features.registry import register_feature

warnings.filterwarnings("ignore")

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    print("⚠️ arch package not available. GARCH features will be disabled.")


def _extract_garch_features_from_close(
    close: pd.Series,
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
    close = pd.to_numeric(close, errors="coerce").astype(float)
    index = close.index

    cols = [
        "garch_volatility",
        "garch_persistence",
        "garch_leverage_gamma",
        "garch_alpha",
        "garch_beta",
    ]

    if not ARCH_AVAILABLE:
        # Return empty features if arch is not available
        return pd.DataFrame(index=index, columns=cols).fillna(0.0)
    
    # Calculate returns
    returns = close.pct_change().dropna()
    
    if len(returns) <= window:
        # Not enough data to fit at least one rolling window
        return pd.DataFrame(index=index, columns=cols).fillna(0.0)
    
    # Initialize result arrays
    n = len(close)
    garch_vol = np.full(n, np.nan)
    persistence = np.full(n, np.nan)
    leverage_gamma = np.full(n, np.nan)
    garch_alpha = np.full(n, np.nan)
    garch_beta = np.full(n, np.nan)
    
    # Rolling GARCH fitting
    returns_arr = returns.values
    returns_index = returns.index
    idx_pos = index.get_indexer(returns_index)
    
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
            df_idx = int(idx_pos[i])
            
            if df_idx < 0 or df_idx >= n:
                continue
            if df_idx < n:
                garch_vol[df_idx] = np.sqrt(forecast.variance.values[-1, 0])
                
                # Get parameters
                alpha = res.params.get("alpha[1]", 0.0)
                beta = res.params.get("beta[1]", 0.0)
                garch_alpha[df_idx] = alpha
                garch_beta[df_idx] = beta
                # Persistence should be non-negative; numerical estimation can produce tiny negatives.
                persistence[df_idx] = max(float(alpha + beta), 0.0)
            
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
                        # Leverage effect coefficient is expected to be non-negative in most formulations;
                        # clip small negative estimates to 0 for stability/consistency.
                        leverage_gamma[df_idx] = max(float(gamma), 0.0)
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
        index=index,
    )

    # IMPORTANT:
    # - Keep `garch_volatility` NaN where the fit fails, so streaming-vs-batch comparisons
    #   can drop those points instead of propagating stale values via ffill.
    # - For parameter-like columns, forward-fill provides stable outputs for downstream usage.
    param_cols = ["garch_persistence", "garch_leverage_gamma", "garch_alpha", "garch_beta"]
    result[param_cols] = result[param_cols].ffill().fillna(0.0)

    # Final safety: keep semantically non-negative columns non-negative
    for col in ("garch_volatility", "garch_persistence", "garch_leverage_gamma"):
        if col in result.columns:
            result[col] = result[col].clip(lower=0.0)
    
    return result


@register_feature("extract_garch_features_from_series", category="garch")
def extract_garch_features_from_series(
    *,
    close: pd.Series,
    window: int = 60,
    garch_p: int = 1,
    garch_q: int = 1,
    use_gjr: bool = True,
    use_figarch: bool = False,
) -> pd.DataFrame:
    """
    Narrow-IO GARCH entrypoint for the feature DAG.

    Uses legacy implementation internally but only constructs a slim DF containing `close`.
    """
    return _extract_garch_features_from_close(
        close=close,
        window=window,
        garch_p=garch_p,
        garch_q=garch_q,
        use_gjr=use_gjr,
        use_figarch=use_figarch,
    )
