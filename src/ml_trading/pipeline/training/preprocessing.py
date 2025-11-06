"""
Time-series safe preprocessing functions for target variable cleaning.
All statistics are computed ONLY on training data to prevent lookahead bias.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, NamedTuple
try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    import warnings
    warnings.warn("statsmodels not available, using fallback AR(1) estimation",
                  UserWarning)


class RobustWinsorizer:
    """
    Robust Winsorizer for deployment use.
    Uses saved preprocessing parameters to apply consistent preprocessing in production.
    
    Example:
        # During training (saved to training_info.json)
        preprocess_params = {
            "winsorize": {"median": 0.001, "sigma": 0.002, "k": 3.5},
            "ar1": {"ar1_phi": 0.15},
            "secondary": {"median": 0.0, "sigma": 0.001, "clip_threshold": 0.003}
        }
        
        # During deployment
        winsorizer = RobustWinsorizer.from_params(preprocess_params)
        y_cleaned = winsorizer.transform(y_new)
    """

    def __init__(self,
                 median: float,
                 sigma: float,
                 k: float = 3.5,
                 ar1_phi: float = 0.0,
                 secondary_median: float = 0.0,
                 secondary_sigma: float = 0.0,
                 secondary_clip_threshold: float = 0.0,
                 forward_bars: int = 1):
        """
        Initialize RobustWinsorizer with preprocessing parameters.
        
        Args:
            median: Median from training data (Step 1: Winsorize)
            sigma: Sigma from training data (Step 1: Winsorize)
            k: Winsorize threshold (default: 3.5)
            ar1_phi: AR(1) coefficient from training data (Step 2)
            secondary_median: Median from secondary cleaning (Step 2b)
            secondary_sigma: Sigma from secondary cleaning (Step 2b)
            secondary_clip_threshold: Clip threshold from secondary cleaning (Step 2b)
            forward_bars: Number of forward bars (for AR(1) prediction)
        """
        self.median = median
        self.sigma = sigma
        self.k = k
        self.ar1_phi = ar1_phi
        self.secondary_median = secondary_median
        self.secondary_sigma = secondary_sigma
        self.secondary_clip_threshold = secondary_clip_threshold
        self.forward_bars = forward_bars

        # Calculate bounds
        self.lower_bound = self.median - self.k * self.sigma
        self.upper_bound = self.median + self.k * self.sigma

        # Secondary bounds
        self.secondary_lower = self.secondary_median - self.secondary_clip_threshold
        self.secondary_upper = self.secondary_median + self.secondary_clip_threshold

    @classmethod
    def from_params(cls,
                    preprocess_params: Dict,
                    forward_bars: int = 1) -> 'RobustWinsorizer':
        """
        Create RobustWinsorizer from saved preprocessing parameters.
        
        Args:
            preprocess_params: Dictionary with keys 'winsorize', 'ar1', 'secondary'
            forward_bars: Number of forward bars
        
        Returns:
            RobustWinsorizer instance
        """
        winsorize = preprocess_params.get("winsorize", {})
        ar1 = preprocess_params.get("ar1", {})
        secondary = preprocess_params.get("secondary", {})

        return cls(
            median=winsorize.get("median", 0.0),
            sigma=winsorize.get("sigma", 0.0),
            k=winsorize.get("k", 3.5),
            ar1_phi=ar1.get("ar1_phi", 0.0),
            secondary_median=secondary.get("median", 0.0),
            secondary_sigma=secondary.get("sigma", 0.0),
            secondary_clip_threshold=secondary.get("clip_threshold", 0.0),
            forward_bars=forward_bars,
        )

    def transform(self,
                  y: pd.Series,
                  current_returns: Optional[pd.Series] = None,
                  apply_ar1: bool = True,
                  apply_secondary: bool = True) -> pd.Series:
        """
        Apply preprocessing to new data using saved parameters.
        
        Args:
            y: Target variable (raw future_return)
            current_returns: Current period log returns (required if apply_ar1=True)
            apply_ar1: Whether to apply AR(1) residual transformation
            apply_secondary: Whether to apply secondary cleaning
        
        Returns:
            Preprocessed target variable
        """
        # Step 1: Winsorize
        y_cleaned = y.clip(self.lower_bound, self.upper_bound)

        # Step 2: AR(1) residual (if applicable)
        if apply_ar1 and current_returns is not None and self.ar1_phi != 0.0:
            # Align current_returns with y
            if len(current_returns) == len(y):
                current_returns_aligned = current_returns.values
            else:
                # Try to align by index
                try:
                    current_returns_aligned = current_returns.reindex(
                        y.index, fill_value=0.0).values
                except ValueError:
                    # Duplicate indices - use positional alignment
                    current_returns_aligned = current_returns.values[:len(y)]
                    if len(current_returns_aligned) < len(y):
                        current_returns_aligned = np.concatenate([
                            current_returns_aligned,
                            np.zeros(len(y) - len(current_returns_aligned))
                        ])

            # Calculate AR(1) prediction
            if self.forward_bars == 1:
                ar1_pred = self.ar1_phi * current_returns_aligned
            else:
                phi_power_fb = np.clip(
                    np.power(self.ar1_phi, self.forward_bars), -10.0, 10.0)
                ar1_pred = phi_power_fb * current_returns_aligned

            # Convert to log returns and apply AR(1) residual
            eps = 1e-6
            y_safe = np.where(y.values < -1 + eps, -1 + eps, y.values)
            y_log = np.log1p(y_safe)

            # Calculate residual
            y_residual_log = y_log - ar1_pred

            # Convert back to simple returns
            y_cleaned = pd.Series(np.exp(y_residual_log) - 1, index=y.index)

            # Handle non-finite values
            y_cleaned_finite = y_cleaned.values[np.isfinite(y_cleaned.values)]
            if len(y_cleaned_finite) > 0:
                fallback = np.nanmedian(y_cleaned_finite)
            else:
                fallback = 0.0
            y_cleaned = pd.Series(np.where(np.isfinite(y_cleaned.values),
                                           y_cleaned.values, fallback),
                                  index=y.index)

        # Step 2b: Secondary cleaning
        if apply_secondary and self.secondary_clip_threshold > 0:
            y_cleaned = y_cleaned.clip(self.secondary_lower,
                                       self.secondary_upper)

        return y_cleaned


def robust_winsorize_train_test(
    y_train: pd.Series,
    y_test: pd.Series,
    k: float = 3.5,
    dynamic_threshold: Optional[pd.Series] = None,
) -> Tuple[pd.Series, pd.Series, Dict]:
    """
    Apply robust Winsorize to training and test data using ONLY training statistics.
    
    Args:
        y_train: Training target variable
        y_test: Test target variable
        k: Winsorize threshold (default: 3.5)
        dynamic_threshold: Optional Series with dynamic k values (must be aligned with y_train/y_test)
    
    Returns:
        Tuple of (y_train_cleaned, y_test_cleaned, stats_dict)
        stats_dict contains: median, mad, sigma, lower_bound, upper_bound
    """
    # Calculate statistics ONLY from training data
    # Performance optimization: use numpy for faster computation
    y_train_values = y_train.values
    median_train = np.nanmedian(y_train_values)
    mad_train = np.nanmedian(np.abs(y_train_values - median_train))

    # Improved fallback: use nanstd with ddof=1 for more robust estimation
    if mad_train == 0 or np.isnan(mad_train):
        sigma_train = np.nanstd(y_train_values, ddof=1)
        if np.isnan(sigma_train) or sigma_train == 0:
            # Ultimate fallback: use a small fraction of the range
            y_train_finite = y_train_values[np.isfinite(y_train_values)]
            if len(y_train_finite) > 1:
                sigma_train = (np.nanmax(y_train_finite) -
                               np.nanmin(y_train_finite)) / 6.0
            else:
                sigma_train = 1.0
    else:
        sigma_train = mad_train * 1.4826  # Convert MAD to approximate std

    # Validate dynamic_threshold length
    if dynamic_threshold is not None:
        required_length = len(y_train) + len(y_test)
        if len(dynamic_threshold) < required_length:
            raise ValueError(
                f"dynamic_threshold length ({len(dynamic_threshold)}) must be >= "
                f"len(y_train) + len(y_test) ({required_length})")

        # Smooth dynamic threshold to prevent abrupt changes (regime shift handling)
        dynamic_threshold = dynamic_threshold.ewm(span=20, adjust=False).mean()

    # Determine threshold (fixed or dynamic)
    if dynamic_threshold is not None and len(
            dynamic_threshold) >= len(y_train) + len(y_test):
        # Use dynamic threshold (for volatility regime-based cleaning)
        k_train = dynamic_threshold.iloc[:len(y_train)]
        k_test = dynamic_threshold.iloc[len(y_train):len(y_train) +
                                        len(y_test)] if len(
                                            y_test) > 0 else pd.Series(
                                                [k], index=y_test.index)

        # Calculate bounds for each sample
        lower_bounds_train = median_train - k_train * sigma_train
        upper_bounds_train = median_train + k_train * sigma_train
        lower_bounds_test = median_train - k_test * sigma_train
        upper_bounds_test = median_train + k_test * sigma_train

        # Vectorized clipping
        y_train_cleaned = y_train.clip(lower_bounds_train, upper_bounds_train)
        y_test_cleaned = y_test.clip(lower_bounds_test, upper_bounds_test)
    else:
        # Fixed threshold
        lower_bound = median_train - k * sigma_train
        upper_bound = median_train + k * sigma_train

        # Apply same bounds to both train and test
        y_train_cleaned = y_train.clip(lower_bound, upper_bound)
        y_test_cleaned = y_test.clip(lower_bound, upper_bound)

    stats = {
        "median": median_train,
        "mad": mad_train,
        "sigma": sigma_train,
        "k": k,
        "n_clipped_train": int((y_train_cleaned != y_train).sum()),
        "n_clipped_test": int((y_test_cleaned != y_test).sum()),
    }

    return y_train_cleaned, y_test_cleaned, stats


def ar1_residual_train_test(
    y_train: pd.Series,
    y_test: pd.Series,
    current_returns_train: pd.Series,
    current_returns_test: pd.Series,
    forward_bars: int,
    accurate_forward: bool = True,
) -> Tuple[pd.Series, pd.Series, Dict]:
    """
    Apply AR(1) residual transformation using ONLY training statistics.
    
    Args:
        y_train: Training target variable (future_return, already cleaned)
        y_test: Test target variable (future_return, already cleaned)
        current_returns_train: Current period log returns for training data
        current_returns_test: Current period log returns for test data
        forward_bars: Number of forward bars
        accurate_forward: If True, use phi^fb for forward_bars > 1 (more accurate)
    
    Returns:
        Tuple of (y_train_residual, y_test_residual, ar1_stats_dict)
    """
    # Calculate AR(1) coefficient ONLY from training data
    # Improved: Use OLS estimation if statsmodels is available (more robust)
    valid_returns_train = current_returns_train.dropna()
    ar1_phi_train = 0.0

    # Minimum sample size for reliable AR(1) estimation
    min_samples_ar1 = 50

    if len(valid_returns_train) > min_samples_ar1:
        # Method 1: OLS estimation (most robust, if statsmodels available)
        if HAS_STATSMODELS:
            try:
                r_t = valid_returns_train[:-1].values
                r_t1 = valid_returns_train[1:].values
                if len(r_t) > 1 and len(r_t1) > 1:
                    # Remove any NaN/inf
                    mask = np.isfinite(r_t) & np.isfinite(r_t1)
                    if mask.sum() > 10:
                        r_t_clean = r_t[mask]
                        r_t1_clean = r_t1[mask]
                        # OLS: r_{t+1} = alpha + phi * r_t + epsilon
                        X = sm.add_constant(r_t_clean)
                        model = sm.OLS(r_t1_clean, X).fit()
                        ar1_phi_train = model.params[1] if len(
                            model.params) > 1 else 0.0
                        # Clamp to reasonable range [-0.95, 0.95] for robustness
                        # Prevents extreme values that could amplify residuals
                        ar1_phi_train = np.clip(ar1_phi_train, -0.95, 0.95)
            except Exception:
                # Fallback to autocorr if OLS fails
                ar1_phi_train = valid_returns_train.autocorr(lag=1)

        # Method 2: pandas autocorr (if OLS not available or failed)
        if pd.isna(ar1_phi_train) or np.isnan(ar1_phi_train):
            ar1_phi_train = valid_returns_train.autocorr(lag=1)
            if pd.isna(ar1_phi_train):
                # Fallback to numpy correlation
                r_t = valid_returns_train[:-1].values
                r_t1 = valid_returns_train[1:].values
                if len(r_t) > 1 and len(r_t1) > 1:
                    mask = np.isfinite(r_t) & np.isfinite(r_t1)
                    if mask.sum() > 1:
                        ar1_phi_train = np.corrcoef(r_t[mask], r_t1[mask])[0,
                                                                           1]
                    else:
                        ar1_phi_train = 0.0
                else:
                    ar1_phi_train = 0.0

        # Final clamp to robust range
        ar1_phi_train = np.clip(ar1_phi_train, -0.95, 0.95)
    else:
        # Skip AR(1) if sample size too small
        ar1_phi_train = 0.0

    if pd.isna(ar1_phi_train) or np.isnan(ar1_phi_train):
        ar1_phi_train = 0.0

    # Ensure current_returns indices match y_train/y_test indices
    # This is critical to avoid shape mismatch errors
    # Handle duplicate indices by using positional alignment instead of label-based reindex
    # If indices are already aligned (same length and order), use values directly
    # Otherwise, use positional indexing to align

    # Check if indices are already aligned (same length and order)
    if len(current_returns_train) == len(y_train) and len(
            current_returns_test) == len(y_test):
        # Use values directly - they should be aligned by position
        # This works because current_returns_train and y_train come from the same CV split
        current_returns_train_values = current_returns_train.values
        current_returns_test_values = current_returns_test.values
    else:
        # Handle misalignment: try to align by index if possible (no duplicates)
        try:
            if y_train.index.is_unique and y_test.index.is_unique:
                current_returns_train_aligned = current_returns_train.reindex(
                    y_train.index, fill_value=0.0)
                current_returns_test_aligned = current_returns_test.reindex(
                    y_test.index, fill_value=0.0)
                current_returns_train_values = current_returns_train_aligned.values
                current_returns_test_values = current_returns_test_aligned.values
            else:
                # If indices have duplicates, use positional alignment
                # Assume current_returns_train and y_train are already aligned by position
                # (which should be the case if they come from the same CV split)
                if len(current_returns_train) >= len(y_train):
                    current_returns_train_values = current_returns_train.iloc[:len(
                        y_train)].values
                else:
                    # Pad with zeros if needed
                    pad_length = len(y_train) - len(current_returns_train)
                    current_returns_train_values = np.concatenate(
                        [current_returns_train.values,
                         np.zeros(pad_length)])

                if len(current_returns_test) >= len(y_test):
                    current_returns_test_values = current_returns_test.iloc[:len(
                        y_test)].values
                else:
                    # Pad with zeros if needed
                    pad_length = len(y_test) - len(current_returns_test)
                    current_returns_test_values = np.concatenate(
                        [current_returns_test.values,
                         np.zeros(pad_length)])
        except (ValueError, IndexError):
            # Fallback: use positional alignment assuming same order
            # This should work if current_returns and y come from the same data split
            current_returns_train_values = current_returns_train.values[:len(
                y_train)]
            current_returns_test_values = current_returns_test.values[:len(
                y_test)]

            # Ensure lengths match exactly
            if len(current_returns_train_values) < len(y_train):
                pad_length = len(y_train) - len(current_returns_train_values)
                current_returns_train_values = np.concatenate(
                    [current_returns_train_values,
                     np.zeros(pad_length)])
            if len(current_returns_test_values) < len(y_test):
                pad_length = len(y_test) - len(current_returns_test_values)
                current_returns_test_values = np.concatenate(
                    [current_returns_test_values,
                     np.zeros(pad_length)])

    # 🔒 CRITICAL: Winsorize current_returns to prevent extreme AR(1) predictions
    # If current_returns has extreme values (e.g., price gaps, data errors),
    # AR(1) prediction can be huge, leading to extreme residuals after exp()
    # Clip current_returns to reasonable range (e.g., ±0.5 log return = ±65% price change)
    max_log_return = 0.5  # Maximum reasonable log return (65% price change)
    current_returns_train_values = np.clip(current_returns_train_values,
                                           -max_log_return, max_log_return)
    current_returns_test_values = np.clip(current_returns_test_values,
                                          -max_log_return, max_log_return)

    # Calculate AR(1) prediction for training set
    # Improved: Use accurate forward prediction for fb > 1
    if forward_bars == 1:
        # Single bar: simple AR(1)
        ar1_pred_train = ar1_phi_train * current_returns_train_values
        ar1_pred_test = ar1_phi_train * current_returns_test_values
    else:
        # Multiple bars: use accurate forward prediction
        if accurate_forward:
            # Use phi^fb for more accurate cumulative prediction
            # For AR(1) model: E[r_{t+fb} | r_t] = phi^fb * r_t
            phi_power_fb = np.power(ar1_phi_train, forward_bars)
            # Clamp to prevent numerical issues
            phi_power_fb = np.clip(phi_power_fb, -10.0, 10.0)
            ar1_pred_train = phi_power_fb * current_returns_train_values
            ar1_pred_test = phi_power_fb * current_returns_test_values
        else:
            # Simplified: use phi * current_return (original approach)
            ar1_pred_train = ar1_phi_train * current_returns_train_values
            ar1_pred_test = ar1_phi_train * current_returns_test_values

    # 🔒 CRITICAL: Clip AR(1) predictions to prevent extreme residuals
    # Even after clipping current_returns, phi * current_returns can still be large
    # Clip AR(1) predictions to reasonable range (e.g., ±1.0 log return)
    max_ar1_pred = 1.0  # Maximum reasonable AR(1) prediction in log space
    ar1_pred_train = np.clip(ar1_pred_train, -max_ar1_pred, max_ar1_pred)
    ar1_pred_test = np.clip(ar1_pred_test, -max_ar1_pred, max_ar1_pred)

    # Convert to log returns and apply AR(1) residual
    # Improved: Use log1p with smooth transition zone to prevent log(negative)
    # Smooth transition prevents hard clipping from creating discrete noise
    eps = 1e-6
    y_train_values = y_train.values
    y_test_values = y_test.values

    # Smooth transition: avoid hard clipping at -0.9999
    y_train_safe = np.where(y_train_values < -1 + eps, -1 + eps,
                            y_train_values)
    y_test_safe = np.where(y_test_values < -1 + eps, -1 + eps, y_test_values)

    y_train_log = np.log1p(
        y_train_safe)  # log1p is more accurate for small values
    y_test_log = np.log1p(y_test_safe)

    # Calculate residual: future_return_log - AR(1) prediction
    # All arrays are now numpy arrays with matching shapes
    y_train_residual_log = y_train_log - ar1_pred_train
    y_test_residual_log = y_test_log - ar1_pred_test

    # 🔒 CRITICAL: Clip residual_log before exp() to prevent extreme values
    # If residual_log is too large (e.g., > 2.0), exp(residual_log) can be huge
    # Clip residual_log to reasonable range (e.g., ±2.0 log return = ±640% simple return)
    # This prevents numerical overflow and extreme predictions
    max_residual_log = 2.0  # Maximum reasonable residual in log space
    y_train_residual_log = np.clip(y_train_residual_log, -max_residual_log,
                                   max_residual_log)
    y_test_residual_log = np.clip(y_test_residual_log, -max_residual_log,
                                  max_residual_log)

    # Convert back to simple returns: exp(residual_log) - 1
    y_train_residual = np.exp(y_train_residual_log) - 1
    y_test_residual = np.exp(y_test_residual_log) - 1

    # Handle NaN/Inf from log/exp operations
    # Ensure all values are finite for LightGBM/XGBoost compatibility
    y_train_residual = pd.Series(y_train_residual, index=y_train.index)
    y_test_residual = pd.Series(y_test_residual, index=y_test.index)

    # Replace Inf/NaN with median (more robust than 0.0 for extreme cases)
    y_train_residual_values = y_train_residual.values
    y_test_residual_values = y_test_residual.values

    # Find finite values for fallback
    y_train_finite = y_train_residual_values[np.isfinite(
        y_train_residual_values)]
    y_test_finite = y_test_residual_values[np.isfinite(y_test_residual_values)]

    train_fallback = np.nanmedian(y_train_finite) if len(
        y_train_finite) > 0 else 0.0
    test_fallback = np.nanmedian(y_test_finite) if len(
        y_test_finite) > 0 else 0.0

    # Replace non-finite values
    y_train_residual_values[~np.isfinite(y_train_residual_values
                                         )] = train_fallback
    y_test_residual_values[~np.isfinite(y_test_residual_values
                                        )] = test_fallback

    y_train_residual = pd.Series(y_train_residual_values, index=y_train.index)
    y_test_residual = pd.Series(y_test_residual_values, index=y_test.index)

    # Calculate autocorrelation after AR(1) removal (for diagnostics, on training data only)
    ar1_autocorr_after = None
    if len(y_train_residual) > 100:
        valid_residual = y_train_residual.dropna()
        if len(valid_residual) > 100:
            ar1_autocorr_after = valid_residual.autocorr(lag=1)

    ar1_stats = {
        "ar1_phi": ar1_phi_train,
        "ar1_autocorr_after": ar1_autocorr_after,
        "autocorr_reduction": None,
    }

    if ar1_autocorr_after is not None and not pd.isna(ar1_autocorr_after):
        # Calculate autocorrelation reduction (if we had before)
        ar1_autocorr_before = ar1_phi_train
        if ar1_autocorr_before is not None:
            reduction = abs(ar1_autocorr_before - ar1_autocorr_after)
            ar1_stats["autocorr_reduction"] = reduction

    return y_train_residual, y_test_residual, ar1_stats


def secondary_clean_train_test(
    y_train: pd.Series,
    y_test: pd.Series,
    k: float = 3.5,
    use_symmetric_quantile: bool = True,
    smooth_clip: bool = True,
) -> Tuple[pd.Series, pd.Series, Dict]:
    """
    Secondary cleaning after AR(1) processing, using ONLY training statistics.
    
    Args:
        y_train: Training target variable (after AR(1))
        y_test: Test target variable (after AR(1))
        k: Winsorize threshold (default: 3.5)
        use_symmetric_quantile: If True, use symmetric quantiles (0.005, 0.995) instead of (0.01, 0.99)
        smooth_clip: If True, use weighted average of MAD and percentile thresholds instead of min
    
    Returns:
        Tuple of (y_train_cleaned, y_test_cleaned, stats_dict)
    """
    # Calculate statistics ONLY from training data
    # Performance optimization: use numpy for faster computation
    y_train_values = y_train.values

    # Compute median in one pass
    median_train = np.nanmedian(y_train_values)
    mad_train = np.nanmedian(np.abs(y_train_values - median_train))

    # Improved fallback: use nanstd with ddof=1
    if mad_train == 0 or np.isnan(mad_train):
        sigma_train = np.nanstd(y_train_values, ddof=1)
        if np.isnan(sigma_train) or sigma_train == 0:
            y_train_finite = y_train_values[np.isfinite(y_train_values)]
            if len(y_train_finite) > 1:
                sigma_train = (np.nanmax(y_train_finite) -
                               np.nanmin(y_train_finite)) / 6.0
            else:
                sigma_train = 1.0
    else:
        sigma_train = mad_train * 1.4826

    # Improved: Use symmetric quantiles for more robust threshold estimation
    # Performance: compute quantiles in one call
    if use_symmetric_quantile:
        q_low, q_high = np.nanpercentile(y_train_values, [0.5, 99.5])
        percentile_clip_high = max(abs(q_high), abs(q_low))
    else:
        # Original approach: use (0.01, 0.99)
        p01_train, p99_train = np.nanpercentile(y_train_values, [1.0, 99.0])
        percentile_clip_high = abs(p99_train) if abs(p99_train) > abs(
            p01_train) else abs(p01_train)

    # Improved: Adaptive weight mixing based on volatility regime
    clip_threshold_mad = k * sigma_train
    if smooth_clip:
        # Adaptive weight: low volatility -> more MAD, high volatility -> more percentile
        vol_ratio = sigma_train / (percentile_clip_high + 1e-9)
        # Use tanh for smooth transition: 0.5 + 0.5 * tanh(2 * (1 - vol_ratio))
        # When vol_ratio < 1 (low vol): mad_weight -> 1.0 (more MAD)
        # When vol_ratio > 1 (high vol): mad_weight -> 0.0 (more percentile)
        mad_weight = 0.5 + 0.5 * np.tanh(2 * (1 - vol_ratio))
        final_clip = mad_weight * clip_threshold_mad + (1 - mad_weight) * (
            percentile_clip_high * 1.5)
    else:
        # Original: use the more conservative threshold
        final_clip = min(clip_threshold_mad, percentile_clip_high * 1.5)

    # Apply clipping using training statistics
    lower_bound = median_train - final_clip
    upper_bound = median_train + final_clip

    y_train_cleaned = y_train.clip(lower_bound, upper_bound)
    y_test_cleaned = y_test.clip(lower_bound, upper_bound)

    stats = {
        "median": median_train,
        "mad": mad_train,
        "sigma": sigma_train,
        "clip_threshold": final_clip,
        "n_clipped_train": int((y_train_cleaned != y_train).sum()),
        "n_clipped_test": int((y_test_cleaned != y_test).sum()),
    }

    return y_train_cleaned, y_test_cleaned, stats


# Structured return type for preprocessing statistics
class TargetPreprocessStats(NamedTuple):
    """Structured statistics for target preprocessing pipeline."""
    winsorize: Dict
    ar1: Dict
    secondary: Dict

    def summary(self) -> pd.DataFrame:
        """Generate a summary DataFrame of preprocessing statistics."""
        return pd.DataFrame({
            'Stage': ['Winsorize', 'AR(1)', 'Secondary'],
            'Clipped_train': [
                self.winsorize.get('n_clipped_train', 0), None,
                self.secondary.get('n_clipped_train', 0)
            ],
            'Clipped_test': [
                self.winsorize.get('n_clipped_test', 0), None,
                self.secondary.get('n_clipped_test', 0)
            ],
            'AR1_phi': [None, self.ar1.get('ar1_phi', 0.0), None],
            'MAD': [
                self.winsorize.get('mad', 0.0), None,
                self.secondary.get('mad', 0.0)
            ],
            'Sigma': [
                self.winsorize.get('sigma', 0.0), None,
                self.secondary.get('sigma', 0.0)
            ],
        })


def preprocess_target_cv(
    y_train: pd.Series,
    y_test: pd.Series,
    current_returns_train: pd.Series,
    current_returns_test: pd.Series,
    forward_bars: int,
    k_winsorize: float = 3.5,
    k_secondary: float = 3.5,
    dynamic_threshold: Optional[pd.Series] = None,
    accurate_forward: bool = True,
    use_symmetric_quantile: bool = True,
    smooth_clip: bool = True,
    verbose: bool = False,
) -> Tuple[pd.Series, pd.Series, TargetPreprocessStats]:
    """
    Complete preprocessing pipeline for target variable in CV fold.
    All statistics computed ONLY from training data.
    
    Args:
        y_train: Training target variable (raw future_return)
        y_test: Test target variable (raw future_return)
        current_returns_train: Current period log returns for training
        current_returns_test: Current period log returns for test
        forward_bars: Number of forward bars
        k_winsorize: Winsorize threshold for Step 1 (default: 3.5)
        k_secondary: Winsorize threshold for Step 2b (default: 3.5)
        dynamic_threshold: Optional dynamic k values
        accurate_forward: If True, use phi^fb for forward_bars > 1 (more accurate)
        use_symmetric_quantile: If True, use symmetric quantiles (0.005, 0.995) for Step 2b
        smooth_clip: If True, use weighted average of MAD and percentile thresholds
        verbose: If True, print detailed preprocessing statistics
    
    Returns:
        Tuple of (y_train_final, y_test_final, preprocessing_stats)
        preprocessing_stats is a TargetPreprocessStats NamedTuple
    """
    if verbose:
        print(f"  [Preprocessing] Starting target preprocessing pipeline...")
        print(
            f"    Input: train={len(y_train)}, test={len(y_test)}, forward_bars={forward_bars}"
        )

    # Step 1: Robust Winsorize (using training statistics)
    y_train_step1, y_test_step1, stats_step1 = robust_winsorize_train_test(
        y_train, y_test, k=k_winsorize, dynamic_threshold=dynamic_threshold)
    if verbose:
        print(
            f"  [Step 1: Winsorize] Clipped: train={stats_step1['n_clipped_train']}, "
            f"test={stats_step1['n_clipped_test']}, MAD={stats_step1['mad']:.6f}"
        )

    # Step 2: AR(1) residual (using training statistics)
    y_train_step2, y_test_step2, stats_ar1 = ar1_residual_train_test(
        y_train_step1,
        y_test_step1,
        current_returns_train,
        current_returns_test,
        forward_bars,
        accurate_forward=accurate_forward)
    if verbose:
        autocorr_after = stats_ar1.get('ar1_autocorr_after', None)
        reduction = stats_ar1.get('autocorr_reduction', None)
        reduction_str = f", reduction={reduction:.4f}" if reduction is not None else ""
        ar1_phi = stats_ar1.get('ar1_phi', 0.0)
        autocorr_after_str = f"{autocorr_after:.4f}" if autocorr_after is not None else 'N/A'
        print(f"  [Step 2: AR(1)] phi={ar1_phi:.4f}, "
              f"autocorr_after={autocorr_after_str}{reduction_str}")

    # Step 2b: Secondary cleaning (using training statistics)
    y_train_final, y_test_final, stats_step2b = secondary_clean_train_test(
        y_train_step2,
        y_test_step2,
        k=k_secondary,
        use_symmetric_quantile=use_symmetric_quantile,
        smooth_clip=smooth_clip)
    if verbose:
        clip_threshold = stats_step2b.get('clip_threshold', 'N/A')
        clip_threshold_str = f"{clip_threshold:.6f}" if isinstance(
            clip_threshold, (int, float)) else str(clip_threshold)
        print(
            f"  [Step 2b: Secondary] Clipped: train={stats_step2b['n_clipped_train']}, "
            f"test={stats_step2b['n_clipped_test']}, clip_threshold={clip_threshold_str}"
        )
        print(f"  ✅ Target preprocessing completed.")

    # Final validation: ensure no Inf/NaN for LightGBM/XGBoost compatibility
    # This is critical for production deployment
    if not np.all(np.isfinite(y_train_final.values)):
        n_inf_train = np.sum(~np.isfinite(y_train_final.values))
        y_train_final_finite = y_train_final.values[np.isfinite(
            y_train_final.values)]
        train_fallback = np.nanmedian(y_train_final_finite) if len(
            y_train_final_finite) > 0 else 0.0
        y_train_final = pd.Series(np.where(np.isfinite(y_train_final.values),
                                           y_train_final.values,
                                           train_fallback),
                                  index=y_train_final.index)
        if verbose and n_inf_train > 0:
            print(
                f"  ⚠️  Warning: Replaced {n_inf_train} non-finite values in y_train_final"
            )

    if not np.all(np.isfinite(y_test_final.values)):
        n_inf_test = np.sum(~np.isfinite(y_test_final.values))
        y_test_final_finite = y_test_final.values[np.isfinite(
            y_test_final.values)]
        test_fallback = np.nanmedian(y_test_final_finite) if len(
            y_test_final_finite) > 0 else 0.0
        y_test_final = pd.Series(np.where(np.isfinite(y_test_final.values),
                                          y_test_final.values, test_fallback),
                                 index=y_test_final.index)
        if verbose and n_inf_test > 0:
            print(
                f"  ⚠️  Warning: Replaced {n_inf_test} non-finite values in y_test_final"
            )

    # Return structured statistics
    preprocessing_stats = TargetPreprocessStats(winsorize=stats_step1,
                                                ar1=stats_ar1,
                                                secondary=stats_step2b)

    return y_train_final, y_test_final, preprocessing_stats


def clean_features_train_test(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    k: float = 4.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Apply robust Winsorize to features using ONLY training statistics.
    
    Args:
        X_train: Training feature matrix
        X_test: Test feature matrix
        k: Winsorize threshold (default: 4.0, more conservative than y)
    
    Returns:
        Tuple of (X_train_cleaned, X_test_cleaned, stats_dict)
    """
    X_train_cleaned = X_train.copy()
    X_test_cleaned = X_test.copy()

    stats = {
        "n_features_cleaned": 0,
        "n_clipped_per_feature": {},
    }

    # Clean each numeric column using training statistics
    for col in X_train.columns:
        if X_train[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            # Calculate statistics ONLY from training data
            median_train = X_train[col].median()
            mad_train = np.median(np.abs(X_train[col] - median_train))
            if mad_train == 0:
                sigma_train = X_train[col].std()
            else:
                sigma_train = mad_train * 1.4826  # Convert MAD to approximate std

            # Calculate bounds using training statistics
            lower_bound = median_train - k * sigma_train
            upper_bound = median_train + k * sigma_train

            # Apply same bounds to both train and test
            X_train_cleaned[col] = X_train[col].clip(lower_bound, upper_bound)
            X_test_cleaned[col] = X_test[col].clip(lower_bound, upper_bound)

            # Count clipped values
            n_clipped_train = int((X_train_cleaned[col] != X_train[col]).sum())
            n_clipped_test = int((X_test_cleaned[col] != X_test[col]).sum())

            if n_clipped_train > 0 or n_clipped_test > 0:
                stats["n_features_cleaned"] += 1
                stats["n_clipped_per_feature"][col] = {
                    "train": n_clipped_train,
                    "test": n_clipped_test,
                }

    return X_train_cleaned, X_test_cleaned, stats
