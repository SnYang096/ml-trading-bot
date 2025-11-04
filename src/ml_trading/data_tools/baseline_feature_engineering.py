from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List


class BaselineFeatureEngineer:
    """Baseline SR and compression features.

    Implements a lightweight, reproducible subset of the spec in
    docs/基线：SR特征和压缩特征.md, focusing on:
      - ATR-based normalization
      - SR proximity via recent rolling highs/lows (proxy for structure)
      - Simple channel via rolling OLS equivalent (EMA bands) and bandwidth
      - Compression features: ATR percentile, compression ratio, volume percentile,
        price direction entropy, compression duration, pre-break silence, composite score

    This class is stateless except for fitted percentiles needed to transform train/test
    consistently (for atr and volume percentiles). Keep it minimal for baseline usage.
    """

    def __init__(self, percentile_window: int = 288, compression_threshold_pct: float = 0.2) -> None:
        self.percentile_window = percentile_window
        self.compression_threshold_pct = compression_threshold_pct
        self._fitted_atr_quantiles: Optional[np.ndarray] = None
        self._fitted_vol_quantiles: Optional[np.ndarray] = None

    @staticmethod
    def _compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)

        prev_close = close.shift(1)
        hl = (high - low).abs()
        hc = (high - prev_close).abs()
        lc = (low - prev_close).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr = tr.rolling(window=window, min_periods=1).mean()
        return atr.rename("atr")

    @staticmethod
    def _rolling_percentile(series: pd.Series, window: int) -> pd.Series:
        # Percentile rank in [0,1] within moving window
        def _rank(x: np.ndarray) -> float:
            if len(x) <= 1 or not np.isfinite(x[-1]):
                return np.nan
            last = x[-1]
            arr = x[np.isfinite(x)]
            if len(arr) == 0:
                return np.nan
            return (arr <= last).sum() / float(len(arr))

        return series.rolling(window=window, min_periods=1).apply(_rank, raw=True)

    @staticmethod
    def _price_entropy(close: pd.Series, window: int = 50) -> pd.Series:
        # Shannon entropy of direction (+1/-1) over window
        ret = close.pct_change().fillna(0.0)
        sign = np.sign(ret).replace(0, 1)  # treat zero as +1 to avoid NaNs
        def _entropy(x: np.ndarray) -> float:
            if len(x) == 0:
                return np.nan
            p_up = (x > 0).mean()
            p_dn = 1.0 - p_up
            eps = 1e-9
            return - (p_up * np.log2(p_up + eps) + p_dn * np.log2(p_dn + eps)) / 1.0

        # Normalize to [0,1] where max entropy at p=0.5 is 1.0 after dividing by 1
        return sign.rolling(window=window, min_periods=1).apply(_entropy, raw=True)

    @staticmethod
    def _ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    def engineer_features(self, df: pd.DataFrame, *, fit: bool = True) -> pd.DataFrame:
        if not {"open", "high", "low", "close", "volume"}.issubset(df.columns):
            raise ValueError("DataFrame must contain open, high, low, close, volume columns")

        data = df.copy()

        # Core ATR
        data["atr"] = self._compute_atr(data, window=14)

        # SR proximity using rolling swing proxies (highest high / lowest low)
        swing_win_short = 20
        swing_win_long = 60
        data["roll_high_s"] = data["high"].rolling(swing_win_short, min_periods=1).max()
        data["roll_low_s"] = data["low"].rolling(swing_win_short, min_periods=1).min()
        data["roll_high_l"] = data["high"].rolling(swing_win_long, min_periods=1).max()
        data["roll_low_l"] = data["low"].rolling(swing_win_long, min_periods=1).min()

        eps = 1e-9
        data["sr_dist_high_s"] = (data["close"] - data["roll_high_s"]) / (data["atr"] + eps)
        data["sr_dist_low_s"] = (data["close"] - data["roll_low_s"]) / (data["atr"] + eps)
        data["sr_dist_high_l"] = (data["close"] - data["roll_high_l"]) / (data["atr"] + eps)
        data["sr_dist_low_l"] = (data["close"] - data["roll_low_l"]) / (data["atr"] + eps)

        # Simple channel via EMAs as OLS proxy
        ema_fast = self._ema(data["close"], span=20)
        ema_slow = self._ema(data["close"], span=60)
        mid = (ema_fast + ema_slow) / 2.0
        band_half = (data["high"].rolling(20, min_periods=1).max() - data["low"].rolling(20, min_periods=1).min()) / 4.0
        upper = mid + band_half
        lower = mid - band_half
        data["channel_upper"] = upper
        data["channel_lower"] = lower
        data["channel_mid"] = mid
        data["channel_bandwidth"] = (upper - lower) / (data["atr"] + eps)
        data["channel_upper_distance"] = (upper - data["close"]) / (data["atr"] + eps)
        data["channel_lower_distance"] = (data["close"] - lower) / (data["atr"] + eps)

        # Compression features
        # ATR percentile (rolling)
        atr_pct = self._rolling_percentile(data["atr"], window=self.percentile_window)
        data["atr_percentile"] = atr_pct

        # ATR compression ratio: mean(ATR_hist)/ATR
        atr_mean_hist = data["atr"].rolling(self.percentile_window, min_periods=1).mean()
        data["atr_compression_ratio"] = (atr_mean_hist / (data["atr"] + eps)).replace([np.inf, -np.inf], np.nan)

        # Volume percentile
        vol_pct = self._rolling_percentile(data["volume"].astype(float), window=self.percentile_window)
        data["volume_percentile"] = vol_pct

        # Price direction entropy
        data["price_entropy"] = self._price_entropy(data["close"], window=50)

        # Compression duration: consecutive bars where ATR below threshold percentile
        threshold = self.compression_threshold_pct
        below = (data["atr_percentile"].fillna(0.0) <= threshold).astype(int)
        # run-length counting of consecutive 1s
        run = np.zeros(len(below), dtype=int)
        cnt = 0
        for i, v in enumerate(below.values):
            if v == 1:
                cnt += 1
            else:
                cnt = 0
            run[i] = cnt
        data["compression_duration"] = run

        # Pre-break silence: mean of ATR percentile over short lookback below threshold
        short_window = 30
        data["pre_break_silence"] = (data["atr_percentile"].rolling(short_window, min_periods=1).mean() <= threshold).astype(float)

        # Internal price density proxy: price variance over small window vs larger window
        small = 20
        large = 100
        var_small = data["close"].rolling(small, min_periods=1).var()
        var_large = data["close"].rolling(large, min_periods=1).var()
        density = 1.0 - (var_small / (var_large + eps))
        data["internal_price_density"] = density.clip(0.0, 1.0)

        # Composite compression confidence
        atr_norm = (data["atr_percentile"].fillna(0.0))
        vol_norm = (data["volume_percentile"].fillna(0.0))
        dens_norm = data["internal_price_density"].fillna(0.0)
        data["compression_confidence"] = 0.5 * (1 - atr_norm) + 0.3 * (1 - vol_norm) + 0.2 * dens_norm

        # Finalize: feature column ordering
        feature_cols: List[str] = [
            # SR distances
            "sr_dist_high_s", "sr_dist_low_s", "sr_dist_high_l", "sr_dist_low_l",
            # Channel features
            "channel_bandwidth", "channel_upper_distance", "channel_lower_distance",
            # Compression set
            "atr_percentile", "atr_compression_ratio", "volume_percentile",
            "price_entropy", "internal_price_density", "compression_duration",
            "pre_break_silence", "compression_confidence",
        ]

        # Clean up and keep OHLCV + features
        keep_cols = ["open", "high", "low", "close", "volume"] + feature_cols
        data = data[keep_cols]
        return data

    def save_scalers(self, path: str) -> None:
        """Save fitted quantiles for consistent train/test transformation.
        
        Args:
            path: Path to save scalers (pickle file)
        """
        import pickle
        
        scalers_data = {
            "fitted_atr_quantiles": self._fitted_atr_quantiles,
            "fitted_vol_quantiles": self._fitted_vol_quantiles,
            "percentile_window": self.percentile_window,
            "compression_threshold_pct": self.compression_threshold_pct,
        }
        
        with open(path, "wb") as f:
            pickle.dump(scalers_data, f)
        print(f"✅ Baseline scalers saved to: {path}")

    def load_scalers(self, path: str) -> None:
        """Load fitted quantiles for consistent train/test transformation.
        
        Args:
            path: Path to load scalers (pickle file)
        """
        import pickle
        
        with open(path, "rb") as f:
            scalers_data = pickle.load(f)
        
        self._fitted_atr_quantiles = scalers_data.get("fitted_atr_quantiles", None)
        self._fitted_vol_quantiles = scalers_data.get("fitted_vol_quantiles", None)
        self.percentile_window = scalers_data.get("percentile_window", 288)
        self.compression_threshold_pct = scalers_data.get("compression_threshold_pct", 0.2)
        print(f"✅ Baseline scalers loaded from: {path}")


def engineer_baseline_features(df: pd.DataFrame, engineer: Optional[BaselineFeatureEngineer] = None, *, fit: bool = True) -> Tuple[pd.DataFrame, BaselineFeatureEngineer]:
    if engineer is None:
        engineer = BaselineFeatureEngineer()
    out = engineer.engineer_features(df, fit=fit)
    return out, engineer


def create_binary_labels_baseline(df: pd.DataFrame, *, forward_bars: int = 3, threshold: float = 0.005) -> pd.DataFrame:
    """Create binary classification labels for baseline (1=Long, 0=not Long).
    
    Args:
        df: DataFrame with OHLCV data
        forward_bars: Number of bars ahead for prediction
        threshold: Threshold for Long signal (future_return > threshold)
    
    Returns:
        DataFrame with 'binary_signal' column (1=Long, 0=not Long)
    """
    df = df.copy()
    df["future_return"] = df["close"].shift(-forward_bars) / df["close"] - 1
    
    # Binary classification: 1=Long (future_return > threshold), 0=not Long
    df["binary_signal"] = (df["future_return"] > threshold).astype(int)
    
    # Keep backward compatibility: signal for legacy code
    df["signal"] = df["binary_signal"]
    
    return df


def get_baseline_feature_columns(df: pd.DataFrame) -> List[str]:
    exclude = {"open", "high", "low", "close", "volume", "signal", "binary_signal", "future_return"}
    # Also exclude multi-horizon label columns
    exclude.update([
        col for col in df.columns 
        if col.startswith("signal_") or 
           col.startswith("binary_signal_") or 
           col.startswith("future_return_")
    ])
    return [c for c in df.columns if c not in exclude]


__all__ = [
    "BaselineFeatureEngineer",
    "engineer_baseline_features",
    "get_baseline_feature_columns",
    "create_binary_labels_baseline",
]


