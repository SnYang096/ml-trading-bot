"""Advanced feature engineering module with wavelet transform and normalization.

基于改进版特征工程，添加了：
1. 小波变换特征（价格和成交量）
2. Hilbert变换特征
3. 频谱分析特征

基础指标计算复用 base_indicators 模块。
"""

import pandas as pd
import numpy as np
from typing import Dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import pickle
import pywt
from scipy import stats
from scipy.signal import hilbert

from .base_indicators import add_basic_indicators


def compute_wavelet_features(
    series: pd.Series, wavelet: str = "db4", levels: int = 4
) -> Dict[str, pd.Series]:
    """
    Compute wavelet transform features.

    Args:
        series: Input time series
        wavelet: Wavelet type ('db4', 'haar', 'coif2', etc.)
        levels: Number of decomposition levels

    Returns:
        Dictionary of wavelet features
    """
    series = pd.to_numeric(series, errors="coerce")
    series = series.dropna()

    if len(series) < 2**levels:
        # Return zeros if not enough data
        return {
            "wavelet_energy": pd.Series(0, index=series.index),
            "wavelet_entropy": pd.Series(0, index=series.index),
            "wavelet_std": pd.Series(0, index=series.index),
            "wavelet_skewness": pd.Series(0, index=series.index),
            "wavelet_kurtosis": pd.Series(0, index=series.index),
        }

    try:
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(series.values, wavelet, level=levels)

        # Extract features from coefficients
        features = {}

        # Energy of each level
        energies = [np.sum(np.square(c)) for c in coeffs]
        total_energy = sum(energies)

        # Relative energy
        relative_energies = [
            e / total_energy if total_energy > 0 else 0 for e in energies
        ]

        # Wavelet energy (sum of all energies)
        features["wavelet_energy"] = pd.Series(total_energy, index=series.index)

        # Wavelet entropy
        entropy = -sum([p * np.log2(p) if p > 0 else 0 for p in relative_energies])
        features["wavelet_entropy"] = pd.Series(entropy, index=series.index)

        # Standard deviation of coefficients
        all_coeffs = np.concatenate(coeffs)
        features["wavelet_std"] = pd.Series(np.std(all_coeffs), index=series.index)

        # Skewness and kurtosis of coefficients
        features["wavelet_skewness"] = pd.Series(
            stats.skew(all_coeffs), index=series.index
        )
        features["wavelet_kurtosis"] = pd.Series(
            stats.kurtosis(all_coeffs), index=series.index
        )

        # Detail coefficients features
        for i, coeff in enumerate(coeffs[1:], 1):  # Skip approximation coefficients
            features[f"wavelet_detail_{i}_energy"] = pd.Series(
                np.sum(np.square(coeff)), index=series.index
            )
            features[f"wavelet_detail_{i}_std"] = pd.Series(
                np.std(coeff), index=series.index
            )
            features[f"wavelet_detail_{i}_mean"] = pd.Series(
                np.mean(coeff), index=series.index
            )

        # Approximation coefficients features
        approx_coeff = coeffs[0]
        features["wavelet_approx_energy"] = pd.Series(
            np.sum(np.square(approx_coeff)), index=series.index
        )
        features["wavelet_approx_std"] = pd.Series(
            np.std(approx_coeff), index=series.index
        )
        features["wavelet_approx_mean"] = pd.Series(
            np.mean(approx_coeff), index=series.index
        )

        return features

    except Exception as e:
        print(f"Warning: Error in wavelet transform: {e}")
        # Return zeros if wavelet transform fails
        return {
            "wavelet_energy": pd.Series(0, index=series.index),
            "wavelet_entropy": pd.Series(0, index=series.index),
            "wavelet_std": pd.Series(0, index=series.index),
            "wavelet_skewness": pd.Series(0, index=series.index),
            "wavelet_kurtosis": pd.Series(0, index=series.index),
        }


def compute_hilbert_features(series: pd.Series) -> Dict[str, pd.Series]:
    """Compute Hilbert transform features for instantaneous frequency and amplitude."""
    series = pd.to_numeric(series, errors="coerce")
    series = series.dropna()

    if len(series) < 10:
        return {
            "hilbert_amplitude": pd.Series(0, index=series.index),
            "hilbert_phase": pd.Series(0, index=series.index),
            "hilbert_frequency": pd.Series(0, index=series.index),
        }

    try:
        # Compute analytic signal
        analytic_signal = hilbert(series.values)

        # Extract amplitude and phase
        amplitude = np.abs(analytic_signal)
        phase = np.angle(analytic_signal)

        # Compute instantaneous frequency
        frequency = np.diff(np.unwrap(phase)) / (2.0 * np.pi)
        frequency = np.concatenate([[frequency[0]], frequency])  # Pad first value

        features = {
            "hilbert_amplitude": pd.Series(amplitude, index=series.index),
            "hilbert_phase": pd.Series(phase, index=series.index),
            "hilbert_frequency": pd.Series(frequency, index=series.index),
        }

        return features

    except Exception as e:
        print(f"Warning: Error in Hilbert transform: {e}")
        return {
            "hilbert_amplitude": pd.Series(0, index=series.index),
            "hilbert_phase": pd.Series(0, index=series.index),
            "hilbert_frequency": pd.Series(0, index=series.index),
        }


def compute_spectral_features(series: pd.Series) -> Dict[str, pd.Series]:
    """Compute spectral features using FFT."""
    series = pd.to_numeric(series, errors="coerce")
    series = series.dropna()

    if len(series) < 10:
        return {
            "spectral_centroid": pd.Series(0, index=series.index),
            "spectral_bandwidth": pd.Series(0, index=series.index),
            "spectral_rolloff": pd.Series(0, index=series.index),
        }

    try:
        # Compute FFT
        fft = np.fft.fft(series.values)
        magnitude = np.abs(fft)
        freqs = np.fft.fftfreq(len(series))

        # Spectral centroid
        spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)

        # Spectral bandwidth
        spectral_bandwidth = np.sqrt(
            np.sum(((freqs - spectral_centroid) ** 2) * magnitude) / np.sum(magnitude)
        )

        # Spectral rolloff (95% of energy)
        cumsum_magnitude = np.cumsum(magnitude)
        rolloff_idx = np.where(cumsum_magnitude >= 0.95 * cumsum_magnitude[-1])[0]
        spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]

        features = {
            "spectral_centroid": pd.Series(spectral_centroid, index=series.index),
            "spectral_bandwidth": pd.Series(spectral_bandwidth, index=series.index),
            "spectral_rolloff": pd.Series(spectral_rolloff, index=series.index),
        }

        return features

    except Exception as e:
        print(f"Warning: Error in spectral analysis: {e}")
        return {
            "spectral_centroid": pd.Series(0, index=series.index),
            "spectral_bandwidth": pd.Series(0, index=series.index),
            "spectral_rolloff": pd.Series(0, index=series.index),
        }


class WaveletFeatureEngineer:
    """Advanced feature engineer with wavelet transform and normalization."""

    def __init__(
        self,
        scaler_type: str = "standard",
        wavelet: str = "db4",
        wavelet_levels: int = 4,
    ):
        """
        Initialize the wavelet feature engineer.

        Args:
            scaler_type: Type of scaler ('standard', 'minmax', 'robust')
            wavelet: Wavelet type for decomposition
            wavelet_levels: Number of wavelet decomposition levels
        """
        self.scaler_type = scaler_type
        self.wavelet = wavelet
        self.wavelet_levels = wavelet_levels
        self.scalers = {}  # Store scalers for each timeframe
        self.feature_stats = {}  # Store feature statistics

        # Choose scaler
        if scaler_type == "standard":
            self.scaler_class = StandardScaler
        elif scaler_type == "minmax":
            self.scaler_class = MinMaxScaler
        elif scaler_type == "robust":
            self.scaler_class = RobustScaler
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")

    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """添加基础技术指标（复用base_indicators）."""
        if data.empty:
            return data

        # 添加所有基础指标（包括改进版的衍生特征）
        df = add_basic_indicators(data)

        if df.empty:
            return df

        # 添加改进版的额外衍生特征
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (
            df["bb_upper"] - df["bb_lower"]
        )
        df["rsi_normalized"] = (df["rsi"] - 50) / 50
        df["macd_normalized"] = df["macd"] / df["close"]
        df["atr_normalized"] = df["atr"] / df["close"]

        df["momentum_5"] = df["close"].pct_change(5)
        df["momentum_10"] = df["close"].pct_change(10)
        df["momentum_20"] = df["close"].pct_change(20)

        df["sma_5"] = df["close"].rolling(window=5).mean()
        df["sma_10"] = df["close"].rolling(window=10).mean()
        df["sma_20"] = df["close"].rolling(window=20).mean()
        df["sma_ratio_5_20"] = df["sma_5"] / df["sma_20"]
        df["sma_ratio_10_20"] = df["sma_10"] / df["sma_20"]

        # Fill NaN
        feature_cols = [
            col
            for col in df.columns
            if col not in ["open", "high", "low", "close", "volume"]
        ]
        for col in feature_cols:
            df[col] = df[col].fillna(0)

        return df

    def add_wavelet_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add wavelet transform features."""
        if data.empty:
            return data

        df = data.copy()

        print(f"Adding wavelet features using {self.wavelet} wavelet...")

        # Wavelet features for close price
        try:
            wavelet_features = compute_wavelet_features(
                df["close"], self.wavelet, self.wavelet_levels
            )
            for name, series in wavelet_features.items():
                df[name] = series
        except Exception as e:
            print(f"Warning: Error computing wavelet features for close: {e}")

        # Wavelet features for volume
        try:
            volume_wavelet = compute_wavelet_features(
                df["volume"], self.wavelet, self.wavelet_levels
            )
            for name, series in volume_wavelet.items():
                df[f"volume_{name}"] = series
        except Exception as e:
            print(f"Warning: Error computing wavelet features for volume: {e}")

        # Hilbert transform features
        try:
            hilbert_features = compute_hilbert_features(df["close"])
            for name, series in hilbert_features.items():
                df[name] = series
        except Exception as e:
            print(f"Warning: Error computing Hilbert features: {e}")

        # Spectral features
        try:
            spectral_features = compute_spectral_features(df["close"])
            for name, series in spectral_features.items():
                df[name] = series
        except Exception as e:
            print(f"Warning: Error computing spectral features: {e}")

        return df

    def normalize_features(
        self, data: pd.DataFrame, timeframe: str, fit: bool = True
    ) -> pd.DataFrame:
        """
        Normalize features using the specified scaler.

        Args:
            data: DataFrame with features
            timeframe: Timeframe identifier
            fit: Whether to fit the scaler (True for training, False for prediction)

        Returns:
            DataFrame with normalized features
        """
        df = data.copy()

        # Get feature columns (exclude OHLCV)
        feature_cols = [
            col
            for col in df.columns
            if col not in ["open", "high", "low", "close", "volume"]
        ]

        if not feature_cols:
            return df

        # Prepare feature matrix
        X = df[feature_cols].values

        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        if fit:
            # Fit scaler on training data
            scaler = self.scaler_class()
            X_scaled = scaler.fit_transform(X)
            self.scalers[timeframe] = scaler

            # Store feature statistics
            self.feature_stats[timeframe] = {
                "mean": np.mean(X, axis=0),
                "std": np.std(X, axis=0),
                "min": np.min(X, axis=0),
                "max": np.max(X, axis=0),
            }
        else:
            # Use existing scaler for prediction
            if timeframe not in self.scalers:
                raise ValueError(
                    f"No scaler found for timeframe {timeframe}. Call fit first."
                )
            scaler = self.scalers[timeframe]
            X_scaled = scaler.transform(X)

        # Update DataFrame with scaled features
        df_scaled = df.copy()
        for i, col in enumerate(feature_cols):
            df_scaled[col] = X_scaled[:, i]

        return df_scaled

    def engineer_features(
        self, multi_tf_data: Dict[str, pd.DataFrame], fit: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Engineer features for multi-timeframe data with wavelet transform and normalization.

        Args:
            multi_tf_data: Dictionary mapping timeframe to DataFrame
            fit: Whether to fit scalers (True for training, False for prediction)

        Returns:
            Dictionary with engineered and normalized features for each timeframe
        """
        engineered_data = {}

        for timeframe, data in multi_tf_data.items():
            print(f"Engineering features for {timeframe}: {data.shape}")

            # Add technical indicators
            df_with_indicators = self.add_technical_indicators(data)
            print(
                f"Added technical indicators for {timeframe}: {df_with_indicators.shape}"
            )

            # Add wavelet features
            df_with_wavelets = self.add_wavelet_features(df_with_indicators)
            print(f"Added wavelet features for {timeframe}: {df_with_wavelets.shape}")

            # Normalize features
            df_normalized = self.normalize_features(
                df_with_wavelets, timeframe, fit=fit
            )
            print(f"Normalized features for {timeframe}: {df_normalized.shape}")

            engineered_data[timeframe] = df_normalized

        return engineered_data

    def save_scalers(self, filepath: str):
        """Save fitted scalers to file."""
        scaler_data = {
            "scalers": self.scalers,
            "feature_stats": self.feature_stats,
            "scaler_type": self.scaler_type,
            "wavelet": self.wavelet,
            "wavelet_levels": self.wavelet_levels,
        }

        with open(filepath, "wb") as f:
            pickle.dump(scaler_data, f)

        print(f"Scalers saved to {filepath}")

    def load_scalers(self, filepath: str):
        """Load fitted scalers from file."""
        with open(filepath, "rb") as f:
            scaler_data = pickle.load(f)

        self.scalers = scaler_data["scalers"]
        self.feature_stats = scaler_data["feature_stats"]
        self.scaler_type = scaler_data["scaler_type"]
        self.wavelet = scaler_data.get("wavelet", "db4")
        self.wavelet_levels = scaler_data.get("wavelet_levels", 4)

        print(f"Scalers loaded from {filepath}")

    def get_feature_importance_info(self, timeframe: str) -> Dict:
        """Get feature statistics for analysis."""
        if timeframe not in self.feature_stats:
            return {}

        stats = self.feature_stats[timeframe]
        return {
            "mean": stats["mean"].tolist(),
            "std": stats["std"].tolist(),
            "min": stats["min"].tolist(),
            "max": stats["max"].tolist(),
            "scaler_type": self.scaler_type,
            "wavelet": self.wavelet,
            "wavelet_levels": self.wavelet_levels,
        }
