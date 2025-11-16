"""Enhanced feature engineering with Wavelet Packet Transform, Energy, Entropy, and Hurst.

对所有信号源（close, open, volume, cvd, taker_buy_ratio）都做：
1. 小波包变换（WPT） - 精细频带分解
2. Hurst指数 - 趋势持续性分析
3. 能量和熵特征
"""

import pandas as pd
import numpy as np
import pywt
from typing import Any, Dict, Tuple, List
from scipy import signal
from scipy.signal import hilbert
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from .baseline_features import add_common_derived_features

try:
    import talib
except ImportError:  # pragma: no cover - optional dependency
    talib = None


class EnhancedFeatureEngineer:
    """增强版特征工程：对所有信号源做WPT+Hurst分解."""

    def __init__(
        self,
        scaler_type: str = "standard",
        wavelet: str = "db4",
        wpt_level: int = 3,
        hurst_window: int = 100,
        feature_shift: int = 0,
        feature_clip_bound: float = 10.0,
        enable_diagnostics: bool = False,
    ):
        """
        Initialize enhanced feature engineer.

        Args:
            scaler_type: Type of scaler ('standard' or 'minmax')
            wavelet: Wavelet type for WPT
            wpt_level: Decomposition level for wavelet packet
            hurst_window: Window size for Hurst exponent calculation
        """
        self.scaler_type = scaler_type
        self.scalers: Dict[str, Any] = {}
        self.wavelet = wavelet
        self.wpt_level = wpt_level
        self.hurst_window = hurst_window
        self.hurst_lags = np.array([5, 10, 20, 40, 80, 160], dtype=int)
        self.feature_shift = feature_shift
        self.feature_clip_bound = float(feature_clip_bound)
        self.enable_diagnostics = enable_diagnostics
        self.diagnostic_report: Dict[str, Dict[str, Dict[str, float]]] = {}

        if scaler_type == "standard":
            self.scaler_class = StandardScaler
        elif scaler_type == "minmax":
            self.scaler_class = MinMaxScaler
        elif scaler_type == "robust":
            self.scaler_class = RobustScaler
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")

        if self.feature_clip_bound <= 0:
            raise ValueError("feature_clip_bound must be positive")

    def _shift_feature(self,
                       series: pd.Series,
                       *,
                       offset: int = 0) -> pd.Series:
        """Apply configurable lag to feature series."""
        total_shift = self.feature_shift + offset
        if total_shift == 0:
            return series
        return series.shift(total_shift)

    def _run_diagnostics(self, timeframe: str, df: pd.DataFrame,
                         feature_cols: List[str]) -> None:
        """Collect diagnostics for clipping/zero saturation per timeframe."""
        report: Dict[str, Dict[str, float]] = {}
        tol_zero = 1e-9
        tol_clip = 1e-3
        clip_bound = float(self.feature_clip_bound)
        near_clip = 0.9 * clip_bound
        clip_markers = {
            "at_pos_bound_ratio": clip_bound,
            "at_neg_bound_ratio": -clip_bound,
            "at_pos1_ratio": 1.0,
            "at_neg1_ratio": -1.0,
        }

        flagged_clip: List[Tuple[str, float, float]] = []
        flagged_zero: List[Tuple[str, float]] = []

        for col in feature_cols:
            series = df[col].replace([np.inf, -np.inf], np.nan)
            if series.empty:
                continue

            metrics: Dict[str, float] = {}
            metrics["nan_ratio"] = float(series.isna().mean())

            valid = series.dropna()
            if valid.empty:
                report[col] = metrics
                continue

            metrics["zero_ratio"] = float(
                np.isclose(valid, 0.0, atol=tol_zero).mean())
            metrics["abs_ge_90pct_ratio"] = float((valid.abs()
                                                   >= near_clip).mean())
            metrics["abs_ge_99pct_ratio"] = float(
                (valid.abs() >= 0.99 * clip_bound).mean())
            metrics["mean"] = float(valid.mean())
            metrics["std"] = float(valid.std())

            for key, marker in clip_markers.items():
                metrics[key] = float(
                    np.isclose(valid, marker, atol=tol_clip).mean())

            report[col] = metrics

            if metrics["abs_ge_90pct_ratio"] > 0.05 or metrics[
                    "abs_ge_99pct_ratio"] > 0.01:
                flagged_clip.append((col, metrics["abs_ge_90pct_ratio"],
                                     metrics["abs_ge_99pct_ratio"]))
            if metrics["zero_ratio"] > 0.5:
                flagged_zero.append((col, metrics["zero_ratio"]))

        self.diagnostic_report[timeframe] = report

        messages: List[str] = []
        if flagged_clip:
            top_clip = sorted(flagged_clip, key=lambda x: x[1],
                              reverse=True)[:5]
            clip_msg = (f"   Possible clipping (abs>={near_clip:.2f}) "
                        "detected:")
            details = ", ".join(
                f"{col}: {ratio:.1%} (abs>={0.99*clip_bound:.2f} {ratio99:.1%})"
                for col, ratio, ratio99 in top_clip)
            messages.append(f"{clip_msg} {details}")

        if flagged_zero:
            top_zero = sorted(flagged_zero, key=lambda x: x[1],
                              reverse=True)[:5]
            zero_msg = "   High zero saturation detected:"
            zero_details = ", ".join(f"{col}: {ratio:.1%}"
                                     for col, ratio in top_zero)
            messages.append(f"{zero_msg} {zero_details}")

        if messages:
            print(f"⚠️  EnhancedFeatureEngineer diagnostics ({timeframe}):")
            for msg in messages:
                print(msg)

    def _get_hurst_lags(self, window: int) -> np.ndarray:
        usable = self.hurst_lags[self.hurst_lags < max(window // 2, 3)]
        if usable.size >= 2:
            return usable
        upper = max(4, window // 2)
        if upper <= 2:
            return np.array([2, 3], dtype=int)
        generated = np.linspace(2, upper, num=4, dtype=int)
        generated = np.unique(generated)
        generated = generated[generated >= 2]
        if generated.size < 2:
            generated = np.array([2, upper], dtype=int)
        return generated

    def _rolling_hurst_fast(
        self,
        series: np.ndarray,
        window: int,
        lags: np.ndarray,
        min_valid_ratio: float = 0.8,
    ) -> np.ndarray:
        series = np.asarray(series, dtype=float)
        n = len(series)
        hurst = np.full(n, np.nan, dtype=float)
        if window <= 0 or n == 0:
            return hurst

        lags = lags[(lags >= 2) & (lags < window)]
        if lags.size < 2:
            return hurst

        log_lags = np.log(lags)

        for end in range(window, n + 1):
            window_slice = series[end - window:end]
            finite_mask = np.isfinite(window_slice)
            if finite_mask.mean() < min_valid_ratio:
                hurst[end - 1] = np.nan
                continue
            ts = window_slice[finite_mask]
            if ts.size < 10:
                hurst[end - 1] = np.nan
                continue

            rs_values: List[float] = []
            for lag in lags:
                max_blocks = ts.size // lag
                if max_blocks < 2:
                    rs_values.append(np.nan)
                    continue
                blocks = ts[:max_blocks * lag].reshape(max_blocks, lag)
                blocks = blocks - blocks.mean(axis=1, keepdims=True)
                cum_dev = np.cumsum(blocks, axis=1)
                ranges = cum_dev.max(axis=1) - cum_dev.min(axis=1)
                stdevs = blocks.std(axis=1, ddof=1)
                valid = stdevs > 1e-8
                if not np.any(valid):
                    rs_values.append(np.nan)
                    continue
                rs = ranges[valid] / (stdevs[valid] + 1e-8)
                if rs.size == 0:
                    rs_values.append(np.nan)
                    continue
                rs_values.append(np.mean(rs))

            rs_arr = np.array(rs_values, dtype=float)
            valid_rs = np.isfinite(rs_arr) & (rs_arr > 0)
            if valid_rs.sum() < 2:
                hurst[end - 1] = np.nan
                continue

            try:
                slope = np.polyfit(
                    log_lags[valid_rs],
                    np.log(rs_arr[valid_rs]),
                    1,
                )[0]
                hurst[end - 1] = np.clip(slope, 0.0, 1.0)
            except Exception:
                hurst[end - 1] = np.nan

        if window > 0:
            hurst[:window - 1] = np.nan
        return hurst

    def calculate_hurst_exponent(self,
                                 ts: np.ndarray,
                                 min_window: int = 10) -> float:
        """
        Calculate Hurst exponent using R/S analysis.

        Args:
            ts: Time series data
            min_window: Minimum window size

        Returns:
            Hurst exponent (0.5 = random, >0.5 = trending, <0.5 = mean-reverting)
        """
        if len(ts) < min_window * 2:
            return 0.5

        # Remove any NaN or inf values
        ts = ts[np.isfinite(ts)]
        if len(ts) < min_window * 2:
            return 0.5

        try:
            # Create range of window sizes
            lags = range(min_window, min(len(ts) // 2, 100))

            # Calculate R/S for each lag
            rs_values = []
            for lag in lags:
                # Split into chunks
                n_chunks = len(ts) // lag
                if n_chunks == 0:
                    continue

                rs_chunk = []
                for i in range(n_chunks):
                    chunk = ts[i * lag:(i + 1) * lag]
                    if len(chunk) < lag:
                        continue

                    # Mean-adjusted series
                    mean_chunk = np.mean(chunk)
                    y = chunk - mean_chunk

                    # Cumulative deviate
                    z = np.cumsum(y)

                    # Range
                    r = np.max(z) - np.min(z)

                    # Standard deviation
                    s = np.std(chunk, ddof=1)

                    if s > 0:
                        rs_chunk.append(r / s)

                if rs_chunk:
                    rs_values.append((lag, np.mean(rs_chunk)))

            if len(rs_values) < 2:
                return 0.5

            # Linear regression on log-log plot
            lags_log = np.log([x[0] for x in rs_values])
            rs_log = np.log([x[1] for x in rs_values])

            # Handle any inf or nan in logs
            valid_mask = np.isfinite(lags_log) & np.isfinite(rs_log)
            if np.sum(valid_mask) < 2:
                return 0.5

            lags_log = lags_log[valid_mask]
            rs_log = rs_log[valid_mask]

            # Fit linear regression
            hurst = np.polyfit(lags_log, rs_log, 1)[0]

            # Clip to reasonable range
            return np.clip(hurst, 0.0, 1.0)

        except Exception as e:
            return 0.5

    def calculate_wavelet_packet_features(
            self, data: np.ndarray) -> Dict[str, float]:
        """
        Calculate wavelet packet transform features.

        Args:
            data: Time series data

        Returns:
            Dictionary of WPT features
        """
        features = {}

        try:
            # Ensure data has enough length and is finite
            if len(data) < 2**self.wpt_level:
                return {}

            # Remove NaN and inf
            data_clean = data[np.isfinite(data)]
            if len(data_clean) < 2**self.wpt_level:
                return {}

            # Perform wavelet packet decomposition
            wp = pywt.WaveletPacket(
                data=data_clean,
                wavelet=self.wavelet,
                mode="symmetric",
                maxlevel=self.wpt_level,
            )

            # Get all nodes at specified level
            nodes = wp.get_level(self.wpt_level, "natural")

            # Calculate energy for each node
            energies = []
            for node in nodes:
                try:
                    coeffs = node.data
                    if len(coeffs) > 0:
                        energy = np.sum(np.square(coeffs))
                        energies.append(energy)

                        # Individual node features
                        features[f"wpt_{node.path}_energy"] = energy
                        features[f"wpt_{node.path}_mean"] = np.mean(coeffs)
                        features[f"wpt_{node.path}_std"] = np.std(coeffs)
                except:
                    continue

            if len(energies) > 0:
                total_energy = np.sum(energies)

                # Energy distribution features
                if total_energy > 0:
                    # Energy ratios
                    for i, node in enumerate(nodes):
                        if i < len(energies):
                            features[f"wpt_{node.path}_energy_ratio"] = (
                                energies[i] / total_energy)

                    # Shannon entropy of energy distribution
                    probs = np.array(energies) / total_energy
                    probs = probs[probs > 1e-10]  # Remove zero probabilities
                    shannon_entropy = -np.sum(probs * np.log(probs + 1e-10))
                    features["wpt_shannon_entropy"] = shannon_entropy

                    # Energy concentration (Gini coefficient-like)
                    sorted_energies = np.sort(energies)[::-1]
                    cumsum_energies = np.cumsum(sorted_energies) / total_energy
                    features["wpt_energy_concentration"] = (
                        cumsum_energies[0] if len(cumsum_energies) > 0 else 0)

                    # High frequency vs low frequency energy ratio
                    n_nodes = len(energies)
                    mid = n_nodes // 2
                    low_freq_energy = np.sum(energies[:mid])
                    high_freq_energy = np.sum(energies[mid:])

                    if low_freq_energy > 0:
                        features["wpt_high_low_ratio"] = (high_freq_energy /
                                                          low_freq_energy)

                    # Dominant frequency band
                    features["wpt_dominant_band"] = np.argmax(energies)

        except Exception as e:
            pass  # Return empty dict on error

        return features

    def add_hurst_features(self,
                           data: pd.DataFrame,
                           window: int = None) -> pd.DataFrame:
        """
        Add Hurst exponent features for ALL signal sources.
        对 close, open, volume, cvd, taker_buy_ratio 都计算Hurst指数
        """
        if window is None:
            window = self.hurst_window

        df = data.copy()

        # 严格校验：必须存在订单流关键列
        required_cols = ["cvd", "taker_buy_ratio"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"缺少订单流数据列: {missing}。这是数据错误，请检查上游数据准备流程。")

        # 定义需要计算Hurst的信号源
        signal_sources = {
            "close": df["close"].values,
            "open": df["open"].values,
            "volume": df["volume"].values,
        }

        # 加入订单流信号（已严格校验）
        signal_sources["cvd"] = df["cvd"].values
        signal_sources["taker_buy_ratio"] = df["taker_buy_ratio"].values

        print(f"      Hurst计算信号源: {list(signal_sources.keys())}")

        # 对每个信号源计算Hurst
        hurst_lags = self._get_hurst_lags(window)

        for source_name, source_data in signal_sources.items():
            hurst_series = self._rolling_hurst_fast(
                source_data,
                window=window,
                lags=hurst_lags,
            )
            hurst_series = np.nan_to_num(hurst_series, nan=0.5)
            df[f"{source_name}_hurst"] = hurst_series

            # Hurst衍生特征
            df[f"{source_name}_hurst_deviation"] = df[
                f"{source_name}_hurst"] - 0.5
            df[f"{source_name}_hurst_trend_signal"] = (
                df[f"{source_name}_hurst"] > 0.55).astype(int)
            df[f"{source_name}_hurst_mean_revert_signal"] = (
                df[f"{source_name}_hurst"] < 0.45).astype(int)
            df[f"{source_name}_hurst_change"] = df[
                f"{source_name}_hurst"].diff()
            df[f"{source_name}_hurst_acceleration"] = df[
                f"{source_name}_hurst_change"].diff()

        return df

    def add_wavelet_packet_features(self,
                                    data: pd.DataFrame,
                                    window: int = 100) -> pd.DataFrame:
        """
        Add wavelet packet transform features for ALL signal sources.
        对 close, open, volume, cvd, taker_buy_ratio 都做WPT分解
        """
        df = data.copy()

        # 定义需要做WPT的信号源
        signal_sources = {
            "close": df["close"].values,
            "open": df["open"].values,
            "volume": df["volume"].values,
        }

        # 如果有订单流数据，也加入
        if "cvd" in df.columns:
            signal_sources["cvd"] = df["cvd"].values
        if "taker_buy_ratio" in df.columns:
            signal_sources["taker_buy_ratio"] = df["taker_buy_ratio"].values

        print(f"      WPT分解信号源: {list(signal_sources.keys())}")

        # 对每个信号源计算WPT特征
        for source_name, source_data in signal_sources.items():
            wpt_features_list = []

            for i in range(len(source_data)):
                if i < window:
                    wpt_features_list.append({})
                else:
                    window_data = source_data[i - window:i]
                    wpt_features = self.calculate_wavelet_packet_features(
                        window_data)

                    # 添加前缀以区分不同信号源
                    prefixed_features = {
                        f"{source_name}_{k}": v
                        for k, v in wpt_features.items()
                    }
                    wpt_features_list.append(prefixed_features)

            # Convert to DataFrame
            wpt_df = pd.DataFrame(wpt_features_list, index=df.index)
            wpt_df = wpt_df.fillna(0)

            # Add to main dataframe
            for col in wpt_df.columns:
                df[col] = wpt_df[col].values

        return df

    def add_hilbert_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add Hilbert transform features for ALL signal sources.
        对 close, open, volume, cvd, taker_buy_ratio 都做Hilbert变换
        """
        df = data.copy()

        # 定义需要计算Hilbert的信号源
        signal_sources = {
            "close": df["close"].values,
            "open": df["open"].values,
            "volume": df["volume"].values,
        }

        if "cvd" in df.columns:
            signal_sources["cvd"] = df["cvd"].values
        if "taker_buy_ratio" in df.columns:
            signal_sources["taker_buy_ratio"] = df["taker_buy_ratio"].values

        print(f"      Hilbert变换信号源: {list(signal_sources.keys())}")

        # 对每个信号源计算Hilbert特征（强制使用 SciPy 实现，忽略 TA-Lib）
        for source_name, source_data in signal_sources.items():
            source_series = pd.Series(source_data, index=df.index, dtype=float)
            # 预清洗：前向填充 + 非有限值置零，避免外部 NaN 传播
            source_series = source_series.replace([np.inf, -np.inf],
                                                  np.nan).ffill()
            source_series = source_series.fillna(0.0)

            try:
                valid_data = source_series.to_numpy()
                valid_mask = np.isfinite(valid_data)
                valid_data = valid_data[valid_mask]

                if len(valid_data) < 10:
                    df[f"{source_name}_hilbert_amplitude"] = 0
                    df[f"{source_name}_hilbert_phase"] = 0
                    df[f"{source_name}_hilbert_frequency"] = 0
                    continue

                analytic_signal = hilbert(valid_data)
                amplitude = np.abs(analytic_signal)
                phase = np.angle(analytic_signal)
                frequency = np.diff(np.unwrap(phase)) / (2.0 * np.pi)
                frequency = np.concatenate([[frequency[0]], frequency])

                # Pad to original length if needed (left pad if we trimmed head)
                if len(amplitude) < len(source_series):
                    pad_len = len(source_series) - len(amplitude)
                    amplitude = np.pad(amplitude, (pad_len, 0), mode="edge")
                    phase = np.pad(phase, (pad_len, 0), mode="edge")
                    frequency = np.pad(frequency, (pad_len, 0), mode="edge")

                # 清理非有限值
                amplitude = np.nan_to_num(amplitude,
                                          nan=0.0,
                                          posinf=0.0,
                                          neginf=0.0)
                phase = np.nan_to_num(phase, nan=0.0, posinf=0.0, neginf=0.0)
                frequency = np.nan_to_num(frequency,
                                          nan=0.0,
                                          posinf=0.0,
                                          neginf=0.0)

                amplitude_series = pd.Series(amplitude[:len(df)],
                                             index=df.index)
                phase_series = pd.Series(phase[:len(df)], index=df.index)
                frequency_series = pd.Series(frequency[:len(df)],
                                             index=df.index)

                df[f"{source_name}_hilbert_amplitude"] = amplitude_series.values
                df[f"{source_name}_hilbert_phase"] = phase_series.values
                df[f"{source_name}_hilbert_frequency"] = frequency_series.values

                # Phase-based enhancements
                phase_sin = np.sin(phase_series.values)
                phase_cos = np.cos(phase_series.values)

                phase_unwrapped = np.unwrap(phase_series.values)
                phase_unwrapped_series = pd.Series(phase_unwrapped,
                                                   index=df.index)
                phase_change = phase_unwrapped_series.diff().fillna(0.0)
                phase_acceleration = phase_change.diff().fillna(0.0)

                df[f"{source_name}_hilbert_phase_sin"] = phase_sin
                df[f"{source_name}_hilbert_phase_cos"] = phase_cos
                df[f"{source_name}_hilbert_phase_change"] = phase_change.values
                df[f"{source_name}_hilbert_phase_acceleration"] = (
                    phase_acceleration.values)

            except Exception as e:
                print(f"      Warning: Hilbert for {source_name} failed: {e}")
                df[f"{source_name}_hilbert_amplitude"] = 0
                df[f"{source_name}_hilbert_phase"] = 0
                df[f"{source_name}_hilbert_frequency"] = 0
                df[f"{source_name}_hilbert_phase_sin"] = 0
                df[f"{source_name}_hilbert_phase_cos"] = 1
                df[f"{source_name}_hilbert_phase_change"] = 0
                df[f"{source_name}_hilbert_phase_acceleration"] = 0

        return df

    def add_spectral_features(self,
                              data: pd.DataFrame,
                              window: int = 100) -> pd.DataFrame:
        """
        Add spectral analysis features for ALL signal sources (FIXED VERSION).
        使用滚动窗口计算spectral特征，生成时间序列

        Args:
            data: DataFrame with OHLCV and other columns
            window: 滚动窗口大小（默认100根K线）
        """
        from scipy.signal import periodogram

        df = data.copy()

        # 定义需要计算光谱的信号源
        signal_sources = {
            "close": df["close"].values,
            "open": df["open"].values,
            "volume": df["volume"].values,
        }

        if "cvd" in df.columns:
            signal_sources["cvd"] = df["cvd"].values
        if "taker_buy_ratio" in df.columns:
            signal_sources["taker_buy_ratio"] = df["taker_buy_ratio"].values

        print(f"      光谱分析信号源: {list(signal_sources.keys())} (滚动窗口={window})")

        # 对每个信号源计算滚动spectral特征
        for source_name, source_data in signal_sources.items():
            # 初始化特征数组
            n_samples = len(source_data)
            spectral_centroid = np.zeros(n_samples)
            spectral_bandwidth = np.zeros(n_samples)
            spectral_rolloff = np.zeros(n_samples)

            # 滚动窗口计算
            for i in range(n_samples):
                if i < window:
                    # 窗口不足，保持为0
                    continue

                try:
                    # 获取窗口数据
                    window_data = source_data[i - window:i]

                    # 移除NaN
                    valid_data = window_data[np.isfinite(window_data)]

                    if len(valid_data) < 10:
                        continue

                    # 使用periodogram计算功率谱密度（更稳定）
                    if source_name in ["close", "open"]:
                        # 对价格信号，先转换为收益率
                        returns = np.diff(valid_data) / (valid_data[:-1] +
                                                         1e-10)
                        if len(returns) < 5:
                            continue
                        freqs, psd = periodogram(returns, fs=1.0)
                    else:
                        # 对volume、cvd等，直接使用
                        freqs, psd = periodogram(valid_data, fs=1.0)

                    # 归一化
                    psd_sum = np.sum(psd)
                    if psd_sum == 0 or not np.isfinite(psd_sum):
                        continue

                    psd_norm = psd / psd_sum

                    # 1. Spectral Centroid（谱质心）
                    spectral_centroid[i] = np.sum(freqs * psd_norm)

                    # 2. Spectral Bandwidth（谱带宽）
                    spectral_bandwidth[i] = np.sqrt(
                        np.sum(((freqs - spectral_centroid[i])**2) * psd_norm))

                    # 3. Spectral Rolloff（谱滚降，95%能量点）
                    cumsum_psd = np.cumsum(psd_norm)
                    rolloff_idx = np.where(cumsum_psd >= 0.95)[0]
                    if len(rolloff_idx) > 0:
                        spectral_rolloff[i] = freqs[rolloff_idx[0]]

                except Exception:
                    # 如果计算失败，保持为0
                    continue

            # 添加到DataFrame（时间序列）
            df[f"{source_name}_spectral_centroid"] = spectral_centroid
            df[f"{source_name}_spectral_bandwidth"] = spectral_bandwidth
            df[f"{source_name}_spectral_rolloff"] = spectral_rolloff

        return df

    def add_advanced_derived_features(self,
                                      data: pd.DataFrame) -> pd.DataFrame:
        """
        Add advanced derived features from baseline model.
        添加基线模型中的高级衍生特征
        """
        df = data.copy()

        print("      计算高级衍生特征...")

        try:
            # 需要的基础特征
            if "bb_upper" not in df.columns or "atr" not in df.columns:
                print("        缺少BB或ATR，跳过部分衍生特征")
                return df

            # 1. BB Width相关
            df["bb_width"] = (df["bb_upper"] - df["bb_lower"]).abs()
            df["bb_width_normalized"] = df["bb_width"] / df["atr"].replace(
                0, np.nan)
            df["bb_width_normalized"] = (df["bb_width_normalized"].replace(
                [np.inf, -np.inf], np.nan).fillna(0))

            # 2. Range ratio
            df["hl"] = df["high"] - df["low"]
            df["range_ratio_5bar"] = df["hl"].rolling(
                5).mean() / df["hl"].rolling(20).mean().replace(0, np.nan)
            df["range_ratio_5bar"] = df["range_ratio_5bar"].fillna(1)

            # 3. Compression duration
            perc = df["bb_width"].rolling(20, min_periods=5).quantile(0.2)
            low_vol = (df["bb_width"] <= perc).astype(int)
            df["compression_duration"] = (low_vol.groupby(
                (low_vol != low_vol.shift()).cumsum()).cumsum()) * low_vol

            # 4. Compression energy
            df["compression_energy"] = (
                1.0 / df["bb_width"].replace(0, np.nan)) * df["volume_ratio"]
            df["compression_energy"] = (df["compression_energy"].replace(
                [np.inf, -np.inf], np.nan).fillna(0))

            # 5. ATR percentile
            def pct_rank(x):
                r = pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
                return r

            df["atr_percentile"] = (df["atr"].rolling(
                100, min_periods=20).apply(pct_rank, raw=False))

            # 6. Volatility reversal score
            atr_mean = df["atr"].rolling(50).mean()
            atr_std = df["atr"].rolling(50).std()
            df["volatility_reversal_score"] = (
                df["atr"] - atr_mean) / atr_std.replace(0, np.nan)
            df["volatility_reversal_score"] = df[
                "volatility_reversal_score"].fillna(0)

            # 7. Volatility squeeze flag
            df["volatility_squeeze_flag"] = (df["bb_width"]
                                             < (2.0 * df["atr"])).astype(int)

            # 8. Price range symmetry
            # Use configurable shift to avoid using current close (data leakage)
            df["price_range_symmetry"] = (self._shift_feature(
                df["high"]) - self._shift_feature(df["close"])) / (
                    (self._shift_feature(df["close"]) -
                     self._shift_feature(df["low"])).replace(0, np.nan))
            df["price_range_symmetry"] = (df["price_range_symmetry"].replace(
                [np.inf, -np.inf], np.nan).fillna(1))

            # 9. Volume anomaly
            df["volume_anomaly"] = (
                df["volume"] /
                df["volume"].ewm(span=20, min_periods=10).mean())

            # 10. Up/Down volume ratio
            # Use configurable shift to avoid using current close (data leakage)
            close_ref = self._shift_feature(df["close"])
            up = (close_ref > self._shift_feature(df["close"],
                                                  offset=1)).astype(int)
            df["up_vol"] = (df["volume"] * up).rolling(20).sum()
            df["down_vol"] = (df["volume"] * (1 - up)).rolling(20).sum()
            df["upvol_downvol_ratio"] = df["up_vol"] / df["down_vol"].replace(
                0, np.nan)
            df["upvol_downvol_ratio"] = (df["upvol_downvol_ratio"].replace(
                [np.inf, -np.inf], np.nan).fillna(1))

            # 11. ROC and acceleration
            # Note: pct_change() already uses past data, configurable shift ensures alignment
            roc_5_raw = df["close"].pct_change(5)
            df["roc_5"] = self._shift_feature(roc_5_raw)
            roc_3 = df["close"].pct_change(3)
            roc_3_shifted = self._shift_feature(roc_3)
            df["acceleration_3"] = roc_3_shifted - self._shift_feature(
                roc_3, offset=1)

            # 12. Price vs EMA distance
            # Use configurable shift to avoid using current close (data leakage)
            df["price_vs_ema_distance"] = (
                (self._shift_feature(df["close"]) -
                 self._shift_feature(df["sma_20"])) /
                self._shift_feature(df["atr"]).replace(0, np.nan))
            df["price_vs_ema_distance"] = (df["price_vs_ema_distance"].replace(
                [np.inf, -np.inf], np.nan).fillna(0))

            # 13. Momentum persistence
            # Use configurable shift to avoid using current close (data leakage)
            sig = np.sign(df["close"].diff())
            sig = self._shift_feature(sig)
            df["momentum_persistence"] = sig.rolling(10).apply(
                lambda x: (np.sum(x > 0) / max(len(x), 1)), raw=True)

            # 14. Slope consistency
            ema10 = df["close"].ewm(span=10).mean()
            ema20 = df["close"].ewm(span=20).mean()
            ema50 = df["close"].ewm(span=50).mean()
            slope10 = np.sign(ema10.diff())
            slope20 = np.sign(ema20.diff())
            slope50 = np.sign(ema50.diff())
            df["slope_consistency_score"] = ((slope10 == slope20).astype(int) +
                                             (slope20 == slope50).astype(int) +
                                             (slope10 == slope50).astype(int))

            # 15. Temporal features (时间特征)
            try:
                idx = df.index
                # 检查索引是否是datetime类型
                if hasattr(idx, "hour") and hasattr(idx, "dayofweek"):
                    df["hour_of_day_sin"] = np.sin(2 * np.pi * idx.hour / 24)
                    df["hour_of_day_cos"] = np.cos(2 * np.pi * idx.hour / 24)
                    df["day_of_week_sin"] = np.sin(2 * np.pi * idx.dayofweek /
                                                   7)
                    df["day_of_week_cos"] = np.cos(2 * np.pi * idx.dayofweek /
                                                   7)
                else:
                    # 如果不是datetime索引，创建虚拟时间特征
                    df["hour_of_day_sin"] = 0
                    df["hour_of_day_cos"] = 1
                    df["day_of_week_sin"] = 0
                    df["day_of_week_cos"] = 1
            except Exception as e:
                print(f"        Warning: 时间特征计算失败: {e}")
                # 创建虚拟时间特征
                df["hour_of_day_sin"] = 0
                df["hour_of_day_cos"] = 1
                df["day_of_week_sin"] = 0
                df["day_of_week_cos"] = 1

            # 16. Structure tension
            # Use configurable shift to avoid using current close (data leakage)
            dist_high = (
                self._shift_feature(df["high"]).rolling(50).max() -
                self._shift_feature(df["close"])).abs() / self._shift_feature(
                    df["close"]).replace(0, np.nan)
            dist_low = (self._shift_feature(df["close"]) - self._shift_feature(
                df["low"]).rolling(50).min()).abs() / self._shift_feature(
                    df["close"]).replace(0, np.nan)
            df["structure_tension"] = (
                dist_high + dist_low) / df["bb_width"].replace(0, np.nan)
            df["structure_tension"] = (df["structure_tension"].replace(
                [np.inf, -np.inf], np.nan).fillna(0))

            # 17. Trend volatility alignment
            df["trend_volatility_alignment"] = np.sign(
                df["roc_5"]).fillna(0) * df["atr_percentile"].fillna(0)

            # 18. Compression to breakout probability
            df["compression_to_breakout_prob"] = df[
                "compression_duration"].fillna(0) * df["roc_5"].fillna(0)

        except Exception as e:
            print(f"      Warning: 高级衍生特征计算失败: {e}")

        return df

    def add_order_flow_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add advanced order flow features.
        添加高级订单流特征：Order Flow Imbalance, Delta Divergence, Liquidity等
        """
        df = data.copy()

        # 严格模式：必须存在订单流数据
        if "cvd" not in df.columns or "taker_buy_ratio" not in df.columns:
            raise ValueError(
                "缺少订单流数据(cvd/taker_buy_ratio)，这是数据错误，请检查上游数据准备流程。")

        print("      计算订单流特征...")

        try:
            # 1. Order Flow Imbalance (买卖压力不平衡)
            # 使用buy_qty和sell_qty如果存在，否则从taker_buy_ratio推导
            if "buy_qty" in df.columns and "sell_qty" in df.columns:
                total_vol = df["buy_qty"] + df["sell_qty"]
                df["order_flow_imbalance"] = (df["buy_qty"] - df["sell_qty"]
                                              ) / total_vol.replace(0, np.nan)
                df["order_flow_imbalance"] = df["order_flow_imbalance"].fillna(
                    0)

                # 改用滚动窗口OFI（避免全局cumsum），并进行Z-score标准化
                ofi_short_raw = df["order_flow_imbalance"].rolling(
                    20, min_periods=1).sum()
                ofi_medium_raw = df["order_flow_imbalance"].rolling(
                    60, min_periods=1).sum()
                ofi_long_raw = df["order_flow_imbalance"].rolling(
                    288, min_periods=1).sum()

                # Z-score标准化
                ofi_short_mean = ofi_short_raw.rolling(100,
                                                       min_periods=20).mean()
                ofi_short_std = ofi_short_raw.rolling(100,
                                                      min_periods=20).std()
                df["ofi_short"] = ((ofi_short_raw - ofi_short_mean) /
                                   ofi_short_std.replace(0, np.nan)).replace(
                                       [np.inf, -np.inf],
                                       np.nan).fillna(0).clip(
                                           -self.feature_clip_bound,
                                           self.feature_clip_bound)

                ofi_medium_mean = ofi_medium_raw.rolling(
                    100, min_periods=20).mean()
                ofi_medium_std = ofi_medium_raw.rolling(100,
                                                        min_periods=20).std()
                df["ofi_medium"] = ((ofi_medium_raw - ofi_medium_mean) /
                                    ofi_medium_std.replace(0, np.nan)).replace(
                                        [np.inf, -np.inf],
                                        np.nan).fillna(0).clip(
                                            -self.feature_clip_bound,
                                            self.feature_clip_bound)

                ofi_long_mean = ofi_long_raw.rolling(100,
                                                     min_periods=20).mean()
                ofi_long_std = ofi_long_raw.rolling(100, min_periods=20).std()
                df["ofi_long"] = ((ofi_long_raw - ofi_long_mean) /
                                  ofi_long_std.replace(0, np.nan)).replace(
                                      [np.inf, -np.inf],
                                      np.nan).fillna(0).clip(
                                          -self.feature_clip_bound,
                                          self.feature_clip_bound)

                # 向后兼容：保留cumulative_ofi，并进行Z-score标准化
                cumulative_ofi_raw = df["order_flow_imbalance"].cumsum()
                cumulative_ofi_mean = cumulative_ofi_raw.rolling(
                    100, min_periods=20).mean()
                cumulative_ofi_std = cumulative_ofi_raw.rolling(
                    100, min_periods=20).std()
                df["cumulative_ofi"] = (
                    (cumulative_ofi_raw - cumulative_ofi_mean) /
                    cumulative_ofi_std.replace(0, np.nan)).replace(
                        [np.inf, -np.inf],
                        np.nan).fillna(0).clip(-self.feature_clip_bound,
                                               self.feature_clip_bound)

                # Order flow momentum (OFI变化率)，已经是[-1,1]范围内，但进一步标准化
                ofi_momentum_5_raw = df["order_flow_imbalance"].rolling(
                    5).mean()
                ofi_momentum_20_raw = df["order_flow_imbalance"].rolling(
                    20).mean()

                # Z-score标准化
                ofi_momentum_5_mean = ofi_momentum_5_raw.rolling(
                    50, min_periods=5).mean()
                ofi_momentum_5_std = ofi_momentum_5_raw.rolling(
                    50, min_periods=5).std()
                df["ofi_momentum_5"] = (
                    (ofi_momentum_5_raw - ofi_momentum_5_mean) /
                    ofi_momentum_5_std.replace(0, np.nan)).replace(
                        [np.inf, -np.inf],
                        np.nan).fillna(0).clip(-self.feature_clip_bound,
                                               self.feature_clip_bound)

                ofi_momentum_20_mean = ofi_momentum_20_raw.rolling(
                    50, min_periods=20).mean()
                ofi_momentum_20_std = ofi_momentum_20_raw.rolling(
                    50, min_periods=20).std()
                df["ofi_momentum_20"] = (
                    (ofi_momentum_20_raw - ofi_momentum_20_mean) /
                    ofi_momentum_20_std.replace(0, np.nan)).replace(
                        [np.inf, -np.inf],
                        np.nan).fillna(0).clip(-self.feature_clip_bound,
                                               self.feature_clip_bound)

                # Order flow volatility，Z-score标准化
                ofi_volatility_raw = df["order_flow_imbalance"].rolling(
                    20).std()
                ofi_volatility_mean = ofi_volatility_raw.rolling(
                    50, min_periods=20).mean()
                ofi_volatility_std = ofi_volatility_raw.rolling(
                    50, min_periods=20).std()
                df["ofi_volatility"] = (
                    (ofi_volatility_raw - ofi_volatility_mean) /
                    ofi_volatility_std.replace(0, np.nan)).replace(
                        [np.inf, -np.inf],
                        np.nan).fillna(0).clip(-self.feature_clip_bound,
                                               self.feature_clip_bound)
            else:
                # 从taker_buy_ratio推导
                df["order_flow_imbalance"] = (df["taker_buy_ratio"] -
                                              0.5) * 2  # 映射到[-1, 1]

                # 改用滚动窗口OFI，并进行Z-score标准化
                ofi_short_raw = df["order_flow_imbalance"].rolling(
                    20, min_periods=1).sum()
                ofi_medium_raw = df["order_flow_imbalance"].rolling(
                    60, min_periods=1).sum()
                ofi_long_raw = df["order_flow_imbalance"].rolling(
                    288, min_periods=1).sum()

                # Z-score标准化
                ofi_short_mean = ofi_short_raw.rolling(100,
                                                       min_periods=20).mean()
                ofi_short_std = ofi_short_raw.rolling(100,
                                                      min_periods=20).std()
                df["ofi_short"] = ((ofi_short_raw - ofi_short_mean) /
                                   ofi_short_std.replace(0, np.nan)).replace(
                                       [np.inf, -np.inf],
                                       np.nan).fillna(0).clip(
                                           -self.feature_clip_bound,
                                           self.feature_clip_bound)

                ofi_medium_mean = ofi_medium_raw.rolling(
                    100, min_periods=20).mean()
                ofi_medium_std = ofi_medium_raw.rolling(100,
                                                        min_periods=20).std()
                df["ofi_medium"] = ((ofi_medium_raw - ofi_medium_mean) /
                                    ofi_medium_std.replace(0, np.nan)).replace(
                                        [np.inf, -np.inf],
                                        np.nan).fillna(0).clip(
                                            -self.feature_clip_bound,
                                            self.feature_clip_bound)

                ofi_long_mean = ofi_long_raw.rolling(100,
                                                     min_periods=20).mean()
                ofi_long_std = ofi_long_raw.rolling(100, min_periods=20).std()
                df["ofi_long"] = ((ofi_long_raw - ofi_long_mean) /
                                  ofi_long_std.replace(0, np.nan)).replace(
                                      [np.inf, -np.inf],
                                      np.nan).fillna(0).clip(
                                          -self.feature_clip_bound,
                                          self.feature_clip_bound)

                # 向后兼容，并进行Z-score标准化
                cumulative_ofi_raw = df["order_flow_imbalance"].cumsum()
                cumulative_ofi_mean = cumulative_ofi_raw.rolling(
                    100, min_periods=20).mean()
                cumulative_ofi_std = cumulative_ofi_raw.rolling(
                    100, min_periods=20).std()
                df["cumulative_ofi"] = (
                    (cumulative_ofi_raw - cumulative_ofi_mean) /
                    cumulative_ofi_std.replace(0, np.nan)).replace(
                        [np.inf, -np.inf],
                        np.nan).fillna(0).clip(-self.feature_clip_bound,
                                               self.feature_clip_bound)

                # Z-score标准化
                ofi_momentum_5_raw = df["order_flow_imbalance"].rolling(
                    5).mean()
                ofi_momentum_20_raw = df["order_flow_imbalance"].rolling(
                    20).mean()

                ofi_momentum_5_mean = ofi_momentum_5_raw.rolling(
                    50, min_periods=5).mean()
                ofi_momentum_5_std = ofi_momentum_5_raw.rolling(
                    50, min_periods=5).std()
                df["ofi_momentum_5"] = (
                    (ofi_momentum_5_raw - ofi_momentum_5_mean) /
                    ofi_momentum_5_std.replace(0, np.nan)).replace(
                        [np.inf, -np.inf],
                        np.nan).fillna(0).clip(-self.feature_clip_bound,
                                               self.feature_clip_bound)

                ofi_momentum_20_mean = ofi_momentum_20_raw.rolling(
                    50, min_periods=20).mean()
                ofi_momentum_20_std = ofi_momentum_20_raw.rolling(
                    50, min_periods=20).std()
                df["ofi_momentum_20"] = (
                    (ofi_momentum_20_raw - ofi_momentum_20_mean) /
                    ofi_momentum_20_std.replace(0, np.nan)).replace(
                        [np.inf, -np.inf],
                        np.nan).fillna(0).clip(-self.feature_clip_bound,
                                               self.feature_clip_bound)

                ofi_volatility_raw = df["order_flow_imbalance"].rolling(
                    20).std()
                ofi_volatility_mean = ofi_volatility_raw.rolling(
                    50, min_periods=20).mean()
                ofi_volatility_std = ofi_volatility_raw.rolling(
                    50, min_periods=20).std()
                df["ofi_volatility"] = (
                    (ofi_volatility_raw - ofi_volatility_mean) /
                    ofi_volatility_std.replace(0, np.nan)).replace(
                        [np.inf, -np.inf],
                        np.nan).fillna(0).clip(-self.feature_clip_bound,
                                               self.feature_clip_bound)

            # 2. Delta Divergence (CVD vs Price背离)
            # 价格动量
            # pct_change() already uses past data; configurable shift ensures alignment
            price_momentum_5_raw = df["close"].pct_change(5)
            price_momentum_20_raw = df["close"].pct_change(20)
            price_momentum_5 = self._shift_feature(price_momentum_5_raw)
            price_momentum_20 = self._shift_feature(price_momentum_20_raw)

            # 优先使用新的CVD滚动窗口特征，如果不存在则使用原始CVD
            if "cvd_change_5" in df.columns and "cvd_change_20" in df.columns:
                # 使用预计算的CVD变化率（更安全，避免全局cumsum）
                cvd_change_5 = df["cvd_change_5"]
                cvd_change_20 = df["cvd_change_20"]
            else:
                # 向后兼容：使用原始CVD的diff
                cvd_change_5 = df["cvd"].diff(5)
                cvd_change_20 = df["cvd"].diff(20)

            # CVD动量标准化
            cvd_change_5_norm = (cvd_change_5 -
                                 cvd_change_5.rolling(50).mean()
                                 ) / cvd_change_5.rolling(50).std()
            cvd_change_20_norm = (cvd_change_20 -
                                  cvd_change_20.rolling(50).mean()
                                  ) / cvd_change_20.rolling(50).std()

            # 价格动量也需要标准化
            price_momentum_5_mean = price_momentum_5.rolling(
                50, min_periods=5).mean()
            price_momentum_5_std = price_momentum_5.rolling(
                50, min_periods=5).std()
            price_momentum_5_norm = (
                (price_momentum_5 - price_momentum_5_mean) /
                price_momentum_5_std.replace(0, np.nan)).replace(
                    [np.inf, -np.inf],
                    np.nan).fillna(0).clip(-self.feature_clip_bound,
                                           self.feature_clip_bound)

            price_momentum_20_mean = price_momentum_20.rolling(
                50, min_periods=20).mean()
            price_momentum_20_std = price_momentum_20.rolling(
                50, min_periods=20).std()
            price_momentum_20_norm = (
                (price_momentum_20 - price_momentum_20_mean) /
                price_momentum_20_std.replace(0, np.nan)).replace(
                    [np.inf, -np.inf],
                    np.nan).fillna(0).clip(-self.feature_clip_bound,
                                           self.feature_clip_bound)

            # Delta divergence = price momentum - CVD momentum（两个都标准化后）
            df["delta_divergence_5"] = price_momentum_5_norm - cvd_change_5_norm
            df["delta_divergence_20"] = price_momentum_20_norm - cvd_change_20_norm

            # Divergence strength (绝对值)，不需要归一化，因为已经是标准化的差值
            df["divergence_strength"] = df["delta_divergence_20"].abs()

            # 3. 多时间框架CVD特征（如果有新的滚动窗口CVD）
            if ("cvd_short" in df.columns and "cvd_medium" in df.columns
                    and "cvd_long" in df.columns):
                # 短期CVD趋势，Z-score标准化
                cvd_short_trend_raw = df["cvd_short"].diff(5)
                cvd_short_trend_mean = cvd_short_trend_raw.rolling(
                    50, min_periods=5).mean()
                cvd_short_trend_std = cvd_short_trend_raw.rolling(
                    50, min_periods=5).std()
                df["cvd_short_trend"] = (
                    (cvd_short_trend_raw - cvd_short_trend_mean) /
                    cvd_short_trend_std.replace(0, np.nan)).replace(
                        [np.inf, -np.inf],
                        np.nan).fillna(0).clip(-self.feature_clip_bound,
                                               self.feature_clip_bound)

                cvd_short_momentum_raw = df["cvd_short_trend"].diff()
                cvd_short_momentum_mean = cvd_short_momentum_raw.rolling(
                    50, min_periods=5).mean()
                cvd_short_momentum_std = cvd_short_momentum_raw.rolling(
                    50, min_periods=5).std()
                df["cvd_short_momentum"] = (
                    (cvd_short_momentum_raw - cvd_short_momentum_mean) /
                    cvd_short_momentum_std.replace(0, np.nan)).replace(
                        [np.inf, -np.inf],
                        np.nan).fillna(0).clip(-self.feature_clip_bound,
                                               self.feature_clip_bound)

                # 中期CVD趋势，Z-score标准化
                cvd_medium_trend_raw = df["cvd_medium"].diff(10)
                cvd_medium_trend_mean = cvd_medium_trend_raw.rolling(
                    50, min_periods=10).mean()
                cvd_medium_trend_std = cvd_medium_trend_raw.rolling(
                    50, min_periods=10).std()
                df["cvd_medium_trend"] = (
                    (cvd_medium_trend_raw - cvd_medium_trend_mean) /
                    cvd_medium_trend_std.replace(0, np.nan)).replace(
                        [np.inf, -np.inf],
                        np.nan).fillna(0).clip(-self.feature_clip_bound,
                                               self.feature_clip_bound)

                cvd_medium_momentum_raw = df["cvd_medium_trend"].diff()
                cvd_medium_momentum_mean = cvd_medium_momentum_raw.rolling(
                    50, min_periods=10).mean()
                cvd_medium_momentum_std = cvd_medium_momentum_raw.rolling(
                    50, min_periods=10).std()
                df["cvd_medium_momentum"] = (
                    (cvd_medium_momentum_raw - cvd_medium_momentum_mean) /
                    cvd_medium_momentum_std.replace(0, np.nan)).replace(
                        [np.inf, -np.inf],
                        np.nan).fillna(0).clip(-self.feature_clip_bound,
                                               self.feature_clip_bound)

                # 长期CVD趋势，Z-score标准化
                cvd_long_trend_raw = df["cvd_long"].diff(20)
                cvd_long_trend_mean = cvd_long_trend_raw.rolling(
                    50, min_periods=20).mean()
                cvd_long_trend_std = cvd_long_trend_raw.rolling(
                    50, min_periods=20).std()
                df["cvd_long_trend"] = (
                    (cvd_long_trend_raw - cvd_long_trend_mean) /
                    cvd_long_trend_std.replace(0, np.nan)).replace(
                        [np.inf, -np.inf],
                        np.nan).fillna(0).clip(-self.feature_clip_bound,
                                               self.feature_clip_bound)

                # CVD跨周期关系，使用log转换和标准化
                cvd_short_medium_ratio_raw = df["cvd_short"] / (
                    df["cvd_medium"].abs() + 1e-10)
                # 使用log转换避免极端值，然后标准化
                cvd_short_medium_ratio_log = np.log1p(
                    np.abs(cvd_short_medium_ratio_raw)) * np.sign(
                        cvd_short_medium_ratio_raw)
                cvd_short_medium_ratio_mean = cvd_short_medium_ratio_log.rolling(
                    50, min_periods=10).mean()
                cvd_short_medium_ratio_std = cvd_short_medium_ratio_log.rolling(
                    50, min_periods=10).std()
                df["cvd_short_medium_ratio"] = (
                    (cvd_short_medium_ratio_log - cvd_short_medium_ratio_mean)
                    / cvd_short_medium_ratio_std.replace(0, np.nan)).replace(
                        [np.inf, -np.inf],
                        np.nan).fillna(0).clip(-self.feature_clip_bound,
                                               self.feature_clip_bound)

                cvd_medium_long_ratio_raw = df["cvd_medium"] / (
                    df["cvd_long"].abs() + 1e-10)
                cvd_medium_long_ratio_log = np.log1p(
                    np.abs(cvd_medium_long_ratio_raw)) * np.sign(
                        cvd_medium_long_ratio_raw)
                cvd_medium_long_ratio_mean = cvd_medium_long_ratio_log.rolling(
                    50, min_periods=10).mean()
                cvd_medium_long_ratio_std = cvd_medium_long_ratio_log.rolling(
                    50, min_periods=10).std()
                df["cvd_medium_long_ratio"] = (
                    (cvd_medium_long_ratio_log - cvd_medium_long_ratio_mean) /
                    cvd_medium_long_ratio_std.replace(0, np.nan)).replace(
                        [np.inf, -np.inf],
                        np.nan).fillna(0).clip(-self.feature_clip_bound,
                                               self.feature_clip_bound)

                # CVD趋势一致性（短中长期同向），已经是0/1，不需要归一化
                cvd_short_sign = np.sign(df["cvd_short"])
                cvd_medium_sign = np.sign(df["cvd_medium"])
                cvd_long_sign = np.sign(df["cvd_long"])
                df["cvd_trend_alignment"] = (
                    (cvd_short_sign == cvd_medium_sign)
                    & (cvd_medium_sign == cvd_long_sign)).astype(int)

            # 4. CVD归一化特征（如果存在）
            if "cvd_normalized" in df.columns:
                df["cvd_norm_momentum"] = df["cvd_normalized"].rolling(
                    5).mean()
                df["cvd_norm_extreme"] = (df["cvd_normalized"].abs()
                                          > 0.6).astype(int)

            # 5. Taker Buy Ratio动量和极值
            # taker_buy_ratio 在 [0, 1] 范围内，diff值在 [-1, 1] 范围内，但需要标准化
            tbr_momentum_5_raw = df["taker_buy_ratio"].diff(5)
            tbr_momentum_20_raw = df["taker_buy_ratio"].diff(20)

            tbr_momentum_5_mean = tbr_momentum_5_raw.rolling(
                50, min_periods=5).mean()
            tbr_momentum_5_std = tbr_momentum_5_raw.rolling(
                50, min_periods=5).std()
            df["tbr_momentum_5"] = (
                (tbr_momentum_5_raw - tbr_momentum_5_mean) /
                tbr_momentum_5_std.replace(0, np.nan)).replace(
                    [np.inf, -np.inf],
                    np.nan).fillna(0).clip(-self.feature_clip_bound,
                                           self.feature_clip_bound)

            tbr_momentum_20_mean = tbr_momentum_20_raw.rolling(
                50, min_periods=20).mean()
            tbr_momentum_20_std = tbr_momentum_20_raw.rolling(
                50, min_periods=20).std()
            df["tbr_momentum_20"] = (
                (tbr_momentum_20_raw - tbr_momentum_20_mean) /
                tbr_momentum_20_std.replace(0, np.nan)).replace(
                    [np.inf, -np.inf],
                    np.nan).fillna(0).clip(-self.feature_clip_bound,
                                           self.feature_clip_bound)

            # TBR极值（买方/卖方主导）
            df["tbr_extreme_buy"] = (df["taker_buy_ratio"]
                                     > 0.7).astype(int)  # 买方主导
            df["tbr_extreme_sell"] = (df["taker_buy_ratio"]
                                      < 0.3).astype(int)  # 卖方主导
            df["tbr_neutral"] = ((df["taker_buy_ratio"] >= 0.45) &
                                 (df["taker_buy_ratio"] <= 0.55)).astype(int)

            # 6. CVD Slope (CVD变化趋势) - 向后兼容，并进行Z-score标准化
            if "cvd_change_1" in df.columns:
                # 使用预计算的delta
                cvd_slope_3_raw = df["cvd_change_1"].rolling(3).sum()
                cvd_slope_10_raw = df["cvd_change_1"].rolling(10).sum()
                cvd_slope_30_raw = df["cvd_change_1"].rolling(30).sum()
            else:
                # 向后兼容
                cvd_slope_3_raw = df["cvd"].diff(3)
                cvd_slope_10_raw = df["cvd"].diff(10)
                cvd_slope_30_raw = df["cvd"].diff(30)

            # Z-score标准化
            cvd_slope_3_mean = cvd_slope_3_raw.rolling(50,
                                                       min_periods=3).mean()
            cvd_slope_3_std = cvd_slope_3_raw.rolling(50, min_periods=3).std()
            df["cvd_slope_3"] = ((cvd_slope_3_raw - cvd_slope_3_mean) /
                                 cvd_slope_3_std.replace(0, np.nan)).replace(
                                     [np.inf, -np.inf], np.nan).fillna(0).clip(
                                         -self.feature_clip_bound,
                                         self.feature_clip_bound)

            cvd_slope_10_mean = cvd_slope_10_raw.rolling(
                50, min_periods=10).mean()
            cvd_slope_10_std = cvd_slope_10_raw.rolling(50,
                                                        min_periods=10).std()
            df["cvd_slope_10"] = ((cvd_slope_10_raw - cvd_slope_10_mean) /
                                  cvd_slope_10_std.replace(0, np.nan)).replace(
                                      [np.inf, -np.inf],
                                      np.nan).fillna(0).clip(
                                          -self.feature_clip_bound,
                                          self.feature_clip_bound)

            cvd_slope_30_mean = cvd_slope_30_raw.rolling(
                50, min_periods=30).mean()
            cvd_slope_30_std = cvd_slope_30_raw.rolling(50,
                                                        min_periods=30).std()
            df["cvd_slope_30"] = ((cvd_slope_30_raw - cvd_slope_30_mean) /
                                  cvd_slope_30_std.replace(0, np.nan)).replace(
                                      [np.inf, -np.inf],
                                      np.nan).fillna(0).clip(
                                          -self.feature_clip_bound,
                                          self.feature_clip_bound)

            # CVD加速度，Z-score标准化
            cvd_acceleration_raw = df["cvd_slope_10"].diff()
            cvd_acceleration_mean = cvd_acceleration_raw.rolling(
                50, min_periods=10).mean()
            cvd_acceleration_std = cvd_acceleration_raw.rolling(
                50, min_periods=10).std()
            df["cvd_acceleration"] = (
                (cvd_acceleration_raw - cvd_acceleration_mean) /
                cvd_acceleration_std.replace(0, np.nan)).replace(
                    [np.inf, -np.inf],
                    np.nan).fillna(0).clip(-self.feature_clip_bound,
                                           self.feature_clip_bound)

            # 7. Liquidity Drain特征（成交量突然下降）
            volume_ma_20 = df["volume"].rolling(20).mean()
            volume_std_20 = df["volume"].rolling(20).std()

            # 流动性枯竭信号：成交量突然低于均值-2std
            df["liquidity_drain"] = (df["volume"]
                                     < (volume_ma_20 -
                                        2 * volume_std_20)).astype(int)

            # 流动性比率，使用log转换和标准化
            liquidity_ratio_raw = df["volume"] / volume_ma_20.replace(
                0, np.nan)
            liquidity_ratio_log = np.log1p(liquidity_ratio_raw.fillna(1))
            liquidity_ratio_mean = liquidity_ratio_log.rolling(
                50, min_periods=20).mean()
            liquidity_ratio_std = liquidity_ratio_log.rolling(
                50, min_periods=20).std()
            df["liquidity_ratio"] = (
                (liquidity_ratio_log - liquidity_ratio_mean) /
                liquidity_ratio_std.replace(0, np.nan)).replace(
                    [np.inf, -np.inf],
                    np.nan).fillna(0).clip(-self.feature_clip_bound,
                                           self.feature_clip_bound)

            # 6. Buy/Sell Pressure Ratio（买卖压力比）
            if "buy_qty" in df.columns and "sell_qty" in df.columns:
                # 滚动窗口的买卖压力
                buy_pressure_20 = df["buy_qty"].rolling(20).sum()
                sell_pressure_20 = df["sell_qty"].rolling(20).sum()

                # 买卖压力比，使用log转换和标准化
                buy_sell_pressure_ratio_raw = buy_pressure_20 / sell_pressure_20.replace(
                    0, np.nan)
                buy_sell_pressure_ratio_raw = buy_sell_pressure_ratio_raw.fillna(
                    1)
                buy_sell_pressure_ratio_log = np.log1p(
                    buy_sell_pressure_ratio_raw)
                buy_sell_pressure_ratio_mean = buy_sell_pressure_ratio_log.rolling(
                    50, min_periods=20).mean()
                buy_sell_pressure_ratio_std = buy_sell_pressure_ratio_log.rolling(
                    50, min_periods=20).std()
                df["buy_sell_pressure_ratio"] = (
                    (buy_sell_pressure_ratio_log -
                     buy_sell_pressure_ratio_mean) /
                    buy_sell_pressure_ratio_std.replace(0, np.nan)).replace(
                        [np.inf, -np.inf],
                        np.nan).fillna(0).clip(-self.feature_clip_bound,
                                               self.feature_clip_bound)

                # 压力差，Z-score标准化
                pressure_diff_raw = buy_pressure_20 - sell_pressure_20
                pressure_diff_mean = pressure_diff_raw.rolling(
                    50, min_periods=20).mean()
                pressure_diff_std = pressure_diff_raw.rolling(
                    50, min_periods=20).std()
                df["pressure_diff"] = (
                    (pressure_diff_raw - pressure_diff_mean) /
                    pressure_diff_std.replace(0, np.nan)).replace(
                        [np.inf, -np.inf],
                        np.nan).fillna(0).clip(-self.feature_clip_bound,
                                               self.feature_clip_bound)

                # 压力差归一化（相对值）
                df["pressure_diff_norm"] = df["pressure_diff"] / (
                    (buy_pressure_20 + sell_pressure_20).replace(0, np.nan) +
                    1e-10)
                df["pressure_diff_norm"] = df["pressure_diff_norm"].fillna(
                    0).clip(-1, 1)

            # 7. Volume-Price Divergence（量价背离）
            # pct_change() already uses past data; configurable shift ensures alignment
            price_change_20_raw = df["close"].pct_change(20)
            volume_change_20_raw = df["volume"].pct_change(20)
            price_change_20 = self._shift_feature(price_change_20_raw)
            volume_change_20 = self._shift_feature(volume_change_20_raw)

            # 标准化
            price_change_norm = (price_change_20 -
                                 price_change_20.rolling(50).mean()
                                 ) / price_change_20.rolling(50).std()
            volume_change_norm = (volume_change_20 -
                                  volume_change_20.rolling(50).mean()
                                  ) / volume_change_20.rolling(50).std()

            # 量价背离 = 价格上涨但成交量下降（或反之）
            df["volume_price_divergence"] = price_change_norm - volume_change_norm

        except Exception as e:
            print(f"      Warning: 订单流特征计算失败: {e}")

        return df

    def add_basic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Ensure baseline indicators and shared derived features are present."""
        return add_common_derived_features(data)

    def _ensure_orderflow_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure minimal order-flow signals exist (cvd, taker_buy_ratio) to allow
        Hurst/WPT/Hilbert to include them as signal sources. Uses robust proxies
        when true order-flow columns are missing.
        """
        out = df.copy()
        # Proxy taker_buy_ratio: rolling up-move frequency as a rough proxy
        if "taker_buy_ratio" not in out.columns:
            up = (out["close"].diff() > 0).astype(float)
            out["taker_buy_ratio"] = up.rolling(
                20, min_periods=5).mean().fillna(0.5)
        # Proxy CVD: cumulative signed volume using price change sign
        if "cvd" not in out.columns:
            signed_vol = np.sign(
                out["close"].diff().fillna(0.0)) * out["volume"].fillna(0.0)
            out["cvd"] = signed_vol.cumsum()
            # Provide simple deltas for downstream usage if needed
            out["cvd_change_1"] = signed_vol
            out["cvd_change_5"] = signed_vol.rolling(5, min_periods=1).sum()
            out["cvd_change_20"] = signed_vol.rolling(20, min_periods=1).sum()
            # Smoothed windows for short/medium/long
            out["cvd_short"] = signed_vol.rolling(20, min_periods=1).sum()
            out["cvd_medium"] = signed_vol.rolling(60, min_periods=1).sum()
            out["cvd_long"] = signed_vol.rolling(288, min_periods=1).sum()
            total_vol = out["volume"].rolling(20, min_periods=1).sum()
            out["cvd_normalized"] = out["cvd_short"] / total_vol.replace(
                0, np.nan)
            out["cvd_normalized"] = out["cvd_normalized"].replace(
                [np.inf, -np.inf], np.nan).fillna(0.0)
        return out

    def engineer_features(self,
                          multi_tf_data: Dict[str, pd.DataFrame],
                          fit: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Engineer features for all timeframes with enhanced WPT, Entropy, and Hurst features.

        对所有信号源（close, open, volume, cvd, taker_buy_ratio）都做：
        1. 小波包变换
        2. Hurst指数
        3. 基础技术指标

        Args:
            multi_tf_data: Dictionary mapping timeframe to OHLCV data
            fit: Whether to fit scalers (True for training, False for testing)

        Returns:
            Dictionary mapping timeframe to engineered features
        """
        engineered_data = {}

        for timeframe, data in multi_tf_data.items():
            print(f"  Engineering enhanced features for {timeframe}...")

            df = data.copy()

            # 1. Basic features
            print(f"    - Adding basic technical indicators...")
            df = self.add_basic_features(df)

            # Strict validation: require order flow columns to be present
            required_of_cols = ["cvd", "taker_buy_ratio"]
            missing = [c for c in required_of_cols if c not in df.columns]
            if missing:
                raise ValueError(
                    f"Missing required order-flow columns: {missing}. Data source error; please ensure preprocessing provides these columns."
                )

            # 2. Hurst exponent features for ALL signal sources
            print(f"    - Calculating Hurst exponent for all sources...")
            df = self.add_hurst_features(df)

            # 3. Wavelet packet features for ALL signal sources
            print(f"    - Calculating WPT features for all sources...")
            df = self.add_wavelet_packet_features(df)

            # 4. Hilbert transform features for ALL signal sources
            print(f"    - Calculating Hilbert transform for all sources...")
            df = self.add_hilbert_features(df)

            # 5. Spectral analysis features for ALL signal sources
            print(f"    - Calculating spectral features for all sources...")
            df = self.add_spectral_features(df)

            # 6. Advanced derived features (moved to baseline model)
            # Note: Advanced derived features are now in BaselineFeatureEngineer._add_advanced_derived_features
            # Uncomment below if you need them in this context, but they will be added by baseline model
            # print(f"    - Calculating advanced derived features...")
            # df = self.add_advanced_derived_features(df)

            # 7. Order flow features (advanced)
            print(f"    - Calculating order flow features...")
            df = self.add_order_flow_features(df)

            # 4. Normalize features
            feature_columns = [
                col for col in df.columns
                if col not in ["open", "high", "low", "close", "volume"]
            ]

            df = self._normalize_features(df, timeframe, feature_columns, fit)

            # Remove NaN rows
            df = df.dropna()

            if self.enable_diagnostics:
                diag_cols = [
                    col for col in feature_columns if col in df.columns
                ]
                self._run_diagnostics(timeframe, df, diag_cols)

            engineered_data[timeframe] = df

            print(
                f"    ✓ {len(feature_columns)} features engineered, {len(df)} samples"
            )

        return engineered_data

    def save_scalers(self, path: str):
        """Save fitted scalers."""
        import pickle

        with open(path, "wb") as f:
            pickle.dump(self.scalers, f)
        print(f"Scalers saved to {path}")

    def load_scalers(self, path: str):
        """Load fitted scalers."""
        import pickle

        with open(path, "rb") as f:
            self.scalers = pickle.load(f)
        print(f"Scalers loaded from {path}")

    def _normalize_features(
        self,
        df: pd.DataFrame,
        timeframe: str,
        feature_columns: list,
        fit: bool,
    ) -> pd.DataFrame:
        if not feature_columns:
            return df

        features = (df[feature_columns].replace([np.inf, -np.inf],
                                                np.nan).fillna(0.0))

        if fit:
            scaler = self.scaler_class()
            scaled = scaler.fit_transform(features.values)
            self.scalers[timeframe] = scaler
        else:
            scaler = self.scalers.get(timeframe)
            if scaler is None:
                raise ValueError(
                    f"No fitted scaler found for timeframe '{timeframe}'. Call with fit=True first."
                )
            scaled = scaler.transform(features.values)

        df.loc[:, feature_columns] = scaled
        return df
