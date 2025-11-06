from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
from sklearn.linear_model import LinearRegression


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

    def __init__(self,
                 percentile_window: int = 288,
                 compression_threshold_pct: float = 0.2) -> None:
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

        return series.rolling(window=window, min_periods=1).apply(_rank,
                                                                  raw=True)

    @staticmethod
    def _trend_r2(prices: pd.Series, window: int = 20) -> pd.Series:
        """
        计算趋势R²特征（基于对数价格序列）
        
        CRITICAL: 趋势应该体现在价格路径上，而不是收益率上。
        收益率序列本质上是白噪声，对收益率计算R²没有意义。
        正确做法：在对数价格序列上计算R²，衡量价格对时间的线性拟合优度。
        
        Args:
            prices: 价格序列（close价格）
            window: 滚动窗口大小
            
        Returns:
            R²值序列（0-1范围），已shift(1)避免未来信息
        """
        # 使用对数价格（更稳定，避免价格水平影响R²）
        # 例如：BTC从10k→20k和从60k→120k，趋势强度应相同
        log_price = np.log(prices.replace(0, np.nan)).ffill()

        def _compute_r2(series):
            """计算线性回归的R²"""
            if len(series) < 3:
                return 0.0
            try:
                x = np.arange(len(series))  # 时间索引: 0,1,2,...,window-1
                y = series.values

                # 简化线性回归：y = a + b*x
                # 使用polyfit更高效
                slope, intercept = np.polyfit(x, y, 1)
                y_pred = slope * x + intercept

                # 计算R²
                ss_res = np.sum((y - y_pred)**2)
                ss_tot = np.sum((y - np.mean(y))**2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

                # 截断到[0,1]范围
                return max(0.0, min(1.0, r2))
            except Exception:
                return 0.0

        # 滚动计算R²，并shift(1)确保在t时刻只能用到t-1及之前的信息
        # 这样在t时刻预测t+1时，这个特征是可用的（基于t-1及之前的数据）
        r2_series = log_price.rolling(window=window,
                                      min_periods=3).apply(_compute_r2,
                                                           raw=False)

        # shift(1)确保不包含当前未完成的K线
        return r2_series.shift(1).fillna(0.0)

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
            return -(p_up * np.log2(p_up + eps) +
                     p_dn * np.log2(p_dn + eps)) / 1.0

        # Normalize to [0,1] where max entropy at p=0.5 is 1.0 after dividing by 1
        return sign.rolling(window=window, min_periods=1).apply(_entropy,
                                                                raw=True)

    @staticmethod
    def _ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    def _add_advanced_derived_features(self,
                                       data: pd.DataFrame) -> pd.DataFrame:
        """
        Add advanced derived features (moved from enhanced features).
        Only adds features that don't already exist to avoid duplicates.
        添加高级衍生特征（从enhanced特征移过来）
        只添加不存在的特征，避免重复
        """
        df = data.copy()

        try:
            # 需要的基础特征
            if "bb_upper" not in df.columns or "atr" not in df.columns:
                # 如果缺少BB或ATR，跳过部分依赖这些的衍生特征
                # 但可以添加不依赖BB/ATR的特征
                pass
            else:
                # 1. BB Width相关（如果不存在）
                if "bb_width" not in df.columns:
                    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]).abs()
                if "bb_width_normalized" not in df.columns:
                    df["bb_width_normalized"] = df["bb_width"] / df[
                        "atr"].replace(0, np.nan)
                    df["bb_width_normalized"] = (
                        df["bb_width_normalized"].replace([np.inf, -np.inf],
                                                          np.nan).fillna(0))

                # 3. Compression duration (BB-based, only if not exists)
                # Note: baseline already has compression_duration (ATR-based), so check for bb_compression_duration
                if "bb_compression_duration" not in df.columns and "bb_width" in df.columns:
                    perc = df["bb_width"].rolling(20,
                                                  min_periods=5).quantile(0.2)
                    low_vol = (df["bb_width"] <= perc).astype(int)
                    df["bb_compression_duration"] = (low_vol.groupby(
                        (low_vol
                         != low_vol.shift()).cumsum()).cumsum()) * low_vol

                # 4. Compression energy (if volume_ratio exists)，使用log转换和标准化
                if "compression_energy" not in df.columns and "volume_ratio" in df.columns and "bb_width" in df.columns:
                    compression_energy_raw = (1.0 / df["bb_width"].replace(
                        0, np.nan)) * df["volume_ratio"]
                    compression_energy_raw = compression_energy_raw.replace(
                        [np.inf, -np.inf], np.nan).fillna(0)
                    # 使用log转换避免极端值，然后标准化（处理正值）
                    compression_energy_log = np.log1p(
                        np.abs(compression_energy_raw)) * np.sign(
                            compression_energy_raw)
                    compression_energy_mean = compression_energy_log.rolling(
                        50, min_periods=10).mean()
                    compression_energy_std = compression_energy_log.rolling(
                        50, min_periods=10).std()
                    df["compression_energy"] = (
                        (compression_energy_log - compression_energy_mean) /
                        compression_energy_std.replace(0, np.nan)).replace(
                            [np.inf, -np.inf], np.nan).fillna(0).clip(-5, 5)

                # 7. Volatility squeeze flag
                if "volatility_squeeze_flag" not in df.columns and "bb_width" in df.columns:
                    df["volatility_squeeze_flag"] = (df["bb_width"]
                                                     < (2.0 *
                                                        df["atr"])).astype(int)

                # 16. Structure tension，使用log转换和标准化
                # FIXED: Use shift(1) to avoid using current close (data leakage)
                if "structure_tension" not in df.columns and "bb_width" in df.columns:
                    dist_high = (df["high"].shift(1).rolling(50).max() -
                                 df["close"].shift(1)).abs(
                                 ) / df["close"].shift(1).replace(0, np.nan)
                    dist_low = (df["close"].shift(1) -
                                df["low"].shift(1).rolling(50).min()).abs(
                                ) / df["close"].shift(1).replace(0, np.nan)
                    structure_tension_raw = (
                        dist_high.fillna(0) +
                        dist_low.fillna(0)) / df["bb_width"].replace(
                            0, np.nan)
                    structure_tension_raw = structure_tension_raw.replace(
                        [np.inf, -np.inf], np.nan).fillna(0)
                    # 使用log转换避免极端值，然后标准化
                    structure_tension_log = np.log1p(structure_tension_raw)
                    structure_tension_mean = structure_tension_log.rolling(
                        50, min_periods=10).mean()
                    structure_tension_std = structure_tension_log.rolling(
                        50, min_periods=10).std()
                    df["structure_tension"] = (
                        (structure_tension_log - structure_tension_mean) /
                        structure_tension_std.replace(0, np.nan)).replace(
                            [np.inf, -np.inf], np.nan).fillna(0).clip(-5, 5)

            # 2. Range ratio (不依赖BB/ATR)，使用log转换和标准化
            if "range_ratio_5bar" not in df.columns:
                if "hl" not in df.columns:
                    df["hl"] = df["high"] - df["low"]
                range_ratio_raw = df["hl"].rolling(
                    5).mean() / df["hl"].rolling(20).mean().replace(0, np.nan)
                range_ratio_raw = range_ratio_raw.fillna(1)
                # 使用log转换避免极端值，然后标准化
                range_ratio_log = np.log1p(range_ratio_raw)
                range_ratio_mean = range_ratio_log.rolling(
                    50, min_periods=5).mean()
                range_ratio_std = range_ratio_log.rolling(50,
                                                          min_periods=5).std()
                df["range_ratio_5bar"] = (
                    (range_ratio_log - range_ratio_mean) /
                    range_ratio_std.replace(0, np.nan)).replace(
                        [np.inf, -np.inf], np.nan).fillna(0).clip(-5, 5)

            # 5. ATR percentile (注意：baseline已有atr_percentile，但这里用不同的窗口，所以跳过或改名)
            # Skip since baseline already has atr_percentile

            # 6. Volatility reversal score
            if "volatility_reversal_score" not in df.columns and "atr" in df.columns:
                atr_mean = df["atr"].rolling(50).mean()
                atr_std = df["atr"].rolling(50).std()
                df["volatility_reversal_score"] = (
                    df["atr"] - atr_mean) / atr_std.replace(0, np.nan)
                df["volatility_reversal_score"] = df[
                    "volatility_reversal_score"].fillna(0)

            # 8. Price range symmetry，使用log转换和标准化
            # FIXED: Use shift(1) to avoid using current close (data leakage)
            if "price_range_symmetry" not in df.columns:
                price_range_symmetry_raw = (
                    df["high"].shift(1) - df["close"].shift(1)) / (
                        (df["close"].shift(1) - df["low"].shift(1)).replace(
                            0, np.nan))
                price_range_symmetry_raw = price_range_symmetry_raw.replace(
                    [np.inf, -np.inf], np.nan).fillna(1)
                # 使用log转换避免极端值，然后标准化
                price_range_symmetry_log = np.log1p(
                    np.abs(price_range_symmetry_raw)) * np.sign(
                        price_range_symmetry_raw)
                price_range_symmetry_mean = price_range_symmetry_log.rolling(
                    50, min_periods=5).mean()
                price_range_symmetry_std = price_range_symmetry_log.rolling(
                    50, min_periods=5).std()
                df["price_range_symmetry"] = (
                    (price_range_symmetry_log - price_range_symmetry_mean) /
                    price_range_symmetry_std.replace(0, np.nan)).replace(
                        [np.inf, -np.inf], np.nan).fillna(0).clip(-5, 5)

            # 9. Volume anomaly，使用log转换和标准化
            if "volume_anomaly" not in df.columns:
                volume_anomaly_raw = df["volume"] / df["volume"].ewm(
                    span=20, min_periods=10).mean().replace(0, np.nan)
                volume_anomaly_raw = volume_anomaly_raw.fillna(1)
                # 使用log转换避免极端值，然后标准化
                volume_anomaly_log = np.log1p(volume_anomaly_raw)
                volume_anomaly_mean = volume_anomaly_log.rolling(
                    50, min_periods=10).mean()
                volume_anomaly_std = volume_anomaly_log.rolling(
                    50, min_periods=10).std()
                df["volume_anomaly"] = (
                    (volume_anomaly_log - volume_anomaly_mean) /
                    volume_anomaly_std.replace(0, np.nan)).replace(
                        [np.inf, -np.inf], np.nan).fillna(0).clip(-5, 5)

            # 10. Up/Down volume ratio，使用log转换和标准化
            if "upvol_downvol_ratio" not in df.columns:
                up = (df["close"] > df["close"].shift()).astype(int)
                df["up_vol"] = (df["volume"] * up).rolling(20).sum()
                df["down_vol"] = (df["volume"] * (1 - up)).rolling(20).sum()
                upvol_downvol_ratio_raw = df["up_vol"] / df[
                    "down_vol"].replace(0, np.nan)
                upvol_downvol_ratio_raw = upvol_downvol_ratio_raw.replace(
                    [np.inf, -np.inf], np.nan).fillna(1)
                # 使用log转换避免极端值，然后标准化
                upvol_downvol_ratio_log = np.log1p(upvol_downvol_ratio_raw)
                upvol_downvol_ratio_mean = upvol_downvol_ratio_log.rolling(
                    50, min_periods=20).mean()
                upvol_downvol_ratio_std = upvol_downvol_ratio_log.rolling(
                    50, min_periods=20).std()
                df["upvol_downvol_ratio"] = (
                    (upvol_downvol_ratio_log - upvol_downvol_ratio_mean) /
                    upvol_downvol_ratio_std.replace(0, np.nan)).replace(
                        [np.inf, -np.inf], np.nan).fillna(0).clip(-5, 5)

            # 11. ROC and acceleration (with normalization)
            if "roc_5" not in df.columns:
                roc_raw = df["close"].pct_change(5)
                # Normalize ROC by rolling std (Z-score) to make it consistent with other features
                # Use a larger window and ensure minimum std to avoid extreme values
                roc_mean = roc_raw.rolling(window=50, min_periods=5).mean()
                roc_std = roc_raw.rolling(window=50, min_periods=5).std()
                # Clip std to avoid division by very small values
                roc_std = roc_std.clip(lower=roc_raw.abs().quantile(0.01))
                df["roc_5"] = ((roc_raw - roc_mean) /
                               roc_std.replace(0, np.nan)).replace(
                                   [np.inf, -np.inf],
                                   np.nan).fillna(0).clip(-5, 5)
            if "acceleration_3" not in df.columns:
                roc_3 = df["close"].pct_change(3)
                roc_3_mean = roc_3.rolling(window=50, min_periods=5).mean()
                roc_3_std = roc_3.rolling(window=50, min_periods=5).std()
                roc_3_std = roc_3_std.clip(lower=roc_3.abs().quantile(0.01))
                roc_3_norm = ((roc_3 - roc_3_mean) /
                              roc_3_std.replace(0, np.nan)).replace(
                                  [np.inf, -np.inf],
                                  np.nan).fillna(0).clip(-5, 5)
                df["acceleration_3"] = roc_3_norm - roc_3_norm.shift(1)

            # 12. Trend R² (R-squared) - 衡量趋势强度
            # CORRECTED: 趋势应该体现在价格路径上，而不是收益率上
            # 收益率序列本质上是白噪声，对收益率计算R²没有意义
            # 正确做法：在对数价格序列上计算R²，并shift(1)避免未来信息
            if "trend_r2_20" not in df.columns:
                df["trend_r2_20"] = self._trend_r2(df["close"], window=20)
            if "trend_r2_50" not in df.columns:
                df["trend_r2_50"] = self._trend_r2(df["close"], window=50)

            # 12.1 Price vs EMA/SMA distance (normalized)
            # FIXED: Use shift(1) to avoid using current close (data leakage)
            if "price_vs_ema_distance" not in df.columns and "sma_20" in df.columns and "atr" in df.columns:
                df["price_vs_ema_distance"] = (
                    (df["close"].shift(1) - df["sma_20"].shift(1)) /
                    df["atr"].shift(1).replace(0, np.nan))
                df["price_vs_ema_distance"] = (
                    df["price_vs_ema_distance"].replace([np.inf, -np.inf],
                                                        np.nan).fillna(0))

            # 12.2 SMA/EMA/WMA distance features (price/sma - 1, normalized)
            # 计算移动平均线（如果不存在）
            if "sma_20" not in df.columns:
                df["sma_20"] = df["close"].rolling(window=20,
                                                   min_periods=1).mean()
            if "sma_50" not in df.columns:
                df["sma_50"] = df["close"].rolling(window=50,
                                                   min_periods=1).mean()
            if "ema_20" not in df.columns:
                df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
            if "ema_50" not in df.columns:
                df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()

            # SMA距离特征（归一化）
            # FIXED: Use shift(1) to avoid using current close (data leakage)
            if "sma_20_distance" not in df.columns:
                df["sma_20_distance"] = (
                    (df["close"].shift(1) /
                     df["sma_20"].shift(1).replace(0, np.nan) - 1)).replace(
                         [np.inf, -np.inf], np.nan).fillna(0)
            if "sma_50_distance" not in df.columns:
                df["sma_50_distance"] = (
                    (df["close"].shift(1) /
                     df["sma_50"].shift(1).replace(0, np.nan) - 1)).replace(
                         [np.inf, -np.inf], np.nan).fillna(0)

            # EMA距离特征（归一化）
            # FIXED: Use shift(1) to avoid using current close (data leakage)
            if "ema_20_distance" not in df.columns:
                df["ema_20_distance"] = (
                    (df["close"].shift(1) /
                     df["ema_20"].shift(1).replace(0, np.nan) - 1)).replace(
                         [np.inf, -np.inf], np.nan).fillna(0)
            if "ema_50_distance" not in df.columns:
                df["ema_50_distance"] = (
                    (df["close"].shift(1) /
                     df["ema_50"].shift(1).replace(0, np.nan) - 1)).replace(
                         [np.inf, -np.inf], np.nan).fillna(0)

            # WMA距离特征（如果存在）
            try:
                if "wma_20" not in df.columns:
                    # 简单加权移动平均
                    weights = np.arange(1, 21)
                    df["wma_20"] = df["close"].rolling(
                        window=20, min_periods=1).apply(
                            lambda x: np.dot(x, weights) / weights.sum(),
                            raw=True)
                if "wma_20" in df.columns and "wma_20_distance" not in df.columns:
                    # FIXED: Use shift(1) to avoid using current close (data leakage)
                    df["wma_20_distance"] = ((
                        df["close"].shift(1) /
                        df["wma_20"].shift(1).replace(0, np.nan) - 1)).replace(
                            [np.inf, -np.inf], np.nan).fillna(0)
            except Exception:
                pass

            # VWAP距离特征（如果存在）
            # FIXED: Use shift(1) to avoid using current close (data leakage)
            if "vwap" in df.columns and "vwap_distance" not in df.columns:
                df["vwap_distance"] = (
                    (df["close"].shift(1) /
                     df["vwap"].shift(1).replace(0, np.nan) - 1)).replace(
                         [np.inf, -np.inf], np.nan).fillna(0)

            # 13. Momentum persistence
            # FIXED: Use shift(1) to avoid using current close (data leakage)
            if "momentum_persistence" not in df.columns:
                sig = np.sign(df["close"].diff().shift(1))
                df["momentum_persistence"] = sig.rolling(10).apply(
                    lambda x: (np.sum(x > 0) / max(len(x), 1)), raw=True)

            # 14. Slope consistency
            if "slope_consistency_score" not in df.columns:
                ema10 = df["close"].ewm(span=10).mean()
                ema20 = df["close"].ewm(span=20).mean()
                ema50 = df["close"].ewm(span=50).mean()
                slope10 = np.sign(ema10.diff())
                slope20 = np.sign(ema20.diff())
                slope50 = np.sign(ema50.diff())
                df["slope_consistency_score"] = (
                    (slope10 == slope20).astype(int) +
                    (slope20 == slope50).astype(int) +
                    (slope10 == slope50).astype(int))

            # 15. Temporal features (时间特征)
            if "hour_of_day_sin" not in df.columns:
                try:
                    idx = df.index
                    if hasattr(idx, "hour") and hasattr(idx, "dayofweek"):
                        df["hour_of_day_sin"] = np.sin(2 * np.pi * idx.hour /
                                                       24)
                        df["hour_of_day_cos"] = np.cos(2 * np.pi * idx.hour /
                                                       24)
                        df["day_of_week_sin"] = np.sin(2 * np.pi *
                                                       idx.dayofweek / 7)
                        df["day_of_week_cos"] = np.cos(2 * np.pi *
                                                       idx.dayofweek / 7)
                    else:
                        df["hour_of_day_sin"] = 0
                        df["hour_of_day_cos"] = 1
                        df["day_of_week_sin"] = 0
                        df["day_of_week_cos"] = 1
                except Exception:
                    df["hour_of_day_sin"] = 0
                    df["hour_of_day_cos"] = 1
                    df["day_of_week_sin"] = 0
                    df["day_of_week_cos"] = 1

            # 17. Trend volatility alignment (需要atr_percentile和roc_5)
            if "trend_volatility_alignment" not in df.columns and "atr_percentile" in df.columns and "roc_5" in df.columns:
                df["trend_volatility_alignment"] = np.sign(
                    df["roc_5"]).fillna(0) * df["atr_percentile"].fillna(0)

            # 18. Compression to breakout probability
            if "compression_to_breakout_prob" not in df.columns and "compression_duration" in df.columns and "roc_5" in df.columns:
                df["compression_to_breakout_prob"] = df[
                    "compression_duration"].fillna(0) * df["roc_5"].fillna(0)

        except Exception as e:
            print(f"      Warning: 高级衍生特征计算失败: {e}")

        return df

    def engineer_features(self,
                          df: pd.DataFrame,
                          *,
                          fit: bool = True) -> pd.DataFrame:
        if not {"open", "high", "low", "close", "volume"}.issubset(df.columns):
            raise ValueError(
                "DataFrame must contain open, high, low, close, volume columns"
            )

        data = df.copy()

        # Core ATR
        data["atr"] = self._compute_atr(data, window=14)

        # SR proximity using rolling swing proxies (highest high / lowest low)
        swing_win_short = 20
        swing_win_long = 60
        data["roll_high_s"] = data["high"].rolling(swing_win_short,
                                                   min_periods=1).max()
        data["roll_low_s"] = data["low"].rolling(swing_win_short,
                                                 min_periods=1).min()
        data["roll_high_l"] = data["high"].rolling(swing_win_long,
                                                   min_periods=1).max()
        data["roll_low_l"] = data["low"].rolling(swing_win_long,
                                                 min_periods=1).min()

        eps = 1e-9
        data["sr_dist_high_s"] = (data["close"] -
                                  data["roll_high_s"]) / (data["atr"] + eps)
        data["sr_dist_low_s"] = (data["close"] -
                                 data["roll_low_s"]) / (data["atr"] + eps)
        data["sr_dist_high_l"] = (data["close"] -
                                  data["roll_high_l"]) / (data["atr"] + eps)
        data["sr_dist_low_l"] = (data["close"] -
                                 data["roll_low_l"]) / (data["atr"] + eps)

        # Simple channel via EMAs as OLS proxy
        ema_fast = self._ema(data["close"], span=20)
        ema_slow = self._ema(data["close"], span=60)
        mid = (ema_fast + ema_slow) / 2.0
        band_half = (data["high"].rolling(20, min_periods=1).max() -
                     data["low"].rolling(20, min_periods=1).min()) / 4.0
        upper = mid + band_half
        lower = mid - band_half
        data["channel_upper"] = upper
        data["channel_lower"] = lower
        data["channel_mid"] = mid
        data["channel_bandwidth"] = (upper - lower) / (data["atr"] + eps)
        data["channel_upper_distance"] = (upper -
                                          data["close"]) / (data["atr"] + eps)
        data["channel_lower_distance"] = (data["close"] -
                                          lower) / (data["atr"] + eps)

        # Compression features
        # ATR percentile (rolling)
        atr_pct = self._rolling_percentile(data["atr"],
                                           window=self.percentile_window)
        data["atr_percentile"] = atr_pct

        # ATR compression ratio: mean(ATR_hist)/ATR
        atr_mean_hist = data["atr"].rolling(self.percentile_window,
                                            min_periods=1).mean()
        data["atr_compression_ratio"] = (atr_mean_hist /
                                         (data["atr"] + eps)).replace(
                                             [np.inf, -np.inf], np.nan)

        # Volume percentile
        vol_pct = self._rolling_percentile(data["volume"].astype(float),
                                           window=self.percentile_window)
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
        data["pre_break_silence"] = (data["atr_percentile"].rolling(
            short_window, min_periods=1).mean() <= threshold).astype(float)

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
        data["compression_confidence"] = 0.5 * (1 - atr_norm) + 0.3 * (
            1 - vol_norm) + 0.2 * dens_norm

        # Advanced derived features (if BB/ATR available)
        data = self._add_advanced_derived_features(data)

        # Time factors (UTC): Cyclic hour encoding, Is_Weekend, Minutes_Since_Last_Trade
        # For tree models (LightGBM/XGBoost), we use cyclic encoding for hour to capture
        # the fact that 23:00 and 00:00 are close in time.
        try:
            idx = data.index
            if hasattr(idx, "hour") and hasattr(idx, "dayofweek"):
                # Hour_of_Day: Cyclic encoding (sin/cos) for tree models
                # This allows the model to learn that 23:00 and 00:00 are close
                try:
                    # If timezone-aware, convert to UTC first
                    if getattr(idx, "tz", None) is not None:
                        utc_idx = idx.tz_convert("UTC")
                        hour = utc_idx.hour.astype(int)
                    else:
                        # Assume timestamps are already UTC
                        hour = idx.hour.astype(int)
                    
                    # Cyclic encoding: sin/cos transformation
                    # This preserves the periodic nature of hours (23 and 0 are close)
                    data["hour_sin"] = np.sin(2 * np.pi * hour / 24)
                    data["hour_cos"] = np.cos(2 * np.pi * hour / 24)
                    
                    # Also keep raw hour for reference (optional, can be excluded later)
                    data["Hour_of_Day"] = hour
                    
                except Exception:
                    data["hour_sin"] = 0.0
                    data["hour_cos"] = 1.0
                    data["Hour_of_Day"] = 0

                # Is_Weekend (Sat/Sun): Binary 0/1, no normalization needed for tree models
                try:
                    data["Is_Weekend"] = (idx.dayofweek >= 5).astype(int)
                except Exception:
                    data["Is_Weekend"] = 0

                # Minutes_Since_Last_Trade: Winsorize + optional binning/boolean features
                # If tick data exists you could compute true minutes since last trade.
                # For bar data, approximate using consecutive zero-volume run length
                # scaled by inferred bar minutes when possible.
                try:
                    zero_vol = (data["volume"].astype(float) == 0).astype(int)
                    run = np.zeros(len(zero_vol), dtype=int)
                    cnt = 0
                    for i, v in enumerate(zero_vol.values):
                        if v == 1:
                            cnt += 1
                        else:
                            cnt = 0
                        run[i] = cnt

                    # Infer bar minutes from timestamp diffs if available
                    inferred_bar_minutes = None
                    try:
                        # Use median to be robust
                        diffs = pd.Series(idx).diff().dt.total_seconds()
                        med_sec = float(diffs.median()) if diffs.notna().any() else None
                        if med_sec is not None and med_sec > 0:
                            inferred_bar_minutes = med_sec / 60.0
                    except Exception:
                        inferred_bar_minutes = None

                    if inferred_bar_minutes is None:
                        # Fallback: just use bars since last trade as minutes proxy
                        minutes_raw = run.astype(float)
                    else:
                        minutes_raw = run.astype(float) * float(inferred_bar_minutes)
                    
                    # Step 1: Winsorize (clip upper bound at 60 minutes)
                    # Beyond 60 minutes, the signal is similar (long-term no-trade)
                    minutes_clipped = minutes_raw.clip(upper=60.0)
                    data["Minutes_Since_Last_Trade"] = minutes_clipped
                    
                    # Step 2: Add boolean features for common thresholds
                    # These are more interpretable and robust for tree models
                    data["no_trade_5min"] = (minutes_clipped > 5).astype(int)
                    data["no_trade_15min"] = (minutes_clipped > 15).astype(int)
                    data["no_trade_30min"] = (minutes_clipped > 30).astype(int)
                    
                    # Step 3: Optional binning (discretization)
                    # This can help tree models by creating clear splits
                    try:
                        # Since we already clipped at 60, values are in [0, 60]
                        # Bins: [0, 1), [1, 5), [5, 15), [15, 60]
                        data["trade_gap_bin"] = pd.cut(
                            minutes_clipped,
                            bins=[0, 1, 5, 15, 60.1],  # 60.1 to include 60.0
                            labels=[0, 1, 2, 3],
                            include_lowest=True,
                            right=False  # [0,1), [1,5), [5,15), [15,60.1)
                        ).astype(float).fillna(0.0)
                    except Exception:
                        # If binning fails, just use the clipped value
                        pass
                        
                except Exception:
                    data["Minutes_Since_Last_Trade"] = 0.0
                    data["no_trade_5min"] = 0
                    data["no_trade_15min"] = 0
                    data["no_trade_30min"] = 0
            else:
                # Fallback values if index doesn't have hour/dayofweek
                data["hour_sin"] = 0.0
                data["hour_cos"] = 1.0
                data["Hour_of_Day"] = 0
                data["Is_Weekend"] = 0
                data["Minutes_Since_Last_Trade"] = 0.0
                data["no_trade_5min"] = 0
                data["no_trade_15min"] = 0
                data["no_trade_30min"] = 0
        except Exception:
            # Fallback values on any error
            data["hour_sin"] = 0.0
            data["hour_cos"] = 1.0
            data["Hour_of_Day"] = 0
            data["Is_Weekend"] = 0
            data["Minutes_Since_Last_Trade"] = 0.0
            data["no_trade_5min"] = 0
            data["no_trade_15min"] = 0
            data["no_trade_30min"] = 0

        # Finalize: feature column ordering
        feature_cols: List[str] = [
            # SR distances
            "sr_dist_high_s",
            "sr_dist_low_s",
            "sr_dist_high_l",
            "sr_dist_low_l",
            # Channel features
            "channel_bandwidth",
            "channel_upper_distance",
            "channel_lower_distance",
            # Compression set
            "atr_percentile",
            "atr_compression_ratio",
            "volume_percentile",
            "price_entropy",
            "internal_price_density",
            "compression_duration",
            "pre_break_silence",
            "compression_confidence",
        ]

        # Clean up and keep OHLCV + features + advanced derived features
        # Collect all feature columns (including advanced derived ones)
        all_feature_cols = feature_cols + [
            col for col in data.columns if col not in
            ["open", "high", "low", "close", "volume", "timestamp", "symbol"]
            and col not in feature_cols and not col.startswith("signal_")
            and not col.startswith("binary_signal_")
            and not col.startswith("future_return_")
        ]
        keep_cols = ["open", "high", "low", "close", "volume"
                     ] + all_feature_cols
        # Only keep columns that exist
        keep_cols = [c for c in keep_cols if c in data.columns]
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

        self._fitted_atr_quantiles = scalers_data.get("fitted_atr_quantiles",
                                                      None)
        self._fitted_vol_quantiles = scalers_data.get("fitted_vol_quantiles",
                                                      None)
        self.percentile_window = scalers_data.get("percentile_window", 288)
        self.compression_threshold_pct = scalers_data.get(
            "compression_threshold_pct", 0.2)
        print(f"✅ Baseline scalers loaded from: {path}")


def engineer_baseline_features(
        df: pd.DataFrame,
        engineer: Optional[BaselineFeatureEngineer] = None,
        *,
        fit: bool = True) -> Tuple[pd.DataFrame, BaselineFeatureEngineer]:
    if engineer is None:
        engineer = BaselineFeatureEngineer()
    out = engineer.engineer_features(df, fit=fit)
    return out, engineer


def create_binary_labels_baseline(df: pd.DataFrame,
                                  *,
                                  forward_bars: int = 3,
                                  threshold: float = 0.005) -> pd.DataFrame:
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
    exclude = {
        "open", "high", "low", "close", "volume", "signal", "binary_signal",
        "future_return"
    }
    # Also exclude multi-horizon label columns
    exclude.update([
        col for col in df.columns
        if (col.startswith("signal_") or col.startswith("binary_signal_")
            or col.startswith("future_return_"))
    ])
    # Exclude unnormalized/raw columns; prefer normalized variants
    exclude.update({
        "bb_upper",
        "bb_lower",
        "bb_middle",
        "bb_width",  # use bb_width_normalized
        "hl",  # helper
        "up_vol",
        "down_vol",  # helpers
        "macd",
        "macd_signal",
        "macd_hist",
        "macd_ext_hist",
        "macd_fix_hist",
        "atr",  # 原始ATR（未归一化），保留natr和atr_normalized
        "channel_mid",  # 原始价格量纲，保留归一化的距离特征
        "channel_upper",  # 原始价格量纲
        "channel_lower",  # 原始价格量纲
        # 原始滚动高低点（使用原始价格），保留标准化的距离特征（sr_dist_*）
        "roll_high_s",  # 使用原始 high，保留 sr_dist_high_s（标准化）
        "roll_low_s",  # 使用原始 low，保留 sr_dist_low_s（标准化）
        "roll_high_l",  # 使用原始 high，保留 sr_dist_high_l（标准化）
        "roll_low_l",  # 使用原始 low，保留 sr_dist_low_l（标准化）
    })
    # Exclude raw-scale prefixes
    raw_prefixes = ("sma_", "ema_", "wma_", "tema_", "kama_", "volume_sma_",
                    "atr_")
    # Exclude unnormalized WPT features (wpt_*_energy, wpt_*_mean, wpt_*_std)
    # but keep normalized WPT features (wpt_*_energy_ratio, wpt_shannon_entropy, etc.)
    wpt_raw_patterns = ("_energy", "_mean", "_std")
    return [
        c for c in df.columns if (c not in exclude and not any(
            c.startswith(p)
            for p in raw_prefixes) and not (c.startswith("wpt_") and any(
                c.endswith(p)
                for p in wpt_raw_patterns) and not c.endswith("_energy_ratio"))
                                  )
    ]


__all__ = [
    "BaselineFeatureEngineer",
    "engineer_baseline_features",
    "get_baseline_feature_columns",
    "create_binary_labels_baseline",
]
