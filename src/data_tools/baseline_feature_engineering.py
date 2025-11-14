from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict
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
                 compression_threshold_pct: float = 0.2,
                 feature_shift: int = 0,
                 feature_clip_bound: float = 10.0,
                 enable_diagnostics: bool = False) -> None:
        self.percentile_window = percentile_window
        self.compression_threshold_pct = compression_threshold_pct
        self.feature_shift = feature_shift
        self.feature_clip_bound = float(feature_clip_bound)
        self.enable_diagnostics = enable_diagnostics
        self.diagnostic_report: Dict[str, Dict[str, float]] = {}
        self._fitted_atr_quantiles: Optional[np.ndarray] = None
        self._fitted_vol_quantiles: Optional[np.ndarray] = None

        if self.feature_clip_bound <= 0:
            raise ValueError("feature_clip_bound must be positive")

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
    def _trend_r2(prices: pd.Series,
                  window: int = 20,
                  *,
                  lag: int = 0) -> pd.Series:
        """
        计算趋势R²特征（基于对数价格序列）
        
        CRITICAL: 趋势应该体现在价格路径上，而不是收益率上。
        收益率序列本质上是白噪声，对收益率计算R²没有意义。
        正确做法：在对数价格序列上计算R²，衡量价格对时间的线性拟合优度。
        
        Args:
            prices: 价格序列（close价格）
            window: 滚动窗口大小
            
        Returns:
            R²值序列（0-1范围），可选shift避免未来信息
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

        # 滚动计算R²，可选shift确保在t时刻只能用到t-1及之前的信息
        # 这样在t时刻预测t+1时，这个特征是可用的（基于历史数据）
        r2_series = log_price.rolling(window=window,
                                      min_periods=3).apply(_compute_r2,
                                                           raw=False)

        # shift(lag)确保不包含当前未完成的K线
        if lag == 0:
            return r2_series.fillna(0.0)
        return r2_series.shift(lag).fillna(0.0)

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
    def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Wilder-style RSI."""
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(alpha=1 / period, adjust=False,
                            min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False,
                            min_periods=period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50.0)
        return rsi.clip(0, 100)

    @staticmethod
    def _rolling_skew(series: pd.Series, window: int) -> pd.Series:
        """Rolling skewness using pandas' Series.skew (biased)."""
        return series.rolling(window=window, min_periods=window).skew()

    @staticmethod
    def _ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    def _shift_feature(self,
                       series: pd.Series,
                       *,
                       offset: int = 0) -> pd.Series:
        """Apply configurable lag to feature series.

        Args:
            series: Series to shift.
            offset: Additional offset relative to base `feature_shift`.

        Returns:
            Shifted series with total lag = feature_shift + offset.
        """
        total_shift = self.feature_shift + offset
        if total_shift == 0:
            return series
        return series.shift(total_shift)

    def _run_diagnostics(self, df: pd.DataFrame,
                         feature_cols: List[str]) -> None:
        """Collect simple diagnostics to detect heavy clipping or zero saturation."""
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
            total = len(series)
            if total == 0:
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

        self.diagnostic_report = report

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
            print("⚠️  BaselineFeatureEngineer diagnostics:")
            for msg in messages:
                print(msg)

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
                            [np.inf, -np.inf],
                            np.nan).fillna(0).clip(-self.feature_clip_bound,
                                                   self.feature_clip_bound)

                # 7. Volatility squeeze flag
                if "volatility_squeeze_flag" not in df.columns and "bb_width" in df.columns:
                    df["volatility_squeeze_flag"] = (df["bb_width"]
                                                     < (2.0 *
                                                        df["atr"])).astype(int)

                # 16. Structure tension，使用log转换和标准化
                # Use configurable shift to avoid using current close (data leakage)
                if "structure_tension" not in df.columns and "bb_width" in df.columns:
                    dist_high = (self._shift_feature(
                        df["high"]).rolling(50).max() - self._shift_feature(
                            df["close"])).abs() / self._shift_feature(
                                df["close"]).replace(0, np.nan)
                    dist_low = (self._shift_feature(df["close"]) -
                                self._shift_feature(df["low"]).rolling(
                                    50).min()).abs() / self._shift_feature(
                                        df["close"]).replace(0, np.nan)
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
                            [np.inf, -np.inf],
                            np.nan).fillna(0).clip(-self.feature_clip_bound,
                                                   self.feature_clip_bound)

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
                        [np.inf, -np.inf],
                        np.nan).fillna(0).clip(-self.feature_clip_bound,
                                               self.feature_clip_bound)

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
            # Use configurable shift to avoid using current close (data leakage)
            if "price_range_symmetry" not in df.columns:
                price_range_symmetry_raw = (self._shift_feature(
                    df["high"]) - self._shift_feature(df["close"])) / (
                        (self._shift_feature(df["close"]) -
                         self._shift_feature(df["low"])).replace(0, np.nan))
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
                        [np.inf, -np.inf],
                        np.nan).fillna(0).clip(-self.feature_clip_bound,
                                               self.feature_clip_bound)

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
                        [np.inf, -np.inf],
                        np.nan).fillna(0).clip(-self.feature_clip_bound,
                                               self.feature_clip_bound)

            # 10. Up/Down volume ratio，使用log转换和标准化
            if "upvol_downvol_ratio" not in df.columns:
                close_ref = self._shift_feature(df["close"])
                up = (close_ref > self._shift_feature(df["close"],
                                                      offset=1)).astype(int)
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
                        [np.inf, -np.inf],
                        np.nan).fillna(0).clip(-self.feature_clip_bound,
                                               self.feature_clip_bound)
                # 归一化的多空成交量分布，便于跨标的比较
                vol_total_20 = df["volume"].rolling(
                    20, min_periods=1).sum().replace(0, np.nan)
                df["up_vol_share_20"] = (df["up_vol"] / vol_total_20).replace(
                    [np.inf, -np.inf], np.nan).fillna(0).clip(0.0, 1.0)
                df["down_vol_share_20"] = (df["down_vol"] /
                                           vol_total_20).replace(
                                               [np.inf, -np.inf],
                                               np.nan).fillna(0).clip(
                                                   0.0, 1.0)
                updown_total = (df["up_vol"] + df["down_vol"]).replace(
                    0, np.nan)
                df["up_down_vol_balance"] = ((df["up_vol"] - df["down_vol"]) /
                                             updown_total).replace(
                                                 [np.inf, -np.inf],
                                                 np.nan).fillna(0).clip(
                                                     -1.0, 1.0)

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
                                   [np.inf, -np.inf], np.nan).fillna(0).clip(
                                       -self.feature_clip_bound,
                                       self.feature_clip_bound)
            if "acceleration_3" not in df.columns:
                roc_3 = df["close"].pct_change(3)
                roc_3_mean = roc_3.rolling(window=50, min_periods=5).mean()
                roc_3_std = roc_3.rolling(window=50, min_periods=5).std()
                roc_3_std = roc_3_std.clip(lower=roc_3.abs().quantile(0.01))
                roc_3_norm = ((roc_3 - roc_3_mean) /
                              roc_3_std.replace(0, np.nan)).replace(
                                  [np.inf, -np.inf], np.nan).fillna(0).clip(
                                      -self.feature_clip_bound,
                                      self.feature_clip_bound)
                current = self._shift_feature(roc_3_norm)
                prev = self._shift_feature(roc_3_norm, offset=1)
                df["acceleration_3"] = current - prev

            # 12. Trend R² (R-squared) - 衡量趋势强度
            # CORRECTED: 趋势应该体现在价格路径上，而不是收益率上
            # 收益率序列本质上是白噪声，对收益率计算R²没有意义
            # 正确做法：在对数价格序列上计算R²，可选shift避免未来信息
            if "trend_r2_20" not in df.columns:
                df["trend_r2_20"] = self._trend_r2(df["close"],
                                                   window=20,
                                                   lag=self.feature_shift)
            if "trend_r2_50" not in df.columns:
                df["trend_r2_50"] = self._trend_r2(df["close"],
                                                   window=50,
                                                   lag=self.feature_shift)

            # 12.1 Price vs EMA/SMA distance (normalized)
            # Use configurable shift to avoid using current close (data leakage)
            if "price_vs_ema_distance" not in df.columns and "sma_20" in df.columns and "atr" in df.columns:
                df["price_vs_ema_distance"] = (
                    (self._shift_feature(df["close"]) -
                     self._shift_feature(df["sma_20"])) /
                    self._shift_feature(df["atr"]).replace(0, np.nan))
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
            # Use configurable shift to avoid using current close (data leakage)
            if "sma_20_distance" not in df.columns:
                df["sma_20_distance"] = (
                    (self._shift_feature(df["close"]) /
                     self._shift_feature(df["sma_20"]).replace(0, np.nan) -
                     1)).replace([np.inf, -np.inf], np.nan).fillna(0)
            if "sma_50_distance" not in df.columns:
                df["sma_50_distance"] = (
                    (self._shift_feature(df["close"]) /
                     self._shift_feature(df["sma_50"]).replace(0, np.nan) -
                     1)).replace([np.inf, -np.inf], np.nan).fillna(0)

            # EMA距离特征（归一化）
            # Use configurable shift to avoid using current close (data leakage)
            if "ema_20_distance" not in df.columns:
                df["ema_20_distance"] = (
                    (self._shift_feature(df["close"]) /
                     self._shift_feature(df["ema_20"]).replace(0, np.nan) -
                     1)).replace([np.inf, -np.inf], np.nan).fillna(0)
            if "ema_50_distance" not in df.columns:
                df["ema_50_distance"] = (
                    (self._shift_feature(df["close"]) /
                     self._shift_feature(df["ema_50"]).replace(0, np.nan) -
                     1)).replace([np.inf, -np.inf], np.nan).fillna(0)

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
                    # Use configurable shift to avoid using current close (data leakage)
                    df["wma_20_distance"] = (
                        (self._shift_feature(df["close"]) /
                         self._shift_feature(df["wma_20"]).replace(0, np.nan) -
                         1)).replace([np.inf, -np.inf], np.nan).fillna(0)
            except Exception:
                pass

            # 归一化均线（相对close），便于跨标的比较
            close_now = df["close"].replace(0, np.nan)

            def _pct_vs_close(series: pd.Series) -> pd.Series:
                return (series / close_now - 1.0).replace([np.inf, -np.inf],
                                                          np.nan).fillna(0.0)

            df["sma_20_pct_close"] = _pct_vs_close(df["sma_20"])
            df["sma_50_pct_close"] = _pct_vs_close(df["sma_50"])
            df["ema_20_pct_close"] = _pct_vs_close(df["ema_20"])
            df["ema_50_pct_close"] = _pct_vs_close(df["ema_50"])
            if "wma_20" in df.columns:
                df["wma_20_pct_close"] = _pct_vs_close(df["wma_20"])

            # VWAP距离特征（如果存在）
            # Use configurable shift to avoid using current close (data leakage)
            if "vwap" in df.columns and "vwap_distance" not in df.columns:
                df["vwap_distance"] = (
                    (self._shift_feature(df["close"]) /
                     self._shift_feature(df["vwap"]).replace(0, np.nan) -
                     1)).replace([np.inf, -np.inf], np.nan).fillna(0)

            # 13. Momentum persistence
            # Use configurable shift to avoid using current close (data leakage)
            if "momentum_persistence" not in df.columns:
                sig = np.sign(df["close"].diff())
                sig = self._shift_feature(sig)
                df["momentum_persistence"] = sig.rolling(10).apply(
                    lambda x: (np.sum(x > 0) / max(len(x), 1)),
                    raw=True).clip(-self.feature_clip_bound,
                                   self.feature_clip_bound)

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

    def engineer_features(
            self,
            df: pd.DataFrame,
            *,
            fit: bool = True,
            required_features: Optional[set] = None) -> pd.DataFrame:
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
        close_safe = data["close"].replace(0, np.nan)
        data["channel_upper_pct_close"] = (upper / close_safe - 1.0).replace(
            [np.inf, -np.inf], np.nan).fillna(0.0)
        data["channel_lower_pct_close"] = (lower / close_safe - 1.0).replace(
            [np.inf, -np.inf], np.nan).fillna(0.0)
        data["channel_mid_pct_close"] = (mid / close_safe - 1.0).replace(
            [np.inf, -np.inf], np.nan).fillna(0.0)

        # Price-volume divergence signals
        if "rsi_14" not in data.columns:
            data["rsi_14"] = self._compute_rsi(data["close"], period=14)

        recent_high = data["close"].rolling(20, min_periods=5).max().ffill()
        recent_rsi_high = data["rsi_14"].rolling(20,
                                                 min_periods=5).max().ffill()
        tol = 1e-8
        divergence_mask = (recent_high.notna()
                           & recent_rsi_high.notna()
                           & (data["close"] >= (recent_high - tol))
                           & (data["rsi_14"] < (recent_rsi_high - tol)))
        data["rsi_divergence"] = divergence_mask.astype(float) * -1.0

        price_vs_past = (data["close"] > data["close"].shift(5)).fillna(False)
        avg_volume_20 = data["volume"].rolling(20, min_periods=5).mean()
        low_volume_mask = (data["volume"] < avg_volume_20).fillna(False)
        volume_div_mask = price_vs_past & low_volume_mask
        data["volume_divergence"] = volume_div_mask.astype(float) * -1.0

        # Compression features
        # ATR percentile (rolling)
        atr_pct = self._rolling_percentile(data["atr"],
                                           window=self.percentile_window)
        data["atr_percentile"] = atr_pct

        # Volatility regime: binary feature indicating high volatility state
        # This helps the model learn: trust trend signals more in high volatility,
        # reduce positions in low volatility
        # Window: 200 bars, threshold: 70th percentile
        volatility_regime_window = 200
        volatility_regime_threshold = 0.7
        atr_quantile_70 = data["atr"].rolling(
            window=volatility_regime_window,
            min_periods=1).quantile(volatility_regime_threshold)
        data["volatility_regime"] = (data["atr"]
                                     > atr_quantile_70).astype(int).fillna(0)

        # Volatility skew features
        returns = data["close"].pct_change().fillna(0.0)
        realized_skew = self._rolling_skew(returns, window=20)
        data["realized_skew"] = realized_skew.fillna(0.0).clip(
            -self.feature_clip_bound, self.feature_clip_bound)

        vol5 = returns.rolling(5, min_periods=1).std()
        vol60 = returns.rolling(60, min_periods=1).std()
        volatility_ratio = (vol5 / vol60.replace(0, np.nan)).replace(
            [np.inf, -np.inf], np.nan).fillna(0.0)
        data["volatility_ratio"] = volatility_ratio.clip(
            -self.feature_clip_bound, self.feature_clip_bound)

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
                        midnight_delta = (utc_idx - utc_idx.normalize()
                                          ).total_seconds() / 60.0
                    else:
                        # Assume timestamps are already UTC
                        hour = idx.hour.astype(int)
                        midnight_delta = (
                            idx - idx.normalize()).total_seconds() / 60.0

                    # Cyclic encoding: sin/cos transformation
                    # This preserves the periodic nature of hours (23 and 0 are close)
                    data["hour_sin"] = np.sin(2 * np.pi * hour / 24)
                    data["hour_cos"] = np.cos(2 * np.pi * hour / 24)

                    # Also keep raw hour for reference (optional, can be excluded later)
                    data["Hour_of_Day"] = hour
                    data["minutes_since_reset"] = (pd.Series(
                        midnight_delta, index=data.index).fillna(0.0))

                except Exception:
                    data["hour_sin"] = 0.0
                    data["hour_cos"] = 1.0
                    data["Hour_of_Day"] = 0
                    data["minutes_since_reset"] = 0.0

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
                        med_sec = float(
                            diffs.median()) if diffs.notna().any() else None
                        if med_sec is not None and med_sec > 0:
                            inferred_bar_minutes = med_sec / 60.0
                    except Exception:
                        inferred_bar_minutes = None

                    if inferred_bar_minutes is None:
                        # Fallback: just use bars since last trade as minutes proxy
                        minutes_raw = run.astype(float)
                    else:
                        minutes_raw = run.astype(float) * float(
                            inferred_bar_minutes)

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
                    data["minutes_since_reset"] = 0.0
            else:
                # Fallback values if index doesn't have hour/dayofweek
                data["hour_sin"] = 0.0
                data["hour_cos"] = 1.0
                data["Hour_of_Day"] = 0
                data["minutes_since_reset"] = 0.0
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
            data["minutes_since_reset"] = 0.0
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
            "rsi_divergence",
            "volume_divergence",
            "realized_skew",
            "volatility_ratio",
            "hour_sin",
            "hour_cos",
            "Hour_of_Day",
            "Is_Weekend",
            "minutes_since_reset",
            "Minutes_Since_Last_Trade",
            "no_trade_5min",
            "no_trade_15min",
            "no_trade_30min",
            "trade_gap_bin",
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

        # 如果指定了required_features，只保留需要的特征
        if required_features is not None:
            data_cols = {
                'open', 'high', 'low', 'close', 'volume', 'timestamp',
                'datetime'
            }
            # 保留数据列和需要的特征列
            cols_to_keep = [
                c for c in data.columns
                if c in data_cols or c in required_features
                or not pd.api.types.is_numeric_dtype(data[c])
            ]
            data = data[cols_to_keep]

        if self.enable_diagnostics:
            diag_cols = [c for c in all_feature_cols if c in data.columns]
            self._run_diagnostics(data, diag_cols)

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
    fit: bool = True,
    required_features: Optional[set] = None
) -> Tuple[pd.DataFrame, BaselineFeatureEngineer]:
    if engineer is None:
        engineer = BaselineFeatureEngineer()
    out = engineer.engineer_features(df,
                                     fit=fit,
                                     required_features=required_features)
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
        "open",
        "high",
        "low",
        "close",
        "volume",
        "signal",
        "binary_signal",
        "future_return",
    }
    exclude.update([
        col for col in df.columns
        if (col.startswith("signal_") or col.startswith("binary_signal_")
            or col.startswith("future_return_"))
    ])
    return [c for c in df.columns if c not in exclude]


__all__ = [
    "BaselineFeatureEngineer",
    "engineer_baseline_features",
    "get_baseline_feature_columns",
    "create_binary_labels_baseline",
]
