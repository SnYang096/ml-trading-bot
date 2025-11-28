"""统一的基础指标和特征工程模块

合并了 base_indicators.py 和 baseline_feature_engineering.py 的功能，
并添加了无量纲特征和优化的依赖关系管理。

主要改进：
1. 合并基础指标和 baseline 特征到一个模块
2. 添加 ZigZag、POC、HAL、Swing High/Low 的无量纲特征
3. 添加基础价格与量能相对变化特征
4. 优化依赖关系管理，支持按需计算
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict
from sklearn.linear_model import LinearRegression

import talib

from .utils_volume_profile import compute_wpt_volume_profile

# ============================================================================
# BaselineFeatureEngineer 类
# ============================================================================


class BaselineFeatureEngineer:
    """Baseline SR and compression features (合并后的版本)"""

    def __init__(
        self,
        percentile_window: int = 288,
        compression_threshold_pct: float = 0.2,
        feature_shift: int = 0,
        feature_clip_bound: float = 10.0,
        vwap_window: int = 160,
        rolling_zscore_windows: List[
            int
        ] = None,  # NEW: Multiple windows for rolling z-score (Feature Stacking)
        enable_diagnostics: bool = False,
    ) -> None:
        self.percentile_window = percentile_window
        self.compression_threshold_pct = compression_threshold_pct
        self.feature_shift = feature_shift
        self.feature_clip_bound = float(feature_clip_bound)
        self.vwap_window = vwap_window
        # Default windows: [50 (short-term), 288 (24h), 500 (long-term)]
        # For 5-minute K-line: 50≈4h, 288=24h, 500≈2 days
        if rolling_zscore_windows is None:
            self.rolling_zscore_windows = [50, 288, 500]
        else:
            self.rolling_zscore_windows = rolling_zscore_windows
        self.enable_diagnostics = enable_diagnostics
        self.diagnostic_report: Dict[str, Dict[str, float]] = {}

        if self.feature_clip_bound <= 0:
            raise ValueError("feature_clip_bound must be positive")
        if not self.rolling_zscore_windows or not all(
            w > 0 for w in self.rolling_zscore_windows
        ):
            raise ValueError(
                "rolling_zscore_windows must be a non-empty list of positive integers"
            )

    # ========================================================================
    # 基础指标计算静态方法
    # ========================================================================

    @staticmethod
    def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """计算相对强弱指数 (RSI)."""
        series = pd.to_numeric(series, errors="coerce").astype(float)
        values = talib.RSI(series.values, timeperiod=period)
        return pd.Series(values, index=series.index)

    @staticmethod
    def compute_macd(
        series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """计算MACD指标."""
        series = pd.to_numeric(series, errors="coerce").astype(float)
        macd_line, signal_line, histogram = talib.MACD(
            series.values, fastperiod=fast, slowperiod=slow, signalperiod=signal
        )
        index = series.index
        return (
            pd.Series(macd_line, index=index),
            pd.Series(signal_line, index=index),
            pd.Series(histogram, index=index),
        )

    @staticmethod
    def compute_bollinger_bands(
        series: pd.Series, period: int = 20, std_dev: int = 2
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """计算布林带."""
        series = pd.to_numeric(series, errors="coerce").astype(float)
        upper, middle, lower = talib.BBANDS(
            series.values, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev, matype=0
        )
        index = series.index
        return (
            pd.Series(upper, index=index),
            pd.Series(middle, index=index),
            pd.Series(lower, index=index),
        )

    @staticmethod
    def compute_atr(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """计算平均真实波幅 (ATR)."""
        high = pd.to_numeric(high, errors="coerce").astype(float)
        low = pd.to_numeric(low, errors="coerce").astype(float)
        close = pd.to_numeric(close, errors="coerce").astype(float)
        atr_values = talib.ATR(high.values, low.values, close.values, timeperiod=period)
        return pd.Series(atr_values, index=high.index)

    @staticmethod
    def compute_zigzag(
        high: pd.Series,
        low: pd.Series,
        threshold: float = 0.05,
        return_high_low: bool = False,
        price_col: Optional[pd.Series] = None,
    ) -> pd.Series | Tuple[pd.Series, pd.Series, pd.Series]:
        """
        计算ZigZag指标（优化版：可同时计算高点和低点）
        
        ✅ 建议：使用 WPT 中高频重构价格（price_col）而非原始价格
        这样可以保留关键拐点，同时去除毛刺噪声。

        Args:
            high: 最高价序列（如果提供了 price_col，此参数将被忽略用于价格计算）
            low: 最低价序列（如果提供了 price_col，此参数将被忽略用于价格计算）
            threshold: 转折阈值（默认 0.05，即 5%）
            return_high_low: 是否同时返回高点和低点序列（默认 False）
            price_col: 可选的价格序列（如 WPT 中高频重构价格）。如果提供，将使用此价格
                       而非原始 high/low。默认 None，使用原始价格（向后兼容）

        Returns:
            如果 return_high_low=False: 返回 zigzag 序列
            如果 return_high_low=True: 返回 (zigzag, zz_high, zz_low) 元组
        """
        high = pd.to_numeric(high, errors="coerce")
        low = pd.to_numeric(low, errors="coerce")
        if len(high) < 2:
            if return_high_low:
                empty = pd.Series(index=high.index, dtype=float)
                return empty, empty, empty
            return pd.Series(index=high.index, dtype=float)

        zigzag = pd.Series(index=high.index, dtype=float)
        zz_high = pd.Series(index=high.index, dtype=float) if return_high_low else None
        zz_low = pd.Series(index=high.index, dtype=float) if return_high_low else None

        # 确定使用的价格序列
        if price_col is not None:
            # 使用 WPT 重构价格（同时作为 high 和 low）
            price_series = price_col
            last_pivot = price_series.iloc[0]
        else:
            # 使用原始价格
            price_series = None
            last_pivot = high.iloc[0]
        
        trend = None
        try:
            for i in range(1, len(high)):
                if price_col is not None:
                    # 使用 WPT 重构价格
                    current_price = price_series.iloc[i]
                    current_high = current_price
                    current_low = current_price
                else:
                    # 使用原始价格
                    current_high = high.iloc[i]
                    current_low = low.iloc[i]
                
                if trend is None:
                    if current_high >= last_pivot * (1 + threshold):
                        trend = "up"
                        last_pivot = current_high
                        zigzag.iloc[i] = current_high
                        if return_high_low:
                            zz_high.iloc[i] = current_high
                    elif current_low <= last_pivot * (1 - threshold):
                        trend = "down"
                        last_pivot = current_low
                        zigzag.iloc[i] = current_low
                        if return_high_low:
                            zz_low.iloc[i] = current_low
                elif trend == "up":
                    if current_low <= last_pivot * (1 - threshold):
                        # 趋势反转：从上涨转为下跌
                        trend = "down"
                        last_pivot = current_low
                        zigzag.iloc[i] = current_low
                        if return_high_low:
                            zz_low.iloc[i] = current_low
                    elif current_high >= last_pivot:
                        # 继续上涨，更新高点
                        last_pivot = current_high
                        zigzag.iloc[i] = current_high
                        if return_high_low:
                            zz_high.iloc[i] = current_high
                else:  # trend == 'down'
                    if current_high >= last_pivot * (1 + threshold):
                        # 趋势反转：从下跌转为上涨
                        trend = "up"
                        last_pivot = current_high
                        zigzag.iloc[i] = current_high
                        if return_high_low:
                            zz_high.iloc[i] = current_high
                    elif current_low <= last_pivot:
                        # 继续下跌，更新低点
                        last_pivot = current_low
                        zigzag.iloc[i] = current_low
                        if return_high_low:
                            zz_low.iloc[i] = current_low

            zigzag = zigzag.ffill()
            if return_high_low:
                zz_high = zz_high.ffill()
                zz_low = zz_low.ffill()
        except Exception:
            zigzag = pd.Series(0, index=high.index, dtype=float)
            if return_high_low:
                zz_high = pd.Series(0, index=high.index, dtype=float)
                zz_low = pd.Series(0, index=high.index, dtype=float)

        if return_high_low:
            return zigzag, zz_high, zz_low
        return zigzag

    @staticmethod
    def compute_poc(
        high: pd.Series,
        low: pd.Series,
        volume: pd.Series,
        window: int = 160,
        bins: int | str = "auto",
        value_area_ratio: float = 0.7,
        price_col: Optional[pd.Series] = None,
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        计算 POC (Point of Control) 和 HAL (Value Area 70% 价格区间的上下界)
        
        ✅ 强烈建议：使用 WPT 低频重构价格（price_col）而非原始价格
        这样可以过滤高频噪声，使 POC/HAL 更接近真实供需平衡点。

        Args:
            high: 最高价序列（如果提供了 price_col，此参数将被忽略用于价格计算）
            low: 最低价序列（如果提供了 price_col，此参数将被忽略用于价格计算）
            volume: 成交量序列（始终使用原始成交量，不应过滤）
            window: 滚动窗口大小
            bins: 价格分档数量。如果为 "auto"（默认），则使用 Freedman-Diaconis rule 自动计算
            value_area_ratio: Value Area 的成交量占比（默认 0.7，即 70%）
            price_col: 可选的价格序列（如 WPT 低频重构价格）。如果提供，将使用此价格
                       而非 (high+low)/2。默认 None，使用原始价格（向后兼容）

        Returns:
            (poc, poc_volume_ratio, hal_high, hal_low):
            - poc: POC 价格序列
            - poc_volume_ratio: POC 价格档的成交量占比序列
            - hal_high: HAL 高点（Value Area 上界）
            - hal_low: HAL 低点（Value Area 下界）
        """
        poc = pd.Series(index=high.index, dtype=float)
        poc_volume_ratio = pd.Series(index=high.index, dtype=float)
        hal_high = pd.Series(index=high.index, dtype=float)
        hal_low = pd.Series(index=high.index, dtype=float)

        for i in range(window, len(high)):
            price_window = (
                price_col.iloc[i - window : i].values
                if price_col is not None
                else ((high.iloc[i - window : i].values + low.iloc[i - window : i].values) / 2.0)
            )
            volume_window = volume.iloc[i - window : i].values

            vp_result = compute_wpt_volume_profile(
                price_window=price_window,
                volume_window=volume_window,
                bins=bins,
            )

            if vp_result is None:
                fallback_price = (
                    float(price_window[-1])
                    if len(price_window) > 0
                    else float("nan")
                )
                poc.iloc[i] = fallback_price
                hal_high.iloc[i] = fallback_price
                hal_low.iloc[i] = fallback_price
                continue

            hist = vp_result.hist
            total_volume = hist.sum()

            if total_volume <= 0:
                continue

            centers = vp_result.centers
            edges = vp_result.edges

            max_vol_idx = int(np.argmax(hist))
            poc.iloc[i] = centers[max_vol_idx]

            poc_volume_ratio.iloc[i] = hist[max_vol_idx] / total_volume

            target_volume = total_volume * value_area_ratio
            accumulated_volume = hist[max_vol_idx]
            upper_idx = max_vol_idx
            lower_idx = max_vol_idx

            while accumulated_volume < target_volume:
                upper_vol = hist[upper_idx + 1] if upper_idx + 1 < len(hist) else 0.0
                lower_vol = hist[lower_idx - 1] if lower_idx - 1 >= 0 else 0.0

                if (upper_vol >= lower_vol and upper_idx + 1 < len(hist)) or lower_idx == 0:
                    if upper_idx + 1 < len(hist):
                        upper_idx += 1
                        accumulated_volume += hist[upper_idx]
                    else:
                        break
                elif lower_idx - 1 >= 0:
                    lower_idx -= 1
                    accumulated_volume += hist[lower_idx]
                else:
                    break

            # HAL 上下界对应价格档的边界
            hal_high_edge_idx = min(upper_idx + 1, len(edges) - 1)
            hal_low_edge_idx = max(lower_idx, 0)
            hal_high.iloc[i] = edges[hal_high_edge_idx]
            hal_low.iloc[i] = edges[hal_low_edge_idx]

        poc = poc.ffill()
        hal_high = hal_high.ffill()
        hal_low = hal_low.ffill()
        return poc, poc_volume_ratio, hal_high, hal_low

    @staticmethod
    def calculate_sqs(
        sr_price: float,
        df: pd.DataFrame,
        window: int = 60,
        tolerance_factor: float = 0.5,
        sr_type: str = "support",  # 必须指定: 'support' 或 'resistance'
        max_lookahead_bars: int = 3,
        use_volume_confirmation: bool = True,  # 是否使用量价确认
        volume_lookback: int = 20,  # 成交量回看窗口（用于计算平均成交量）
        min_volume_ratio: float = 1.0,  # 最小成交量比率（低于此值不计入有效反应）
    ) -> float:
        """
        计算支撑/阻力位的质量评分（Structure Quality Score, SQS）

        ⚠️ 要求：df 必须是截止到【当前决策时刻之前】的历史数据（即不含未来K线）
        例如，在时刻 i 决策时，df = data.iloc[:i]

        Args:
            sr_price: 支撑或阻力价位
            df: 历史K线数据，必须包含 ['high', 'low', 'close', 'atr', 'volume']，索引为时间
            window: 回看窗口长度（单位：K线根数）
            tolerance_factor: ATR 容忍带系数，默认 0.5
            sr_type: 必须为 'support' 或 'resistance'
            max_lookahead_bars: 最大反应观察期（不超过窗口剩余长度）
            use_volume_confirmation: 是否使用量价确认（默认 True）
            volume_lookback: 成交量回看窗口（用于计算平均成交量）
            min_volume_ratio: 最小成交量比率（低于此值不计入有效反应，默认 1.0 表示不强制要求放量）

        Returns:
            SQS 分数（>=0，越高越好；无有效测试时返回 0.0）
        """
        if sr_type not in {"support", "resistance"}:
            raise ValueError("sr_type must be 'support' or 'resistance'")

        if len(df) < window or "atr" not in df.columns or df["atr"].empty:
            return 0.0

        # 使用窗口内最后一个 ATR（即最新可用ATR）
        current_atr = df["atr"].iloc[-1]
        if current_atr <= 0:
            return 0.0

        tolerance = current_atr * tolerance_factor
        window_df = df.tail(window).copy()

        # 1. 找出触及 SR 区域的K线（价格区间与 [sr_price ± tolerance] 有交集）
        near_sr = (window_df["low"] <= sr_price + tolerance) & (
            window_df["high"] >= sr_price - tolerance
        )
        test_indices = window_df[near_sr].index.tolist()
        if not test_indices:
            return 0.0

        reactions = []
        n = len(window_df)

        for idx in test_indices:
            try:
                pos = window_df.index.get_loc(idx)
            except KeyError:
                continue

            # 确保后面还有至少1根K线用于观察反应
            if pos >= n - 1:
                continue

            # 动态确定实际可观察的反应期（不超过 max_lookahead_bars，也不越界）
            actual_lookahead = min(max_lookahead_bars, n - pos - 1)
            if actual_lookahead <= 0:
                continue

            future_slice = window_df.iloc[pos + 1 : pos + 1 + actual_lookahead]
            close_at_touch = window_df.loc[idx, "close"]

            # 【安全实现量价加权】：在反应循环内计算成交量统计
            # 关键原则：
            # 1. current_vol（测试点K线的成交量）可以使用，因为在该K线结束后是已知的
            # 2. avg_vol（基准平均成交量）必须来自更早的数据（不含当前K线）
            # 3. 使用 pos - volume_lookback : pos 确保不包含当前K线
            if use_volume_confirmation and "volume" in window_df.columns:
                # 用测试点之前的 volume_lookback 根K线计算平均成交量（不含当前K线）
                if pos >= volume_lookback:
                    # 有足够历史数据：使用 pos - volume_lookback : pos（不包含 pos）
                    ref_vols = window_df.iloc[pos - volume_lookback : pos]["volume"]
                else:
                    # 数据不足：使用可用数据（至少1根，但不包含当前K线）
                    ref_vols = window_df.iloc[:max(1, pos)]["volume"]

                avg_vol = ref_vols.mean() if len(ref_vols) > 0 else 1.0
                current_vol = window_df.loc[idx, "volume"]  # 当前K线成交量（可以使用）
                vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0

                # 体积确认因子（抑制极端值，限制在3倍以内）
                vol_factor = min(vol_ratio, 3.0)
            else:
                vol_factor = 1.0
                vol_ratio = 1.0

            if sr_type == "resistance":
                # 阻力：期望价格下跌 → 反应 = 触及时收盘价 - 未来最低价
                reaction = close_at_touch - future_slice["low"].min()
                if reaction > 0:  # 仅当确实下跌时才计入
                    # 只有放量且方向正确的反应才计入
                    if use_volume_confirmation and "volume" in window_df.columns:
                        if vol_ratio >= min_volume_ratio:
                            # 使用平方根加权，避免极端值影响过大
                            weighted_reaction = (reaction / current_atr) * np.sqrt(vol_factor)
                            reactions.append(weighted_reaction)
                    else:
                        # 不使用量价确认，直接归一化
                        reactions.append(reaction / current_atr)
            else:  # support
                # 支撑：期望价格上涨 → 反应 = 未来最高价 - 触及时收盘价
                reaction = future_slice["high"].max() - close_at_touch
                if reaction > 0:  # 仅当确实反弹时才计入
                    # 只有放量且方向正确的反应才计入
                    if use_volume_confirmation and "volume" in window_df.columns:
                        if vol_ratio >= min_volume_ratio:
                            # 使用平方根加权，避免极端值影响过大
                            weighted_reaction = (reaction / current_atr) * np.sqrt(vol_factor)
                            reactions.append(weighted_reaction)
                    else:
                        # 不使用量价确认，直接归一化
                        reactions.append(reaction / current_atr)

        # 汇总指标
        test_count = len(test_indices)
        valid_reaction_count = len(reactions)
        avg_reaction = np.mean(reactions) if reactions else 0.0
        recent_test_count = near_sr.tail(20).sum()  # 近20根K线内的测试次数

        # 标准化打分（抑制极端值，强调有效反应）
        test_score = min(test_count, 5) * 0.4
        reaction_score = min(avg_reaction * 2, 3.0) * 0.4  # avg_reaction=1.5 → 满分
        recency_score = min(recent_test_count, 3) * 0.2

        sqs = test_score + reaction_score + recency_score
        return float(sqs)

    @staticmethod
    def evaluate_level_quality_bidirectional(
        sr_price: float,
        df: pd.DataFrame,
        window: int = 60,
        tolerance_factor: float = 0.5,
        max_lookahead_bars: int = 3,
        use_volume_confirmation: bool = True,
        volume_lookback: int = 20,
        min_volume_ratio: float = 1.0,
    ) -> Dict[str, float | str]:
        """
        对未知类型的价格水平进行双向 SQS 评估（Bidirectional Testing）

        适用于：
        - 历史高点/低点（突破后角色可能转换）
        - 成交密集区中轴（POC）
        - OLS 通道中线、VWAP 等动态中轴
        - 其他无法明确判断是支撑还是阻力的水平

        Args:
            sr_price: 价格水平
            df: 历史K线数据，必须包含 ['high', 'low', 'close', 'atr', 'volume']
            window: 回看窗口长度
            tolerance_factor: ATR 容忍带系数
            max_lookahead_bars: 最大反应观察期
            use_volume_confirmation: 是否使用量价确认
            volume_lookback: 成交量回看窗口
            min_volume_ratio: 最小成交量比率

        Returns:
            包含以下键的字典：
            - sqs: 最佳 SQS 分数（支撑和阻力中的较高者）
            - type: 最佳类型（'support' 或 'resistance'）
            - support_sqs: 作为支撑的 SQS 分数
            - resistance_sqs: 作为阻力的 SQS 分数
        """
        # 分别计算支撑和阻力质量
        support_sqs = BaselineFeatureEngineer.calculate_sqs(
            sr_price,
            df,
            window=window,
            tolerance_factor=tolerance_factor,
            sr_type="support",
            max_lookahead_bars=max_lookahead_bars,
            use_volume_confirmation=use_volume_confirmation,
            volume_lookback=volume_lookback,
            min_volume_ratio=min_volume_ratio,
        )

        resistance_sqs = BaselineFeatureEngineer.calculate_sqs(
            sr_price,
            df,
            window=window,
            tolerance_factor=tolerance_factor,
            sr_type="resistance",
            max_lookahead_bars=max_lookahead_bars,
            use_volume_confirmation=use_volume_confirmation,
            volume_lookback=volume_lookback,
            min_volume_ratio=min_volume_ratio,
        )

        # 选择更优角色
        if support_sqs >= resistance_sqs:
            best_sqs = support_sqs
            best_type = "support"
        else:
            best_sqs = resistance_sqs
            best_type = "resistance"

        return {
            "sqs": float(best_sqs),
            "type": best_type,
            "support_sqs": float(support_sqs),
            "resistance_sqs": float(resistance_sqs),
        }

    @staticmethod
    def calculate_volume_price_confirmation(
        df: pd.DataFrame,
        breakout_idx: int,
        sr_price: float,
        lookback: int = 20,
        vol_threshold: float = 1.5,
        confirmation_bars: int = 3,
        sr_type: Optional[str] = None,
    ) -> float:
        """
        计算量价配合度（Volume-Price Confirmation）
        在突破发生时，评估量能是否支持

        Args:
            df: 包含 high, low, close, volume 的 K 线数据
            breakout_idx: 突破发生的索引位置
            sr_price: 突破的支撑/阻力价格
            lookback: 回看窗口大小（用于计算平均成交量）
            vol_threshold: 成交量倍数阈值（默认 1.5）
            confirmation_bars: 确认站稳的 K 线数量（默认 3）

        Returns:
            量价配合度分数（0-1，1 表示完全确认）
        """
        if breakout_idx < lookback or breakout_idx >= len(df) - confirmation_bars:
            return 0.0

        # 1. 成交量确认
        current_vol = df["volume"].iloc[breakout_idx]
        avg_vol = df["volume"].iloc[breakout_idx - lookback : breakout_idx].mean()
        if avg_vol <= 0:
            vol_ratio = 0.0
        else:
            vol_ratio = current_vol / avg_vol
        vol_confirmed = vol_ratio > vol_threshold

        # 2. 站稳确认：后几根 K 线是否持续在突破方向？
        if sr_type == "resistance":
            direction = 1
        elif sr_type == "support":
            direction = -1
        else:
            direction = 1 if df["close"].iloc[breakout_idx] >= sr_price else -1
        confirmed = True
        for i in range(1, min(confirmation_bars + 1, len(df) - breakout_idx)):
            if breakout_idx + i < len(df):
                price_diff = (df["close"].iloc[breakout_idx + i] - sr_price) * direction
                if price_diff <= 0:
                    confirmed = False
                    break

        # 综合评分：成交量确认权重 0.6，站稳确认权重 0.4
        score = 0.6 * (1.0 if vol_confirmed else 0.0) + 0.4 * (
            1.0 if confirmed else 0.0
        )

        return score

    @staticmethod
    def calculate_failed_breakout_reversal(
        df: pd.DataFrame,
        breakout_idx: int,
        sr_price: float,
        sr_type: str,
        lookback: int = 20,
        confirmation_bars: int = 3,
        reversal_bars: int = 3,
    ) -> float:
        """
        计算失败突破反转强度（Failed Breakout Reversal）

        检测价格突破边界但未站稳，然后反向运动的情况。
        例如：价格突破 VWAP/均线/OLS 通道，但没站稳，然后反向运动回到边界内。

        Args:
            df: 包含 high, low, close, volume, atr 的 K 线数据
            breakout_idx: 突破发生的索引位置
            sr_price: 突破的支撑/阻力价格
            sr_type: 边界类型 ("support" 或 "resistance")
            lookback: 回看窗口大小（用于计算平均成交量）
            confirmation_bars: 确认站稳的 K 线数量（默认 3）
            reversal_bars: 确认反向运动的 K 线数量（默认 3）

        Returns:
            失败突破反转强度分数（0-1，1 表示强烈反转）
        """
        if breakout_idx < lookback or breakout_idx >= len(df) - max(
            confirmation_bars, reversal_bars
        ):
            return 0.0

        if "atr" not in df.columns or df["atr"].iloc[breakout_idx] <= 0:
            return 0.0

        # 1. 确认发生了突破
        if sr_type == "resistance":
            # 阻力：价格应该突破（close > sr_price）
            broke_out = df["close"].iloc[breakout_idx] > sr_price
            direction = 1  # 向上突破
        else:  # support
            # 支撑：价格应该突破（close < sr_price）
            broke_out = df["close"].iloc[breakout_idx] < sr_price
            direction = -1  # 向下突破

        if not broke_out:
            return 0.0

        # 2. 检测是否没站稳（在确认期内价格回到边界内）
        failed_confirmation = False
        for i in range(1, min(confirmation_bars + 1, len(df) - breakout_idx)):
            if breakout_idx + i < len(df):
                if sr_type == "resistance":
                    # 阻力：如果价格回到阻力位下方，说明没站稳
                    if df["close"].iloc[breakout_idx + i] <= sr_price:
                        failed_confirmation = True
                        break
                else:  # support
                    # 支撑：如果价格回到支撑位上方，说明没站稳
                    if df["close"].iloc[breakout_idx + i] >= sr_price:
                        failed_confirmation = True
                        break

        if not failed_confirmation:
            return 0.0  # 站稳了，不是失败突破

        # 3. 检测反向运动：后续 K 线是否持续反向（远离突破方向，回到边界内）
        reversal_confirmed = True
        reversal_start_idx = breakout_idx + confirmation_bars

        if reversal_start_idx >= len(df) - reversal_bars:
            return 0.0

        for i in range(reversal_bars):
            if reversal_start_idx + i < len(df):
                if sr_type == "resistance":
                    # 阻力：价格应该持续下跌（回到阻力位下方）
                    if (
                        df["close"].iloc[reversal_start_idx + i]
                        > df["close"].iloc[reversal_start_idx]
                    ):
                        reversal_confirmed = False
                        break
                    # 确认回到阻力位下方
                    if df["close"].iloc[reversal_start_idx + i] > sr_price:
                        reversal_confirmed = False
                        break
                else:  # support
                    # 支撑：价格应该持续上涨（回到支撑位上方）
                    if (
                        df["close"].iloc[reversal_start_idx + i]
                        < df["close"].iloc[reversal_start_idx]
                    ):
                        reversal_confirmed = False
                        break
                    # 确认回到支撑位上方
                    if df["close"].iloc[reversal_start_idx + i] < sr_price:
                        reversal_confirmed = False
                        break

        if not reversal_confirmed:
            return 0.0

        # 4. 计算反转幅度（以 ATR 为单位）
        if sr_type == "resistance":
            # 阻力：计算从突破点到反转后的跌幅
            breakout_price = df["close"].iloc[breakout_idx]
            reversal_price = df["close"].iloc[reversal_start_idx + reversal_bars - 1]
            reversal_magnitude = (breakout_price - reversal_price) / df["atr"].iloc[
                breakout_idx
            ]
        else:  # support
            # 支撑：计算从突破点到反转后的涨幅
            breakout_price = df["close"].iloc[breakout_idx]
            reversal_price = df["close"].iloc[reversal_start_idx + reversal_bars - 1]
            reversal_magnitude = (reversal_price - breakout_price) / df["atr"].iloc[
                breakout_idx
            ]

        # 5. 成交量确认：突破时成交量是否放大，反转时是否缩量或放量
        breakout_vol = df["volume"].iloc[breakout_idx]
        avg_vol = df["volume"].iloc[breakout_idx - lookback : breakout_idx].mean()
        if avg_vol <= 0:
            vol_ratio = 0.0
        else:
            vol_ratio = breakout_vol / avg_vol

        # 反转时的成交量
        reversal_vol = (
            df["volume"]
            .iloc[reversal_start_idx : reversal_start_idx + reversal_bars]
            .mean()
        )
        reversal_vol_ratio = reversal_vol / avg_vol if avg_vol > 0 else 0.0

        # 6. 综合评分
        # 突破幅度（0-1）：突破幅度越大，失败反转的信号越强
        if sr_type == "resistance":
            breakout_magnitude = (df["close"].iloc[breakout_idx] - sr_price) / df[
                "atr"
            ].iloc[breakout_idx]
        else:
            breakout_magnitude = (sr_price - df["close"].iloc[breakout_idx]) / df[
                "atr"
            ].iloc[breakout_idx]
        breakout_score = min(1.0, breakout_magnitude / 2.0)  # 2*ATR 的突破幅度为满分

        # 反转幅度（0-1）：反转幅度越大，信号越强
        reversal_score = min(1.0, reversal_magnitude / 2.0)  # 2*ATR 的反转幅度为满分

        # 成交量模式（0-1）：突破时放量，反转时缩量或放量都是信号
        vol_score = 0.0
        if vol_ratio > 1.5:  # 突破时放量
            if reversal_vol_ratio < 0.8:  # 反转时缩量（更典型）
                vol_score = 1.0
            elif reversal_vol_ratio > 1.2:  # 反转时也放量（恐慌性反转）
                vol_score = 0.8

        # 综合评分
        score = (
            0.3 * breakout_score  # 突破幅度权重 30%
            + 0.4 * reversal_score  # 反转幅度权重 40%
            + 0.3 * vol_score  # 成交量模式权重 30%
        )

        return score

    @staticmethod
    def _get_sr_boundary_definitions(data: pd.DataFrame) -> List[Dict[str, str]]:
        """收集所有可用的 SR 边界定义"""
        boundaries: List[Dict[str, str]] = []

        def _register(name: str, column: str, sr_type: str, category: str) -> None:
            if column in data.columns:
                boundaries.append(
                    {
                        "name": name,
                        "column": column,
                        "type": sr_type,
                        "category": category,
                    }
                )

        _register("swing_high_s", "roll_high_s", "resistance", "swing_short")
        _register("swing_low_s", "roll_low_s", "support", "swing_short")
        _register("swing_high_l", "roll_high_l", "resistance", "swing_long")
        _register("swing_low_l", "roll_low_l", "support", "swing_long")
        _register("zigzag_high", "zz_high_value", "resistance", "zigzag")
        _register("zigzag_low", "zz_low_value", "support", "zigzag")
        _register("hal_high", "hal_high", "resistance", "hal")
        _register("hal_low", "hal_low", "support", "hal")
        _register("poc_level", "poc", "mid", "poc")
        _register("boll_upper", "bb_upper", "resistance", "bollinger")
        _register("boll_lower", "bb_lower", "support", "bollinger")
        _register("ols_upper", "ols_channel_upper", "resistance", "ols")
        _register("ols_lower", "ols_channel_lower", "support", "ols")
        _register("ols_mid", "ols_channel_mid", "mid", "ols")
        _register("vwap_level", "vwap", "mid", "vwap")

        return boundaries

    @staticmethod
    def _compute_boundary_strengths(
        data: pd.DataFrame,
        boundaries: List[Dict[str, str]],
        window: int = 60,
        tolerance_factor: float = 0.5,
        cluster_weight: float = 0.15,
        compression_series: Optional[pd.Series] = None,
    ) -> Dict[str, pd.Series]:
        """
        计算每个边界的 SQS 强度，并考虑边界重合与压缩质量
        
        对于 mid 类型的边界（如 poc, ols_mid, vwap），不仅输出加权平均的 base，
        还输出原始分量和上下文特征，让模型学习交互关系：
        - {name}_support_sqs: 支撑方向的 SQS
        - {name}_resistance_sqs: 阻力方向的 SQS
        - {name}_price_above: 价格是否在边界上方 (1.0/0.0)
        - {name}_trend_down: 价格是否向下走 (1.0/0.0)
        - {name}_weight_support: 支撑权重
        - {name}_weight_resistance: 阻力权重
        - sqs_{name}: 加权平均的 base（保留用于向后兼容）
        """
        if "atr" not in data.columns or not boundaries:
            return {}

        atr_series = data["atr"].ffill()
        strengths: Dict[str, pd.Series] = {}
        sr_values = {b["name"]: data[b["column"]] for b in boundaries}
        comp_series = (
            compression_series.ffill()
            if compression_series is not None
            else pd.Series(0.0, index=data.index)
        )

        for boundary in boundaries:
            name = boundary["name"]
            column = boundary["column"]
            sr_type = boundary["type"]
            sr_series = sr_values[name]
            strength = pd.Series(0.0, index=data.index, dtype=float)
            
            # 对于 mid 类型，初始化额外的特征序列
            if sr_type == "mid":
                support_sqs_series = pd.Series(0.0, index=data.index, dtype=float)
                resistance_sqs_series = pd.Series(0.0, index=data.index, dtype=float)
                price_above_series = pd.Series(0.0, index=data.index, dtype=float)
                trend_down_series = pd.Series(0.0, index=data.index, dtype=float)
                weight_support_series = pd.Series(0.5, index=data.index, dtype=float)
                weight_resistance_series = pd.Series(0.5, index=data.index, dtype=float)

            for i in range(window, len(data)):
                sr_price = sr_series.iloc[i]
                if pd.isna(sr_price):
                    continue

                # 【关键修复】：window_slice 不包含当前时刻 i，只使用历史数据 [i-window, i)
                # 这样 calculate_sqs 在计算反应强度时，可以使用 [i-window, i) 范围内的数据
                # 对于窗口内的每个测试点，可以使用该点之后、窗口结束之前的数据来计算反应
                window_slice = data.iloc[max(0, i - window) : i]
                try:
                    # 对于 "mid" 类型的边界（如 poc, ols_mid, vwap），使用双向测试
                    # 因为它们既可能是支撑也可能是阻力，取决于价格相对位置
                    if sr_type == "mid":
                        # 使用双向测试，自动识别当前市场角色
                        level_quality = (
                            BaselineFeatureEngineer.evaluate_level_quality_bidirectional(
                                sr_price,
                                window_slice,
                                window=window,
                                tolerance_factor=tolerance_factor,
                                use_volume_confirmation=True,
                            )
                        )
                        # 【关键修复】：对于 mid 类型，使用 support_sqs 和 resistance_sqs 的加权平均
                        # 而不是只选择较大的那个，避免偏向某一方向
                        # 权重根据价格趋势动态调整：
                        # - 价格在边界上方且向下走 → 边界作为支撑 → support 权重更高
                        # - 价格在边界下方且向上走 → 边界作为阻力 → resistance 权重更高
                        support_sqs_val = level_quality.get("support_sqs", 0.0)
                        resistance_sqs_val = level_quality.get("resistance_sqs", 0.0)
                        
                        # 判断价格位置和趋势
                        current_price = data["close"].iloc[i]
                        if not pd.isna(current_price) and not pd.isna(sr_price):
                            price_above = current_price > sr_price
                            
                            # 计算价格趋势（使用最近几根K线的平均变化）
                            lookback_trend = 3  # 使用最近3根K线判断趋势
                            if i >= lookback_trend:
                                recent_prices = data["close"].iloc[i - lookback_trend : i]
                                if len(recent_prices) > 1 and recent_prices.notna().all():
                                    price_trend = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
                                    # price_trend > 0 表示上涨，< 0 表示下跌
                                    
                                    # 动态权重分配：
                                    # 1. 价格在边界上方且向下走 → 边界作为支撑 → support 权重更高
                                    # 2. 价格在边界下方且向上走 → 边界作为阻力 → resistance 权重更高
                                    if price_above and price_trend < 0:
                                        # 价格在边界上方且向下走，边界作为支撑
                                        weight_support = 0.7
                                        weight_resistance = 0.3
                                    elif not price_above and price_trend > 0:
                                        # 价格在边界下方且向上走，边界作为阻力
                                        weight_support = 0.3
                                        weight_resistance = 0.7
                                    else:
                                        # 其他情况（价格远离边界或趋势不明显），使用平衡权重
                                        weight_support = 0.5
                                        weight_resistance = 0.5
                                else:
                                    # 数据不足，使用位置判断
                                    if price_above:
                                        # 价格在边界上方，更可能作为支撑（如果回落）
                                        weight_support = 0.6
                                        weight_resistance = 0.4
                                    else:
                                        # 价格在边界下方，更可能作为阻力（如果反弹）
                                        weight_support = 0.4
                                        weight_resistance = 0.6
                            else:
                                # 数据不足，使用位置判断
                                if price_above:
                                    weight_support = 0.6
                                    weight_resistance = 0.4
                                else:
                                    weight_support = 0.4
                                    weight_resistance = 0.6
                            
                            base = (
                                support_sqs_val * weight_support
                                + resistance_sqs_val * weight_resistance
                            )
                            
                            # 【增强方案】：不仅输出加权平均的 base，还输出原始分量和上下文特征
                            # 让模型自己学习交互关系，而不是依赖预定义的权重
                            support_sqs_series.iloc[i] = support_sqs_val
                            resistance_sqs_series.iloc[i] = resistance_sqs_val
                            price_above_series.iloc[i] = 1.0 if price_above else 0.0
                            trend_down_series.iloc[i] = 1.0 if price_trend < 0 else 0.0
                            weight_support_series.iloc[i] = weight_support
                            weight_resistance_series.iloc[i] = weight_resistance
                        else:
                            # 如果无法判断价格位置，使用简单平均
                            base = (support_sqs_val + resistance_sqs_val) / 2.0
                            # 仍然记录原始分量
                            support_sqs_series.iloc[i] = support_sqs_val
                            resistance_sqs_series.iloc[i] = resistance_sqs_val
                    else:
                        # 对于明确类型的边界，使用量价确认增强的 SQS
                        base = BaselineFeatureEngineer.calculate_sqs(
                            sr_price,
                            window_slice,
                            window=window,
                            tolerance_factor=tolerance_factor,
                            sr_type=sr_type,
                            use_volume_confirmation=True,  # 启用量价确认
                        )
                except Exception:
                    base = 0.0
                    # 如果发生异常，对于 mid 类型，保持特征为默认值（已在初始化时设置）

                tolerance = (
                    atr_series.iloc[i] * tolerance_factor
                    if not pd.isna(atr_series.iloc[i])
                    else np.nan
                )
                cluster_bonus = 0.0
                if not np.isnan(tolerance):
                    for other in boundaries:
                        if other["name"] == name:
                            continue
                        other_val = sr_values[other["name"]].iloc[i]
                        if (
                            pd.notna(other_val)
                            and abs(other_val - sr_price) <= tolerance
                        ):
                            cluster_bonus += cluster_weight

                compression_bonus = (
                    comp_series.iloc[i] if not np.isnan(comp_series.iloc[i]) else 0.0
                )
                score = base * (1.0 + cluster_bonus) + 0.2 * compression_bonus
                strength.iloc[i] = score

            # 对于 mid 类型，不仅输出加权平均的 base，还输出原始分量和上下文特征
            if sr_type == "mid":
                strengths[f"sqs_{name}"] = strength.shift(1).fillna(0.0)  # 保留加权平均的 base
                # 输出原始分量和上下文特征，让模型学习交互关系
                strengths[f"{name}_support_sqs"] = support_sqs_series.shift(1).fillna(0.0)
                strengths[f"{name}_resistance_sqs"] = resistance_sqs_series.shift(1).fillna(0.0)
                strengths[f"{name}_price_above"] = price_above_series.shift(1).fillna(0.0)
                strengths[f"{name}_trend_down"] = trend_down_series.shift(1).fillna(0.0)
                strengths[f"{name}_weight_support"] = weight_support_series.shift(1).fillna(0.5)
                strengths[f"{name}_weight_resistance"] = weight_resistance_series.shift(1).fillna(0.5)
            else:
                strengths[f"sqs_{name}"] = strength.shift(1).fillna(0.0)

        return strengths

    @staticmethod
    def _compute_breakout_confirmation_and_role_flip(
        data: pd.DataFrame,
        boundaries: List[Dict[str, str]],
        lookback: int = 20,
        confirmation_bars: int = 3,
        max_retest_bars: int = 10,
    ) -> Dict[str, pd.Series]:
        """
        计算突破确认和角色转换特征
        
        包括：
        1. 突破确认概率：基于量价关系判断真伪突破
        2. 角色转换概率：支撑/阻力角色转换的概率
        3. 转换状态显式标记：post_breakout_retest, post_breakdown_retest 等
        
        这些特征帮助模型理解"同一个位置，在不同市场环境下会扮演完全相反的角色"
        """
        if "atr" not in data.columns or not boundaries:
            return {}
        
        features: Dict[str, pd.Series] = {}
        atr_series = data["atr"].ffill()
        
        for boundary in boundaries:
            name = boundary["name"]
            column = boundary["column"]
            sr_type = boundary["type"]
            sr_series = data[column]
            
            # 初始化特征序列
            breakout_confirmation = pd.Series(0.0, index=data.index, dtype=float)
            role_flip_prob = pd.Series(0.0, index=data.index, dtype=float)
            post_breakout_retest = pd.Series(0.0, index=data.index, dtype=float)
            post_breakdown_retest = pd.Series(0.0, index=data.index, dtype=float)
            
            # 记录最近的突破事件（用于检测回踩）
            last_breakout_idx = -1
            last_breakout_direction = 0  # 1=向上突破, -1=向下突破
            last_breakout_price = np.nan
            
            for i in range(lookback + confirmation_bars + max_retest_bars, len(data)):
                sr_price = sr_series.iloc[i]
                if pd.isna(sr_price):
                    continue
                
                current_price = data["close"].iloc[i]
                current_high = data["high"].iloc[i]
                current_low = data["low"].iloc[i]
                current_volume = data["volume"].iloc[i]
                
                # 计算 ATR 用于归一化
                current_atr = atr_series.iloc[i] if not pd.isna(atr_series.iloc[i]) else 1.0
                
                # 1. 检测突破（使用历史数据）
                breakout_idx = i - confirmation_bars - max_retest_bars
                if breakout_idx >= 0:
                    prev_close = data["close"].iloc[breakout_idx - 1] if breakout_idx > 0 else current_price
                    breakout_close = data["close"].iloc[breakout_idx]
                    breakout_high = data["high"].iloc[breakout_idx]
                    breakout_low = data["low"].iloc[breakout_idx]
                    breakout_volume = data["volume"].iloc[breakout_idx]
                    
                    # 检测突破方向
                    detected_breakout = False
                    breakout_direction = 0
                    
                    if sr_type == "resistance":
                        if prev_close <= sr_price and breakout_high > sr_price:
                            detected_breakout = True
                            breakout_direction = 1
                    elif sr_type == "support":
                        if prev_close >= sr_price and breakout_low < sr_price:
                            detected_breakout = True
                            breakout_direction = -1
                    elif sr_type == "mid":
                        if (prev_close <= sr_price and breakout_close > sr_price) or \
                           (prev_close >= sr_price and breakout_close < sr_price):
                            detected_breakout = True
                            breakout_direction = 1 if breakout_close > sr_price else -1
                    
                    if detected_breakout:
                        last_breakout_idx = breakout_idx
                        last_breakout_direction = breakout_direction
                        last_breakout_price = sr_price
                        
                        # 计算突破确认概率（基于量价关系）
                        # 使用历史数据计算平均成交量
                        if breakout_idx >= lookback:
                            avg_vol = data["volume"].iloc[breakout_idx - lookback : breakout_idx].mean()
                        else:
                            avg_vol = data["volume"].iloc[:breakout_idx].mean() if breakout_idx > 0 else 1.0
                        
                        volume_ratio = breakout_volume / avg_vol if avg_vol > 0 else 1.0
                        
                        # 突破幅度（归一化）
                        breakout_size = abs(breakout_close - sr_price) / current_atr if current_atr > 0 else 0.0
                        
                        # 突破后回踩速度（在 confirmation_bars 内是否回踩）
                        retrace_speed = 0.0
                        if breakout_idx + confirmation_bars < i:
                            post_breakout_slice = data.iloc[breakout_idx + 1 : breakout_idx + 1 + confirmation_bars]
                            if len(post_breakout_slice) > 0:
                                if breakout_direction == 1:  # 向上突破
                                    min_after = post_breakout_slice["low"].min()
                                    retrace_pct = (sr_price - min_after) / current_atr if current_atr > 0 else 0.0
                                    retrace_speed = max(0.0, retrace_pct)  # 回踩越深，速度越快
                                else:  # 向下突破
                                    max_after = post_breakout_slice["high"].max()
                                    retrace_pct = (max_after - sr_price) / current_atr if current_atr > 0 else 0.0
                                    retrace_speed = max(0.0, retrace_pct)
                        
                        # 突破确认概率 = sigmoid(量能验证 * 0.5 + 突破幅度 * 0.3 - 回踩速度 * 0.2)
                        import math
                        confirmation_score = (
                            min(volume_ratio, 3.0) * 0.5 +
                            min(breakout_size, 2.0) * 0.3 -
                            min(retrace_speed, 1.5) * 0.2
                        )
                        breakout_confirmation.iloc[i] = 1.0 / (1.0 + math.exp(-confirmation_score))  # Sigmoid
                
                # 2. 检测回踩（突破后回踩原边界）
                if last_breakout_idx >= 0 and i > last_breakout_idx:
                    # 检查是否回踩到原边界附近（在 ATR 范围内）
                    tolerance = current_atr * 0.5
                    near_original_sr = abs(current_price - last_breakout_price) <= tolerance
                    
                    if last_breakout_direction == 1:  # 向上突破后回踩
                        if near_original_sr and current_low <= last_breakout_price + tolerance:
                            post_breakout_retest.iloc[i] = 1.0
                    elif last_breakout_direction == -1:  # 向下突破后回踩
                        if near_original_sr and current_high >= last_breakout_price - tolerance:
                            post_breakdown_retest.iloc[i] = 1.0
                
                # 3. 计算角色转换概率（仅对 mid 类型）
                if sr_type == "mid":
                    # 获取双向 SQS（如果已计算）
                    support_sqs_col = f"{name}_support_sqs"
                    resistance_sqs_col = f"{name}_resistance_sqs"
                    
                    if support_sqs_col in data.columns and resistance_sqs_col in data.columns:
                        support_sqs = data[support_sqs_col].iloc[i] if i < len(data) else 0.0
                        resistance_sqs = data[resistance_sqs_col].iloc[i] if i < len(data) else 0.0
                        
                        # 支撑/阻力主导强度差
                        strength_diff = abs(support_sqs - resistance_sqs)
                        
                        # 价格位置（+1=在边界上方，-1=在边界下方）
                        price_position = 1.0 if current_price > sr_price else -1.0
                        
                        # 角色转换临界点（价格突破后回踩原阻力/支撑）
                        flip_zone = 0.0
                        if last_breakout_idx >= 0 and i > last_breakout_idx:
                            if (last_breakout_direction == 1 and price_position < 0) or \
                               (last_breakout_direction == -1 and price_position > 0):
                                flip_zone = 1.0
                        
                        # 转换概率 = sigmoid(强度差 * 0.7 + 位置验证 * 1.2)
                        import math
                        flip_score = strength_diff * 0.7 + flip_zone * 1.2
                        role_flip_prob.iloc[i] = 1.0 / (1.0 + math.exp(-flip_score))  # Sigmoid
            
            # 保存特征（shift(1) 确保因果性）
            features[f"{name}_breakout_confirmation"] = breakout_confirmation.shift(1).fillna(0.0)
            features[f"{name}_role_flip_prob"] = role_flip_prob.shift(1).fillna(0.0)
            features[f"{name}_post_breakout_retest"] = post_breakout_retest.shift(1).fillna(0.0)
            features[f"{name}_post_breakdown_retest"] = post_breakdown_retest.shift(1).fillna(0.0)
        
        return features

    @staticmethod
    def _add_breakout_quality_features(
        data: pd.DataFrame,
        boundaries: List[Dict[str, str]],
    ) -> pd.DataFrame:
        """
        添加4类12个核心特征，用于让模型自动学习突破质量判断
        
        整体特征体系设计（共4类12个核心特征）：
        
        A. 结构质量（3个）
        1. sqs - SR测试次数+反应强度+时间衰减（已有）
        2. dist_to_nearest_sr - 当前价距最近SR的距离（已有）
        3. sr_confluence - 是否多个周期SR重合（新增）
        
        B. 突破动能（3个）
        1. vol_ratio - 突破K线量比（已有 volume_ratio）
        2. order_flow_delta - 主动买卖差（新增，基于 delta 或 buy_qty - sell_qty）
        3. breakout_speed - 突破K线实体/影线比（新增）
        
        C. 动能持续性（3个）
        1. follow_through_1 - 第2根K线是否继续新高/新低（新增）
        2. follow_through_2 - 第3根K线是否站稳（新增）
        3. momentum_decay - 突破后3根K线的斜率变化（新增）
        
        D. 市场环境（3个）
        1. compression_score - 布林带宽度 / ATR 比值（已有 compression_confidence）
        2. trend_strength - ADX(14) 或 slope of MA50（新增）
        3. time_phase - 是否在活跃交易时段（已有 hour_sin, hour_cos）
        """
        if data.empty:
            return data
        
        # 确保有必要的列
        if "atr" not in data.columns:
            data["atr"] = BaselineFeatureEngineer._compute_atr(data)
        
        # A.3. SR重合度（sr_confluence）
        # 检查是否有多个边界在相近位置（ATR范围内）
        sr_confluence = pd.Series(0.0, index=data.index, dtype=float)
        if boundaries:
            for i in range(len(data)):
                current_price = data["close"].iloc[i]
                current_atr = data["atr"].iloc[i] if not pd.isna(data["atr"].iloc[i]) else 1.0
                tolerance = current_atr * 0.5
                
                # 收集所有非NaN的边界价格
                nearby_boundaries = []
                for boundary in boundaries:
                    col = boundary["column"]
                    if col in data.columns:
                        sr_price = data[col].iloc[i]
                        if not pd.isna(sr_price) and abs(sr_price - current_price) <= tolerance * 2:
                            nearby_boundaries.append(sr_price)
                
                # 计算在 tolerance 范围内的边界数量
                if len(nearby_boundaries) >= 2:
                    # 检查有多少个边界在 tolerance 范围内
                    count = 0
                    for sr1 in nearby_boundaries:
                        for sr2 in nearby_boundaries:
                            if sr1 != sr2 and abs(sr1 - sr2) <= tolerance:
                                count += 1
                    sr_confluence.iloc[i] = min(count / 2.0, 3.0) / 3.0  # 归一化到 [0, 1]
        
        data["sr_confluence"] = sr_confluence.shift(1).fillna(0.0)
        
        # B.2. 订单流差值（order_flow_delta）
        # 注意：如果已经加载了 order_flow features，可以直接使用：
        # - cvd_normalized（单根K线，归一化，等同于 order_flow_delta）
        # - cvd_change_1（单根K线，原始值）
        # - cvd_change_5（5根K线周期）
        # - cvd_change_20（20根K线周期）
        # 
        # 这里为了保持特征命名一致性，优先使用 cvd_normalized，如果没有则计算
        if "cvd_normalized" in data.columns:
            # 直接使用已有的 cvd_normalized（单根K线，归一化）
            data["order_flow_delta"] = data["cvd_normalized"].shift(1).fillna(0.0)
        elif "cvd_change_1" in data.columns:
            # 使用 cvd_change_1（单根K线，原始值），需要归一化
            if "volume" in data.columns:
                total_vol = data["volume"].replace(0, np.nan)
                order_flow_delta_normalized = (data["cvd_change_1"] / total_vol).fillna(0.0)
            else:
                order_flow_delta_normalized = data["cvd_change_1"]
            data["order_flow_delta"] = order_flow_delta_normalized.shift(1).fillna(0.0)
        elif "delta" in data.columns:
            # 使用 delta，需要归一化
            if "volume" in data.columns:
                total_vol = data["volume"].replace(0, np.nan)
                order_flow_delta_normalized = (data["delta"] / total_vol).fillna(0.0)
            else:
                order_flow_delta_normalized = data["delta"]
            data["order_flow_delta"] = order_flow_delta_normalized.shift(1).fillna(0.0)
        elif "buy_qty" in data.columns and "sell_qty" in data.columns:
            # 从 buy_qty 和 sell_qty 计算
            order_flow_delta = data["buy_qty"] - data["sell_qty"]
            if "volume" in data.columns:
                total_vol = data["volume"].replace(0, np.nan)
                order_flow_delta_normalized = (order_flow_delta / total_vol).fillna(0.0)
            else:
                order_flow_delta_normalized = order_flow_delta
            data["order_flow_delta"] = order_flow_delta_normalized.shift(1).fillna(0.0)
        else:
            # 如果没有订单流数据，使用0
            data["order_flow_delta"] = pd.Series(0.0, index=data.index)
        
        # B.3. 突破速度（breakout_speed）
        # 突破K线实体/影线比
        breakout_speed = pd.Series(0.0, index=data.index, dtype=float)
        for i in range(1, len(data)):
            high = data["high"].iloc[i]
            low = data["low"].iloc[i]
            open_price = data["open"].iloc[i]
            close = data["close"].iloc[i]
            
            # 实体大小
            body = abs(close - open_price)
            # 上影线
            upper_shadow = high - max(open_price, close)
            # 下影线
            lower_shadow = min(open_price, close) - low
            # 总影线
            total_shadow = upper_shadow + lower_shadow
            
            # 突破速度 = 实体 / (实体 + 影线)
            if body + total_shadow > 0:
                speed = body / (body + total_shadow)
            else:
                speed = 0.0
            
            breakout_speed.iloc[i] = speed
        
        data["breakout_speed"] = breakout_speed.shift(1).fillna(0.0)
        
        # C. 动能持续性特征
        # 需要检测突破事件，然后计算后续K线的表现
        follow_through_1 = pd.Series(0.0, index=data.index, dtype=float)
        follow_through_2 = pd.Series(0.0, index=data.index, dtype=float)
        momentum_decay = pd.Series(0.0, index=data.index, dtype=float)
        
        # 检测突破事件（相对于最近SR）
        if "dist_to_nearest_sr" in data.columns and len(boundaries) > 0:
            # 找到最近的SR边界
            nearest_sr = pd.Series(index=data.index, dtype=float)
            for boundary in boundaries:
                col = boundary["column"]
                if col in data.columns:
                    if nearest_sr.isna().all():
                        nearest_sr = data[col]
                    else:
                        # 选择距离当前价格更近的边界
                        current_price = data["close"]
                        dist1 = abs(nearest_sr - current_price)
                        dist2 = abs(data[col] - current_price)
                        nearest_sr = np.where(dist2 < dist1, data[col], nearest_sr)
            
            for i in range(3, len(data)):
                if pd.isna(nearest_sr.iloc[i]):
                    continue
                
                sr_price = nearest_sr.iloc[i]
                prev_close = data["close"].iloc[i - 1]
                curr_close = data["close"].iloc[i]
                curr_high = data["high"].iloc[i]
                curr_low = data["low"].iloc[i]
                
                # 检测突破方向
                breakout_direction = 0
                if prev_close <= sr_price and curr_high > sr_price:
                    breakout_direction = 1  # 向上突破
                elif prev_close >= sr_price and curr_low < sr_price:
                    breakout_direction = -1  # 向下突破
                
                if breakout_direction != 0:
                    # C.1. follow_through_1: 第2根K线是否继续新高/新低
                    if i + 1 < len(data):
                        next_high = data["high"].iloc[i + 1]
                        next_low = data["low"].iloc[i + 1]
                        if breakout_direction == 1:
                            # 向上突破：第2根K线是否创新高
                            follow_through_1.iloc[i + 1] = 1.0 if next_high > curr_high else 0.0
                        else:
                            # 向下突破：第2根K线是否创新低
                            follow_through_1.iloc[i + 1] = 1.0 if next_low < curr_low else 0.0
                    
                    # C.2. follow_through_2: 第3根K线是否站稳
                    if i + 2 < len(data):
                        third_close = data["close"].iloc[i + 2]
                        if breakout_direction == 1:
                            # 向上突破：第3根K线收盘价是否仍在SR上方
                            follow_through_2.iloc[i + 2] = 1.0 if third_close > sr_price else 0.0
                        else:
                            # 向下突破：第3根K线收盘价是否仍在SR下方
                            follow_through_2.iloc[i + 2] = 1.0 if third_close < sr_price else 0.0
                    
                    # C.3. momentum_decay: 突破后3根K线的斜率变化
                    if i + 3 < len(data):
                        # 计算突破后3根K线的价格变化
                        prices_after = data["close"].iloc[i + 1 : i + 4].values
                        if len(prices_after) == 3 and all(~np.isnan(prices_after)):
                            # 计算斜率（使用线性回归）
                            x = np.array([1, 2, 3])
                            y = prices_after
                            slope = np.polyfit(x, y, 1)[0]
                            
                            # 归一化斜率（除以ATR）
                            current_atr = data["atr"].iloc[i] if not pd.isna(data["atr"].iloc[i]) else 1.0
                            normalized_slope = slope / current_atr if current_atr > 0 else 0.0
                            
                            # 动能衰减 = 1 - abs(斜率)（斜率越小，衰减越大）
                            momentum_decay.iloc[i + 3] = 1.0 - min(abs(normalized_slope), 1.0)
        
        data["follow_through_1"] = follow_through_1.shift(1).fillna(0.0)
        data["follow_through_2"] = follow_through_2.shift(1).fillna(0.0)
        data["momentum_decay"] = momentum_decay.shift(1).fillna(0.0)
        
        # D.2. 趋势强度（trend_strength）
        # 使用 ADX(14) 或 MA50 斜率
        if "close" in data.columns:
            # 计算 MA50
            ma50 = data["close"].rolling(window=50, min_periods=1).mean()
            
            # 计算 MA50 斜率（使用线性回归）
            trend_strength = pd.Series(0.0, index=data.index, dtype=float)
            for i in range(50, len(data)):
                if i >= 14:
                    ma_window = ma50.iloc[i - 13 : i + 1].values
                    if len(ma_window) == 14 and all(~np.isnan(ma_window)):
                        x = np.arange(14)
                        slope = np.polyfit(x, ma_window, 1)[0]
                        # 归一化斜率（除以当前价格）
                        current_price = data["close"].iloc[i]
                        normalized_slope = slope / current_price if current_price > 0 else 0.0
                        trend_strength.iloc[i] = normalized_slope * 100  # 放大100倍便于观察
            
            # 如果可以使用 TA-Lib，优先使用 ADX
            try:
                import talib
                high = data["high"].values
                low = data["low"].values
                close = data["close"].values
                adx = talib.ADX(high, low, close, timeperiod=14)
                # ADX 范围是 0-100，归一化到 [0, 1]
                trend_strength = pd.Series(adx / 100.0, index=data.index)
            except Exception:
                pass  # 如果 TA-Lib 不可用，使用 MA50 斜率
        
        data["trend_strength"] = trend_strength.shift(1).fillna(0.0)
        
        return data

    @staticmethod
    def _compute_boundary_volume_confirmations(
        data: pd.DataFrame,
        boundaries: List[Dict[str, str]],
        lookback: int = 20,
        vol_threshold: float = 1.5,
        confirmation_bars: int = 3,
    ) -> Dict[str, pd.Series]:
        """计算每个边界的量价配合度序列"""
        confirmations: Dict[str, pd.Series] = {}
        if not boundaries:
            return confirmations

        for boundary in boundaries:
            column = boundary["column"]
            sr_type = boundary["type"]
            sr_series = data[column]
            conf = pd.Series(0.0, index=data.index, dtype=float)

            # 【关键修复】：在时刻 i，只能使用历史数据 [0, i] 来检测突破和计算确认
            # 不能使用未来数据来确认是否站稳，因为这会导致数据泄漏
            # 解决方案：在突破发生后的 confirmation_bars 根K线之后，才计算确认分数
            # 这样在时刻 i，我们使用的是历史突破事件（发生在 i - confirmation_bars 之前）的确认结果
            for i in range(lookback + confirmation_bars, len(data)):
                # 检测突破：使用历史数据（i - confirmation_bars 时刻的突破）
                # 这样在时刻 i，我们计算的是历史突破的确认结果
                breakout_check_idx = i - confirmation_bars
                if breakout_check_idx < 0:
                    continue

                sr_price = sr_series.iloc[breakout_check_idx - 1]
                if pd.isna(sr_price):
                    continue

                prev_close = data["close"].iloc[breakout_check_idx - 1]
                curr_close = data["close"].iloc[breakout_check_idx]
                breakout = False
                direction = 0

                if (
                    sr_type == "resistance"
                    and prev_close <= sr_price
                    and curr_close > sr_price
                ):
                    breakout = True
                    direction = 1
                elif (
                    sr_type == "support"
                    and prev_close >= sr_price
                    and curr_close < sr_price
                ):
                    breakout = True
                    direction = -1
                elif sr_type == "mid":
                    if (curr_close - sr_price) * (prev_close - sr_price) <= 0:
                        breakout = True
                        direction = 1 if curr_close > sr_price else -1

                if breakout:
                    # 【修复】：只使用历史数据 [0, i] 来计算确认
                    # 在时刻 i，我们已经有了 [breakout_check_idx, i] 的数据来确认是否站稳
                    try:
                        score = (
                            BaselineFeatureEngineer.calculate_volume_price_confirmation(
                                data.iloc[: i + 1],  # 只使用历史数据，不包含未来
                                breakout_check_idx,  # 突破发生在 breakout_check_idx
                                sr_price,
                                lookback=lookback,
                                vol_threshold=vol_threshold,
                                confirmation_bars=confirmation_bars,
                                sr_type=(
                                    sr_type
                                    if sr_type != "mid"
                                    else ("resistance" if direction == 1 else "support")
                                ),
                            )
                        )
                    except Exception:
                        score = 0.0
                    conf.iloc[i] = score

            confirmations[f"volume_price_confirmation_{boundary['name']}"] = conf.shift(
                1
            ).fillna(0.0)

        return confirmations

    @staticmethod
    def _add_price_action_features(
        data: pd.DataFrame,
        boundaries: List[Dict[str, str]],
        compression_series: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        添加价格行为特征（Price Action Features）

        包括：
        1. 突破状态（相对于最近SR）
        2. 反转信号（未到SR就回头）
        3. 假突破迹象
        4. SR结构特征（SR密度）
        5. 多时间框架上下文（趋势、压缩、波动率）
        """
        if data.empty or not boundaries:
            return data

        # 确保有必要的列
        if "atr" not in data.columns:
            data["atr"] = BaselineFeatureEngineer._compute_atr(data)

        # 1. 找到最近的SR边界
        # 收集所有边界价格
        all_boundary_prices = []
        for boundary in boundaries:
            col = boundary["column"]
            if col in data.columns:
                all_boundary_prices.append(data[col])

        if not all_boundary_prices:
            return data

        # 合并所有边界价格，找到最近的非NaN值
        boundary_df = pd.concat(all_boundary_prices, axis=1)

        # 计算到最近边界的距离和方向
        nearest_sr = pd.Series(index=data.index, dtype=float)
        dist_to_sr = pd.Series(index=data.index, dtype=float)
        direction_to_sr = pd.Series(index=data.index, dtype=float)
        nearest_sr_type = pd.Series(index=data.index, dtype=str)

        for i in range(len(data)):
            # 获取当前时刻所有非NaN的边界价格
            valid_boundaries = boundary_df.iloc[i].dropna()
            if len(valid_boundaries) == 0:
                continue

            current_price = data["close"].iloc[i]
            # 找到最近的边界
            distances = abs(valid_boundaries - current_price)
            nearest_idx = distances.idxmin()
            nearest_price = valid_boundaries[nearest_idx]

            nearest_sr.iloc[i] = nearest_price
            dist_to_sr.iloc[i] = (
                (current_price - nearest_price) / current_price
                if current_price > 0
                else 0.0
            )
            direction_to_sr.iloc[i] = 1.0 if current_price < nearest_price else -1.0

            # 找到对应的边界类型
            for boundary in boundaries:
                if boundary["column"] == nearest_idx:
                    nearest_sr_type.iloc[i] = boundary["type"]
                    break

        # 2. 突破状态（相对于最近SR）
        breakout_status = pd.Series(0, index=data.index, dtype=int)
        prev_close = data["close"].shift(1)

        for i in range(1, len(data)):
            if pd.isna(nearest_sr.iloc[i]) or pd.isna(prev_close.iloc[i]):
                continue

            current_high = data["high"].iloc[i]
            current_low = data["low"].iloc[i]
            current_close = data["close"].iloc[i]
            nearest_sr_price = nearest_sr.iloc[i]
            sr_type = nearest_sr_type.iloc[i]

            if sr_type == "resistance":
                # 刚上破阻力
                if (
                    current_high > nearest_sr_price
                    and prev_close.iloc[i] <= nearest_sr_price
                ):
                    breakout_status.iloc[i] = 1
            elif sr_type == "support":
                # 刚下破支撑
                if (
                    current_low < nearest_sr_price
                    and prev_close.iloc[i] >= nearest_sr_price
                ):
                    breakout_status.iloc[i] = -1
            elif sr_type == "mid":
                # 对于mid类型（如VWAP），检测穿越
                if (
                    current_close > nearest_sr_price
                    and prev_close.iloc[i] <= nearest_sr_price
                ):
                    breakout_status.iloc[i] = 1
                elif (
                    current_close < nearest_sr_price
                    and prev_close.iloc[i] >= nearest_sr_price
                ):
                    breakout_status.iloc[i] = -1

        data["breakout_status"] = breakout_status.shift(1).fillna(0)

        # 3. 反转信号（未到SR就回头）
        price_reversed_before_sr = pd.Series(False, index=data.index, dtype=bool)
        volume_spike_threshold = 1.5

        for i in range(1, len(data)):
            if pd.isna(nearest_sr.iloc[i]) or pd.isna(dist_to_sr.iloc[i]):
                continue

            # 应上涨但回落（距离SR为正，方向为正，但价格下跌）
            if dist_to_sr.iloc[i] > 0 and direction_to_sr.iloc[i] == 1:
                if data["close"].iloc[i] < data["close"].iloc[i - 1]:
                    # 检查成交量是否放大
                    if i >= 20:
                        avg_vol = data["volume"].iloc[i - 20 : i].mean()
                        if (
                            avg_vol > 0
                            and data["volume"].iloc[i] / avg_vol
                            > volume_spike_threshold
                        ):
                            price_reversed_before_sr.iloc[i] = True
            # 应下跌但反弹（距离SR为负，方向为负，但价格上涨）
            elif dist_to_sr.iloc[i] < 0 and direction_to_sr.iloc[i] == -1:
                if data["close"].iloc[i] > data["close"].iloc[i - 1]:
                    # 检查成交量是否放大
                    if i >= 20:
                        avg_vol = data["volume"].iloc[i - 20 : i].mean()
                        if (
                            avg_vol > 0
                            and data["volume"].iloc[i] / avg_vol
                            > volume_spike_threshold
                        ):
                            price_reversed_before_sr.iloc[i] = True

        data["price_reversed_before_sr"] = (
            price_reversed_before_sr.shift(1).fillna(False).astype(int)
        )

        # 4. 假突破迹象（突破后3根K线是否收回？）
        # 【关键修复】：在时刻 i，只能使用历史数据来判断假突破
        # 解决方案：在时刻 i，检查发生在 i - 3 的突破是否在后续被收回
        fake_breakout = pd.Series(False, index=data.index, dtype=bool)

        for i in range(3, len(data)):
            # 检查发生在 i - 3 的突破是否在后续被收回
            check_idx = i - 3
            if check_idx < 0 or breakout_status.iloc[check_idx] == 0:
                continue

            nearest_sr_price = nearest_sr.iloc[check_idx]
            if pd.isna(nearest_sr_price):
                continue

            # 检查突破后3根K线是否收回（使用历史数据）
            if breakout_status.iloc[check_idx] == 1:  # 向上突破
                # 如果后续收盘价回到阻力位下方，可能是假突破
                # 在时刻 i，我们已经有了 [check_idx, i] 的数据来判断
                if i >= check_idx + 1:
                    # 检查从 check_idx + 1 到 i 的收盘价是否回到阻力位下方
                    if (
                        data["close"].iloc[check_idx + 1 : i + 1] < nearest_sr_price
                    ).any():
                        fake_breakout.iloc[i] = True
            elif breakout_status.iloc[check_idx] == -1:  # 向下突破
                # 如果后续收盘价回到支撑位上方，可能是假突破
                if i >= check_idx + 1:
                    # 检查从 check_idx + 1 到 i 的收盘价是否回到支撑位上方
                    if (
                        data["close"].iloc[check_idx + 1 : i + 1] > nearest_sr_price
                    ).any():
                        fake_breakout.iloc[i] = True

        data["fake_breakout"] = fake_breakout.shift(1).fillna(False).astype(int)

        # 5. SR密度（是否处于SR密集区？）
        sr_density = pd.Series(0.0, index=data.index, dtype=float)
        tolerance_window = 0.5  # ATR倍数

        for i in range(len(data)):
            if pd.isna(data["atr"].iloc[i]) or data["atr"].iloc[i] <= 0:
                continue

            current_price = data["close"].iloc[i]
            tolerance = data["atr"].iloc[i] * tolerance_window

            # 计算在当前价格 ± tolerance 范围内的边界数量
            count = 0
            for boundary in boundaries:
                col = boundary["column"]
                if col in data.columns:
                    sr_price = data[col].iloc[i]
                    if pd.notna(sr_price):
                        if abs(sr_price - current_price) <= tolerance:
                            count += 1

            sr_density.iloc[i] = count

        data["sr_density"] = sr_density.shift(1).fillna(0.0)

        # 6. 多时间框架上下文
        # 6.1 趋势方向（基于均线，简化：使用50和200周期均线）
        if "close" in data.columns:
            ma50 = data["close"].rolling(window=50, min_periods=1).mean()
            ma200 = data["close"].rolling(window=200, min_periods=1).mean()
            trend_4h = pd.Series(0, index=data.index, dtype=int)
            trend_4h[ma50 > ma200] = 1
            trend_4h[ma50 < ma200] = -1
            data["trend_context"] = trend_4h.shift(1).fillna(0)

        # 6.2 压缩状态（基于布林带宽度）
        if (
            "bb_upper" in data.columns
            and "bb_lower" in data.columns
            and "bb_middle" in data.columns
        ):
            boll_width = (data["bb_upper"] - data["bb_lower"]) / data[
                "bb_middle"
            ].replace(0, np.nan)
            compression_score = 1.0 / (1.0 + boll_width)
            data["compression_score"] = compression_score.shift(1).fillna(0.0)
        else:
            # 如果没有布林带，使用ATR作为替代
            if "atr" in data.columns and "close" in data.columns:
                atr_normalized = data["atr"] / data["close"].replace(0, np.nan)
                compression_score = 1.0 / (1.0 + atr_normalized * 10)  # 缩放因子
                data["compression_score"] = compression_score.shift(1).fillna(0.0)

        # 6.3 波动率状态
        if "atr" in data.columns:
            atr_20_avg = data["atr"].rolling(window=20, min_periods=1).mean()
            volatility_regime = data["atr"] / atr_20_avg.replace(0, np.nan)
            data["volatility_regime"] = volatility_regime.shift(1).fillna(1.0)

        # 标准化距离特征
        data["dist_to_nearest_sr"] = dist_to_sr.shift(1).fillna(0.0)
        data["direction_to_nearest_sr"] = direction_to_sr.shift(1).fillna(0.0)
        
        # 【新增】：添加4类12个核心特征，用于让模型自动学习突破质量判断
        data = BaselineFeatureEngineer._add_breakout_quality_features(
            data, boundaries
        )

        return data

    @staticmethod
    def compute_bb_width_features(
        df: pd.DataFrame,
        *,
        period: int = 20,
        std_dev: int = 2,
        atr_window: int = 14,
    ) -> pd.DataFrame:
        """计算布林带宽度及其归一化特征。"""
        if "bb_upper" not in df.columns or "bb_lower" not in df.columns:
            upper, middle, lower = BaselineFeatureEngineer.compute_bollinger_bands(
                df["close"], period=period, std_dev=std_dev
            )
            df["bb_upper"] = upper
            df["bb_middle"] = middle
            df["bb_lower"] = lower

        width = (df["bb_upper"] - df["bb_lower"]).abs()
        df["bb_width"] = width

        if "atr" not in df.columns:
            df["atr"] = BaselineFeatureEngineer.compute_atr(
                df["high"], df["low"], df["close"], period=atr_window
            )

        df["bb_width_normalized"] = (
            (width / df["atr"].replace(0, np.nan))
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        return df

    @staticmethod
    def compute_range_ratio_5bar(df: pd.DataFrame) -> pd.DataFrame:
        """计算 5/20 Bar 区间比率 z-score。"""
        if "hl" not in df.columns:
            df["hl"] = df["high"] - df["low"]

        short_range = df["hl"].rolling(5).mean()
        long_range = df["hl"].rolling(20).mean()
        ratio = (short_range / long_range.replace(0, np.nan)).replace(
            [np.inf, -np.inf], np.nan
        )
        ratio = ratio.fillna(1.0)
        ratio_log = np.log1p(ratio)
        mean = ratio_log.rolling(50, min_periods=5).mean()
        std = ratio_log.rolling(50, min_periods=5).std()
        df["range_ratio_5bar"] = (
            ((ratio_log - mean) / std.replace(0, np.nan))
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        return df

    @staticmethod
    def compute_volatility_reversal_score(df: pd.DataFrame) -> pd.DataFrame:
        """ATR 回落 z-score，用于识别波动率反转。"""
        if "atr" not in df.columns:
            df["atr"] = BaselineFeatureEngineer.compute_atr(
                df["high"], df["low"], df["close"]
            )
        atr_mean = df["atr"].rolling(50).mean()
        atr_std = df["atr"].rolling(50).std()
        df["volatility_reversal_score"] = (
            (df["atr"] - atr_mean) / atr_std.replace(0, np.nan)
        ).fillna(0.0)
        return df

    @staticmethod
    def compute_price_range_symmetry(
        df: pd.DataFrame,
        *,
        feature_shift: int = 0,
    ) -> pd.DataFrame:
        """价格区间对称性（高/低/收盘关系），衡量上下影线不对称。"""
        high = df["high"].shift(feature_shift)
        low = df["low"].shift(feature_shift)
        close = df["close"].shift(feature_shift)

        numerator = (high - close)
        denominator = (close - low).replace(0, np.nan)
        raw = (numerator / denominator).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        log_val = np.log1p(np.abs(raw)) * np.sign(raw)
        mean = log_val.rolling(50, min_periods=5).mean()
        std = log_val.rolling(50, min_periods=5).std()
        df["price_range_symmetry"] = (
            ((log_val - mean) / std.replace(0, np.nan))
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        return df

    @staticmethod
    def compute_volume_anomaly(df: pd.DataFrame) -> pd.DataFrame:
        """成交量异常 z-score。"""
        vol_ratio = df["volume"] / df["volume"].ewm(span=20, min_periods=10).mean()
        vol_ratio = vol_ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        vol_log = np.log1p(vol_ratio)
        mean = vol_log.rolling(50, min_periods=10).mean()
        std = vol_log.rolling(50, min_periods=10).std()
        df["volume_anomaly"] = (
            ((vol_log - mean) / std.replace(0, np.nan))
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        return df

    @staticmethod
    def compute_roc_5(df: pd.DataFrame) -> pd.DataFrame:
        """5 Bar ROC z-score。"""
        if "roc_5" in df.columns:
            return df
        roc_raw = df["close"].pct_change(5)
        roc_mean = roc_raw.rolling(window=50, min_periods=5).mean()
        roc_std = roc_raw.rolling(window=50, min_periods=5).std()
        roc_std = roc_std.clip(lower=roc_raw.abs().quantile(0.01))
        df["roc_5"] = (
            ((roc_raw - roc_mean) / roc_std.replace(0, np.nan))
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        return df

    @staticmethod
    def compute_acceleration_3(df: pd.DataFrame, feature_shift: int = 0) -> pd.DataFrame:
        """计算 acceleration_3 特征：ROC(3) 归一化后的差分（动量加速度）。"""
        if "acceleration_3" in df.columns:
            return df

        roc_3 = df["close"].pct_change(3)
        roc_3_mean = roc_3.rolling(window=50, min_periods=5).mean()
        roc_3_std = roc_3.rolling(window=50, min_periods=5).std()
        roc_3_std = roc_3_std.clip(lower=roc_3.abs().quantile(0.01))
        roc_3_norm = (
            ((roc_3 - roc_3_mean) / roc_3_std.replace(0, np.nan))
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        current = roc_3_norm.shift(feature_shift) if feature_shift > 0 else roc_3_norm
        prev = roc_3_norm.shift(feature_shift + 1)
        df["acceleration_3"] = current - prev
        return df

    @staticmethod
    def compute_trend_r2_20(
        df: pd.DataFrame,
        *,
        feature_shift: int = 0,
    ) -> pd.DataFrame:
        """计算 20 Bar 趋势 R²。"""
        df["trend_r2_20"] = BaselineFeatureEngineer._trend_r2(
            df["close"], window=20, lag=feature_shift
        )
        return df

    @staticmethod
    def compute_trend_r2_50(
        df: pd.DataFrame,
        *,
        feature_shift: int = 0,
    ) -> pd.DataFrame:
        """计算 50 Bar 趋势 R²。"""
        df["trend_r2_50"] = BaselineFeatureEngineer._trend_r2(
            df["close"], window=50, lag=feature_shift
        )
        return df

    @staticmethod
    def compute_slope_consistency_score(df: pd.DataFrame) -> pd.DataFrame:
        """多均线斜率一致性（EMA10/20/50）。"""
        ema10 = df["close"].ewm(span=10).mean()
        ema20 = df["close"].ewm(span=20).mean()
        ema50 = df["close"].ewm(span=50).mean()
        slope10 = np.sign(ema10.diff())
        slope20 = np.sign(ema20.diff())
        slope50 = np.sign(ema50.diff())
        df["slope_consistency_score"] = (
            (slope10 == slope20).astype(int)
            + (slope20 == slope50).astype(int)
            + (slope10 == slope50).astype(int)
        )
        return df

    @staticmethod
    def compute_atr_percentile(
        df: pd.DataFrame,
        *,
        window: int = 288,
        shift: int = 1,
    ) -> pd.DataFrame:
        """ATR 百分位（压缩检测）。"""
        if "atr" not in df.columns:
            df["atr"] = BaselineFeatureEngineer.compute_atr(
                df["high"], df["low"], df["close"]
            )
        if "atr_percentile" in df.columns:
            return df

        series = df["atr"].astype(float)
        eps = 1e-9

        def _percentile(arr: np.ndarray) -> float:
            if len(arr) == 0:
                return 0.5
            current = arr[-1]
            return float(np.mean(arr <= current))

        pct = (
            series.rolling(window=window, min_periods=window)
            .apply(_percentile, raw=True)
            .fillna(0.5)
        )
        if shift:
            pct = pct.shift(shift)
        df["atr_percentile"] = pct.clip(0.0, 1.0).fillna(0.5)
        return df

    @staticmethod
    def compute_trend_volatility_alignment(
        df: pd.DataFrame,
        *,
        feature_shift: int = 0,
        atr_percentile_window: int = 288,
    ) -> pd.DataFrame:
        """趋势方向与波动率状态的一致性。"""
        if "roc_5" not in df.columns:
            df = BaselineFeatureEngineer.compute_roc_5(df)
        if "atr_percentile" not in df.columns:
            df = BaselineFeatureEngineer.compute_atr_percentile(
                df, window=atr_percentile_window
            )
        df["trend_volatility_alignment"] = (
            np.sign(df["roc_5"].shift(feature_shift)).fillna(0.0)
            * df["atr_percentile"].fillna(0.0)
        )
        return df

    @staticmethod
    def compute_compression_to_breakout_prob(df: pd.DataFrame) -> pd.DataFrame:
        """压缩持续时间与未来动量的联动。"""
        if "compression_duration" not in df.columns or "roc_5" not in df.columns:
            return df
        df["compression_to_breakout_prob"] = (
            df["compression_duration"].fillna(0.0) * df["roc_5"].fillna(0.0)
        )
        return df
    
    @staticmethod
    def _compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
        """计算 ATR（内部方法，使用类的静态方法）"""
        return BaselineFeatureEngineer.compute_atr(
            df["high"], df["low"], df["close"], period=window
        )

    # ========================================================================
    # 特征添加静态方法
    # ========================================================================

    @staticmethod
    def add_basic_indicators(
        df: pd.DataFrame, required_features: Optional[set] = None
    ) -> pd.DataFrame:
        """
        添加基础技术指标到DataFrame（优化版：支持按需计算）

        Args:
            df: 包含OHLCV数据的DataFrame
            required_features: 需要计算的指标集合，None 表示计算所有
        """
        if df.empty:
            return df

        result = df.copy()

        # 确保所有列都是数值类型
        for col in ["open", "high", "low", "close", "volume"]:
            if col in result.columns:
                result[col] = pd.to_numeric(result[col], errors="coerce")

        result = result.dropna(subset=["open", "high", "low", "close", "volume"])
        if result.empty:
            return result

        # 按需计算 RSI
        if required_features is None or "rsi" in required_features:
            if "rsi" not in result.columns:
                result["rsi"] = BaselineFeatureEngineer.compute_rsi(result["close"])

        # 按需计算 MACD
        need_macd = required_features is None or any(
            f in required_features for f in ["macd", "macd_signal", "macd_histogram"]
        )
        if need_macd and "macd" not in result.columns:
            try:
                macd_line, signal_line, histogram = (
                    BaselineFeatureEngineer.compute_macd(result["close"])
                )
                result["macd"] = macd_line
                result["macd_signal"] = signal_line
                result["macd_histogram"] = histogram
            except Exception as e:
                print(f"Warning: Error computing MACD: {e}")
                result["macd"] = 0
                result["macd_signal"] = 0
                result["macd_histogram"] = 0

        # 按需计算 Bollinger Bands
        need_bb = required_features is None or any(
            f in required_features for f in ["bb_upper", "bb_middle", "bb_lower"]
        )
        if need_bb and "bb_upper" not in result.columns:
            try:
                upper_band, middle_band, lower_band = (
                    BaselineFeatureEngineer.compute_bollinger_bands(result["close"])
                )
                result["bb_upper"] = upper_band
                result["bb_middle"] = middle_band
                result["bb_lower"] = lower_band
            except Exception as e:
                print(f"Warning: Error computing Bollinger Bands: {e}")
                result["bb_upper"] = result["close"]
                result["bb_middle"] = result["close"]
                result["bb_lower"] = result["close"]

        # 按需计算 ATR
        if required_features is None or "atr" in required_features:
            if "atr" not in result.columns:
                try:
                    result["atr"] = BaselineFeatureEngineer.compute_atr(
                        result["high"], result["low"], result["close"]
                    )
                except Exception as e:
                    print(f"Warning: Error computing ATR: {e}")
                    result["atr"] = 0

        # 按需计算 ZigZag
        # 注意：这里只计算 zigzag，不计算高点和低点（高点和低点在 add_zigzag_dimensionless_features 中计算）
        if required_features is None or "zigzag" in required_features:
            if "zigzag" not in result.columns:
                try:
                    result["zigzag"] = BaselineFeatureEngineer.compute_zigzag(
                        result["high"], result["low"], return_high_low=False
                    )
                except Exception as e:
                    print(f"Warning: Error computing ZigZag: {e}")
                    result["zigzag"] = 0

        # 按需计算价格变化和波动率
        if required_features is None or "price_change" in required_features:
            if "price_change" not in result.columns:
                result["price_change"] = result["close"].pct_change()

        if required_features is None or "volatility" in required_features:
            if "volatility" not in result.columns:
                price_change_numeric = pd.to_numeric(
                    result["price_change"], errors="coerce"
                ).astype(float)
                values = talib.STDDEV(
                    price_change_numeric.values, timeperiod=14, nbdev=1
                )
                # 使用 shift(1) 确保时间对齐，避免使用未来信息
                result["volatility"] = pd.Series(
                    values, index=price_change_numeric.index
                ).shift(1)

        # 按需计算成交量特征
        if (
            required_features is None
            or "volume_sma" in required_features
            or "volume_ratio" in required_features
        ):
            if "volume_sma" not in result.columns:
                volume_numeric = pd.to_numeric(
                    result["volume"], errors="coerce"
                ).astype(float)
                values = talib.SMA(volume_numeric.values, timeperiod=20)
                result["volume_sma"] = pd.Series(values, index=volume_numeric.index)
            if "volume_ratio" not in result.columns:
                result["volume_ratio"] = result["volume"] / result[
                    "volume_sma"
                ].replace(0, np.nan)

        return result

    @staticmethod
    def ensure_basic_indicators(
        df: pd.DataFrame, required_features: Optional[set] = None
    ) -> pd.DataFrame:
        """确保基础指标存在（优化版：支持按需计算）"""
        if df.empty:
            return df

        # 检查需要的指标是否都已存在
        if required_features:
            missing = required_features - set(df.columns)
            if not missing:
                return df

        return BaselineFeatureEngineer.add_basic_indicators(df, required_features)

    @staticmethod
    def add_zigzag_dimensionless_features(
        df: pd.DataFrame, required_features: Optional[set] = None
    ) -> pd.DataFrame:
        """
        添加 ZigZag 相关的无量纲特征

        新增特征：
        - price_to_zz_high_pct: 当前价格到最近 ZigZag 高点的相对距离
        - price_to_zz_low_pct: 当前价格到最近 ZigZag 低点的相对距离
        - zz_amplitude_pct: ZigZag 波幅（相对）
        - zz_duration: ZigZag 持续时间（bar 数，无量纲）
        - zz_slope: ZigZag 斜率（归一化）
        """
        if df.empty:
            return df

        result = df.copy()

        # 确保 zigzag 存在
        if "zigzag" not in result.columns:
            if required_features and any(
                "zz_" in f or "zigzag" in f for f in required_features
            ):
                result = BaselineFeatureEngineer.ensure_basic_indicators(
                    result, {"zigzag"}
                )
            else:
                return result

        # 确保 atr 存在（zz_slope 需要 atr）
        if "atr" not in result.columns:
            if required_features and "zz_slope" in required_features:
                result = BaselineFeatureEngineer.ensure_basic_indicators(
                    result, {"atr"}
                )
            elif required_features is None:
                # 如果没有指定 required_features，也确保 atr 存在
                result = BaselineFeatureEngineer.ensure_basic_indicators(
                    result, {"atr"}
                )

        close = result["close"].replace(0, np.nan)

        # 确定使用的价格序列（优先使用 WPT 中高频重构价格）
        price_series = None
        if "wpt_price_reconstructed" in result.columns:
            # 自动检测 WPT 重构价格（中高频，保留关键拐点）
            price_series = result["wpt_price_reconstructed"]
        
        # 优化：直接计算 zigzag + 高点和低点（一次性完成）
        # 如果 zigzag 已存在，重新计算以确保高点和低点正确（性能影响可忽略）
        zigzag, zz_high, zz_low = BaselineFeatureEngineer.compute_zigzag(
            result["high"], result["low"], return_high_low=True, price_col=price_series
        )
        result["zigzag"] = zigzag
        result["zz_high_value"] = zz_high
        result["zz_low_value"] = zz_low

        # 1. 当前价格距离最近 ZigZag 高/低点的相对距离
        if required_features is None or "price_to_zz_high_pct" in required_features:
            if "price_to_zz_high_pct" not in result.columns:
                result["price_to_zz_high_pct"] = ((zz_high - close) / close).replace(
                    [np.inf, -np.inf], np.nan
                )

        if required_features is None or "price_to_zz_low_pct" in required_features:
            if "price_to_zz_low_pct" not in result.columns:
                result["price_to_zz_low_pct"] = ((close - zz_low) / close).replace(
                    [np.inf, -np.inf], np.nan
                )

        # 2. ZigZag 波幅（相对）
        if required_features is None or "zz_amplitude_pct" in required_features:
            if "zz_amplitude_pct" not in result.columns:
                zz_low_safe = zz_low.replace(0, np.nan)
                result["zz_amplitude_pct"] = ((zz_high - zz_low) / zz_low_safe).replace(
                    [np.inf, -np.inf], np.nan
                )

        # 3. ZigZag 持续时间（从上一个转折点至今的 bar 数）
        if required_features is None or "zz_duration" in required_features:
            if "zz_duration" not in result.columns:
                zigzag_diff = zigzag.diff()
                turn_points = (zigzag_diff * zigzag_diff.shift(1) < 0) | (
                    (zigzag_diff != 0) & (zigzag_diff.shift(1) == 0)
                )

                duration = pd.Series(index=zigzag.index, dtype=float)
                last_turn_idx = 0
                for i in range(len(zigzag)):
                    if turn_points.iloc[i]:
                        last_turn_idx = i
                    duration.iloc[i] = i - last_turn_idx
                result["zz_duration"] = duration.fillna(0.0)

        # 4. ZigZag 斜率（归一化）
        if required_features is None or "zz_slope" in required_features:
            if "zz_slope" not in result.columns:
                if "atr" not in result.columns:
                    raise ValueError(
                        "ATR is required for computing zz_slope. "
                        "Please ensure 'atr' is computed before calling this function."
                    )

                window = 5
                zz_slope_raw = zigzag.diff(window) / window

                # 归一化：使用 ATR
                atr_safe = result["atr"].replace(0, np.nan)
                result["zz_slope"] = (zz_slope_raw / atr_safe).replace(
                    [np.inf, -np.inf], np.nan
                )

        return result

    @staticmethod
    def add_poc_hal_dimensionless_features(
        df: pd.DataFrame, 
        required_features: Optional[set] = None, 
        poc_window: int = 160,
        price_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        添加 POC (Point of Control) 和 HAL (Value Area 70% 价格区间的上下界) 相关的无量纲特征
        
        ✅ 强烈建议：使用 WPT 低频重构价格（price_col='wpt_price_reconstructed'）
        这样可以过滤高频噪声，使 POC/HAL 更接近真实供需平衡点。

        注意：POC 和 HAL 的计算合并在一起，因为它们都基于相同的 volume profile 计算，
        避免重复计算浪费性能。

        POC 相关特征：
        - price_to_poc_pct: 当前价格到 POC 的相对距离
        - poc_position_ratio: POC 在价格区间中的位置（0-1）
        - poc_volume_ratio: POC 位置的成交量占比

        HAL 相关特征：
        - price_to_hal_high_pct: 当前价格到 HAL 高点的相对距离
        - price_to_hal_low_pct: 当前价格到 HAL 低点的相对距离
        - price_to_hal_mid_pct: 当前价格到 HAL 中点的相对距离
        - hal_bandwidth_pct: HAL 带宽（相对）
        
        Args:
            df: 输入 DataFrame
            required_features: 需要的特征集合（可选）
            poc_window: POC 计算窗口大小
            price_col: 可选的价格列名（如 'wpt_price_reconstructed'）。如果提供，将使用此列
                       而非原始 high/low。默认 None，使用原始价格（向后兼容）
        """
        if df.empty:
            return df

        result = df.copy()

        # 检查是否需要计算 POC 或 HAL
        need_poc = required_features is None or any(
            "poc" in f for f in required_features
        )
        need_hal = required_features is None or any(
            "hal" in f for f in required_features
        )

        if not (need_poc or need_hal):
            return result

        # 确定使用的价格序列
        # 优先使用 WPT 低频重构价格，如果不存在则使用原始价格
        price_series = None
        if price_col and price_col in result.columns:
            price_series = result[price_col]
        elif "wpt_price_reconstructed" in result.columns:
            # 自动检测 WPT 重构价格
            price_series = result["wpt_price_reconstructed"]
        
        # 计算 POC 和 HAL（一次性计算，避免重复）
        need_compute = False
        if need_poc and (
            "poc" not in result.columns or "poc_volume_ratio" not in result.columns
        ):
            need_compute = True
        if need_hal and (
            "hal_high" not in result.columns or "hal_low" not in result.columns
        ):
            need_compute = True

        if need_compute:
            poc, poc_volume_ratio, hal_high, hal_low = (
                BaselineFeatureEngineer.compute_poc(
                    result["high"], 
                    result["low"], 
                    result["volume"], 
                    window=poc_window,
                    price_col=price_series,  # 传入 WPT 重构价格（如果存在）
                )
            )
            result["poc"] = poc
            result["poc_volume_ratio"] = poc_volume_ratio
            result["hal_high"] = hal_high
            result["hal_low"] = hal_low
            result["hal_mid"] = (hal_high + hal_low) / 2.0

        close = result["close"].replace(0, np.nan)
        poc = result["poc"]
        hal_high = result["hal_high"]
        hal_low = result["hal_low"]
        hal_mid = result["hal_mid"]
        high = result["high"]
        low = result["low"]

        # ========== POC 相关特征 ==========
        # 1. 当前价格到 POC 的相对距离
        if required_features is None or "price_to_poc_pct" in required_features:
            if "price_to_poc_pct" not in result.columns:
                result["price_to_poc_pct"] = ((poc - close) / close).replace(
                    [np.inf, -np.inf], np.nan
                )

        # 2. POC 在价格区间中的位置（0-1）
        if required_features is None or "poc_position_ratio" in required_features:
            if "poc_position_ratio" not in result.columns:
                price_range = (high - low).replace(0, np.nan)
                result["poc_position_ratio"] = (
                    ((poc - low) / price_range)
                    .replace([np.inf, -np.inf], np.nan)
                    .clip(0.0, 1.0)
                )

        # ========== HAL 相关特征 ==========
        # 1. 当前价格到 HAL 的相对距离
        if required_features is None or "price_to_hal_high_pct" in required_features:
            if "price_to_hal_high_pct" not in result.columns:
                result["price_to_hal_high_pct"] = ((hal_high - close) / close).replace(
                    [np.inf, -np.inf], np.nan
                )

        if required_features is None or "price_to_hal_low_pct" in required_features:
            if "price_to_hal_low_pct" not in result.columns:
                result["price_to_hal_low_pct"] = ((close - hal_low) / close).replace(
                    [np.inf, -np.inf], np.nan
                )

        if required_features is None or "price_to_hal_mid_pct" in required_features:
            if "price_to_hal_mid_pct" not in result.columns:
                # 使用 shift(1) 确保时间对齐，避免使用未来信息
                result["price_to_hal_mid_pct"] = (
                    ((hal_mid - close) / close)
                    .replace([np.inf, -np.inf], np.nan)
                    .shift(1)
                )

        # 2. HAL 带宽（相对）
        if required_features is None or "hal_bandwidth_pct" in required_features:
            if "hal_bandwidth_pct" not in result.columns:
                hal_mid_safe = hal_mid.replace(0, np.nan)
                result["hal_bandwidth_pct"] = (
                    (hal_high - hal_low) / hal_mid_safe
                ).replace([np.inf, -np.inf], np.nan)

        return result

    @staticmethod
    def add_swing_dimensionless_features(
        df: pd.DataFrame,
        required_features: Optional[set] = None,
        swing_win_short: int = 20,
        swing_win_long: int = 60,
        price_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        添加 Swing High/Low 相关的无量纲特征
        
        ✅ 建议：使用 WPT 中频重构价格（price_col='wpt_price_reconstructed'）
        这样可以捕捉中期结构，同时过滤高频噪声。

        新增特征：
        - swing_high_pct_close: Swing High 相对收盘价的比率
        - swing_low_pct_close: Swing Low 相对收盘价的比率
        - swing_amplitude_pct: Swing 波幅（相对）
        
        Args:
            df: 输入 DataFrame
            required_features: 需要的特征集合（可选）
            swing_win_short: 短期 Swing 窗口大小
            swing_win_long: 长期 Swing 窗口大小
            price_col: 可选的价格列名（如 'wpt_price_reconstructed'）。如果提供，将使用此列
                       而非原始 high/low。默认 None，使用原始价格（向后兼容）
        """
        if df.empty:
            return df

        result = df.copy()

        close = result["close"].replace(0, np.nan)

        # 确定使用的价格序列（优先使用 WPT 中频重构价格）
        swing_price = None
        if price_col and price_col in result.columns:
            swing_price = result[price_col]
        elif "wpt_price_reconstructed" in result.columns:
            # 自动检测 WPT 重构价格（中频，捕捉中期结构）
            swing_price = result["wpt_price_reconstructed"]

        # 计算 Swing High/Low（如果不存在）
        if "roll_high_s" not in result.columns:
            if required_features and any("swing" in f for f in required_features):
                if swing_price is not None:
                    # 使用 WPT 重构价格
                    result["roll_high_s"] = (
                        swing_price.rolling(swing_win_short, min_periods=1).max()
                    )
                    result["roll_low_s"] = (
                        swing_price.rolling(swing_win_short, min_periods=1).min()
                    )
                    result["roll_high_l"] = (
                        swing_price.rolling(swing_win_long, min_periods=1).max()
                    )
                    result["roll_low_l"] = (
                        swing_price.rolling(swing_win_long, min_periods=1).min()
                    )
                else:
                    # 使用原始价格
                    result["roll_high_s"] = (
                        result["high"].rolling(swing_win_short, min_periods=1).max()
                    )
                    result["roll_low_s"] = (
                        result["low"].rolling(swing_win_short, min_periods=1).min()
                    )
                    result["roll_high_l"] = (
                        result["high"].rolling(swing_win_long, min_periods=1).max()
                    )
                    result["roll_low_l"] = (
                        result["low"].rolling(swing_win_long, min_periods=1).min()
                    )
            else:
                return result

        # 1. Swing High/Low 相对收盘价的比率
        if required_features is None or "swing_high_pct_close" in required_features:
            if "swing_high_pct_close" not in result.columns:
                result["swing_high_pct_close"] = (
                    (result["roll_high_s"] - close) / close.replace(0, np.nan)
                ).replace([np.inf, -np.inf], np.nan)

        if required_features is None or "swing_low_pct_close" in required_features:
            if "swing_low_pct_close" not in result.columns:
                result["swing_low_pct_close"] = (
                    (close - result["roll_low_s"]) / close.replace(0, np.nan)
                ).replace([np.inf, -np.inf], np.nan)

        # 2. Swing 波幅（相对）
        if required_features is None or "swing_amplitude_pct" in required_features:
            if "swing_amplitude_pct" not in result.columns:
                roll_low_s_safe = result["roll_low_s"].replace(0, np.nan)
                result["swing_amplitude_pct"] = (
                    (result["roll_high_s"] - result["roll_low_s"]) / roll_low_s_safe
                ).replace([np.inf, -np.inf], np.nan)

        return result

    @staticmethod
    def add_ols_channel_features(
        df: pd.DataFrame,
        required_features: Optional[set] = None,
        window: int = 96,
    ) -> pd.DataFrame:
        """
        添加 OLS 通道特征（线性回归通道）

        生成：
        - ols_channel_mid：OLS 拟合的中心线
        - ols_channel_upper / lower：中心线 ± 残差标准差
        - ols_channel_width：通道宽度
        """
        if df.empty or "close" not in df.columns:
            return df

        need_channel = required_features is None or any(
            key in (required_features or set())
            for key in {
                "ols_channel_mid",
                "ols_channel_upper",
                "ols_channel_lower",
                "ols_channel_width",
            }
        )
        if not need_channel:
            return df

        result = df.copy()
        close = result["close"].astype(float)
        mid = pd.Series(np.nan, index=result.index, dtype=float)
        upper = pd.Series(np.nan, index=result.index, dtype=float)
        lower = pd.Series(np.nan, index=result.index, dtype=float)
        width = pd.Series(np.nan, index=result.index, dtype=float)

        x = np.arange(window)
        for i in range(window, len(result)):
            window_slice = close.iloc[i - window : i]
            if window_slice.isna().any():
                continue
            try:
                slope, intercept = np.polyfit(x, window_slice.values, 1)
                fitted = slope * x + intercept
                mid_val = slope * (window - 1) + intercept
                resid = window_slice.values - fitted
                resid_std = np.std(resid)
                mid.iloc[i] = mid_val
                upper.iloc[i] = mid_val + resid_std
                lower.iloc[i] = mid_val - resid_std
                width.iloc[i] = 2 * resid_std
            except Exception:
                continue

        result["ols_channel_mid"] = mid.ffill()
        result["ols_channel_upper"] = upper.ffill()
        result["ols_channel_lower"] = lower.ffill()
        result["ols_channel_width"] = width.ffill()

        return result

    @staticmethod
    def add_price_volume_relative_features(
        df: pd.DataFrame, required_features: Optional[set] = None
    ) -> pd.DataFrame:
        """
        添加基础价格与量能相对变化特征

        新增特征：
        - ret_1h, ret_4h, ret_24h: 对数收益率（1小时、4小时、24小时）
        - rv_4h, rv_24h: 已实现波动率
        - vol_ma_ratio: 成交量移动平均比率
        - vol_zscore: 成交量 Z-score
        """
        if df.empty:
            return df

        result = df.copy()
        close = result["close"].replace(0, np.nan)
        volume = result["volume"]

        # 1. 对数收益率（常用）
        # 注意：这里假设数据是 5 分钟 K 线，1h=12根，4h=48根，24h=288根
        # 实际应该根据时间框架动态计算
        periods_1h = 12  # 假设 5 分钟 K 线
        periods_4h = 48
        periods_24h = 288

        if required_features is None or "ret_1h" in required_features:
            if "ret_1h" not in result.columns:
                # 使用 shift(1) 确保时间对齐，避免使用未来信息
                result["ret_1h"] = np.log(close / close.shift(periods_1h)).shift(1)

        if required_features is None or "ret_4h" in required_features:
            if "ret_4h" not in result.columns:
                # 使用 shift(1) 确保时间对齐，避免使用未来信息
                result["ret_4h"] = np.log(close / close.shift(periods_4h)).shift(1)

        if required_features is None or "ret_24h" in required_features:
            if "ret_24h" not in result.columns:
                # 使用 shift(1) 确保时间对齐，避免使用未来信息
                result["ret_24h"] = np.log(close / close.shift(periods_24h)).shift(1)

        # 2. 已实现波动率（基于 ret_1h）
        if required_features is None or "rv_4h" in required_features:
            if "rv_4h" not in result.columns and "ret_1h" in result.columns:
                result["rv_4h"] = (
                    result["ret_1h"]
                    .rolling(window=periods_4h // periods_1h, min_periods=1)
                    .std()
                )

        if required_features is None or "rv_24h" in required_features:
            if "rv_24h" not in result.columns and "ret_1h" in result.columns:
                result["rv_24h"] = (
                    result["ret_1h"]
                    .rolling(window=periods_24h // periods_1h, min_periods=1)
                    .std()
                )

        # 3. 成交量异常度
        if required_features is None or "vol_ma_ratio" in required_features:
            if "vol_ma_ratio" not in result.columns:
                vol_ma = volume.rolling(
                    window=periods_24h, min_periods=periods_24h
                ).mean()
                vol_ma_ratio = (
                    (volume / vol_ma.replace(0, np.nan))
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(1.0)
                )
                # 滚动统计衍生特征统一 shift(1)，确保在 t 时刻仅使用 t-1 及之前的数据
                result["vol_ma_ratio"] = vol_ma_ratio.shift(1)

        if required_features is None or "vol_zscore" in required_features:
            if "vol_zscore" not in result.columns:
                result["vol_zscore"] = BaselineFeatureEngineer._rolling_zscore(
                    volume.astype(float),
                    window=periods_24h,
                    min_periods=periods_24h,
                )

        return result

    @staticmethod
    def add_common_derived_features(
        df: pd.DataFrame,
        required_features: Optional[set] = None,
        rolling_zscore_windows: List[int] = None,
    ) -> pd.DataFrame:
        """
        添加常用衍生特征（优化版：支持按需计算，不强制计算所有基础指标）
        """
        if df.empty:
            return df

        result = df.copy()
        close = result["close"].replace(0, np.nan)

        # 解析依赖关系：确定需要哪些基础指标
        needed_basic = set()
        if required_features:
            # 分析需要哪些基础指标
            if any("rsi" in f for f in required_features):
                needed_basic.add("rsi")
            if any("macd" in f for f in required_features):
                needed_basic.update(["macd", "macd_signal", "macd_histogram"])
            if any("bb_" in f for f in required_features):
                needed_basic.update(["bb_upper", "bb_lower", "bb_middle"])
            if any("atr" in f for f in required_features):
                needed_basic.add("atr")
        else:
            # 如果没有指定，只确保必要的基础指标
            needed_basic = {"rsi", "atr"}  # 最小集合

        # 按需计算基础指标
        if needed_basic:
            result = BaselineFeatureEngineer.ensure_basic_indicators(
                result, needed_basic
            )

        # 只在需要时计算特征
        if not required_features or "returns" in required_features:
            if "returns" not in result.columns:
                result["returns"] = close.pct_change()

        if not required_features or "log_returns" in required_features:
            if "log_returns" not in result.columns:
                shifted = close.shift(1).replace(0, np.nan)
                result["log_returns"] = np.log(close / shifted)

        if not required_features or "price_change" in required_features:
            if "price_change" not in result.columns:
                result["price_change"] = close.diff()

        if not required_features or "volatility" in required_features:
            if "volatility" not in result.columns:
                if "returns" in result.columns:
                    returns_numeric = pd.to_numeric(
                        result["returns"], errors="coerce"
                    ).astype(float)
                    values = talib.STDDEV(
                        returns_numeric.values, timeperiod=20, nbdev=1
                    )
                    result["volatility"] = pd.Series(
                        values, index=returns_numeric.index
                    )
                else:
                    price_change = close.pct_change()
                    price_change_numeric = pd.to_numeric(
                        price_change, errors="coerce"
                    ).astype(float)
                    values = talib.STDDEV(
                        price_change_numeric.values, timeperiod=20, nbdev=1
                    )
                    result["volatility"] = pd.Series(
                        values, index=price_change_numeric.index
                    )

        # BB 相关特征
        if {"bb_upper", "bb_lower"}.issubset(result.columns):
            if not required_features or "bb_position" in required_features:
                if "bb_position" not in result.columns:
                    denom = (result["bb_upper"] - result["bb_lower"]).replace(0, np.nan)
                    result["bb_position"] = (
                        ((close - result["bb_lower"]) / denom)
                        .replace([np.inf, -np.inf], np.nan)
                        .fillna(0.5)
                    )

            if not required_features or "bb_width" in required_features:
                if "bb_width" not in result.columns:
                    # bb_width 归一化：除以 bb_middle（布林带中线）或 close
                    # 这样消除量纲，使其成为相对值
                    bb_width_raw = (result["bb_upper"] - result["bb_lower"]).abs()
                    if "bb_middle" in result.columns:
                        bb_middle_safe = result["bb_middle"].replace(0, np.nan)
                        result["bb_width"] = (bb_width_raw / bb_middle_safe).replace(
                            [np.inf, -np.inf], np.nan
                        )
                    else:
                        # 如果没有 bb_middle，使用 close
                        close_safe = close.replace(0, np.nan)
                        result["bb_width"] = (bb_width_raw / close_safe).replace(
                            [np.inf, -np.inf], np.nan
                        )

        # 归一化特征（价格归一化，保留用于向后兼容）
        # 注意：RSI 本身就是 0~100 的标准范围，不需要归一化
        if not required_features or "macd_normalized" in required_features:
            if "macd_normalized" not in result.columns and "macd" in result.columns:
                result["macd_normalized"] = (
                    result["macd"] / close.replace(0, np.nan)
                ).replace([np.inf, -np.inf], np.nan)

        if not required_features or "atr_normalized" in required_features:
            if "atr_normalized" not in result.columns and "atr" in result.columns:
                result["atr_normalized"] = (
                    result["atr"] / close.replace(0, np.nan)
                ).replace([np.inf, -np.inf], np.nan)

        # ========== 滚动 Z-score 特征（推荐：比价格归一化更优）==========
        # 使用多个滚动窗口标准化（Feature Stacking），让模型学习不同时间尺度的信息
        # 默认窗口：[50 (短期), 288 (24h), 500 (长期)]
        # 对于 5 分钟 K 线：50≈4h, 288=24h, 500≈2天
        if rolling_zscore_windows is None:
            rolling_zscore_windows = [50, 288, 500]

        # Helper function to generate z-score features for multiple windows
        def add_multi_window_zscore(
            base_col: str,
            feature_prefix: str,
            windows: List[int],
            required_features: Optional[set],
        ) -> None:
            """为某个基础指标生成多个窗口的 z-score 特征"""
            if base_col not in result.columns:
                return

            for window in windows:
                zscore_col = f"{feature_prefix}_zscore_w{window}"
                # Check if this specific feature is required
                if required_features is not None:
                    # Check if any zscore variant is requested or this specific one
                    if not any(
                        f"{feature_prefix}_zscore" in f for f in required_features
                    ):
                        continue
                    if zscore_col not in required_features and not any(
                        f.startswith(f"{feature_prefix}_zscore")
                        and f.endswith(f"_w{window}")
                        for f in required_features
                    ):
                        # If specific windows are requested, only generate those
                        if any(
                            f"{feature_prefix}_zscore_w" in f for f in required_features
                        ):
                            continue

                if zscore_col not in result.columns:
                    # 【关键修复】：强制 min_periods=window，确保满窗才输出，减少早期噪声
                    result[zscore_col] = BaselineFeatureEngineer._rolling_zscore(
                        result[base_col], window=window, min_periods=window
                    )

        # 1. RSI 滚动 Z-score（虽然 RSI 本身是 0-100，但 Z-score 能突出极端值）
        # 生成多个窗口：rsi_zscore_w50, rsi_zscore_w288, rsi_zscore_w500
        if required_features is None or any(
            "rsi_zscore" in f for f in required_features or [""]
        ):
            add_multi_window_zscore(
                "rsi", "rsi", rolling_zscore_windows, required_features
            )

        # 2. MACD 滚动 Z-score（MACD 绝对值随价格变化，Z-score 标准化更优）
        if required_features is None or any(
            "macd_zscore" in f for f in required_features or [""]
        ):
            add_multi_window_zscore(
                "macd", "macd", rolling_zscore_windows, required_features
            )

        # 3. MACD Histogram 滚动 Z-score（波动更大，Z-score 能突出极端动量变化）
        if required_features is None or any(
            "macd_histogram_zscore" in f for f in required_features or [""]
        ):
            add_multi_window_zscore(
                "macd_histogram",
                "macd_histogram",
                rolling_zscore_windows,
                required_features,
            )

        # 4. Momentum 滚动 Z-score（不同资产的 ROC 量级差异巨大，Z-score 必须）
        for period in [5, 10, 20]:
            momentum_col = f"momentum_{period}"
            if required_features is None or any(
                f"momentum_{period}_zscore" in f for f in required_features or [""]
            ):
                add_multi_window_zscore(
                    momentum_col,
                    f"momentum_{period}",
                    rolling_zscore_windows,
                    required_features,
                )

        # 5. ATR 滚动 Z-score（ATR 绝对值与价格成正比，Z-score 可判断波动率异常）
        if required_features is None or any(
            "atr_zscore" in f for f in required_features or [""]
        ):
            add_multi_window_zscore(
                "atr", "atr", rolling_zscore_windows, required_features
            )

        # 6. Volume 滚动 Z-score（交易量绝对值与流动性相关，Z-score 捕捉相对变化）
        if required_features is None or any(
            "volume_zscore" in f for f in required_features or [""]
        ):
            add_multi_window_zscore(
                "volume", "volume", rolling_zscore_windows, required_features
            )

        # 7. BB Width 滚动 Z-score（布林带宽度反映波动性，Z-score 标准化）
        if required_features is None or any(
            "bb_width_zscore" in f for f in required_features or [""]
        ):
            add_multi_window_zscore(
                "bb_width", "bb_width", rolling_zscore_windows, required_features
            )

        # 8. Volatility 滚动 Z-score（波动率指标的 Z-score）
        if required_features is None or any(
            "volatility_zscore" in f for f in required_features or [""]
        ):
            add_multi_window_zscore(
                "volatility", "volatility", rolling_zscore_windows, required_features
            )

        # Momentum features
        for period in [5, 10, 20]:
            momentum_col = f"momentum_{period}"
            if not required_features or momentum_col in required_features:
                if momentum_col not in result.columns:
                    result[momentum_col] = close.pct_change(period)

        # SMA features
        sma_map = {5: "sma_5", 10: "sma_10", 20: "sma_20"}
        for window, col_name in sma_map.items():
            if not required_features or col_name in required_features:
                if col_name not in result.columns:
                    close_numeric = pd.to_numeric(close, errors="coerce").astype(float)
                    values = talib.SMA(close_numeric.values, timeperiod=window)
                    result[col_name] = pd.Series(
                        values, index=close_numeric.index
                    ).fillna(close)

        # SMA/EMA 相对 close 的百分比
        close_safe = close.replace(0, np.nan)
        for col_name in [
            "sma_5",
            "sma_10",
            "sma_20",
            "ema_5",
            "ema_10",
            "ema_20",
            "ema_50",
            "wma_20",
        ]:
            pct_col = f"{col_name}_pct_close"
            if not required_features or pct_col in required_features:
                if col_name in result.columns and pct_col not in result.columns:
                    result[pct_col] = ((result[col_name] / close_safe - 1.0)).replace(
                        [np.inf, -np.inf], np.nan
                    )

        # SMA ratios
        if not required_features or "sma_ratio_5_20" in required_features:
            if {"sma_5", "sma_20"}.issubset(
                result.columns
            ) and "sma_ratio_5_20" not in result.columns:
                # 使用 shift(1) 确保时间对齐，避免使用未来信息
                result["sma_ratio_5_20"] = (
                    (result["sma_5"] / result["sma_20"].replace(0, np.nan))
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(1.0)
                    .shift(1)
                )

        if not required_features or "sma_ratio_10_20" in required_features:
            if {"sma_10", "sma_20"}.issubset(
                result.columns
            ) and "sma_ratio_10_20" not in result.columns:
                # 使用 shift(1) 确保时间对齐，避免使用未来信息
                result["sma_ratio_10_20"] = (
                    (result["sma_10"] / result["sma_20"].replace(0, np.nan))
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(1.0)
                    .shift(1)
                )

        # Volume features
        if not required_features or "volume_sma_20" in required_features:
            if "volume_sma_20" not in result.columns:
                volume_numeric = pd.to_numeric(
                    result["volume"], errors="coerce"
                ).astype(float)
                values = talib.SMA(volume_numeric.values, timeperiod=20)
                result["volume_sma_20"] = pd.Series(
                    values, index=volume_numeric.index
                ).fillna(result["volume"])

        if not required_features or "volume_ratio" in required_features:
            if "volume_ratio" not in result.columns:
                if "volume_sma_20" in result.columns:
                    denom = result["volume_sma_20"].replace(0, np.nan)
                    # 使用 shift(1) 确保时间对齐，避免使用未来信息
                    result["volume_ratio"] = (
                        (result["volume"] / denom)
                        .replace([np.inf, -np.inf], np.nan)
                        .fillna(1.0)
                        .shift(1)
                    )

        # Final cleanup: 处理所有数值列的 inf，保留 NaN 到预处理阶段
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in result.columns:
                result[col] = result[col].replace([np.inf, -np.inf], np.nan)

        return result

    @staticmethod
    def _rolling_percentile(
        series: pd.Series, window: int, min_periods: int = None, shift: bool = True
    ) -> pd.Series:
        """
        安全滚动百分位排名（严格因果，无未来信息）

        【核心原则：滚动窗口统计特征需要 shift(1)】
        - 虽然我们在 t 时刻可以使用 close[t]，但对于"滚动窗口统计特征"仍需 shift(1)
        - 原因：避免将当前值包含在历史分布中，即使我们已经排除了当前值
        - 更严格的做法：在 t 时刻使用的特征基于 t-1 及之前的数据计算

        【关键区分】
        - 基础滚动指标（sma, ema, atr, bb_width）：不需要 shift(1)
        - 滚动窗口统计特征（zscore, percentile, entropy）：需要 shift(1)

        Args:
            series: 输入序列
            window: 滚动窗口长度
            min_periods: 最小观测数才开始输出（默认=window，最稳健）
            shift: 是否 shift(1) 以确保完全因果（默认=True，推荐）

        Returns:
            滚动百分位排名序列（0~1，早期不足窗口处为 NaN）
        """
        if min_periods is None:
            min_periods = window  # 默认：必须满窗才输出，最稳健

        min_periods = min(min_periods, window)

        def _percentile(x: np.ndarray) -> float:
            """
            计算当前值在历史窗口中的百分位排名（严格因果，无自我参照偏差）

            【实现说明】
            - current = x[-1]：当前值（如 close[t]），作为"新来的考生"
            - history = x[:-1]：历史窗口（如 [t-N, t-1]），作为"老考生的成绩分数线"
            - percentile = (history <= current).sum() / len(history)
              表示：当前值在历史中的相对位置，完全基于历史评估当前状态
            """
            if len(x) < 2 or not np.isfinite(x[-1]):
                return np.nan
            current = x[-1]  # 当前值（如 close[t]），作为"新来的考生"
            history = x[
                :-1
            ]  # ← 关键：只用历史（如 [t-N, t-1]），作为"老考生的成绩分数线"
            history = history[np.isfinite(history)]
            if len(history) == 0:
                return np.nan
            # 当前值在历史中的分位：(历史中 ≤ 当前值的数量) / 历史总数量
            # 这表示：当前值相对于历史的位置，完全基于历史评估当前状态
            return (history <= current).sum() / float(len(history))

        percentile_series = series.rolling(
            window=window, min_periods=min_periods
        ).apply(_percentile, raw=True)

        # 【关键修复】：对滚动窗口统计特征强制 shift(1)，确保完全因果
        # 在 t 时刻使用的特征基于 t-1 及之前的数据计算
        if shift:
            percentile_series = percentile_series.shift(1)

        return percentile_series

    @staticmethod
    def _rolling_zscore(
        series: pd.Series,
        window: int,
        min_periods: int = None,
        return_quality: bool = False,
        shift: bool = True,
    ):
        """
        安全滚动 Z-score（严格因果，无未来信息）

        【核心原则：滚动窗口统计特征需要 shift(1)】
        - 虽然我们在 t 时刻可以使用 close[t]，但对于"滚动窗口统计特征"仍需 shift(1)
        - 原因：避免将当前值包含在历史分布中，即使 rolling 本身是因果的
        - 更严格的做法：在 t 时刻使用的特征基于 t-1 及之前的数据计算

        【关键区分】
        - 基础滚动指标（sma, ema, atr, bb_width）：不需要 shift(1)
        - 滚动窗口统计特征（zscore, percentile, entropy）：需要 shift(1)

        【说明】
        - min_periods 小会导致早期统计量不稳定（小样本噪声）
        - 但 rolling() 只使用历史和当前数据，绝不包含未来
        - 提高 min_periods 能降低虚假相关，因为剔除了高噪声样本

        Args:
            series: 输入时间序列（如 ATR、volatility）
            window: 滚动窗口长度（如 288）
            min_periods: 最小观测数才开始输出（默认=window，最稳健）
            return_quality: 是否同时返回质量分数（0~1，1=完整窗口）
            shift: 是否 shift(1) 以确保完全因果（默认=True，推荐）

        Returns:
            zscore: 标准化后的序列（早期不足窗口处为 NaN）
            quality (可选): 每个点的统计质量 = 实际样本数 / window
        """
        if min_periods is None:
            min_periods = window  # 默认：必须满窗才输出，最稳健

        min_periods = min(min_periods, window)

        # 滚动统计量（只依赖历史和当前，无未来信息）
        rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
        rolling_std = series.rolling(window=window, min_periods=min_periods).std()

        # 计算 Z-score: (x - mean) / std
        zscore = (series - rolling_mean) / rolling_std.replace(0, np.nan)

        # 【关键修复】：对滚动窗口统计特征强制 shift(1)，确保完全因果
        # 在 t 时刻使用的特征基于 t-1 及之前的数据计算
        if shift:
            zscore = zscore.shift(1)

        if return_quality:
            # 计算质量分数：实际样本数 / window（0~1，1表示使用了完整窗口）
            count = series.rolling(window=window, min_periods=1).count()
            quality = count / window
            # Quality 也需要 shift(1) 以保持对齐
            if shift:
                quality = quality.shift(1)
            return zscore, quality

        # 处理 inf 和 NaN：将 inf 替换为 NaN，然后保留 NaN（不填充，让下游处理）
        zscore = zscore.replace([np.inf, -np.inf], np.nan)

        return zscore

    @staticmethod
    def _trend_r2(prices: pd.Series, window: int = 20, *, lag: int = 0) -> pd.Series:
        """计算趋势R²特征（基于对数价格序列）"""
        log_price = np.log(prices.replace(0, np.nan)).ffill()

        def _compute_r2(series):
            if len(series) < 3:
                return 0.0
            try:
                x = np.arange(len(series))
                y = series.values
                slope, intercept = np.polyfit(x, y, 1)
                y_pred = slope * x + intercept
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
                return max(0.0, min(1.0, r2))
            except Exception:
                return 0.0

        r2_series = log_price.rolling(window=window, min_periods=3).apply(
            _compute_r2, raw=False
        )
        if lag == 0:
            return r2_series
        return r2_series.shift(lag)

    @staticmethod
    def _price_entropy(close: pd.Series, window: int = 50) -> pd.Series:
        """价格方向熵"""
        ret = close.pct_change().fillna(0.0)
        sign = np.sign(ret).replace(0, 1)

        def _entropy(x: np.ndarray) -> float:
            if len(x) == 0:
                return np.nan
            p_up = (x > 0).mean()
            p_dn = 1.0 - p_up
            eps = 1e-9
            return -(p_up * np.log2(p_up + eps) + p_dn * np.log2(p_dn + eps)) / 1.0

        return sign.rolling(window=window, min_periods=1).apply(_entropy, raw=True)

    @staticmethod
    def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Wilder-style RSI."""
        return BaselineFeatureEngineer.compute_rsi(series, period)

    @staticmethod
    def _rolling_skew(series: pd.Series, window: int) -> pd.Series:
        """Rolling skewness"""
        return series.rolling(window=window, min_periods=window).skew()

    @staticmethod
    def _ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    def _shift_feature(self, series: pd.Series, *, offset: int = 0) -> pd.Series:
        """Apply configurable lag to feature series"""
        total_shift = self.feature_shift + offset
        if total_shift == 0:
            return series
        return series.shift(total_shift)

    def _run_diagnostics(self, df: pd.DataFrame, feature_cols: List[str]) -> None:
        """收集诊断信息"""
        report: Dict[str, Dict[str, float]] = {}
        tol_zero = 1e-9
        tol_clip = 1e-3
        clip_bound = float(self.feature_clip_bound)
        near_clip = 0.9 * clip_bound

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

            metrics["zero_ratio"] = float(np.isclose(valid, 0.0, atol=tol_zero).mean())
            metrics["abs_ge_90pct_ratio"] = float((valid.abs() >= near_clip).mean())
            metrics["mean"] = float(valid.mean())
            metrics["std"] = float(valid.std())
            report[col] = metrics

        self.diagnostic_report = report

    def _add_advanced_derived_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """添加高级衍生特征"""
        df = data.copy()
        try:
            # 需要的基础特征
            if "bb_upper" not in df.columns or "atr" not in df.columns:
                return df

            # 1. BB Width相关
            if "bb_width" not in df.columns:
                df["bb_width"] = (df["bb_upper"] - df["bb_lower"]).abs()
            if "bb_width_normalized" not in df.columns:
                df["bb_width_normalized"] = df["bb_width"] / df["atr"].replace(
                    0, np.nan
                )
                df["bb_width_normalized"] = (
                    df["bb_width_normalized"]
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0)
                )

            # 2. Range ratio
            if "range_ratio_5bar" not in df.columns:
                if "hl" not in df.columns:
                    df["hl"] = df["high"] - df["low"]
                range_ratio_raw = df["hl"].rolling(5).mean() / df["hl"].rolling(
                    20
                ).mean().replace(0, np.nan)
                range_ratio_raw = range_ratio_raw.fillna(1)
                range_ratio_log = np.log1p(range_ratio_raw)
                range_ratio_mean = range_ratio_log.rolling(50, min_periods=5).mean()
                range_ratio_std = range_ratio_log.rolling(50, min_periods=5).std()
                df["range_ratio_5bar"] = (
                    (
                        (range_ratio_log - range_ratio_mean)
                        / range_ratio_std.replace(0, np.nan)
                    )
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                )

            # 3. Volatility reversal score
            if "volatility_reversal_score" not in df.columns:
                atr_mean = df["atr"].rolling(50).mean()
                atr_std = df["atr"].rolling(50).std()
                df["volatility_reversal_score"] = (
                    (df["atr"] - atr_mean) / atr_std.replace(0, np.nan)
                ).fillna(0)

            # 4. Price range symmetry
            if "price_range_symmetry" not in df.columns:
                price_range_symmetry_raw = (
                    self._shift_feature(df["high"]) - self._shift_feature(df["close"])
                ) / (
                    (
                        self._shift_feature(df["close"])
                        - self._shift_feature(df["low"])
                    ).replace(0, np.nan)
                )
                price_range_symmetry_raw = price_range_symmetry_raw.replace(
                    [np.inf, -np.inf], np.nan
                ).fillna(1)
                price_range_symmetry_log = np.log1p(
                    np.abs(price_range_symmetry_raw)
                ) * np.sign(price_range_symmetry_raw)
                price_range_symmetry_mean = price_range_symmetry_log.rolling(
                    50, min_periods=5
                ).mean()
                price_range_symmetry_std = price_range_symmetry_log.rolling(
                    50, min_periods=5
                ).std()
                df["price_range_symmetry"] = (
                    (
                        (price_range_symmetry_log - price_range_symmetry_mean)
                        / price_range_symmetry_std.replace(0, np.nan)
                    )
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                )

            # 5. Volume anomaly
            if "volume_anomaly" not in df.columns:
                volume_anomaly_raw = df["volume"] / df["volume"].ewm(
                    span=20, min_periods=10
                ).mean().replace(0, np.nan)
                volume_anomaly_raw = volume_anomaly_raw.fillna(1)
                volume_anomaly_log = np.log1p(volume_anomaly_raw)
                volume_anomaly_mean = volume_anomaly_log.rolling(
                    50, min_periods=10
                ).mean()
                volume_anomaly_std = volume_anomaly_log.rolling(
                    50, min_periods=10
                ).std()
                df["volume_anomaly"] = (
                    (
                        (volume_anomaly_log - volume_anomaly_mean)
                        / volume_anomaly_std.replace(0, np.nan)
                    )
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                )

            # 6. ROC and acceleration
            if "roc_5" not in df.columns:
                roc_raw = df["close"].pct_change(5)
                roc_mean = roc_raw.rolling(window=50, min_periods=5).mean()
                roc_std = roc_raw.rolling(window=50, min_periods=5).std()
                roc_std = roc_std.clip(lower=roc_raw.abs().quantile(0.01))
                df["roc_5"] = (
                    ((roc_raw - roc_mean) / roc_std.replace(0, np.nan))
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                )

            if "acceleration_3" not in df.columns:
                roc_3 = df["close"].pct_change(3)
                roc_3_mean = roc_3.rolling(window=50, min_periods=5).mean()
                roc_3_std = roc_3.rolling(window=50, min_periods=5).std()
                roc_3_std = roc_3_std.clip(lower=roc_3.abs().quantile(0.01))
                roc_3_norm = (
                    ((roc_3 - roc_3_mean) / roc_3_std.replace(0, np.nan))
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                )
                current = self._shift_feature(roc_3_norm)
                prev = self._shift_feature(roc_3_norm, offset=1)
                df["acceleration_3"] = current - prev

            # 7. Trend R²
            if "trend_r2_20" not in df.columns:
                df["trend_r2_20"] = self._trend_r2(
                    df["close"], window=20, lag=self.feature_shift
                )
            if "trend_r2_50" not in df.columns:
                df["trend_r2_50"] = self._trend_r2(
                    df["close"], window=50, lag=self.feature_shift
                )

            # 8. Slope consistency
            if "slope_consistency_score" not in df.columns:
                ema10 = df["close"].ewm(span=10).mean()
                ema20 = df["close"].ewm(span=20).mean()
                ema50 = df["close"].ewm(span=50).mean()
                slope10 = np.sign(ema10.diff())
                slope20 = np.sign(ema20.diff())
                slope50 = np.sign(ema50.diff())
                df["slope_consistency_score"] = (
                    (slope10 == slope20).astype(int)
                    + (slope20 == slope50).astype(int)
                    + (slope10 == slope50).astype(int)
                )

            # 9. Trend volatility alignment
            if "trend_volatility_alignment" not in df.columns:
                if "atr_percentile" in df.columns and "roc_5" in df.columns:
                    df["trend_volatility_alignment"] = np.sign(df["roc_5"]).fillna(
                        0
                    ) * df["atr_percentile"].fillna(0)

            # 10. Compression to breakout probability
            if "compression_to_breakout_prob" not in df.columns:
                if "compression_duration" in df.columns and "roc_5" in df.columns:
                    df["compression_to_breakout_prob"] = df[
                        "compression_duration"
                    ].fillna(0) * df["roc_5"].fillna(0)

        except Exception as e:
            print(f"      Warning: 高级衍生特征计算失败: {e}")
        return df

    def engineer_features(
        self,
        df: pd.DataFrame,
        *,
        fit: bool = True,
        required_features: Optional[set] = None,
    ) -> pd.DataFrame:
        """工程特征（合并后的版本，包含所有新特征）

        对于多资产数据，rolling 操作会按 symbol 分组进行，避免跨资产数据泄露。
        """
        if not {"open", "high", "low", "close", "volume"}.issubset(df.columns):
            raise ValueError(
                "DataFrame must contain open, high, low, close, volume columns"
            )

        # Check if this is multi-asset data (has _symbol or symbol column)
        has_symbol_col = "_symbol" in df.columns or "symbol" in df.columns
        symbol_col = (
            "_symbol"
            if "_symbol" in df.columns
            else ("symbol" if "symbol" in df.columns else None)
        )
        is_multi_asset = (
            has_symbol_col and df[symbol_col].nunique() > 1 if symbol_col else False
        )

        if is_multi_asset:
            # Multi-asset: process each symbol separately to avoid cross-asset leakage
            print(
                f"   🔒 Multi-asset detected ({df[symbol_col].nunique()} symbols): processing each symbol separately"
            )
            processed_groups = []
            for symbol in df[symbol_col].unique():
                symbol_mask = df[symbol_col] == symbol
                symbol_df = df[symbol_mask].copy()
                # Process this symbol's data
                symbol_processed = self._engineer_features_single_asset(
                    symbol_df, fit=fit, required_features=required_features
                )
                processed_groups.append(symbol_processed)
            # Combine all symbols
            data = pd.concat(processed_groups, axis=0).sort_index()
        else:
            # Single asset: process directly
            data = self._engineer_features_single_asset(
                df.copy(), fit=fit, required_features=required_features
            )

        return data

    def _engineer_features_single_asset(
        self,
        df: pd.DataFrame,
        *,
        fit: bool = True,
        required_features: Optional[set] = None,
    ) -> pd.DataFrame:
        """工程特征（单资产版本，内部方法）"""
        data = df.copy()

        # Core ATR（按需计算）
        if required_features is None or any(
            "atr" in f for f in required_features or [""]
        ):
            if "atr" not in data.columns:
                data["atr"] = self._compute_atr(data, window=14)

        # SR proximity using rolling swing proxies
        swing_win_short = 20
        swing_win_long = 60

        need_swing = required_features is None or any(
            "swing" in f or "sr_dist" in f for f in required_features or [""]
        )

        if need_swing:
            data["roll_high_s"] = (
                data["high"].rolling(swing_win_short, min_periods=1).max()
            )
            data["roll_low_s"] = (
                data["low"].rolling(swing_win_short, min_periods=1).min()
            )
            data["roll_high_l"] = (
                data["high"].rolling(swing_win_long, min_periods=1).max()
            )
            data["roll_low_l"] = (
                data["low"].rolling(swing_win_long, min_periods=1).min()
            )

            eps = 1e-9
            data["sr_dist_high_s"] = (data["close"] - data["roll_high_s"]) / (
                data["atr"] + eps
            )
            data["sr_dist_low_s"] = (data["close"] - data["roll_low_s"]) / (
                data["atr"] + eps
            )
            data["sr_dist_high_l"] = (data["close"] - data["roll_high_l"]) / (
                data["atr"] + eps
            )
            data["sr_dist_low_l"] = (data["close"] - data["roll_low_l"]) / (
                data["atr"] + eps
            )

        # VWAP (Volume Weighted Average Price) 价格偏离特征
        # 当价格通常"贴着 VWAP 走"时，直接使用"价格对 VWAP 的偏离度"作为特征更简洁、有效
        need_vwap = True

        if need_vwap:
            # 计算滚动 VWAP（滚动窗口成交量加权平均价格）
            # VWAP = Σ(price × volume) / Σ(volume) over rolling window
            # 使用典型价格 (high + low + close) / 3 作为价格基准
            typical_price = (data["high"] + data["low"] + data["close"]) / 3.0

            # 计算价格×成交量
            price_volume = typical_price * data["volume"]

            # 滚动求和：价格×成交量的滚动和
            rolling_pv = price_volume.rolling(
                window=self.vwap_window, min_periods=1
            ).sum()
            # 滚动求和：成交量的滚动和
            rolling_vol = (
                data["volume"].rolling(window=self.vwap_window, min_periods=1).sum()
            )

            # 滚动 VWAP = 滚动价格×成交量总和 / 滚动成交量总和
            vwap = (rolling_pv / rolling_vol.replace(0, np.nan)).fillna(typical_price)

            # 保存原始值（用于计算归一化特征，但会被排除）
            data["vwap"] = vwap  # VWAP 原始值（有量纲，会被排除）

            eps = 1e-9
            # VWAP 价格偏离程度特征（无量纲）
            vwap_safe = vwap.replace(0, np.nan)
            # 价格相对 VWAP 的百分比偏离
            data["price_to_vwap_pct"] = (
                (data["close"] - vwap_safe) / vwap_safe
            ).replace([np.inf, -np.inf], np.nan)
            # 价格相对 VWAP 的 ATR 归一化偏离
            data["price_to_vwap_atr"] = (
                (data["close"] - vwap_safe) / (data["atr"] + eps)
            ).replace([np.inf, -np.inf], np.nan)
            # VWAP 偏离的 Z-score（相对于历史分布）
            vwap_deviation = (data["close"] - vwap_safe) / vwap_safe
            vwap_dev_mean = vwap_deviation.rolling(window=50, min_periods=10).mean()
            vwap_dev_std = vwap_deviation.rolling(window=50, min_periods=10).std()
            price_to_vwap_zscore = (
                (vwap_deviation - vwap_dev_mean) / vwap_dev_std.replace(0, np.nan)
            ).replace([np.inf, -np.inf], np.nan)
            data["price_to_vwap_zscore"] = price_to_vwap_zscore.shift(1)

            # VWAP 趋势（VWAP 的变化率，无量纲）
            # 使用 6 根 K 线的 VWAP 变化率作为趋势指标
            # 如果数据不足无法计算，保留 NaN 表示"无法计算"
            data["vwap_trend_6h"] = vwap.pct_change(periods=6)

        # Price-volume divergence signals
        if required_features is None or any(
            "divergence" in f for f in required_features or [""]
        ):
            # 使用 rsi（如果不存在则计算，默认 period=14）
            if "rsi" not in data.columns:
                data["rsi"] = self._compute_rsi(data["close"], period=14)

            # RSI 背离检测
            # 顶背离：价格创新高但 RSI 没有创新高（看跌信号）
            recent_high = data["close"].rolling(20, min_periods=5).max().ffill()
            recent_rsi_high = data["rsi"].rolling(20, min_periods=5).max().ffill()
            tol = 1e-8
            top_divergence_mask = (
                recent_high.notna()
                & recent_rsi_high.notna()
                & (data["close"] >= (recent_high - tol))
                & (data["rsi"] < (recent_rsi_high - tol))
            )

            # 底背离：价格创新低但 RSI 没有创新低（看涨信号）
            recent_low = data["close"].rolling(20, min_periods=5).min().ffill()
            recent_rsi_low = data["rsi"].rolling(20, min_periods=5).min().ffill()
            bottom_divergence_mask = (
                recent_low.notna()
                & recent_rsi_low.notna()
                & (data["close"] <= (recent_low + tol))
                & (data["rsi"] > (recent_rsi_low + tol))
            )

            # 合并背离信号：顶背离=-1（看跌），底背离=+1（看涨），无背离=0
            data["rsi_divergence"] = bottom_divergence_mask.astype(
                float
            ) - top_divergence_mask.astype(float)

            # 成交量背离检测
            # 顶背离：价格上涨但成交量下降（看跌信号）
            price_vs_past_up = (data["close"] > data["close"].shift(5)).fillna(False)
            avg_volume_20 = data["volume"].rolling(20, min_periods=5).mean()
            low_volume_mask = (data["volume"] < avg_volume_20).fillna(False)
            volume_top_div_mask = price_vs_past_up & low_volume_mask

            # 底背离：价格下跌但成交量上升（看涨信号）
            price_vs_past_down = (data["close"] < data["close"].shift(5)).fillna(False)
            high_volume_mask = (data["volume"] > avg_volume_20).fillna(False)
            volume_bottom_div_mask = price_vs_past_down & high_volume_mask

            # 合并背离信号：底背离=+1（看涨），顶背离=-1（看跌），无背离=0
            data["volume_divergence"] = volume_bottom_div_mask.astype(
                float
            ) - volume_top_div_mask.astype(float)

        # Compression features
        need_compression = required_features is None or any(
            "compression" in f or "atr_percentile" in f
            for f in required_features or [""]
        )

        if need_compression:
            atr_pct = self._rolling_percentile(
                data["atr"], window=self.percentile_window
            )
            data["atr_percentile"] = atr_pct

            volatility_regime_window = 200
            volatility_regime_threshold = 0.7
            atr_quantile_70 = (
                data["atr"]
                .rolling(window=volatility_regime_window, min_periods=1)
                .quantile(volatility_regime_threshold)
            )
            data["volatility_regime"] = (
                (data["atr"] > atr_quantile_70).astype(int).fillna(0)
            )

            returns = data["close"].pct_change()
            # 使用 shift(1) 确保时间对齐，避免使用未来信息
            realized_skew = self._rolling_skew(returns, window=20)
            data["realized_skew"] = realized_skew.shift(1)

            vol5 = returns.rolling(5, min_periods=1).std()
            vol60 = returns.rolling(60, min_periods=1).std()
            volatility_ratio = (vol5 / vol60.replace(0, np.nan)).replace(
                [np.inf, -np.inf], np.nan
            )
            data["volatility_ratio"] = volatility_ratio.shift(1)

            # 【说明】：这不是数据泄漏问题，而是统计可靠性问题。
            # - min_periods 小会导致早期统计量不稳定（小样本噪声）
            # - 但 rolling() 只使用历史和当前数据，绝不包含未来
            # - 提高 min_periods 能降低虚假相关，因为剔除了高噪声样本
            # 使用 min_periods=window（最稳健），确保早期数据使用完整的统计量
            atr_mean_hist = (
                data["atr"]
                .rolling(self.percentile_window, min_periods=self.percentile_window)
                .mean()
            )
            eps = 1e-9
            # 【关键修复】：atr_compression_ratio 是比率特征，也需要 shift(1) 确保完全因果
            # 在 t 时刻使用的特征基于 t-1 及之前的数据计算
            atr_compression_ratio_raw = (atr_mean_hist / (data["atr"] + eps)).replace(
                [np.inf, -np.inf], np.nan
            )
            data["atr_compression_ratio"] = atr_compression_ratio_raw.shift(1)

            vol_pct = self._rolling_percentile(
                data["volume"].astype(float), window=self.percentile_window
            )
            data["volume_percentile"] = vol_pct

            data["price_entropy"] = self._price_entropy(data["close"], window=50)

            threshold = self.compression_threshold_pct
            # 如果百分位是 NaN，使用 0.5（中位数）作为默认值
            below = (data["atr_percentile"].fillna(0.5) <= threshold).astype(int)
            run = np.zeros(len(below), dtype=int)
            cnt = 0
            for i, v in enumerate(below.values):
                if v == 1:
                    cnt += 1
                else:
                    cnt = 0
                run[i] = cnt
            data["compression_duration"] = run

            short_window = 30
            data["pre_break_silence"] = (
                data["atr_percentile"].rolling(short_window, min_periods=1).mean()
                <= threshold
            ).astype(float)

            small = 20
            large = 100
            var_small = data["close"].rolling(small, min_periods=1).var()
            var_large = data["close"].rolling(large, min_periods=1).var()
            density = 1.0 - (var_small / (var_large + eps))
            # 使用 shift(1) 确保时间对齐，避免使用未来信息
            data["internal_price_density"] = density.clip(0.0, 1.0).shift(1)

            # 如果百分位是 NaN，使用 0.5（中位数）作为默认值
            # 【说明】：compression_confidence 使用了已 shift(1) 的 atr_percentile 和 volume_percentile
            # 所以它本身已经是因果的，不需要再次 shift
            atr_norm = data["atr_percentile"].fillna(0.5)
            vol_norm = data["volume_percentile"].fillna(0.5)
            dens_norm = data["internal_price_density"].fillna(0.0)
            data["compression_confidence"] = (
                0.5 * (1 - atr_norm) + 0.3 * (1 - vol_norm) + 0.2 * dens_norm
            )

        # Advanced derived features
        data = self._add_advanced_derived_features(data)

        # Time factors
        # 注意：时间特征本身不应该有数据泄漏（它们只依赖于时间戳）
        # 如果时间特征与未来收益有相关性，可能是真实的时间模式（如不同时段的交易行为差异）
        # 在打乱测试中，由于时间特征是确定性的，即使打乱收益，相关性仍然存在
        # 这不是数据泄漏，而是特征的性质。需要通过其他方法（如滞后测试、walk-forward）验证其预测价值
        try:
            idx = data.index
            if hasattr(idx, "hour") and hasattr(idx, "dayofweek"):
                try:
                    if getattr(idx, "tz", None) is not None:
                        utc_idx = idx.tz_convert("UTC")
                        hour = utc_idx.hour.astype(int)
                        midnight_delta = (
                            utc_idx - utc_idx.normalize()
                        ).total_seconds() / 60.0
                    else:
                        hour = idx.hour.astype(int)
                        midnight_delta = (idx - idx.normalize()).total_seconds() / 60.0

                    # 时间特征：使用当前时间点的时间信息
                    # 这些特征本身不包含未来信息，但如果与未来收益相关，可能是真实的时间模式
                    data["hour_sin"] = np.sin(2 * np.pi * hour / 24)
                    data["hour_cos"] = np.cos(2 * np.pi * hour / 24)
                    data["Hour_of_Day"] = hour
                    data["minutes_since_reset"] = pd.Series(
                        midnight_delta, index=data.index
                    ).fillna(0.0)
                    data["Is_Weekend"] = (idx.dayofweek >= 5).astype(int)
                except Exception:
                    data["hour_sin"] = 0.0
                    data["hour_cos"] = 1.0
                    data["Hour_of_Day"] = 0
                    data["minutes_since_reset"] = 0.0
                    data["Is_Weekend"] = 0
            else:
                data["hour_sin"] = 0.0
                data["hour_cos"] = 1.0
                data["Hour_of_Day"] = 0
                data["minutes_since_reset"] = 0.0
                data["Is_Weekend"] = 0
        except Exception:
            data["hour_sin"] = 0.0
            data["hour_cos"] = 1.0
            data["Hour_of_Day"] = 0
            data["minutes_since_reset"] = 0.0
            data["Is_Weekend"] = 0

        # 添加新的无量纲特征
        # ZigZag 无量纲特征
        if required_features is None or any(
            "zz_" in f or "zigzag" in f for f in required_features or [""]
        ):
            data = BaselineFeatureEngineer.add_zigzag_dimensionless_features(
                data, required_features
            )

        # POC 和 HAL 无量纲特征（合并计算，避免重复计算 volume profile）
        if required_features is None or any(
            "poc" in f or "hal" in f for f in required_features or [""]
        ):
            data = BaselineFeatureEngineer.add_poc_hal_dimensionless_features(
                data, required_features
            )

        # Swing 无量纲特征
        if required_features is None or any(
            "swing" in f for f in required_features or [""]
        ):
            data = BaselineFeatureEngineer.add_swing_dimensionless_features(
                data, required_features
            )

        # OLS 通道特征
        data = BaselineFeatureEngineer.add_ols_channel_features(data, required_features)

        # 基础价格与量能相对变化特征
        if required_features is None or any(
            f in required_features or ""
            for f in [
                "ret_1h",
                "ret_4h",
                "ret_24h",
                "rv_4h",
                "rv_24h",
                "vol_ma_ratio",
                "vol_zscore",
            ]
        ):
            data = BaselineFeatureEngineer.add_price_volume_relative_features(
                data, required_features
            )

        # 常用衍生特征（包括多个窗口的滚动 Z-score）
        data = BaselineFeatureEngineer.add_common_derived_features(
            data, required_features, rolling_zscore_windows=self.rolling_zscore_windows
        )

        # 统一 SR 边界强度与突破确认特征
        boundaries = BaselineFeatureEngineer._get_sr_boundary_definitions(data)
        compression_series = data.get("compression_confidence")

        boundary_strengths = BaselineFeatureEngineer._compute_boundary_strengths(
            data,
            boundaries,
            window=60,
            tolerance_factor=0.5,
            compression_series=compression_series,
        )

        strength_columns: List[str] = []
        for name, series in boundary_strengths.items():
            data[name] = series
            strength_columns.append(name)

        if strength_columns:
            data["sr_strength_max"] = data[strength_columns].max(axis=1)
            data["sr_strength_sum"] = data[strength_columns].sum(axis=1)

        boundary_confirmations = (
            BaselineFeatureEngineer._compute_boundary_volume_confirmations(
                data,
                boundaries,
                lookback=20,
                vol_threshold=1.5,
                confirmation_bars=3,
            )
        )

        confirmation_columns: List[str] = []
        compression_effect = (
            compression_series.ffill()
            if compression_series is not None
            else pd.Series(0.0, index=data.index)
        ).fillna(0.0)

        for name, series in boundary_confirmations.items():
            data[name] = series
            confirmation_columns.append(name)

            boundary_name = name.replace("volume_price_confirmation_", "")
            strength_col = f"sqs_{boundary_name}"
            if strength_col in data.columns:
                # 【关键修复】：虽然 sqs_* 和 volume_price_confirmation_* 都已经 shift(1) 了
                # 但相乘后的结果需要确保时间对齐，这里不需要再次 shift
                # 因为两个已经 shift 过的序列相乘，结果仍然是正确对齐的
                breakout_quality = (
                    data[strength_col] * series * (1.0 + compression_effect)
                )
                data[f"sr_breakout_quality_{boundary_name}"] = breakout_quality

        quality_columns = [
            col for col in data.columns if col.startswith("sr_breakout_quality_")
        ]
        if quality_columns:
            # 【关键修复】：确保聚合后的特征也正确对齐
            # 由于所有 quality_columns 都是基于已 shift 的特征计算的，这里不需要再次 shift
            quality_df = data[quality_columns]
            data["sr_breakout_quality_max"] = quality_df.max(axis=1)
            data["sr_breakout_quality_sum"] = quality_df.sum(axis=1)

        if confirmation_columns:
            data["sr_breakout_confirm_any"] = (
                data[confirmation_columns].max(axis=1) > 0
            ).astype(int)

        # 【新增】：计算突破确认和角色转换特征
        # 这些特征帮助模型理解"同一个位置，在不同市场环境下会扮演完全相反的角色"
        breakout_role_features = BaselineFeatureEngineer._compute_breakout_confirmation_and_role_flip(
            data,
            boundaries,
            lookback=20,
            confirmation_bars=3,
            max_retest_bars=10,
        )
        
        for name, series in breakout_role_features.items():
            data[name] = series

        # 价格行为特征（Price Action Features）
        # 这些特征基于当前15m级别的价格行为，结合SR结构信息
        if required_features is None or any(
            "breakout_status" in f
            or "price_reversed" in f
            or "fake_breakout" in f
            or "sr_density" in f
            or "trend_context" in f
            for f in required_features or [""]
        ):
            data = BaselineFeatureEngineer._add_price_action_features(
                data, boundaries, compression_series
            )

        # 如果指定了required_features，只保留需要的特征
        if required_features is not None:
            data_cols = {
                "open",
                "high",
                "low",
                "close",
                "volume",
                "timestamp",
                "datetime",
            }
            cols_to_keep = [
                c
                for c in data.columns
                if c in data_cols
                or c in required_features
                or not pd.api.types.is_numeric_dtype(data[c])
            ]
            data = data[cols_to_keep]

        if self.enable_diagnostics:
            feature_cols = [
                c
                for c in data.columns
                if c
                not in ["open", "high", "low", "close", "volume", "timestamp", "symbol"]
            ]
            self._run_diagnostics(data, feature_cols)

        return data

    def save_scalers(self, path: str) -> None:
        """保存标准化器（仅保存配置参数，percentile 是实时计算的）"""
        import pickle

        scalers_data = {
            "percentile_window": self.percentile_window,
            "compression_threshold_pct": self.compression_threshold_pct,
            "vwap_window": self.vwap_window,
            "feature_clip_bound": self.feature_clip_bound,
            "rolling_zscore_windows": self.rolling_zscore_windows,
        }
        with open(path, "wb") as f:
            pickle.dump(scalers_data, f)
        print(f"✅ Baseline scalers saved to: {path}")

    def load_scalers(self, path: str) -> None:
        """加载标准化器（仅加载配置参数）"""
        import pickle

        with open(path, "rb") as f:
            scalers_data = pickle.load(f)
        # 向后兼容：如果存在旧的 quantiles 字段，忽略它们
        self.percentile_window = scalers_data.get("percentile_window", 288)
        self.compression_threshold_pct = scalers_data.get(
            "compression_threshold_pct", 0.2
        )
        self.vwap_window = scalers_data.get("vwap_window", 160)
        self.feature_clip_bound = scalers_data.get("feature_clip_bound", 10.0)
        # Backward compatibility: support both old single window and new multiple windows
        if "rolling_zscore_windows" in scalers_data:
            self.rolling_zscore_windows = scalers_data["rolling_zscore_windows"]
        elif "rolling_zscore_window" in scalers_data:
            # Migrate from old single window to new multiple windows
            old_window = scalers_data["rolling_zscore_window"]
            self.rolling_zscore_windows = [
                50,
                old_window,
                500,
            ]  # Add short and long term
        else:
            self.rolling_zscore_windows = [50, 288, 500]  # Default
        print(f"✅ Baseline scalers loaded from: {path}")


# ============================================================================
# 便捷函数（保持向后兼容）
# ============================================================================


def engineer_baseline_features(
    df: pd.DataFrame,
    engineer: Optional[BaselineFeatureEngineer] = None,
    *,
    fit: bool = True,
    required_features: Optional[set] = None,
) -> Tuple[pd.DataFrame, BaselineFeatureEngineer]:
    """工程 baseline 特征"""
    if engineer is None:
        engineer = BaselineFeatureEngineer()
    out = engineer.engineer_features(df, fit=fit, required_features=required_features)
    return out, engineer


def create_binary_labels_baseline(
    df: pd.DataFrame, *, forward_bars: int = 3, threshold: float = 0.005
) -> pd.DataFrame:
    """创建二分类标签"""
    df = df.copy()
    df["future_return"] = df["close"].shift(-forward_bars) / df["close"] - 1
    df["binary_signal"] = (df["future_return"] > threshold).astype(int)
    df["signal"] = df["binary_signal"]
    return df


def get_baseline_feature_columns(df: pd.DataFrame) -> List[str]:
    """获取 baseline 特征列"""
    exclude = {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "signal",
        "binary_signal",
        "future_return",
        # Note: _symbol is now included as a categorical feature (not excluded)
        # This allows the model to learn both shared patterns and asset-specific behavior
        # 排除原始布林带值（有量纲），保留归一化的 bb_position 和 bb_width
        "bb_upper",
        "bb_lower",
        "bb_middle",
        # 排除原始 ATR（有量纲），保留归一化的 atr_normalized
        "atr",
        # 排除原始 zigzag（有量纲），保留归一化的 zigzag 特征
        "zigzag",
        # 排除 returns 和 log_returns（虽然无量纲，但不同资产分布差异大）
        # 它们主要用于计算其他特征（如 volatility），不作为最终特征
        "returns",
        "log_returns",
        # 排除 price_change（有量纲），保留归一化的特征
        "price_change",
        # 排除原始均线值（有量纲），保留归一化的 _pct_close 和 _ratio 特征
        "sma_5",
        "sma_10",
        "sma_20",
        "ema_5",
        "ema_10",
        "ema_20",
        "ema_50",
        "wma_20",
        # 注意：momentum_5/10/20 是百分比（pct_change），已经是无量纲的，保留
        # 排除原始成交量值（有量纲），保留归一化的 volume_ratio, vol_ma_ratio, vol_zscore 等
        "volume_sma_20",
        "volume_sma",  # 如果存在的话
        # 排除原始 MACD 值（有量纲），保留归一化的 macd_normalized
        "macd",
        "macd_signal",
        "macd_histogram",
        # 排除原始 POC/HAL 价格值（有量纲），保留归一化的 price_to_poc_pct 等
        "poc",
        "hal_high",
        "hal_low",
        "hal_mid",
        # 排除原始 VWAP 值（有量纲），保留归一化的 price_to_vwap_* 特征
        "vwap",
        # 排除滚动高低点（有量纲），保留归一化的 swing_*_pct_close 和 sr_dist_* 特征
        "roll_high_s",
        "roll_low_s",
        "roll_high_l",
        "roll_low_l",
        # 排除 ZigZag 原始值、OLS 通道等边界值（作为中间计算使用）
        "zz_high_value",
        "zz_low_value",
        "ols_channel_mid",
        "ols_channel_upper",
        "ols_channel_lower",
        # 排除 volatility（虽然基于收益率，但不同资产分布差异大）
        "volatility",
        # 注意：时间特征保留，它们可能包含真实的时间模式信息
    }
    exclude.update(
        [
            col
            for col in df.columns
            if (
                col.startswith("signal_")
                or col.startswith("binary_signal_")
                or col.startswith("future_return_")
            )
        ]
    )
    return [c for c in df.columns if c not in exclude]


# ============================================================================
# 导出（保持向后兼容）
# ============================================================================

__all__ = [
    # BaselineFeatureEngineer 类（包含所有静态方法和特征添加方法）
    "BaselineFeatureEngineer",
    # 便捷函数
    "engineer_baseline_features",
    "get_baseline_feature_columns",
    "create_binary_labels_baseline",
]
