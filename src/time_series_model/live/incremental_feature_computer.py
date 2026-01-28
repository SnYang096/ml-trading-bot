"""
增量特征计算器（事件驱动）

用于实盘和回测，支持流式处理 tick 和 bar 数据，计算特征增量更新。
与批处理版本共享核心算法，但维护状态以支持增量计算。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any, Deque, Set, List
from collections import deque
from datetime import datetime, timedelta
import time
import os
from src.features.time_series.baseline_features import (
    compute_price_range_symmetry_from_series,
    compute_wick_ratios_from_series,
    compute_range_ratio_5bar_from_series,
)
from src.features.time_series.utils_order_flow_features import (
    extract_order_flow_features,
    compute_trade_cluster_derived_features_from_base,
    compute_vpin_derived_features_from_base,
)
from src.features.loader.feature_wrappers import compute_footprint_features
from src.features.time_series.utils_interaction_features import (
    compute_bb_width_ratio_from_price_from_series,
    compute_compression_score_from_series,
)
from src.features.time_series.utils_volatility_features import (
    extract_volume_profile_volatility_features_from_series,
)

try:
    from nautilus_trader.model import TradeTick, Bar
    from nautilus_trader.model.enums import AggressorSide

    NAUTILUS_AVAILABLE = True
except ImportError:
    NAUTILUS_AVAILABLE = False
    TradeTick = None
    Bar = None
    AggressorSide = None


class IncrementalFeatureComputer:
    """
    增量特征计算器

    支持：
    1. Tick 级特征（VPIN、订单流不平衡等）
    2. Bar 级特征（技术指标、时间框架特征等）
    3. 状态维护（滚动窗口、跨月连续性等）
    """

    def __init__(
        self,
        tick_window_minutes: int = 30,
        bar_window_size: int = 1000,
        vpin_bucket_volume: Optional[float] = None,
        vpin_bucket_volume_usd: Optional[float] = None,
        vpin_n_buckets: int = 50,
        live_feature_plan_path: Optional[str] = None,
        primary_timeframe: Optional[str] = None,
    ):
        """
        Args:
            tick_window_minutes: Tick 数据保留窗口（分钟）
            bar_window_size: Bar 数据保留窗口（条数）
            vpin_bucket_volume: VPIN bucket volume（数量）
            vpin_bucket_volume_usd: VPIN bucket volume（USD）
            vpin_n_buckets: VPIN 滚动窗口大小
        """
        self.tick_window_minutes = tick_window_minutes
        self.bar_window_size = bar_window_size

        # Tick 缓冲区
        self.tick_buffer: Deque[Dict[str, Any]] = deque(maxlen=20000)

        # CVD per-bar accumulation (tick-driven)
        self._cvd_bar_delta: float = 0.0
        self._cvd_bar_total_flow: float = 0.0
        self._cvd_cum: float = 0.0
        self._cvd_change_hist: Deque[float] = deque(maxlen=20)
        self._cvd_total_flow_hist: Deque[float] = deque(maxlen=20)

        # Bar 缓冲区
        self.bar_buffer: Deque[Dict[str, Any]] = deque(maxlen=bar_window_size)

        # VPIN 状态
        self.vpin_bucket_volume = vpin_bucket_volume
        self.vpin_bucket_volume_usd = vpin_bucket_volume_usd
        self.vpin_n_buckets = vpin_n_buckets
        self.vpin_buckets: Deque[tuple] = deque(
            maxlen=vpin_n_buckets * 2
        )  # (timestamp, vpin_value)
        self.vpin_bucket_state = {
            "current_buy": 0.0,
            "current_sell": 0.0,
            "filled_value": 0.0,
        }

        # 当前特征值
        self.current_features: Dict[str, float] = {}

        # 时间框架特征（按时间框架分组）
        self.timeframe_features: Dict[str, Dict[str, float]] = {}

        # Live feature plan (optional)
        self.primary_timeframe = str(primary_timeframe) if primary_timeframe else None
        self.live_feature_set: Set[str] = set()
        try:
            from src.time_series_model.live.live_feature_plan import (
                load_live_feature_plan,
                load_live_feature_nodes,
            )

            plan_path = (
                live_feature_plan_path
                if live_feature_plan_path is not None
                else os.getenv(
                    "MLBOT_LIVE_FEATURE_PLAN_YAML",
                    "config/live/live_feature_plan.yaml",
                )
            )
            self.live_feature_set = load_live_feature_plan(plan_path=plan_path) or set()
            self.live_feature_nodes = load_live_feature_nodes(plan_path=plan_path) or []
        except Exception:
            self.live_feature_set = set()
            self.live_feature_nodes = []

        self._feature_loader = None
        self._feature_deps = None
        self._last_missing_log_ts: Optional[float] = None
        self._last_skipped_nodes: List[str] = []
        if self.live_feature_nodes:
            try:
                from src.features.loader.strategy_feature_loader import (
                    StrategyFeatureLoader,
                )

                self._feature_loader = StrategyFeatureLoader(
                    feature_deps_path="config/feature_dependencies.yaml",
                    strategy_config_path=None,
                    cache_dir=None,
                    use_disk_cache=False,
                    use_memory_cache=False,
                    use_monthly_cache=False,
                    max_workers=None,
                    parallel_backend="process",
                    normalization_contract_mode="warn",
                )
                self._feature_deps = self._feature_loader.feature_deps or {}
            except Exception:
                self._feature_loader = None
                self._feature_deps = None

    def _want(self, key: str) -> bool:
        return (not self.live_feature_set) or (key in self.live_feature_set)

    def on_tick(self, tick: Any) -> None:
        """
        处理 tick 数据

        Args:
            tick: TradeTick 对象或字典
        """
        if not NAUTILUS_AVAILABLE:
            return

        # 转换为统一格式
        if isinstance(tick, TradeTick):
            tick_data = {
                "ts": tick.ts_event,
                "price": float(tick.price),
                "volume": float(tick.size),
                "side": 1 if tick.aggressor_side == AggressorSide.BUYER else -1,
            }
        elif isinstance(tick, dict):
            tick_data = tick
        else:
            return

        # 添加到缓冲区
        self.tick_buffer.append(tick_data)

        # CVD accumulation for the current bar
        side = tick_data.get("side")
        volume = tick_data.get("volume")
        if side in (1, -1) and volume is not None:
            vol = float(volume)
            self._cvd_bar_delta += vol if side == 1 else -vol
            self._cvd_bar_total_flow += abs(vol)
            self._cvd_cum += vol if side == 1 else -vol

        # 更新 VPIN
        self._update_vpin(tick_data)

        # 更新订单流特征
        self._update_orderflow_features()

    def on_bar(self, bar: Any, timeframe: str = "1H") -> None:
        """
        处理 bar 数据

        Args:
            bar: Bar 对象或字典
            timeframe: 时间框架（如 "15T", "1H"）
        """
        if not NAUTILUS_AVAILABLE:
            return

        # 转换为统一格式
        if isinstance(bar, Bar):
            bar_data = {
                "ts": bar.ts_event,
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            }
        elif isinstance(bar, dict):
            bar_data = bar
        else:
            return

        # Attach tick-driven orderflow to bar (if available)
        bar_data["cvd_change_1"] = float(self._cvd_bar_delta)
        bar_data["cvd"] = float(self._cvd_cum)
        self._cvd_change_hist.append(float(self._cvd_bar_delta))
        self._cvd_total_flow_hist.append(float(self._cvd_bar_total_flow))
        if len(self._cvd_change_hist) > 0:
            cvd_change_5 = float(np.sum(list(self._cvd_change_hist)[-5:]))
        else:
            cvd_change_5 = 0.0
        if len(self._cvd_total_flow_hist) > 0:
            total_flow_5 = float(np.sum(list(self._cvd_total_flow_hist)[-5:]))
        else:
            total_flow_5 = 0.0
        bar_data["cvd_change_5"] = cvd_change_5
        if total_flow_5 > 0:
            bar_data["cvd_change_5_normalized"] = cvd_change_5 / total_flow_5
        else:
            bar_data["cvd_change_5_normalized"] = 0.0
        # reset accumulators for next bar
        self._cvd_bar_delta = 0.0
        self._cvd_bar_total_flow = 0.0

        # 添加到缓冲区
        self.bar_buffer.append(bar_data)

        # 更新时间框架特征
        self.primary_timeframe = str(timeframe)
        self._update_timeframe_features(bar_data, timeframe)

    def _update_vpin(self, tick_data: Dict[str, Any]) -> None:
        """更新 VPIN（增量计算）"""
        if self.vpin_bucket_volume is None and self.vpin_bucket_volume_usd is None:
            return

        price = tick_data["price"]
        volume = tick_data["volume"]
        side = tick_data["side"]

        # 确定 bucket volume
        if self.vpin_bucket_volume_usd is not None:
            tick_value = price * volume
            target_bucket = self.vpin_bucket_volume_usd
        else:
            tick_value = volume
            target_bucket = self.vpin_bucket_volume

        # 更新 bucket 状态
        remaining = tick_value
        while remaining > 0:
            space_left = target_bucket - self.vpin_bucket_state["filled_value"]
            trade_value = min(remaining, space_left)

            if side == 1:
                self.vpin_bucket_state["current_buy"] += trade_value
            else:
                self.vpin_bucket_state["current_sell"] += trade_value

            self.vpin_bucket_state["filled_value"] += trade_value
            remaining -= trade_value

            # 检查 bucket 是否填满
            BUCKET_COMPLETION_TOLERANCE = 1e-6
            if (
                self.vpin_bucket_state["filled_value"]
                >= target_bucket - BUCKET_COMPLETION_TOLERANCE
            ):
                # 计算 VPIN
                imbalance = abs(
                    self.vpin_bucket_state["current_buy"]
                    - self.vpin_bucket_state["current_sell"]
                )
                vpin_value = imbalance / target_bucket

                # 添加到 buckets
                self.vpin_buckets.append((tick_data["ts"], vpin_value))

                # 重置 bucket 状态
                self.vpin_bucket_state = {
                    "current_buy": 0.0,
                    "current_sell": 0.0,
                    "filled_value": 0.0,
                }

        # 更新当前 VPIN（滚动平均）
        if len(self.vpin_buckets) > 0:
            recent_buckets = list(self.vpin_buckets)[-self.vpin_n_buckets :]
            if recent_buckets:
                vpin_values = [v for _, v in recent_buckets]
                self.current_features["vpin"] = float(np.mean(vpin_values))
            else:
                self.current_features["vpin"] = 0.0
        else:
            self.current_features["vpin"] = 0.0

    def _update_orderflow_features(self) -> None:
        """更新订单流特征（基于最近 N 分钟的 tick）"""
        if not self.tick_buffer:
            return

        # 获取最近 N 分钟的 tick
        cutoff_ns = (
            self.tick_buffer[-1]["ts"] - self.tick_window_minutes * 60 * 1_000_000_000
        )
        recent_ticks = [t for t in self.tick_buffer if t["ts"] >= cutoff_ns]

        if not recent_ticks:
            return

        # 计算买卖量
        buy_vol = sum(t["volume"] for t in recent_ticks if t["side"] == 1)
        sell_vol = sum(t["volume"] for t in recent_ticks if t["side"] == -1)
        total_vol = buy_vol + sell_vol

        # 计算不平衡度
        if total_vol > 0:
            imbalance = (buy_vol - sell_vol) / total_vol
            self.current_features["orderflow_imbalance"] = float(imbalance)
            self.current_features["orderflow_total_vol"] = float(total_vol)
        else:
            self.current_features["orderflow_imbalance"] = 0.0
            self.current_features["orderflow_total_vol"] = 0.0

    def _update_timeframe_features(
        self, bar_data: Dict[str, Any], timeframe: str
    ) -> None:
        """更新时间框架特征（技术指标等）"""
        if timeframe not in self.timeframe_features:
            self.timeframe_features[timeframe] = {}

        # 转换为 DataFrame 用于计算
        bars_df = pd.DataFrame(list(self.bar_buffer))
        if len(bars_df) < 2:
            return
        bars_df_indexed = bars_df.copy()
        bars_df_indexed.index = pd.to_datetime(
            bars_df_indexed["ts"], unit="ns", utc=True
        )

        # 计算简单技术指标（示例）
        closes = bars_df["close"].values

        # RSI（简化版，需要至少 14 根 bar）
        if len(closes) >= 14:
            delta = np.diff(closes)
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)

            avg_gain = np.mean(gain[-14:])
            avg_loss = np.mean(loss[-14:])

            if avg_loss > 0:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                if self._want("rsi"):
                    self.timeframe_features[timeframe]["rsi"] = float(rsi)

        # ATR（简化版）
        if len(bars_df) >= 14:
            high = bars_df["high"].values[-14:]
            low = bars_df["low"].values[-14:]
            close = bars_df["close"].values[-14:]
            close_prev = (
                bars_df["close"].values[-15:-1] if len(bars_df) > 14 else close[:-1]
            )

            # 确保数组长度一致
            min_len = min(len(high), len(low), len(close), len(close_prev))
            if min_len > 0:
                high = high[-min_len:]
                low = low[-min_len:]
                close = close[-min_len:]
                close_prev = (
                    close_prev[-min_len:] if len(close_prev) >= min_len else close[:-1]
                )

                tr1 = high - low
                tr2 = np.abs(high - close_prev)
                tr3 = np.abs(low - close_prev)

                tr = np.maximum(tr1, np.maximum(tr2, tr3))
                atr = np.mean(tr)
                if self._want("atr"):
                    self.timeframe_features[timeframe]["atr"] = float(atr)

        # 价格位置
        if self._want("open"):
            self.timeframe_features[timeframe]["open"] = float(bar_data["open"])
        if self._want("high"):
            self.timeframe_features[timeframe]["high"] = float(bar_data["high"])
        if self._want("low"):
            self.timeframe_features[timeframe]["low"] = float(bar_data["low"])
        if self._want("close"):
            self.timeframe_features[timeframe]["close"] = float(bar_data["close"])
        if self._want("volume"):
            self.timeframe_features[timeframe]["volume"] = float(bar_data["volume"])

        # Wick ratios and price range symmetry
        try:
            if self._want("price_range_symmetry"):
                sym = compute_price_range_symmetry_from_series(
                    high=bars_df["high"],
                    low=bars_df["low"],
                    close=bars_df["close"],
                )
                self.timeframe_features[timeframe]["price_range_symmetry"] = float(
                    sym.iloc[-1]
                )
                bars_df_indexed["price_range_symmetry"] = sym
            if self._want("wick_upper_ratio") or self._want("wick_lower_ratio"):
                wick_df = compute_wick_ratios_from_series(
                    open=bars_df["open"],
                    high=bars_df["high"],
                    low=bars_df["low"],
                    close=bars_df["close"],
                )
                if self._want("wick_upper_ratio"):
                    self.timeframe_features[timeframe]["wick_upper_ratio"] = float(
                        wick_df["wick_upper_ratio"].iloc[-1]
                    )
                if self._want("wick_lower_ratio"):
                    self.timeframe_features[timeframe]["wick_lower_ratio"] = float(
                        wick_df["wick_lower_ratio"].iloc[-1]
                    )
                bars_df_indexed["wick_upper_ratio"] = wick_df["wick_upper_ratio"]
                bars_df_indexed["wick_lower_ratio"] = wick_df["wick_lower_ratio"]
        except Exception:
            pass

        # Tick-driven orderflow and footprint features (bar-aligned)
        need_orderflow = any(
            k.startswith("vpin") or k.startswith("trade_cluster")
            for k in (self.live_feature_set or [])
        ) or any(
            "vpin" in str(n) or "trade_cluster" in str(n)
            for n in (self.live_feature_nodes or [])
        )
        need_footprint = any(
            k.startswith("fp_") for k in (self.live_feature_set or [])
        ) or any("footprint" in str(n) for n in (self.live_feature_nodes or []))
        if (need_orderflow or need_footprint) and self.tick_buffer:
            try:
                ticks_df = pd.DataFrame(list(self.tick_buffer))
                if not ticks_df.empty and "ts" in ticks_df.columns:
                    ticks_df.index = pd.to_datetime(ticks_df["ts"], unit="ns", utc=True)
                if need_orderflow and not ticks_df.empty:
                    of_df = extract_order_flow_features(
                        bars_df_indexed,
                        ticks=ticks_df,
                        freq=pd.infer_freq(bars_df_indexed.index),
                        include_trade_clustering=True,
                        compute_vpin_derived=True,
                        vpin_bucket_volume=self.vpin_bucket_volume,
                        vpin_bucket_volume_usd=self.vpin_bucket_volume_usd,
                        vpin_n_buckets=self.vpin_n_buckets,
                    )
                    bars_df_indexed = bars_df_indexed.join(of_df, how="left")
                    try:
                        vpin_derived = compute_vpin_derived_features_from_base(
                            bars_df_indexed
                        )
                        for c in vpin_derived.columns:
                            if c not in bars_df_indexed.columns:
                                bars_df_indexed[c] = vpin_derived[c]
                    except Exception:
                        pass
                    try:
                        tc_derived = compute_trade_cluster_derived_features_from_base(
                            bars_df_indexed
                        )
                        for c in tc_derived.columns:
                            if c not in bars_df_indexed.columns:
                                bars_df_indexed[c] = tc_derived[c]
                    except Exception:
                        pass
                    last = of_df.iloc[-1].to_dict()
                    for k, v in last.items():
                        if self._want(str(k)):
                            try:
                                self.timeframe_features[timeframe][str(k)] = float(v)
                            except Exception:
                                continue
                if need_footprint and not ticks_df.empty:
                    fp_df = compute_footprint_features(
                        bars_df_indexed.tail(200),
                        ticks=ticks_df,
                        persist_monthly=False,
                    )
                    bars_df_indexed = bars_df_indexed.join(fp_df, how="left")
                    last = fp_df.iloc[-1].to_dict()
                    for k, v in last.items():
                        if self._want(str(k)):
                            try:
                                self.timeframe_features[timeframe][str(k)] = float(v)
                            except Exception:
                                continue
                    # Ensure footprint columns are available for downstream nodes
                    for c in fp_df.columns:
                        if c not in bars_df_indexed.columns:
                            bars_df_indexed[c] = fp_df[c]
            except Exception:
                pass

        # BB width ratio (needed by compression_score_f)
        try:
            if "bb_width_ratio" not in bars_df_indexed.columns:
                bb_ratio = compute_bb_width_ratio_from_price_from_series(
                    close=bars_df_indexed["close"]
                )
                bars_df_indexed["bb_width_ratio"] = bb_ratio
            if "compression_score" not in bars_df_indexed.columns:
                comp = compute_compression_score_from_series(
                    bb_width_ratio=bars_df_indexed["bb_width_ratio"]
                )
                bars_df_indexed["compression_score"] = comp
                if self._want("compression_score"):
                    self.timeframe_features[timeframe]["compression_score"] = float(
                        comp.iloc[-1]
                    )
        except Exception:
            pass

        # Range ratio (5-bar)
        try:
            if "range_ratio_5bar" not in bars_df_indexed.columns:
                rr = compute_range_ratio_5bar_from_series(
                    high=bars_df_indexed["high"], low=bars_df_indexed["low"]
                )
                bars_df_indexed["range_ratio_5bar"] = rr
                if self._want("range_ratio_5bar"):
                    self.timeframe_features[timeframe]["range_ratio_5bar"] = float(
                        rr.iloc[-1]
                    )
        except Exception:
            pass

        # Volume profile volatility features (vp_* entropy/skewness etc.)
        try:
            if any(k.startswith("vp_") for k in (self.live_feature_set or [])):
                vp_df = extract_volume_profile_volatility_features_from_series(
                    close=bars_df_indexed["close"],
                    volume=bars_df_indexed["volume"],
                )
                bars_df_indexed = bars_df_indexed.join(vp_df, how="left")
                last = vp_df.iloc[-1].to_dict()
                for k, v in last.items():
                    if self._want(str(k)):
                        try:
                            self.timeframe_features[timeframe][str(k)] = float(v)
                        except Exception:
                            continue
        except Exception:
            pass

        # CVD change (5 bars) + normalized, if underlying flow columns exist
        if self._want("cvd_change_5") or self._want("cvd_change_5_normalized"):
            net_buy = None
            total_flow = None
            if "buy_qty" in bars_df.columns and "sell_qty" in bars_df.columns:
                buy = pd.to_numeric(bars_df["buy_qty"], errors="coerce").fillna(0.0)
                sell = pd.to_numeric(bars_df["sell_qty"], errors="coerce").fillna(0.0)
                net_buy = buy - sell
                total_flow = (buy + sell).replace(0, np.nan)
            elif "cvd_change_1" in bars_df.columns:
                net_buy = pd.to_numeric(
                    bars_df["cvd_change_1"], errors="coerce"
                ).fillna(0.0)
                total_flow = pd.to_numeric(
                    bars_df.get("volume", 0.0), errors="coerce"
                ).fillna(0.0)
                total_flow = total_flow.replace(0, np.nan)
            elif "cvd" in bars_df.columns:
                cvd = pd.to_numeric(bars_df["cvd"], errors="coerce").fillna(0.0)
                net_buy = cvd.diff().fillna(0.0)
                total_flow = pd.to_numeric(
                    bars_df.get("volume", 0.0), errors="coerce"
                ).fillna(0.0)
                total_flow = total_flow.replace(0, np.nan)

            if net_buy is not None:
                cvd_change_5 = net_buy.rolling(window=5, min_periods=1).sum()
                if self._want("cvd_change_5"):
                    self.timeframe_features[timeframe]["cvd_change_5"] = float(
                        cvd_change_5.iloc[-1]
                    )
                bars_df_indexed["cvd_change_5"] = cvd_change_5
                if self._want("cvd_change_5_normalized"):
                    if total_flow is None:
                        cvd_norm = 0.0
                    else:
                        total_flow_5 = total_flow.rolling(window=5, min_periods=1).sum()
                        cvd_norm = (
                            (cvd_change_5 / total_flow_5)
                            .replace([np.inf, -np.inf], np.nan)
                            .fillna(0.0)
                        )
                    self.timeframe_features[timeframe]["cvd_change_5_normalized"] = (
                        float(
                            cvd_norm.iloc[-1] if hasattr(cvd_norm, "iloc") else cvd_norm
                        )
                    )
                    if hasattr(cvd_norm, "iloc"):
                        bars_df_indexed["cvd_change_5_normalized"] = cvd_norm

        # Extra live features from base training plan
        try:
            from src.features.time_series.baseline_features import (
                compute_atr_percentile,
                compute_bb_width_features_from_series,
                compute_trend_r2_20_from_series,
                compute_volume_ratio_from_series,
            )
            from src.features.time_series.utils_volume_profile import (
                compute_volume_profile_vpvr_from_series,
            )

            df = bars_df.copy()
            if self._want("trend_r2_20"):
                out = compute_trend_r2_20_from_series(close=df["close"])
                val = out.iloc[-1]
                if isinstance(val, pd.Series):
                    val = val.iloc[0]
                self.timeframe_features[timeframe]["trend_r2_20"] = float(val)
                bars_df_indexed["trend_r2_20"] = out
            if self._want("volume_ratio"):
                out = compute_volume_ratio_from_series(volume=df["volume"])
                self.timeframe_features[timeframe]["volume_ratio"] = float(
                    out["volume_ratio"].iloc[-1]
                )
                bars_df_indexed["volume_ratio"] = out["volume_ratio"]
            if self._want("bb_width_normalized") or self._want("bb_position"):
                out = compute_bb_width_features_from_series(
                    close=df["close"],
                    high=df["high"],
                    low=df["low"],
                )
                if self._want("bb_width_normalized"):
                    self.timeframe_features[timeframe]["bb_width_normalized"] = float(
                        out["bb_width_normalized"].iloc[-1]
                    )
                    bars_df_indexed["bb_width_normalized"] = out["bb_width_normalized"]
                if self._want("bb_position"):
                    self.timeframe_features[timeframe]["bb_position"] = float(
                        out["bb_position"].iloc[-1]
                    )
                    bars_df_indexed["bb_position"] = out["bb_position"]
            if self._want("atr_percentile"):
                out = compute_atr_percentile(df)
                self.timeframe_features[timeframe]["atr_percentile"] = float(
                    out["atr_percentile"].iloc[-1]
                )
                bars_df_indexed["atr_percentile"] = out["atr_percentile"]
            if any(k.startswith("vpvr_") for k in self.live_feature_set) or self._want(
                "vpvr_pvp"
            ):
                out = compute_volume_profile_vpvr_from_series(
                    close=df["close"],
                    high=df["high"],
                    low=df["low"],
                    volume=df["volume"],
                )
                last = out.iloc[-1].to_dict()
                for k, v in last.items():
                    if self._want(str(k)):
                        self.timeframe_features[timeframe][str(k)] = float(v)
                for k in out.columns:
                    bars_df_indexed[str(k)] = out[k]
        except Exception:
            pass

        # Batch fallback: compute tier features using FeatureComputer on bar buffer.
        if self._feature_loader is not None and self.live_feature_nodes:
            try:
                req = list(self.live_feature_nodes)
                # Filter nodes that require columns not present in bars_df.
                feats_cfg = (self._feature_deps or {}).get("features") or {}
                bar_cols = set(bars_df_indexed.columns)
                filtered = []
                skipped = []
                for n in req:
                    info = feats_cfg.get(n)
                    if isinstance(info, dict):
                        req_cols = set(info.get("required_columns") or [])
                        deps = info.get("dependencies") or []
                        if req_cols and not req_cols.issubset(bar_cols) and not deps:
                            skipped.append(str(n))
                            continue
                    filtered.append(n)
                self._last_skipped_nodes = skipped
                df2 = bars_df_indexed.copy()
                df2 = self._feature_loader.load_features_from_requested(
                    df2, requested_features=filtered, fit=False
                )
                last = df2.iloc[-1].to_dict()
                for k, v in last.items():
                    if self._want(str(k)):
                        try:
                            self.timeframe_features[timeframe][str(k)] = float(v)
                        except Exception:
                            continue
            except Exception:
                pass

        # Final backfill: include any computed columns in the live feature set
        try:
            last_row = bars_df_indexed.iloc[-1].to_dict()
            for k in self.live_feature_set:
                if k in last_row and k not in self.timeframe_features[timeframe]:
                    v = last_row.get(k)
                    if v is None:
                        continue
                    try:
                        if np.isscalar(v):
                            self.timeframe_features[timeframe][str(k)] = float(v)
                    except Exception:
                        continue
        except Exception:
            pass

    def get_features(self) -> Dict[str, float]:
        """
        获取当前所有特征

        Returns:
            特征字典
        """
        features = {}

        # Tick 级特征
        for k, v in (self.current_features or {}).items():
            if self._want(str(k)):
                features[str(k)] = float(v)

        # 时间框架特征（扁平化）
        for tf, tf_features in self.timeframe_features.items():
            for key, value in tf_features.items():
                pref = f"{tf}_{key}"
                if self._want(pref):
                    features[pref] = value
                if self.primary_timeframe and str(tf) == str(self.primary_timeframe):
                    if self._want(str(key)):
                        features[str(key)] = value

        self._log_missing_features(features)

        return features

    def _log_missing_features(self, features: Dict[str, float]) -> None:
        if not self.live_feature_set:
            return
        missing = [k for k in self.live_feature_set if k not in features]
        now = time.time()
        if self._last_missing_log_ts is None or now - self._last_missing_log_ts >= 60:
            if missing:
                preview = ", ".join(missing[:10])
                print(f"⚠️ live_feature_missing ({len(missing)}): {preview}")
            if self._last_skipped_nodes:
                preview = ", ".join(self._last_skipped_nodes[:10])
                print(
                    f"⚠️ live_feature_nodes_skipped ({len(self._last_skipped_nodes)}): {preview}"
                )
            self._last_missing_log_ts = now

    def get_orderflow_features(self, window_minutes: int = 15) -> Dict[str, float]:
        """
        获取订单流特征（指定时间窗口）

        Args:
            window_minutes: 时间窗口（分钟）

        Returns:
            订单流特征字典
        """
        if not self.tick_buffer:
            return {"vpin": 0.0, "imbalance": 0.0, "total_vol": 0.0}

        cutoff_ns = self.tick_buffer[-1]["ts"] - window_minutes * 60 * 1_000_000_000
        recent_ticks = [t for t in self.tick_buffer if t["ts"] >= cutoff_ns]

        if not recent_ticks:
            return {"vpin": 0.0, "imbalance": 0.0, "total_vol": 0.0}

        buy_vol = sum(t["volume"] for t in recent_ticks if t["side"] == 1)
        sell_vol = sum(t["volume"] for t in recent_ticks if t["side"] == -1)
        total_vol = buy_vol + sell_vol

        vpin = self.current_features.get("vpin", 0.0)
        imbalance = (buy_vol - sell_vol) / (total_vol + 1e-12) if total_vol > 0 else 0.0

        return {
            "vpin": float(vpin),
            "imbalance": float(imbalance),
            "total_vol": float(total_vol),
        }

    def get_recent_bars(self, n: int = 200) -> list[Dict[str, Any]]:
        """
        Return recent bar records (oldest -> newest).

        Used by heuristic execution rules to compute simple structure signals
        without introducing full batch feature dependencies.
        """
        if n <= 0:
            return []
        xs = list(self.bar_buffer)
        if not xs:
            return []
        return xs[-int(n) :]

    def get_last_tick_ts_ns(self) -> Optional[int]:
        if not self.tick_buffer:
            return None
        try:
            return int(self.tick_buffer[-1]["ts"])
        except Exception:
            return None

    def reset(self) -> None:
        """重置所有状态"""
        self.tick_buffer.clear()
        self.bar_buffer.clear()
        self.vpin_buckets.clear()
        self.vpin_bucket_state = {
            "current_buy": 0.0,
            "current_sell": 0.0,
            "filled_value": 0.0,
        }
        self.current_features.clear()
        self.timeframe_features.clear()
