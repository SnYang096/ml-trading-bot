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

# Nautilus Trader types (optional — dict format is always supported)
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
        archetypes_dir: Optional[str] = None,
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
        self.live_feature_nodes: List[str] = []

        # Strategy A: archetypes auto-detect (preferred — no NN dependency)
        if archetypes_dir is not None:
            try:
                from src.time_series_model.live.live_feature_plan import (
                    extract_features_from_archetypes,
                )

                feat_set, feat_nodes = extract_features_from_archetypes(
                    archetypes_dir=archetypes_dir,
                )
                self.live_feature_set = feat_set
                self.live_feature_nodes = feat_nodes
                print(
                    f"   \U0001f4cb Archetypes auto-detect: "
                    f"{len(feat_set)} columns, {len(feat_nodes)} nodes "
                    f"(from {archetypes_dir})"
                )
            except Exception as e:
                print(f"   \u26a0\ufe0f Archetypes auto-detect failed: {e}")
                self.live_feature_set = set()
                self.live_feature_nodes = []
        else:
            # Strategy B: legacy live_feature_plan.yaml (NN tier-based)
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
                self.live_feature_set = (
                    load_live_feature_plan(plan_path=plan_path) or set()
                )
                self.live_feature_nodes = (
                    load_live_feature_nodes(plan_path=plan_path) or []
                )
            except Exception:
                self.live_feature_set = set()
                self.live_feature_nodes = []

        self._feature_loader = None
        self._feature_deps = None
        self._last_missing_log_ts: Optional[float] = None
        self._last_skipped_nodes: List[str] = []
        self._warmup_mode: bool = False  # warmup 期间跳过重型特征计算
        self._batch_features: Dict[str, float] = {}  # 批量计算结果缓存
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
                    verbose=False,  # 实盘模式：只打印摘要+异常
                )
                self._feature_deps = self._feature_loader.feature_deps or {}
            except Exception:
                self._feature_loader = None
                self._feature_deps = None

    def _want(self, key: str) -> bool:
        return (not self.live_feature_set) or (key in self.live_feature_set)

    def _detect_tick_dependent_nodes(self, feats_cfg: dict) -> set:
        """从 feature_dependencies 动态检测需要 tick 数据的特征节点。

        通过 inspect.signature 检查 compute_func 是否接受 `ticks` 或
        `ticks_loader_json` 参数来判断，与 train_strategy_pipeline.py 中
        的检测逻辑保持一致，避免硬编码节点名。
        """
        if hasattr(self, "_tick_dependent_nodes_cache"):
            return self._tick_dependent_nodes_cache

        import inspect

        tick_nodes: set = set()
        try:
            from src.features.registry import get_compute_func
        except ImportError:
            self._tick_dependent_nodes_cache = tick_nodes
            return tick_nodes

        for node_name, node_cfg in feats_cfg.items():
            if not isinstance(node_cfg, dict):
                continue
            compute_func_name = node_cfg.get("compute_func")
            if not compute_func_name:
                continue
            try:
                cfn = get_compute_func(compute_func_name)
                if cfn is None:
                    continue
                sig = inspect.signature(cfn)
                if "ticks" in sig.parameters or "ticks_loader_json" in sig.parameters:
                    tick_nodes.add(node_name)
            except Exception:
                continue

        self._tick_dependent_nodes_cache = tick_nodes
        return tick_nodes

    def on_tick(self, tick: Any) -> None:
        """
        处理 tick 数据

        仅维护 CVD 累加器（用于 1min bar 的 cvd_change_1/cvd 列）。
        VPIN / 订单流特征已改为磁盘批量计算（compute_features_batch），
        不再在 tick 级增量更新。

        Args:
            tick: dict 或 TradeTick 对象
        """
        # 转换为统一格式
        if isinstance(tick, dict):
            tick_data = tick
        elif (
            NAUTILUS_AVAILABLE and TradeTick is not None and isinstance(tick, TradeTick)
        ):
            tick_data = {
                "ts": tick.ts_event,
                "price": float(tick.price),
                "volume": float(tick.size),
                "side": 1 if tick.aggressor_side == AggressorSide.BUYER else -1,
            }
        else:
            return

        # 添加到缓冲区（保留用于 get_orderflow_features 实时查询）
        self.tick_buffer.append(tick_data)

        # CVD accumulation for the current bar
        side = tick_data.get("side")
        volume = tick_data.get("volume")
        if side in (1, -1) and volume is not None:
            vol = float(volume)
            self._cvd_bar_delta += vol if side == 1 else -vol
            self._cvd_bar_total_flow += abs(vol)
            self._cvd_cum += vol if side == 1 else -vol

        # NOTE: _update_vpin / _update_orderflow_features 已移除
        # VPIN 和订单流特征现在通过 compute_features_batch() 从磁盘批量计算

    def on_bar(self, bar: Any, timeframe: str = "1H") -> None:
        """
        处理 bar 数据

        Args:
            bar: dict 或 Bar 对象
            timeframe: 时间框架（如 "15T", "1H")
        """
        # 转换为统一格式
        if isinstance(bar, dict):
            bar_data = bar
        elif NAUTILUS_AVAILABLE and Bar is not None and isinstance(bar, Bar):
            bar_data = {
                "ts": bar.ts_event,
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            }
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

        # 添加到缓冲区（用于 get_recent_bars() — BPC 执行规则需要）
        self.bar_buffer.append(bar_data)

        # NOTE: 特征计算已改为磁盘批量模式 (compute_features_batch)
        # on_bar 只维护 bar_buffer + CVD，不再触发 _update_timeframe_features

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
        获取当前所有特征（兼容接口）

        注意：实盘主路径已改为 compute_features_batch()，
        此方法保留用于后向兼容。

        Returns:
            特征字典
        """
        # 优先返回批量计算结果（如果有）
        if self._batch_features:
            return dict(self._batch_features)

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

    # ── 共享核心：步骤 1-3 的单一实现 ──────────────────────────
    def _compute_features_core(
        self,
        bars_1min: pd.DataFrame,
        ticks_1min: pd.DataFrame,
        primary_timeframe: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """共享的特征计算核心 (steps 1-3).

        compute_features_batch 和 compute_features_dataframe 的唯一入口，
        消除代码重复，确保逻辑一致。

        Returns:
            bars_tf DataFrame (所有计算完的列)，或 None（数据不足）。
        """
        import logging

        _logger = logging.getLogger(__name__)

        ptf = primary_timeframe or self.primary_timeframe or "240T"

        if bars_1min is None or bars_1min.empty:
            _logger.warning("_compute_features_core: bars_1min is empty")
            return None

        # ── 1. 重采样 1min bars → primary_timeframe (4h) ──
        bars = bars_1min.copy()

        # 确保 DatetimeIndex
        if not isinstance(bars.index, pd.DatetimeIndex):
            if "timestamp" in bars.columns:
                bars.index = pd.to_datetime(bars["timestamp"], utc=True)
            else:
                _logger.warning("_compute_features_core: no timestamp column")
                return None

        # 确保 tz-aware (UTC)
        if bars.index.tz is None:
            bars.index = bars.index.tz_localize("UTC")

        # 重采样
        agg_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        for col, agg in [
            ("buy_volume", "sum"),
            ("sell_volume", "sum"),
            ("buy_count", "sum"),
            ("sell_count", "sum"),
            ("trade_count", "sum"),
            ("delta", "sum"),
        ]:
            if col in bars.columns:
                agg_dict[col] = agg

        bars_tf = bars.resample(ptf).agg(agg_dict).dropna(subset=["close"])

        # 添加 buy_qty / sell_qty（研发格式，用于 CVD 等特征节点）
        if "buy_volume" in bars_tf.columns and "sell_volume" in bars_tf.columns:
            bars_tf["buy_qty"] = bars_tf["buy_volume"]
            bars_tf["sell_qty"] = bars_tf["sell_volume"]

            # 计算 CVD 相关特征（研发格式）
            delta = bars_tf["buy_volume"] - bars_tf["sell_volume"]
            total_flow = bars_tf["buy_volume"] + bars_tf["sell_volume"]

            bars_tf["cvd_change_1"] = delta
            bars_tf["cvd_change_5"] = delta.rolling(window=5, min_periods=1).sum()
            bars_tf["cvd_change_20"] = delta.rolling(window=20, min_periods=1).sum()
            bars_tf["cvd_roll20"] = delta.rolling(window=20, min_periods=1).sum()
            bars_tf["cvd_roll60"] = delta.rolling(window=60, min_periods=1).sum()
            bars_tf["cvd_roll288"] = delta.rolling(window=288, min_periods=1).sum()
            bars_tf["cvd"] = delta.cumsum()
            bars_tf["cvd_normalized"] = (delta / total_flow.replace(0, np.nan)).fillna(
                0
            )
            total_flow_5 = total_flow.rolling(window=5, min_periods=1).sum()
            bars_tf["cvd_change_5_normalized"] = (
                bars_tf["cvd_change_5"] / total_flow_5.replace(0, np.nan)
            ).fillna(0)
            bars_tf["taker_buy_ratio"] = (
                bars_tf["buy_volume"] / total_flow.replace(0, np.nan)
            ).fillna(0.5)

        _logger.info(
            "_compute_features_core: %d 1min bars → %d %s bars",
            len(bars_1min),
            len(bars_tf),
            ptf,
        )

        if len(bars_tf) < 10:
            _logger.warning("_compute_features_core: too few bars (%d)", len(bars_tf))
            return None

        # ── 2. 计算订单流特征 (VPIN / Trade Clustering / Footprint) ──
        if ticks_1min is not None and not ticks_1min.empty:
            try:
                ticks = ticks_1min.copy()
                if not isinstance(ticks.index, pd.DatetimeIndex):
                    if "timestamp" in ticks.columns:
                        ticks.index = pd.to_datetime(ticks["timestamp"], utc=True)
                if ticks.index.tz is None:
                    ticks.index = ticks.index.tz_localize("UTC")

                of_df = extract_order_flow_features(
                    bars_tf,
                    ticks=ticks,
                    freq=ptf,
                    include_trade_clustering=True,
                    compute_vpin_derived=True,
                )
                existing_cols = set(bars_tf.columns)
                new_cols = [c for c in of_df.columns if c not in existing_cols]
                if new_cols:
                    bars_tf = bars_tf.join(of_df[new_cols], how="left")

                try:
                    vpin_derived = compute_vpin_derived_features_from_base(bars_tf)
                    for c in vpin_derived.columns:
                        if c not in bars_tf.columns:
                            bars_tf[c] = vpin_derived[c]
                except Exception:
                    pass

                try:
                    tc_derived = compute_trade_cluster_derived_features_from_base(
                        bars_tf
                    )
                    for c in tc_derived.columns:
                        if c not in bars_tf.columns:
                            bars_tf[c] = tc_derived[c]
                except Exception:
                    pass

                try:
                    fp_df = compute_footprint_features(
                        bars_tf.tail(200),
                        ticks=ticks,
                        persist_monthly=False,
                    )
                    fp_new_cols = [c for c in fp_df.columns if c not in bars_tf.columns]
                    if fp_new_cols:
                        bars_tf = bars_tf.join(fp_df[fp_new_cols], how="left")
                except Exception:
                    pass

                _logger.info(
                    "  OF features: %d cols from %d ticks",
                    len(of_df.columns),
                    len(ticks_1min),
                )
            except Exception as e:
                _logger.warning("  OF features failed: %s", e)

        # ── 3. StrategyFeatureLoader 批量计算 ──
        if self._feature_loader is not None and self.live_feature_nodes:
            try:
                req = list(self.live_feature_nodes)
                feats_cfg = (self._feature_deps or {}).get("features") or {}
                bar_cols = set(bars_tf.columns)

                tick_dependent_nodes = self._detect_tick_dependent_nodes(feats_cfg)

                def _has_tick_dependency(node_name: str, visited: set = None) -> bool:
                    if visited is None:
                        visited = set()
                    if node_name in visited:
                        return False
                    visited.add(node_name)
                    if node_name in tick_dependent_nodes:
                        return True
                    info = feats_cfg.get(node_name)
                    if isinstance(info, dict):
                        deps = info.get("dependencies") or []
                        for dep in deps:
                            if _has_tick_dependency(dep, visited):
                                return True
                    return False

                filtered = []
                skipped = []
                for n in req:
                    if _has_tick_dependency(n):
                        skipped.append(str(n))
                        continue
                    info = feats_cfg.get(n)
                    if isinstance(info, dict):
                        req_cols = set(info.get("required_columns") or [])
                        deps = info.get("dependencies") or []
                        if req_cols and not req_cols.issubset(bar_cols) and not deps:
                            skipped.append(str(n))
                            continue
                    filtered.append(n)

                # ── 3a-fix. skipped 节点的非-tick 依赖加入 filtered ──
                filtered_set = set(filtered)
                for n in skipped:
                    if n in tick_dependent_nodes:
                        continue
                    info = feats_cfg.get(n)
                    if isinstance(info, dict):
                        for dep in info.get("dependencies") or []:
                            if dep not in filtered_set and not _has_tick_dependency(
                                dep
                            ):
                                filtered.append(dep)
                                filtered_set.add(dep)

                bars_tf = self._feature_loader.load_features_from_requested(
                    bars_tf, requested_features=filtered, fit=False
                )
                _logger.info(
                    "  Loader: %d nodes (%d skipped tick-dep) → %d cols",
                    len(filtered),
                    len(skipped),
                    len(bars_tf.columns),
                )

                # ── 3b. Second pass ──
                if skipped:
                    bar_cols_updated = set(bars_tf.columns)
                    second_pass = []
                    for n in skipped:
                        if n in tick_dependent_nodes:
                            continue
                        info = feats_cfg.get(n)
                        if not isinstance(info, dict):
                            continue
                        req_cols = set(info.get("required_columns") or [])
                        if req_cols.issubset(bar_cols_updated):
                            second_pass.append(n)

                    if second_pass:
                        from src.features.registry import get_compute_func
                        from src.features.loader.feature_computer import (
                            _build_call_args,
                        )
                        import inspect

                        for n in second_pass:
                            try:
                                info = feats_cfg.get(n)
                                compute_func_name = info.get("compute_func", n)
                                cfn = get_compute_func(compute_func_name)
                                if cfn is None:
                                    continue
                                info_filtered = dict(info)
                                raw_mappings = info.get("column_mappings") or {}
                                if raw_mappings:
                                    avail_mappings = {}
                                    for param, src in raw_mappings.items():
                                        if (
                                            isinstance(src, str)
                                            and src in bar_cols_updated
                                        ):
                                            avail_mappings[param] = src
                                        elif isinstance(src, list) and all(
                                            s in bar_cols_updated for s in src
                                        ):
                                            avail_mappings[param] = src
                                    info_filtered["column_mappings"] = avail_mappings
                                call_args, call_kwargs = _build_call_args(
                                    info_filtered, bars_tf, n
                                )
                                sig = inspect.signature(cfn)
                                accepts_var_kw = any(
                                    p.kind == inspect.Parameter.VAR_KEYWORD
                                    for p in sig.parameters.values()
                                )
                                if not accepts_var_kw and call_kwargs:
                                    allowed = set(sig.parameters.keys())
                                    call_kwargs = {
                                        k: v
                                        for k, v in call_kwargs.items()
                                        if k in allowed
                                    }
                                result = cfn(*call_args, **call_kwargs)
                                output_cols = info.get("output_columns", [n])
                                if isinstance(result, tuple):
                                    if len(result) == len(output_cols):
                                        result = pd.DataFrame(
                                            dict(zip(output_cols, result))
                                        )
                                if isinstance(result, pd.DataFrame):
                                    new_cols = [
                                        c
                                        for c in result.columns
                                        if c not in bars_tf.columns
                                    ]
                                    if new_cols:
                                        aligned = result[new_cols].reindex(
                                            bars_tf.index
                                        )
                                        bars_tf = pd.concat([bars_tf, aligned], axis=1)
                                elif isinstance(result, pd.Series):
                                    if result.name not in bars_tf.columns:
                                        bars_tf[result.name] = result.reindex(
                                            bars_tf.index
                                        )
                                _logger.info("  Second pass: %s → OK", n)
                            except Exception as e:
                                _logger.warning("  Second pass failed for %s: %s", n, e)
            except Exception as e:
                _logger.warning("  Loader failed: %s", e)

        return bars_tf

    # ── 元数据驱动的 warmup 特征识别 ─────────────────────────
    def _get_warmup_check_features(self) -> frozenset:
        """从 feature_dependencies.yaml 元数据识别需要大窗口 warmup 的特征。

        判定规则 (基于配置元数据，不猜后缀):
          1. compute_params 包含 percentile_window >= 100
          2. 或 compute_func 名含 "percentile" (如 compute_atr_percentile_from_series)

        这些特征使用大窗口 rolling + min_periods=window，
        warmup 数据不足时最后一行必然为 NaN。
        """
        if not self._feature_deps or not self.live_feature_set:
            return frozenset()

        feats_cfg = (self._feature_deps or {}).get("features") or {}
        check_cols: set = set()

        for node_name, info in feats_cfg.items():
            if not isinstance(info, dict):
                continue
            params = info.get("compute_params") or {}
            func_name = str(info.get("compute_func", "")).lower()

            # 规则 1: 有 percentile_window >= 100 的节点
            has_large_pct_window = params.get("percentile_window", 0) >= 100
            # 规则 2: compute_func 名含 "percentile"
            is_pct_func = "percentile" in func_name

            if has_large_pct_window or is_pct_func:
                for col in info.get("output_columns") or []:
                    check_cols.add(col)

        return frozenset(check_cols & self.live_feature_set)

    def _validate_warmup(
        self,
        bars_tf: pd.DataFrame,
        features: Optional[Dict[str, float]] = None,
    ) -> None:
        """校验大窗口百分位特征是否因 warmup 不足而为 NaN。

        在 compute_features_batch (实盘) 和 compute_features_dataframe (事件回测)
        中共享调用，确保两条路径的校验逻辑一致。

        Args:
            bars_tf: 计算完成的特征 DataFrame
            features: 最后一行特征字典 (batch 模式提供);
                      为 None 时自动从 bars_tf 最后一行提取。
        """
        import logging

        _logger = logging.getLogger(__name__)

        warmup_features = self._get_warmup_check_features()
        if not warmup_features:
            return

        # 如果没传 features dict，从 DataFrame 最后一行构建
        if features is None and len(bars_tf) > 0:
            last_row = bars_tf.iloc[-1]
            features = {
                str(k): float(v)
                for k, v in last_row.items()
                if v is not None
                and np.isscalar(v)
                and not (isinstance(v, float) and np.isnan(v))
            }
        elif features is None:
            features = {}

        nan_features = []
        for k in warmup_features:
            if k not in features:
                in_cols = k in bars_tf.columns
                last_val = bars_tf[k].iloc[-1] if in_cols else "N/A"
                nonnull = int(bars_tf[k].notna().sum()) if in_cols else 0
                _logger.warning(
                    "  ⚠️ warmup_check: %s missing from features "
                    "(in_cols=%s, last_val=%s, nonnull_rows=%d/%d)",
                    k,
                    in_cols,
                    last_val,
                    nonnull,
                    len(bars_tf),
                )
                nan_features.append(k)

        if nan_features:
            raise RuntimeError(
                f"Warmup 不足: {len(nan_features)} 个大窗口百分位特征为 NaN "
                f"(rolling window 未满足 min_periods). "
                f"缺失特征: {nan_features}. "
                f"当前 bars: {len(bars_tf)}. "
                f"请运行: bash live/scripts/prepare_warmup_ticks.sh highcap 6"
            )

    # ── 公开接口 ─────────────────────────────────────────────
    def compute_features_batch(
        self,
        bars_1min: pd.DataFrame,
        ticks_1min: pd.DataFrame,
        primary_timeframe: Optional[str] = None,
    ) -> Dict[str, float]:
        """实盘入口: 返回最后一行特征字典。

        每 15 分钟调用一次 (order_flow_listener)。
        内部调用 _compute_features_core (共享步骤 1-3)，
        然后提取最后一行 + warmup 校验。
        """
        import logging

        _logger = logging.getLogger(__name__)

        bars_tf = self._compute_features_core(bars_1min, ticks_1min, primary_timeframe)
        if bars_tf is None or bars_tf.empty:
            return {}

        # ── 4. 提取最后一行 ──
        features: Dict[str, float] = {}
        last_row = bars_tf.iloc[-1].to_dict()
        for k, v in last_row.items():
            if self._want(str(k)):
                try:
                    if v is not None and np.isscalar(v) and not pd.isna(v):
                        features[str(k)] = float(v)
                except (ValueError, TypeError):
                    continue

        # ── 5. 校验大窗口百分位特征 (基于 feature_dependencies 元数据) ──
        self._validate_warmup(bars_tf, features)

        # 缓存结果（用于 get_features() 兼容接口）
        self._batch_features = features

        _logger.info("  Final: %d features", len(features))
        self._log_missing_features(features)
        return features

    def compute_features_dataframe(
        self,
        bars_1min: pd.DataFrame,
        ticks_1min: pd.DataFrame,
        primary_timeframe: Optional[str] = None,
    ) -> pd.DataFrame:
        """事件回测入口: 返回完整 DataFrame（所有行）。

        与 compute_features_batch 共享相同的 _compute_features_core，
        保证逻辑一致。用于一次性计算全量特征后逐行喂给 decide()。
        """
        import logging

        _logger = logging.getLogger(__name__)

        bars_tf = self._compute_features_core(bars_1min, ticks_1min, primary_timeframe)
        if bars_tf is None or bars_tf.empty:
            return pd.DataFrame()

        # ── 4. 过滤列，只保留 live_feature_set 需要的 ──
        if self.live_feature_set:
            keep = [c for c in bars_tf.columns if self._want(str(c))]
            bars_tf = bars_tf[keep]

        # ── 5. warmup 校验 (与实盘一致) ──
        self._validate_warmup(bars_tf)

        _logger.info(
            "  compute_features_dataframe done: %d rows × %d cols",
            len(bars_tf),
            len(bars_tf.columns),
        )
        return bars_tf

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
        self._batch_features.clear()
