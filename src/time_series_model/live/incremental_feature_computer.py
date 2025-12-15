"""
增量特征计算器（事件驱动）

用于实盘和回测，支持流式处理 tick 和 bar 数据，计算特征增量更新。
与批处理版本共享核心算法，但维护状态以支持增量计算。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any, Deque
from collections import deque
from datetime import datetime, timedelta

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

        # 添加到缓冲区
        self.bar_buffer.append(bar_data)

        # 更新时间框架特征
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
                self.timeframe_features[timeframe]["atr"] = float(atr)

        # 价格位置
        self.timeframe_features[timeframe]["close"] = float(bar_data["close"])
        self.timeframe_features[timeframe]["volume"] = float(bar_data["volume"])

    def get_features(self) -> Dict[str, float]:
        """
        获取当前所有特征

        Returns:
            特征字典
        """
        features = {}

        # Tick 级特征
        features.update(self.current_features)

        # 时间框架特征（扁平化）
        for tf, tf_features in self.timeframe_features.items():
            for key, value in tf_features.items():
                features[f"{tf}_{key}"] = value

        return features

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
