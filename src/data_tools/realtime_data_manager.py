"""
实时数据管理器

整合历史数据（Parquet）和实时数据（QuestDB），提供统一的数据接口。

职责：
1. 从 Parquet 加载历史数据（Warmup）
2. 从 QuestDB 加载实时数据
3. 合并数据供特征计算使用
4. 将实时数据写入 QuestDB
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from pathlib import Path
import socket
import time

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("⚠️ requests not installed. Install with: pip install requests")

from src.data_tools.data_utils import load_raw_data


class QuestDBWriter:
    """
    QuestDB 数据写入器

    使用 ILP (InfluxDB Line Protocol) 格式写入，性能最优。
    QuestDB ILP 默认端口：9009
    """

    def __init__(self, host: str = "localhost", port: int = 9009):
        self.host = host
        self.port = port

    def write_bar(self, bar: Dict[str, Any]) -> bool:
        """
        写入 K线数据

        Args:
            bar: K线数据字典，包含：
                - symbol: 交易对符号
                - timeframe: 时间框架（如 '15m'）
                - timestamp: 时间戳（毫秒或秒）
                - open, high, low, close, volume: OHLCV 数据
                - 其他字段（可选）：trade_count, buy_qty, sell_qty, delta, etc.

        Returns:
            是否写入成功
        """
        try:
            # 转换时间戳为纳秒
            if isinstance(bar["timestamp"], (int, float)):
                if bar["timestamp"] < 1e12:  # 毫秒时间戳
                    timestamp_ns = int(bar["timestamp"] * 1_000_000)
                else:  # 秒时间戳
                    timestamp_ns = int(bar["timestamp"] * 1_000_000_000)
            else:
                timestamp_ns = int(pd.Timestamp(bar["timestamp"]).value)

            # 构建 ILP 消息
            symbol = bar.get("symbol", "UNKNOWN")
            timeframe = bar.get("timeframe", "1m")

            # 基础字段
            fields = [
                f"open={bar['open']}",
                f"high={bar['high']}",
                f"low={bar['low']}",
                f"close={bar['close']}",
                f"volume={bar.get('volume', 0)}",
            ]

            # 可选字段
            if "trade_count" in bar:
                fields.append(f"trade_count={int(bar['trade_count'])}")
            if "buy_qty" in bar:
                fields.append(f"buy_qty={bar['buy_qty']}")
            if "sell_qty" in bar:
                fields.append(f"sell_qty={bar['sell_qty']}")
            if "delta" in bar:
                fields.append(f"delta={bar['delta']}")
            if "taker_buy_ratio" in bar:
                fields.append(f"taker_buy_ratio={bar['taker_buy_ratio']}")
            if "cvd" in bar:
                fields.append(f"cvd={bar['cvd']}")

            ilp_message = (
                f"klines,symbol={symbol},timeframe={timeframe} "
                f"{','.join(fields)} "
                f"{timestamp_ns}\n"
            )

            return self._send_ilp(ilp_message)

        except Exception as e:
            print(f"❌ 写入 K线数据失败: {e}")
            return False

    def write_tick(self, tick: Dict[str, Any]) -> bool:
        """
        写入订单流数据（Tick数据）

        Args:
            tick: Tick数据字典，包含：
                - symbol: 交易对符号
                - timestamp: 时间戳
                - price: 价格
                - size: 数量
                - side: 方向（'buy' 或 'sell'）
                - trade_id: 交易ID（可选）

        Returns:
            是否写入成功
        """
        try:
            # 转换时间戳为纳秒
            if isinstance(tick["timestamp"], (int, float)):
                if tick["timestamp"] < 1e12:  # 毫秒时间戳
                    timestamp_ns = int(tick["timestamp"] * 1_000_000)
                else:  # 秒时间戳
                    timestamp_ns = int(tick["timestamp"] * 1_000_000_000)
            else:
                timestamp_ns = int(pd.Timestamp(tick["timestamp"]).value)

            symbol = tick.get("symbol", "UNKNOWN")
            side = tick.get("side", "unknown")
            trade_id = tick.get("trade_id", "")

            ilp_message = (
                f"ticks,symbol={symbol},side={side} "
                f"price={tick['price']},size={tick['size']}"
            )

            if trade_id:
                ilp_message += f",trade_id={trade_id}"

            ilp_message += f" {timestamp_ns}\n"

            return self._send_ilp(ilp_message)

        except Exception as e:
            print(f"❌ 写入 Tick 数据失败: {e}")
            return False

    def _send_ilp(self, message: str) -> bool:
        """发送 ILP 消息到 QuestDB"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)  # 1秒超时
            sock.connect((self.host, self.port))
            sock.sendall(message.encode())
            sock.close()
            return True
        except Exception as e:
            print(f"❌ 发送 ILP 消息失败: {e}")
            return False


class RealtimeDataManager:
    """
    实时数据管理器

    整合历史数据（Parquet）和实时数据（QuestDB），提供统一的数据接口。
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        questdb_url: str = "http://localhost:9000",
        questdb_ilp_host: str = "localhost",
        questdb_ilp_port: int = 9009,
        parquet_data_path: str = "data/parquet_data",
        warmup_bars: int = 1000,
    ):
        """
        Args:
            symbol: 交易对符号（如 "BTCUSDT"）
            timeframe: 时间框架（如 "15T" 或 "15m"）
            questdb_url: QuestDB HTTP 查询 URL
            questdb_ilp_host: QuestDB ILP 写入主机
            questdb_ilp_port: QuestDB ILP 写入端口
            parquet_data_path: Parquet 数据路径
            warmup_bars: Warmup 需要的 K线数量
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.questdb_url = questdb_url
        self.parquet_data_path = parquet_data_path
        self.warmup_bars = warmup_bars

        # QuestDB 写入器
        self.questdb_writer = QuestDBWriter(questdb_ilp_host, questdb_ilp_port)

        # 数据缓存（内存中的滑动窗口）
        self.history_df: Optional[pd.DataFrame] = None

    def initialize(self) -> pd.DataFrame:
        """
        初始化：加载 Warmup 数据

        策略：
        1. 优先从 QuestDB 加载最近的数据（如果有）
        2. 如果 QuestDB 数据不足，从 Parquet 补充
        3. 合并数据，去重，排序

        Returns:
            包含历史数据的 DataFrame
        """
        print(f"📊 初始化数据管理器: {self.symbol} {self.timeframe}")
        print(f"   需要 Warmup 数据: {self.warmup_bars} 条")

        # 1. 尝试从 QuestDB 加载最近的数据
        questdb_data = self._load_from_questdb(self.warmup_bars)

        if len(questdb_data) >= self.warmup_bars:
            print(f"✅ 从 QuestDB 加载了 {len(questdb_data)} 条数据（足够）")
            self.history_df = questdb_data
            return questdb_data

        # 2. QuestDB 数据不足，从 Parquet 补充
        print(f"⚠️ QuestDB 数据不足 ({len(questdb_data)} 条)，从 Parquet 补充")

        needed_bars = self.warmup_bars - len(questdb_data)
        parquet_data = self._load_from_parquet(needed_bars)

        # 3. 合并数据
        if len(questdb_data) > 0 and len(parquet_data) > 0:
            # 合并，去重，排序
            combined = pd.concat([parquet_data, questdb_data])
            combined = combined.drop_duplicates(subset=["timestamp"]).sort_values(
                "timestamp"
            )
            self.history_df = combined.tail(self.warmup_bars)
            print(
                f"✅ 合并数据完成: Parquet {len(parquet_data)} 条 + QuestDB {len(questdb_data)} 条 = {len(self.history_df)} 条"
            )
        elif len(questdb_data) > 0:
            self.history_df = questdb_data
            print(f"✅ 使用 QuestDB 数据: {len(self.history_df)} 条")
        else:
            self.history_df = parquet_data
            print(f"✅ 使用 Parquet 数据: {len(self.history_df)} 条")

        return self.history_df

    def append_bar(self, bar: Dict[str, Any]) -> pd.DataFrame:
        """
        追加新的 K线数据

        Args:
            bar: K线数据字典

        Returns:
            更新后的 DataFrame（包含新数据）
        """
        # 1. 转换为 DataFrame
        new_bar_df = pd.DataFrame(
            [
                {
                    "timestamp": (
                        pd.Timestamp.fromtimestamp(bar["timestamp"] / 1000)
                        if isinstance(bar["timestamp"], (int, float))
                        else pd.Timestamp(bar["timestamp"])
                    ),
                    "datetime": (
                        pd.Timestamp.fromtimestamp(bar["timestamp"] / 1000)
                        if isinstance(bar["timestamp"], (int, float))
                        else pd.Timestamp(bar["timestamp"])
                    ),
                    "open": float(bar["open"]),
                    "high": float(bar["high"]),
                    "low": float(bar["low"]),
                    "close": float(bar["close"]),
                    "volume": float(bar.get("volume", 0)),
                    "symbol": self.symbol,
                }
            ]
        )

        # 添加可选字段
        if "trade_count" in bar:
            new_bar_df["trade_count"] = int(bar["trade_count"])
        if "buy_qty" in bar:
            new_bar_df["buy_qty"] = float(bar["buy_qty"])
        if "sell_qty" in bar:
            new_bar_df["sell_qty"] = float(bar["sell_qty"])
        if "delta" in bar:
            new_bar_df["delta"] = float(bar["delta"])
        if "taker_buy_ratio" in bar:
            new_bar_df["taker_buy_ratio"] = float(bar["taker_buy_ratio"])
        if "cvd" in bar:
            new_bar_df["cvd"] = float(bar["cvd"])

        # 2. 写入 QuestDB（异步，不阻塞主流程）
        try:
            self.questdb_writer.write_bar(
                {
                    **bar,
                    "symbol": self.symbol,
                    "timeframe": self.timeframe,
                }
            )
        except Exception as e:
            print(f"⚠️ 写入 QuestDB 失败: {e}")

        # 3. 追加到历史数据（内存中的滑动窗口）
        if self.history_df is None:
            self.history_df = new_bar_df
        else:
            # 检查是否重复（避免重复追加）
            last_timestamp = self.history_df["timestamp"].iloc[-1]
            new_timestamp = new_bar_df["timestamp"].iloc[0]

            if new_timestamp > last_timestamp:
                # 新数据，追加
                self.history_df = pd.concat([self.history_df, new_bar_df])

                # 维护滑动窗口（只保留最近 N 条，避免内存无限增长）
                max_bars = self.warmup_bars * 2
                if len(self.history_df) > max_bars:
                    self.history_df = self.history_df.tail(max_bars)
            elif new_timestamp == last_timestamp:
                # 相同时间戳，更新（可能是数据修正）
                self.history_df.iloc[-1] = new_bar_df.iloc[0]
            # else: 旧数据，忽略

        return self.history_df

    def append_tick(self, tick: Dict[str, Any]):
        """
        追加订单流数据（只写入 QuestDB，不保留在内存）

        Args:
            tick: Tick数据字典
        """
        try:
            self.questdb_writer.write_tick(
                {
                    **tick,
                    "symbol": self.symbol,
                }
            )
        except Exception as e:
            print(f"⚠️ 写入 Tick 数据失败: {e}")

    def get_dataframe(self) -> pd.DataFrame:
        """
        获取当前数据 DataFrame

        Returns:
            当前历史数据的副本
        """
        return self.history_df.copy() if self.history_df is not None else pd.DataFrame()

    def _load_from_questdb(self, limit: int) -> pd.DataFrame:
        """从 QuestDB 加载数据"""
        if not REQUESTS_AVAILABLE:
            return pd.DataFrame()

        # 转换时间框架格式（15T -> 15m）
        timeframe_qdb = (
            self.timeframe.replace("T", "m")
            if "T" in self.timeframe
            else self.timeframe
        )

        query = f"""
        SELECT 
            timestamp,
            open, high, low, close, volume,
            trade_count, buy_qty, sell_qty, delta,
            taker_buy_ratio, cvd
        FROM klines
        WHERE symbol = '{self.symbol}' AND timeframe = '{timeframe_qdb}'
        ORDER BY timestamp DESC
        LIMIT {limit}
        """

        try:
            response = requests.post(
                f"{self.questdb_url}/exec",
                data=query,
                timeout=5,
            )
            response.raise_for_status()

            import io

            df = pd.read_csv(io.StringIO(response.text))

            if len(df) > 0:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.sort_values("timestamp").reset_index(drop=True)
                return df

            return pd.DataFrame()

        except Exception as e:
            print(f"⚠️ 从 QuestDB 加载数据失败: {e}")
            return pd.DataFrame()

    def _load_from_parquet(self, needed_bars: int) -> pd.DataFrame:
        """从 Parquet 文件加载数据"""
        try:
            # 使用现有的数据加载函数
            # 这里需要根据你的实际实现调整
            df = load_raw_data(
                symbol=self.symbol,
                data_path=self.parquet_data_path,
                # 其他参数...
            )

            if df is None or len(df) == 0:
                return pd.DataFrame()

            # 只取最近 needed_bars 条
            if len(df) > needed_bars:
                df = df.tail(needed_bars)

            return df.reset_index(drop=True)

        except Exception as e:
            print(f"⚠️ 从 Parquet 加载数据失败: {e}")
            return pd.DataFrame()


# 使用示例
if __name__ == "__main__":
    # 创建数据管理器
    data_manager = RealtimeDataManager(
        symbol="BTCUSDT",
        timeframe="15T",
        warmup_bars=1000,
    )

    # 初始化（加载 Warmup 数据）
    warmup_df = data_manager.initialize()
    print(f"Warmup 数据: {len(warmup_df)} 条")

    # 模拟接收新的 K线数据
    new_bar = {
        "timestamp": int(time.time() * 1000),
        "open": 50000.0,
        "high": 50100.0,
        "low": 49900.0,
        "close": 50050.0,
        "volume": 100.0,
    }

    # 追加新数据
    updated_df = data_manager.append_bar(new_bar)
    print(f"更新后数据: {len(updated_df)} 条")

    # 模拟接收 Tick 数据
    new_tick = {
        "timestamp": int(time.time() * 1000),
        "price": 50050.0,
        "size": 0.1,
        "side": "buy",
        "trade_id": "12345",
    }

    # 写入 Tick 数据（只写入 QuestDB，不保留在内存）
    data_manager.append_tick(new_tick)
