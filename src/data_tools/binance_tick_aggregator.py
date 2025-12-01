"""
币安 Tick 数据聚合器

从 WebSocket 接收实时 tick 数据，按 100ms 窗口聚合后写入 QuestDB。

聚合后的数据包含：
- OHLCV（开高低收成交量）
- 订单流统计（买卖成交量、交易次数、delta 等）
"""

import asyncio
import json
import time
from typing import Dict, List, Optional

import pandas as pd

try:
    import websockets

    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    websockets = None  # 用于类型检查


class BinanceTickAggregator:
    """
    币安 Tick 数据聚合器

    功能：
    1. 从 WebSocket 接收实时 tick 数据
    2. 按 100ms 窗口聚合（OHLCV + 订单流统计）
    3. 批量写入 QuestDB
    """

    def __init__(
        self,
        ws_url: str,
        symbol: str,
        questdb_client,
        aggregation_ms: int = 100,  # 聚合窗口（毫秒）
        batch_size: int = 1000,  # 批量写入大小
        flush_seconds: int = 5,  # 强制刷新间隔（秒）
    ):
        """
        Args:
            ws_url: WebSocket URL
            symbol: 交易对符号
            questdb_client: QuestDB 客户端实例
            aggregation_ms: 聚合窗口大小（毫秒），默认 100ms
            batch_size: 批量写入大小
            flush_seconds: 强制刷新间隔（秒）
        """
        self.ws_url = ws_url
        self.symbol = symbol.upper()
        self.questdb_client = questdb_client
        self.aggregation_ms = aggregation_ms
        self.batch_size = batch_size
        self.flush_seconds = flush_seconds

        # 聚合缓冲区：{window_start_ms: {tick_data...}}
        self.aggregation_buffer: Dict[int, Dict] = {}

        # 已完成的聚合数据（等待写入）
        self.completed_aggregates: List[Dict] = []

        # 当前窗口的开始时间（毫秒时间戳）
        self.current_window_start: Optional[int] = None

        # 最后刷新时间
        self.last_flush_time = time.time()

    def _get_window_start(self, timestamp_ms: int) -> int:
        """
        计算时间戳所属的聚合窗口开始时间

        Args:
            timestamp_ms: 时间戳（毫秒）

        Returns:
            窗口开始时间（毫秒）
        """
        # 向下取整到最近的聚合窗口
        return (timestamp_ms // self.aggregation_ms) * self.aggregation_ms

    def _aggregate_tick(self, tick_data: Dict):
        """
        将 tick 数据聚合到对应的窗口

        Args:
            tick_data: tick 数据字典
        """
        timestamp_ms = tick_data["ts_ms"]
        window_start = self._get_window_start(timestamp_ms)

        # 初始化窗口数据（如果不存在）
        if window_start not in self.aggregation_buffer:
            self.aggregation_buffer[window_start] = {
                "window_start": window_start,
                "open": tick_data["price"],
                "high": tick_data["price"],
                "low": tick_data["price"],
                "close": tick_data["price"],
                "volume": 0.0,
                "trade_count": 0,
                "buy_volume": 0.0,
                "sell_volume": 0.0,
                "buy_count": 0,
                "sell_count": 0,
                "first_trade_id": tick_data.get("trade_id", 0),
                "last_trade_id": tick_data.get("trade_id", 0),
            }

        window = self.aggregation_buffer[window_start]

        # 更新 OHLC
        window["high"] = max(window["high"], tick_data["price"])
        window["low"] = min(window["low"], tick_data["price"])
        window["close"] = tick_data["price"]

        # 更新成交量统计
        qty = tick_data["qty"]
        window["volume"] += qty
        window["trade_count"] += 1

        # 更新买卖方向统计
        is_buyer_maker = tick_data.get("is_buyer_maker", False)
        if is_buyer_maker:
            # 卖方主动（sell）
            window["sell_volume"] += qty
            window["sell_count"] += 1
        else:
            # 买方主动（buy）
            window["buy_volume"] += qty
            window["buy_count"] += 1

        # 更新 trade_id
        trade_id = tick_data.get("trade_id", 0)
        if trade_id > 0:
            window["last_trade_id"] = max(window["last_trade_id"], trade_id)
            if window["first_trade_id"] == 0:
                window["first_trade_id"] = trade_id

    def _finalize_completed_windows(self, current_time_ms: int):
        """
        完成已结束的窗口（当前时间已超过窗口结束时间）

        Args:
            current_time_ms: 当前时间（毫秒）
        """
        # 计算当前窗口的结束时间
        current_window_end = (
            self._get_window_start(current_time_ms) + self.aggregation_ms
        )

        # 找出所有已完成的窗口（窗口结束时间 < 当前窗口结束时间）
        completed_windows = []
        for window_start, window_data in list(self.aggregation_buffer.items()):
            window_end = window_start + self.aggregation_ms
            if window_end < current_window_end:
                completed_windows.append((window_start, window_data))

        # 按时间排序
        completed_windows.sort(key=lambda x: x[0])

        # 移动到已完成列表
        for window_start, window_data in completed_windows:
            # 添加时间戳和符号
            window_data["timestamp"] = window_start
            window_data["symbol"] = self.symbol

            # 计算订单流指标
            total_volume = window_data["volume"]
            if total_volume > 0:
                window_data["buy_ratio"] = window_data["buy_volume"] / total_volume
                window_data["sell_ratio"] = window_data["sell_volume"] / total_volume
                window_data["delta"] = (
                    window_data["buy_volume"] - window_data["sell_volume"]
                )
            else:
                window_data["buy_ratio"] = 0.0
                window_data["sell_ratio"] = 0.0
                window_data["delta"] = 0.0

            self.completed_aggregates.append(window_data)
            del self.aggregation_buffer[window_start]

    async def _flush_aggregates(self):
        """
        将已完成的聚合数据写入 QuestDB
        """
        if not self.completed_aggregates:
            return

        # 转换为 DataFrame
        df = pd.DataFrame(self.completed_aggregates)

        # 转换时间戳格式（QuestDB 需要 datetime 对象）
        # 使用 timestamp 字段（窗口开始时间，毫秒）转换为 datetime
        df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

        # 保留原始毫秒时间戳作为 timestamp_ms
        df["timestamp_ms"] = df["timestamp"]

        # 确保所有必要的列都存在
        required_columns = [
            "ts",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "trade_count",
            "buy_volume",
            "sell_volume",
            "buy_count",
            "sell_count",
            "buy_ratio",
            "sell_ratio",
            "delta",
            "first_trade_id",
            "last_trade_id",
        ]

        # 检查缺失的列
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            print(f"[aggregator] 警告：缺少列 {missing_columns}")

        # 写入 QuestDB
        try:
            await self.questdb_client.insert_aggregated_ticks(df)
            print(f"[aggregator] 写入 {len(df)} 条聚合数据")
        except Exception as e:
            print(f"[aggregator] 写入失败: {e}")
            import traceback

            traceback.print_exc()

        # 清空已完成列表
        self.completed_aggregates.clear()

    async def run(self, stop_event: asyncio.Event):
        """
        运行聚合器

        Args:
            stop_event: 停止事件
        """
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets 模块未安装，请安装: pip install websockets")

        print(f"[aggregator] 连接 WebSocket: {self.ws_url}")
        print(f"[aggregator] 聚合窗口: {self.aggregation_ms}ms")
        print(f"[aggregator] 批量大小: {self.batch_size}")

        retry_delay = 3

        async for ws in websockets.connect(
            self.ws_url,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=10,
        ):
            if not ws:
                continue

            try:
                async for msg in ws:
                    if not msg:
                        continue

                    try:
                        data = json.loads(msg)

                        # 只处理 trade 事件
                        if data.get("e") != "trade":
                            continue

                    except json.JSONDecodeError as e:
                        print(f"[aggregator] JSON 解析失败: {e}")
                        continue

                    # 解析 Binance Futures 'trade' stream
                    # {'e': 'trade', 'E': 1755174443114, 'T': 1755174443114, 's': 'BTCUSDT',
                    #  't': 6554030773, 'p': '120950.20', 'q': '0.008', 'X': 'MARKET', 'm': False}

                    timestamp_ms = int(data.get("T") or data.get("E"))
                    tick_data = {
                        "ts_ms": timestamp_ms,
                        "symbol": data.get("s", self.symbol),
                        "price": float(data["p"]),
                        "qty": float(data["q"]),
                        "is_buyer_maker": bool(data.get("m", False)),
                        "trade_id": int(data.get("t", 0)),
                    }

                    # 聚合 tick 数据
                    self._aggregate_tick(tick_data)

                    # 完成已结束的窗口
                    self._finalize_completed_windows(timestamp_ms)

                    # 检查是否需要刷新
                    current_time = time.time()
                    should_flush = (
                        len(self.completed_aggregates) >= self.batch_size
                        or (current_time - self.last_flush_time) >= self.flush_seconds
                    )

                    if should_flush:
                        await self._flush_aggregates()
                        self.last_flush_time = current_time

                    if stop_event.is_set():
                        break

                if stop_event.is_set():
                    break

            except websockets.ConnectionClosed as e:
                print(f"[aggregator] 连接关闭: {e.code}, 重连中...")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 60)  # 上限 60 秒

            except Exception as e:
                print(f"[aggregator] 错误: {e}")
                import traceback

                traceback.print_exc()
                break

        # 最终刷新：完成所有窗口并写入
        current_time_ms = int(time.time() * 1000)
        self._finalize_completed_windows(current_time_ms)

        if self.completed_aggregates:
            await self._flush_aggregates()

        print("[aggregator] 已停止")


# QuestDB 客户端接口（需要实现）
class QuestDBClient:
    """
    QuestDB 客户端接口

    需要实现 insert_aggregated_ticks 方法
    """

    async def insert_aggregated_ticks(self, df: pd.DataFrame):
        """
        插入聚合后的 tick 数据到 QuestDB

        Args:
            df: 聚合后的 DataFrame，包含以下列：
                - timestamp: 时间戳（ISO 格式字符串）
                - symbol: 交易对符号
                - open, high, low, close: OHLC 价格
                - volume: 总成交量
                - trade_count: 交易次数
                - buy_volume, sell_volume: 买卖成交量
                - buy_count, sell_count: 买卖次数
                - buy_ratio, sell_ratio: 买卖比例
                - delta: 买卖差量
                - first_trade_id, last_trade_id: 首尾交易 ID
        """
        raise NotImplementedError("需要实现 insert_aggregated_ticks 方法")


# 使用示例
if __name__ == "__main__":
    # 示例：创建聚合器
    ws_url = "wss://fstream.binance.com/ws/btcusdt@trade"
    symbol = "BTCUSDT"

    # 创建 QuestDB 客户端（需要实现）
    # questdb_client = YourQuestDBClient(...)

    # 创建聚合器
    aggregator = BinanceTickAggregator(
        ws_url=ws_url,
        symbol=symbol,
        questdb_client=None,  # 替换为实际的 QuestDB 客户端
        aggregation_ms=100,  # 100ms 聚合窗口
        batch_size=1000,
        flush_seconds=5,
    )

    # 运行聚合器
    # stop_event = asyncio.Event()
    # asyncio.run(aggregator.run(stop_event))
