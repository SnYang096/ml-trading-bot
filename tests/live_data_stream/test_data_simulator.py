"""
测试数据模拟器

从parquet_data_1s读取1秒tick数据并转换为TradeTick对象，支持流式发送和模拟socket中断
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional, AsyncIterator, Iterator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

try:
    from nautilus_trader.model import TradeTick
    from nautilus_trader.model.enums import AggressorSide
    from nautilus_trader.model.objects import Price, Quantity

    NAUTILUS_AVAILABLE = True
except ImportError:
    NAUTILUS_AVAILABLE = False
    TradeTick = None
    AggressorSide = None
    Price = None
    Quantity = None


class MockTradeTick:
    """
    Mock TradeTick对象（当Nautilus Trader不可用时使用）
    """

    def __init__(self, timestamp: pd.Timestamp, price: float, volume: float, side: int):
        """
        Args:
            timestamp: 时间戳
            price: 价格
            volume: 成交量
            side: 买卖方向（1=buy, -1=sell）
        """
        self.ts_init_ns = int(timestamp.value)
        self.price = MockPrice(price)
        self.size = MockQuantity(volume)
        self.aggressor_side = (
            MockAggressorSide.BUY if side == 1 else MockAggressorSide.SELL
        )


class MockPrice:
    """Mock Price对象"""

    def __init__(self, value: float):
        self.value = value

    def __float__(self):
        return self.value


class MockQuantity:
    """Mock Quantity对象"""

    def __init__(self, value: float):
        self.value = value

    def __float__(self):
        return self.value


class MockAggressorSide:
    """Mock AggressorSide枚举"""

    BUY = "BUY"
    SELL = "SELL"


class TickDataSimulator:
    """
    Tick数据模拟器

    从parquet_data_1s读取1秒tick数据并转换为TradeTick对象
    """

    def __init__(
        self,
        symbol: str,
        data_dir: Path,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        max_ticks: Optional[int] = None,
    ):
        """
        Args:
            symbol: 交易对符号
            data_dir: parquet_data_1s目录路径
            start_date: 开始日期
            end_date: 结束日期
            max_ticks: 最大tick数量（用于控制测试数据量）
        """
        self.symbol = symbol
        self.data_dir = Path(data_dir)
        self.start_date = start_date
        self.end_date = end_date
        self.max_ticks = max_ticks
        self.df: Optional[pd.DataFrame] = None

    def load_data(self) -> pd.DataFrame:
        """
        加载parquet数据

        Returns:
            DataFrame with columns: timestamp, price, volume, side, symbol
        """
        if self.df is not None:
            return self.df

        # 计算需要加载的月份
        if self.start_date is None or self.end_date is None:
            # 默认加载最近1个月，但使用实际存在的数据（2024年）
            # 如果当前日期是2026年，使用2024年12月的数据
            now = datetime.now()
            if now.year >= 2025:
                # 使用2024年12月的数据
                end_date = datetime(2024, 12, 31)
                start_date = datetime(2024, 12, 1)
            else:
                end_date = now
                start_date = end_date - timedelta(days=30)
        else:
            start_date = self.start_date
            end_date = self.end_date

        # 生成月份列表
        months = pd.period_range(
            start=pd.Period(start_date, freq="M"),
            end=pd.Period(end_date, freq="M"),
            freq="M",
        )

        dfs = []
        for period in months:
            year = period.year
            month = period.month
            file_path = self.data_dir / f"{self.symbol}_{year}-{month:02d}.parquet"

            if not file_path.exists():
                print(f"⚠️ 文件不存在: {file_path}")
                continue

            try:
                df_month = pd.read_parquet(file_path)
                # 过滤时间范围
                if "timestamp" in df_month.columns:
                    df_month = df_month[
                        (df_month["timestamp"] >= pd.Timestamp(start_date))
                        & (df_month["timestamp"] <= pd.Timestamp(end_date))
                    ]
                if len(df_month) > 0:
                    dfs.append(df_month)
            except Exception as e:
                print(f"⚠️ 加载文件失败 {file_path}: {e}")
                continue

        if not dfs:
            raise ValueError(f"没有找到数据文件: {self.symbol}")

        # 合并数据
        self.df = pd.concat(dfs, ignore_index=True)
        self.df = self.df.sort_values("timestamp").reset_index(drop=True)

        # 限制数据量
        if self.max_ticks and len(self.df) > self.max_ticks:
            self.df = self.df.head(self.max_ticks)

        print(f"✅ 加载了 {len(self.df)} 条tick数据")
        print(
            f"   时间范围: {self.df['timestamp'].min()} 到 {self.df['timestamp'].max()}"
        )

        return self.df

    def _to_tick(self, row: pd.Series) -> TradeTick | MockTradeTick:
        """
        将DataFrame行转换为TradeTick对象

        Args:
            row: DataFrame行

        Returns:
            TradeTick或MockTradeTick对象
        """
        timestamp = pd.Timestamp(row["timestamp"])
        price = float(row["price"])
        volume = float(row["volume"])
        side = int(row["side"])  # 1=buy, -1=sell

        if NAUTILUS_AVAILABLE:
            # 使用真实的Nautilus Trader对象
            from nautilus_trader.model.identifiers import InstrumentId
            from nautilus_trader.model.objects import Price, Quantity

            # 创建InstrumentId（从symbol推断，如BTCUSDT -> BTCUSDT-PERP.BINANCE）
            symbol_str = row.get("symbol", "BTCUSDT")
            if "USDT" in symbol_str:
                instrument_str = f"{symbol_str}-PERP.BINANCE"
            else:
                instrument_str = f"{symbol_str}.BINANCE"

            try:
                instrument_id = InstrumentId.from_str(instrument_str)
            except Exception:
                # 如果创建失败，使用Mock对象
                return MockTradeTick(timestamp, price, volume, side)

            # 创建TradeTick对象
            try:
                from nautilus_trader.model.identifiers import TradeId

                return TradeTick(
                    instrument_id=instrument_id,
                    price=Price(price, precision=2),
                    size=Quantity(volume, precision=8),
                    aggressor_side=(
                        AggressorSide.BUYER if side == 1 else AggressorSide.SELLER
                    ),
                    trade_id=TradeId("0"),  # 使用TradeId对象
                    ts_event=timestamp.value,
                    ts_init=timestamp.value,
                )
            except Exception as e:
                # 如果创建失败，使用Mock对象
                print(f"⚠️ 创建TradeTick失败，使用Mock对象: {e}")
                return MockTradeTick(timestamp, price, volume, side)
        else:
            # 使用Mock对象
            return MockTradeTick(timestamp, price, volume, side)

    def stream_ticks(self) -> Iterator[TradeTick | MockTradeTick]:
        """
        流式发送tick数据（同步版本）

        Yields:
            TradeTick或MockTradeTick对象
        """
        if self.df is None:
            self.load_data()

        for _, row in self.df.iterrows():
            yield self._to_tick(row)

    async def stream_ticks_async(self) -> AsyncIterator[TradeTick | MockTradeTick]:
        """
        流式发送tick数据（异步版本）

        Yields:
            TradeTick或MockTradeTick对象
        """
        if self.df is None:
            self.load_data()

        for _, row in self.df.iterrows():
            yield self._to_tick(row)
            # 添加小延迟，模拟实时数据流
            await asyncio.sleep(0.001)  # 1ms延迟


class InterruptibleDataStream:
    """
    可中断的数据流

    支持在指定时间点模拟socket中断
    """

    def __init__(
        self,
        simulator: TickDataSimulator,
        interrupt_at: Optional[pd.Timestamp] = None,
    ):
        """
        Args:
            simulator: TickDataSimulator实例
            interrupt_at: 中断时间点（如果为None，不中断）
        """
        self.simulator = simulator
        self.interrupt_at = interrupt_at

    def stream(self) -> Iterator[TradeTick | MockTradeTick]:
        """
        流式发送tick数据，在指定时间点中断

        Yields:
            TradeTick或MockTradeTick对象

        Raises:
            ConnectionError: 在interrupt_at时间点抛出
        """
        for tick in self.simulator.stream_ticks():
            # 检查是否到达中断时间点
            if self.interrupt_at is not None:
                # 支持Nautilus Trader的TradeTick（使用ts_init）和Mock对象（使用ts_init_ns）
                if hasattr(tick, "ts_init"):
                    tick_ts = pd.Timestamp(tick.ts_init, unit="ns", tz="UTC")
                elif hasattr(tick, "ts_init_ns"):
                    tick_ts = pd.Timestamp(tick.ts_init_ns, unit="ns", tz="UTC")
                else:
                    tick_ts = pd.Timestamp.now(tz="UTC")

                # 确保interrupt_at也是时区感知的
                interrupt_at = self.interrupt_at
                if isinstance(interrupt_at, pd.Timestamp):
                    if interrupt_at.tz is None:
                        interrupt_at = interrupt_at.tz_localize("UTC")
                    elif interrupt_at.tz != tick_ts.tz:
                        interrupt_at = interrupt_at.tz_convert(tick_ts.tz)
                else:
                    interrupt_at = pd.Timestamp(interrupt_at, tz="UTC")

                if tick_ts >= interrupt_at:
                    raise ConnectionError(f"Socket interrupted at {tick_ts}")

            yield tick

    async def stream_async(self) -> AsyncIterator[TradeTick | MockTradeTick]:
        """
        异步流式发送tick数据，在指定时间点中断

        Yields:
            TradeTick或MockTradeTick对象

        Raises:
            ConnectionError: 在interrupt_at时间点抛出
        """
        async for tick in self.simulator.stream_ticks_async():
            # 检查是否到达中断时间点
            if self.interrupt_at is not None:
                # 支持Nautilus Trader的TradeTick（使用ts_init）和Mock对象（使用ts_init_ns）
                if hasattr(tick, "ts_init"):
                    tick_ts = pd.Timestamp(tick.ts_init, unit="ns", tz="UTC")
                elif hasattr(tick, "ts_init_ns"):
                    tick_ts = pd.Timestamp(tick.ts_init_ns, unit="ns", tz="UTC")
                else:
                    tick_ts = pd.Timestamp.now(tz="UTC")

                # 确保interrupt_at也是时区感知的
                interrupt_at = self.interrupt_at
                if isinstance(interrupt_at, pd.Timestamp):
                    if interrupt_at.tz is None:
                        interrupt_at = interrupt_at.tz_localize("UTC")
                    elif interrupt_at.tz != tick_ts.tz:
                        interrupt_at = interrupt_at.tz_convert(tick_ts.tz)
                else:
                    interrupt_at = pd.Timestamp(interrupt_at, tz="UTC")

                if tick_ts >= interrupt_at:
                    raise ConnectionError(f"Socket interrupted at {tick_ts}")

            yield tick
