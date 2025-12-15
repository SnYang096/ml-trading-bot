"""
事件驱动回测（基于 Nautilus Trader）

使用 Nautilus Trader 的事件驱动架构进行回测，与实盘策略完全一致。
支持：
1. 从历史数据文件加载 tick 和 bar 数据
2. 事件驱动特征计算
3. 与实盘策略相同的信号生成逻辑
4. 详细的交易记录和性能分析
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

try:
    from nautilus_trader.backtest.node import BacktestNode
    from nautilus_trader.backtest.config import BacktestDataConfig
    from nautilus_trader.backtest.config import BacktestEngineConfig
    from nautilus_trader.backtest.config import BacktestVenueConfig
    from nautilus_trader.model import InstrumentId
    from nautilus_trader.model import BarType
    from nautilus_trader.model import BarSpecification
    from nautilus_trader.model import BarAggregation
    from nautilus_trader.model import PriceType
    from nautilus_trader.model import AggregationSource
    from nautilus_trader.persistence.catalog import ParquetDataCatalog
    from nautilus_trader.persistence.loaders import QuoteTickDataLoader
    from nautilus_trader.persistence.loaders import TradeTickDataLoader
    from nautilus_trader.persistence.loaders import BarDataLoader

    NAUTILUS_AVAILABLE = True
except ImportError:
    NAUTILUS_AVAILABLE = False
    print(
        "⚠️ Nautilus Trader is not installed. Install with: pip install nautilus-trader"
    )


@dataclass
class BacktestResult:
    """回测结果"""

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    trades: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "total_pnl": self.total_pnl,
            "total_return": self.total_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "profit_factor": self.profit_factor,
        }


class EventDrivenBacktest:
    """
    事件驱动回测

    使用 Nautilus Trader 的回测引擎，与实盘策略使用相同的代码。
    """

    def __init__(
        self,
        strategy_class,
        strategy_config: Dict[str, Any],
        data_catalog_path: str,
        instrument_id: InstrumentId,
        bar_types: Dict[str, BarType],
        start_time: datetime,
        end_time: datetime,
    ):
        """
        Args:
            strategy_class: 策略类（如 EventDrivenStrategy）
            strategy_config: 策略配置
            data_catalog_path: 数据目录路径
            instrument_id: 交易标的
            bar_types: 时间框架字典
            start_time: 回测开始时间
            end_time: 回测结束时间
        """
        if not NAUTILUS_AVAILABLE:
            raise ImportError(
                "Nautilus Trader is required for event-driven backtesting"
            )

        self.strategy_class = strategy_class
        self.strategy_config = strategy_config
        self.data_catalog_path = data_catalog_path
        self.instrument_id = instrument_id
        self.bar_types = bar_types
        self.start_time = start_time
        self.end_time = end_time

        self.catalog: Optional[ParquetDataCatalog] = None
        self.node: Optional[BacktestNode] = None
        self.result: Optional[BacktestResult] = None

    def prepare_data_catalog(self) -> None:
        """准备数据目录"""
        catalog_path = Path(self.data_catalog_path)
        if not catalog_path.exists():
            raise FileNotFoundError(f"Data catalog not found: {catalog_path}")

        self.catalog = ParquetDataCatalog(catalog_path)
        print(f"✅ Loaded data catalog: {catalog_path}")

    def run(self) -> BacktestResult:
        """运行回测"""
        if self.catalog is None:
            self.prepare_data_catalog()

        # 1. 配置回测引擎
        engine_config = BacktestEngineConfig(
            strategies=[self.strategy_class.__name__],
            venues=[
                BacktestVenueConfig(
                    name="BINANCE",
                    oms_type="NETTING",
                    account_type="MARGIN",
                    base_currency="USDT",
                    starting_balances=["100000 USDT"],
                )
            ],
        )

        # 2. 配置数据
        data_configs = []

        # Trade ticks
        data_configs.append(
            BacktestDataConfig(
                catalog_path=str(self.catalog.path),
                data_clients=["BINANCE"],
                catalog_fs_protocol="file",
                instrument_ids=[str(self.instrument_id)],
                start_time=self.start_time,
                end_time=self.end_time,
            )
        )

        # Bars
        for timeframe, bar_type in self.bar_types.items():
            data_configs.append(
                BacktestDataConfig(
                    catalog_path=str(self.catalog.path),
                    data_clients=["BINANCE"],
                    catalog_fs_protocol="file",
                    bar_types=[str(bar_type)],
                    start_time=self.start_time,
                    end_time=self.end_time,
                )
            )

        # 3. 创建回测节点
        self.node = BacktestNode(
            config=engine_config,
            data_configs=data_configs,
        )

        # 4. 创建策略实例
        strategy = self.strategy_class(**self.strategy_config)

        # 5. 添加策略
        self.node.trader.add_strategy(strategy)

        # 6. 运行回测
        print(f"🚀 Starting backtest: {self.start_time} to {self.end_time}")
        self.node.run()

        # 7. 收集结果
        self.result = self._collect_results(strategy)

        return self.result

    def _collect_results(self, strategy) -> BacktestResult:
        """收集回测结果"""
        result = BacktestResult()

        # 从策略或交易账户中收集交易记录
        # TODO: 实现具体的交易记录收集逻辑
        # 这里简化处理

        portfolio = self.node.trader.portfolio
        account = portfolio.account("BINANCE-001")

        if account is not None:
            # 计算性能指标
            result.total_pnl = float(account.balance_total().as_f64())
            result.total_return = (
                result.total_pnl / 100000.0
            ) * 100  # 假设初始资金 100k

        return result

    def save_results(self, output_path: str) -> None:
        """保存回测结果"""
        if self.result is None:
            raise ValueError("No backtest results to save. Run backtest first.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存为 JSON
        import json

        with open(output_path, "w") as f:
            json.dump(self.result.to_dict(), f, indent=2)

        print(f"✅ Backtest results saved to: {output_path}")


def run_event_driven_backtest(
    strategy_name: str,
    symbol: str,
    data_path: str,
    start_date: str,
    end_date: str,
    timeframes: List[str] = ["15T", "1H"],
    trade_size: float = 0.001,
    model_path: Optional[str] = None,
    output_dir: str = "results/backtest",
) -> BacktestResult:
    """
    运行事件驱动回测（便捷函数）

    Args:
        strategy_name: 策略名称
        symbol: 交易标的（如 "BTCUSDT-PERP"）
        data_path: 数据目录路径
        start_date: 开始日期（YYYY-MM-DD）
        end_date: 结束日期（YYYY-MM-DD）
        timeframes: 时间框架列表
        trade_size: 交易规模
        model_path: 模型路径
        output_dir: 输出目录

    Returns:
        回测结果
    """
    from src.time_series_model.live.event_driven_strategy import EventDrivenStrategy

    # 1. 创建 instrument ID
    instrument_id = InstrumentId.from_str(f"{symbol}.BINANCE")

    # 2. 创建 bar types
    bar_types = {}
    for tf in timeframes:
        if tf.endswith("T"):
            minutes = int(tf.rstrip("T"))
            bar_spec = BarSpecification(minutes, BarAggregation.MINUTE, PriceType.LAST)
        elif tf.endswith("H"):
            hours = int(tf.rstrip("H"))
            bar_spec = BarSpecification(hours, BarAggregation.HOUR, PriceType.LAST)
        else:
            raise ValueError(f"Unsupported timeframe: {tf}")

        bar_type = BarType(
            instrument_id=instrument_id,
            bar_spec=bar_spec,
            aggregation_source=AggregationSource.EXTERNAL,
        )
        bar_types[tf] = bar_type

    # 3. 解析日期
    start_time = pd.to_datetime(start_date).to_pydatetime()
    end_time = pd.to_datetime(end_date).to_pydatetime()

    # 4. 策略配置
    strategy_config = {
        "strategy_name": strategy_name,
        "instrument_id": instrument_id,
        "bar_types": bar_types,
        "trade_size": trade_size,
        "model_path": model_path,
    }

    # 5. 创建回测实例
    backtest = EventDrivenBacktest(
        strategy_class=EventDrivenStrategy,
        strategy_config=strategy_config,
        data_catalog_path=data_path,
        instrument_id=instrument_id,
        bar_types=bar_types,
        start_time=start_time,
        end_time=end_time,
    )

    # 6. 运行回测
    result = backtest.run()

    # 7. 保存结果
    output_path = (
        Path(output_dir) / f"{strategy_name}_{symbol}_{start_date}_{end_date}.json"
    )
    backtest.save_results(str(output_path))

    return result
