"""
简单实时交易引擎 - 自己实现版本

这个实现展示了如何自己实现实时交易模块，直接整合现有的：
- 特征计算系统
- 模型预测
- 决策引擎

使用 ccxt 处理交易所 API，保持轻量级和灵活性。
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import pickle

import pandas as pd
import numpy as np

try:
    import ccxt

    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    print("⚠️ ccxt not installed. Install with: pip install ccxt")

from src.features.loader.strategy_feature_loader import StrategyFeatureLoader
from src.strategy_config import StrategyConfigLoader
from src.time_series_model.strategies.trade_decision_engine import (
    TradeDecision,
    TradeDecisionEngine,
    create_decision_engine,
)
from src.time_series_model.strategies.live.realtime_feature_integration_example import (
    RealtimeFeatureManager,
)
from src.data_tools.realtime_data_manager import RealtimeDataManager


@dataclass
class Position:
    """持仓信息"""

    symbol: str
    direction: int  # 1=Long, -1=Short
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    decision: TradeDecision
    entry_time: float


class SimpleOrderManager:
    """
    简单的订单管理器

    使用 ccxt 处理交易所 API，实现：
    - 下单
    - 撤单
    - 查询订单状态
    - 查询持仓
    """

    def __init__(self, exchange: Any):
        """
        Args:
            exchange: ccxt Exchange 实例
        """
        if not CCXT_AVAILABLE:
            raise ImportError("ccxt is required for SimpleOrderManager")

        self.exchange = exchange
        self.open_orders: Dict[str, Dict] = {}
        self.positions: Dict[str, Position] = {}

    def submit_order(self, decision: TradeDecision, symbol: str) -> Optional[str]:
        """
        提交订单

        Args:
            decision: 交易决策
            symbol: 交易对符号（ccxt 格式，如 "BTC/USDT:USDT"）

        Returns:
            订单 ID 或 None
        """
        try:
            side = "buy" if decision.direction > 0 else "sell"

            # 创建限价单
            order = self.exchange.create_order(
                symbol=symbol,
                type="limit",
                side=side,
                amount=decision.position_size,
                price=decision.entry_price,
            )

            order_id = order["id"]
            self.open_orders[order_id] = {
                "order": order,
                "decision": decision,
                "symbol": symbol,
            }

            print(
                f"✅ 订单已提交: {order_id} {side} {decision.position_size} @ {decision.entry_price}"
            )
            return order_id

        except Exception as e:
            print(f"❌ 提交订单失败: {e}")
            return None

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """取消订单"""
        try:
            self.exchange.cancel_order(order_id, symbol)
            if order_id in self.open_orders:
                del self.open_orders[order_id]
            print(f"✅ 订单已取消: {order_id}")
            return True
        except Exception as e:
            print(f"❌ 取消订单失败: {e}")
            return False

    def get_position(self, symbol: str) -> Optional[Position]:
        """获取持仓"""
        return self.positions.get(symbol)

    def update_position(
        self,
        symbol: str,
        decision: TradeDecision,
        entry_price: float,
        quantity: float,
    ):
        """更新持仓"""
        self.positions[symbol] = Position(
            symbol=symbol,
            direction=decision.direction,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=decision.stop_loss,
            take_profit=decision.take_profit,
            decision=decision,
            entry_time=time.time(),
        )

    def close_position(self, symbol: str):
        """平仓"""
        if symbol in self.positions:
            del self.positions[symbol]
            print(f"✅ 持仓已平仓: {symbol}")


class SimpleRealtimeEngine:
    """
    简单实时交易引擎

    整合：
    - WebSocket 数据接收（需要自己实现或使用 ccxt）
    - 特征计算（复用现有系统）
    - 模型预测
    - 决策引擎（复用现有系统）
    - 订单执行
    """

    def __init__(
        self,
        strategy_name: str,
        symbol: str,
        exchange: Any,
        data_manager: Optional[RealtimeDataManager] = None,
        model_path: Optional[str] = None,
        config_base_path: str = "config/strategies",
        history_window: int = 1000,
        timeframe: str = "15T",
    ):
        """
        Args:
            strategy_name: 策略名称
            symbol: 交易对符号（ccxt 格式，如 "BTC/USDT:USDT"）
            exchange: ccxt Exchange 实例
            data_manager: 数据管理器（如果为 None，会自动创建）
            model_path: 模型文件路径
            config_base_path: 配置基础路径
            history_window: 历史窗口大小
            timeframe: 时间框架（如 "15T"）
        """
        self.strategy_name = strategy_name
        self.symbol = symbol
        self.exchange = exchange
        self.timeframe = timeframe

        # 初始化数据管理器（如果未提供）
        if data_manager is None:
            # 从 symbol 提取基础符号（去掉交易所后缀）
            base_symbol = symbol.split(":")[0].replace("/", "")
            data_manager = RealtimeDataManager(
                symbol=base_symbol,
                timeframe=timeframe,
                warmup_bars=history_window,
            )
        self.data_manager = data_manager

        # 初始化特征管理器
        self.feature_manager = RealtimeFeatureManager(
            strategy_name=strategy_name,
            history_window=history_window,
            config_base_path=config_base_path,
        )

        # 初始化决策引擎
        self.decision_engine = create_decision_engine(strategy_name)

        # 初始化订单管理器
        self.order_manager = SimpleOrderManager(exchange)

        # 加载模型
        self.model = None
        if model_path and Path(model_path).exists():
            self.model = self._load_model(model_path)
            print(f"✅ 模型已加载: {model_path}")
        else:
            print("⚠️ 未加载模型，将使用规则逻辑")

        # 运行标志
        self.running = False
        self.initialized = False

    def _load_model(self, model_path: str) -> Any:
        """加载模型"""
        with open(model_path, "rb") as f:
            return pickle.load(f)

    def _bar_to_dataframe(self, bar: Dict[str, Any]) -> pd.DataFrame:
        """
        将 K线数据转换为 DataFrame

        Args:
            bar: K线数据字典，包含 open, high, low, close, volume, timestamp

        Returns:
            DataFrame
        """
        return pd.DataFrame(
            {
                "timestamp": [bar.get("timestamp", time.time() * 1000)],
                "datetime": [
                    pd.Timestamp.fromtimestamp(
                        bar.get("timestamp", time.time() * 1000) / 1000
                    )
                ],
                "open": [float(bar["open"])],
                "high": [float(bar["high"])],
                "low": [float(bar["low"])],
                "close": [float(bar["close"])],
                "volume": [float(bar.get("volume", 0))],
                "symbol": [self.symbol],
            }
        )

    def initialize(self):
        """初始化：加载 Warmup 数据并初始化特征管理器"""
        if self.initialized:
            return

        print("🚀 初始化实时交易引擎...")

        # 1. 加载 Warmup 数据
        warmup_df = self.data_manager.initialize()

        if len(warmup_df) == 0:
            print("⚠️ 警告：没有加载到 Warmup 数据，特征计算可能不准确")
            return

        # 2. 初始化特征管理器（使用 Warmup 数据）
        self.feature_manager.compute_features(warmup_df)
        print(f"✅ 特征管理器已初始化，使用 {len(warmup_df)} 条历史数据")

        self.initialized = True

    def on_bar(self, bar: Dict[str, Any]):
        """
        处理新的 K线数据

        Args:
            bar: K线数据字典
        """
        try:
            # 确保已初始化
            if not self.initialized:
                self.initialize()

            # 1. 追加到数据管理器（自动写入 QuestDB）
            df = self.data_manager.append_bar(bar)

            # 2. 获取最新数据用于特征计算
            # 使用数据管理器中的完整历史数据（包含新数据）
            bar_df = df.tail(1)  # 只取最新一条用于追加

            # 2. 计算特征（复用现有系统）
            features_df = self.feature_manager.compute_features(bar_df)
            latest_features = self.feature_manager.get_latest_features()

            if latest_features is None or len(latest_features) == 0:
                print("⚠️ 特征计算失败或历史数据不足")
                return

            current_price = float(bar["close"])

            # 3. 模型预测
            model_output = None
            if self.model:
                feature_cols = self.feature_manager.get_feature_columns()
                X = latest_features[feature_cols].values
                model_output = self.model.predict(X)[0]

                # 如果有概率预测
                if hasattr(self.model, "predict_proba"):
                    proba = self.model.predict_proba(X)[0]
                    model_output = proba[1] if len(proba) > 1 else model_output

            # 4. 生成交易决策（复用现有系统）
            # 获取 ATR（如果存在）
            atr = latest_features.get("atr", current_price * 0.01)  # 默认 1%

            decision = self.decision_engine.generate_decision(
                model_output=model_output if model_output is not None else 0.5,
                features=latest_features.iloc[0],
                current_price=current_price,
                atr=atr,
                # 策略特定参数
                sr_type=(
                    "support" if model_output and model_output > 0.5 else "resistance"
                ),
            )

            # 5. 执行交易
            if decision:
                print(f"📊 交易信号: {decision.direction} @ {current_price}")
                print(f"   止损: {decision.stop_loss}, 止盈: {decision.take_profit}")
                print(
                    f"   仓位: {decision.position_size}, 置信度: {decision.confidence:.2f}"
                )

                # 检查是否已有持仓
                position = self.order_manager.get_position(self.symbol)

                if position is None:
                    # 开新仓
                    order_id = self.order_manager.submit_order(decision, self.symbol)
                    if order_id:
                        # 假设订单立即成交（实际应该监听订单状态）
                        self.order_manager.update_position(
                            self.symbol,
                            decision,
                            current_price,
                            decision.position_size,
                        )
                else:
                    # 已有持仓，检查是否需要加仓或减仓
                    if decision.add_position:
                        print("📈 加仓信号")
                        # 实现加仓逻辑
                    elif decision.reduce_position:
                        print("📉 减仓信号")
                        # 实现减仓逻辑

        except Exception as e:
            print(f"❌ 处理 K线数据时出错: {e}")
            import traceback

            traceback.print_exc()

    def monitor_positions(self):
        """监控持仓，检查止盈止损"""
        for symbol, position in list(self.order_manager.positions.items()):
            try:
                # 获取当前价格（实际应该从 WebSocket 获取）
                ticker = self.exchange.fetch_ticker(symbol)
                current_price = ticker["last"]

                # 检查止损
                if position.direction > 0:  # Long
                    if current_price <= position.stop_loss:
                        print(f"🛑 触发止损: {symbol} @ {current_price}")
                        self._close_position(symbol, "stop_loss")
                        continue
                    elif current_price >= position.take_profit:
                        print(f"🎯 触发止盈: {symbol} @ {current_price}")
                        self._close_position(symbol, "take_profit")
                        continue
                else:  # Short
                    if current_price >= position.stop_loss:
                        print(f"🛑 触发止损: {symbol} @ {current_price}")
                        self._close_position(symbol, "stop_loss")
                        continue
                    elif current_price <= position.take_profit:
                        print(f"🎯 触发止盈: {symbol} @ {current_price}")
                        self._close_position(symbol, "take_profit")
                        continue

            except Exception as e:
                print(f"❌ 监控持仓时出错: {e}")

    def _close_position(self, symbol: str, reason: str):
        """平仓"""
        position = self.order_manager.get_position(symbol)
        if not position:
            return

        try:
            # 创建平仓订单
            side = "sell" if position.direction > 0 else "buy"
            order = self.exchange.create_order(
                symbol=symbol,
                type="market",
                side=side,
                amount=position.quantity,
            )

            print(f"✅ 平仓订单已提交: {order['id']} ({reason})")
            self.order_manager.close_position(symbol)

        except Exception as e:
            print(f"❌ 平仓失败: {e}")

    def start(self):
        """启动交易引擎"""
        self.running = True
        print(f"🚀 启动实时交易引擎: {self.strategy_name}")
        print(f"   交易对: {self.symbol}")

        # 启动监控任务
        asyncio.create_task(self._monitor_loop())

    async def _monitor_loop(self):
        """监控循环"""
        while self.running:
            try:
                self.monitor_positions()
                await asyncio.sleep(1)  # 每秒检查一次
            except Exception as e:
                print(f"❌ 监控循环出错: {e}")
                await asyncio.sleep(5)

    def stop(self):
        """停止交易引擎"""
        self.running = False
        print("🛑 交易引擎已停止")


# 使用示例
if __name__ == "__main__":
    if not CCXT_AVAILABLE:
        print("❌ 请先安装 ccxt: pip install ccxt")
        exit(1)

    # 创建交易所实例（使用测试网）
    exchange = ccxt.binance(
        {
            "apiKey": "your_api_key",
            "secret": "your_secret",
            "enableRateLimit": True,
            "options": {
                "defaultType": "future",  # 使用期货
            },
            "sandbox": True,  # 使用测试网
        }
    )

    # 创建交易引擎
    engine = SimpleRealtimeEngine(
        strategy_name="sr_reversal",
        symbol="BTC/USDT:USDT",
        exchange=exchange,
        model_path="models/sr_reversal/model.pkl",  # 可选
    )

    # 模拟接收 K线数据（实际应该从 WebSocket 接收）
    # 这里只是示例，实际使用时需要实现 WebSocket 客户端
    def simulate_bar():
        """模拟 K线数据"""
        ticker = exchange.fetch_ticker("BTC/USDT:USDT")
        return {
            "open": ticker["open"],
            "high": ticker["high"],
            "low": ticker["low"],
            "close": ticker["last"],
            "volume": ticker["quoteVolume"],
            "timestamp": exchange.milliseconds(),
        }

    # 启动引擎
    engine.start()

    # 模拟接收数据（实际应该从 WebSocket 接收）
    try:
        while True:
            bar = simulate_bar()
            engine.on_bar(bar)
            time.sleep(60)  # 每分钟处理一次（实际应该实时）
    except KeyboardInterrupt:
        engine.stop()
