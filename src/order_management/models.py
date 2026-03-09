"""
数据模型定义
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum


class PositionSide(str, Enum):
    """仓位方向"""

    LONG = "long"
    SHORT = "short"


class PositionStatus(str, Enum):
    """仓位状态"""

    OPEN = "open"
    CLOSED = "closed"
    PARTIAL = "partial"


class OrderSide(str, Enum):
    """订单方向"""

    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """订单类型"""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_MARKET = "stop_market"
    TAKE_PROFIT = "take_profit"
    TAKE_PROFIT_MARKET = "take_profit_market"


class OrderStatus(str, Enum):
    """订单状态"""

    PENDING = "pending"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    SHADOW = "shadow"  # Shadow 模式: 只记录不下单


class OperationType(str, Enum):
    """仓位操作类型"""

    ADD = "add"
    REDUCE = "reduce"
    STOP_LOSS_MOVE = "stop_loss_move"
    TAKE_PROFIT_MOVE = "take_profit_move"


@dataclass
class Position:
    """
    仓位模型

    archetype: 策略原型（如 'trend', 'mean_reversion'）
    - 相同 archetype 的订单视为加仓
    - 不同 archetype 的订单是独立仓位

    add_count: 加仓次数（用于 slot 控制）
    parent_position_id: 父仓位 ID（如果是加仓）
    """

    position_id: str
    symbol: str
    side: PositionSide
    entry_time: datetime
    exit_time: Optional[datetime] = None
    entry_price: float = 0.0
    exit_price: Optional[float] = None
    initial_size: float = 0.0
    current_size: float = 0.0
    total_cost: float = 0.0
    total_value: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    realized_pnl: float = 0.0
    status: PositionStatus = PositionStatus.OPEN
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    trailing_stop_config: Optional[Dict[str, Any]] = None
    exit_reason: Optional[str] = None
    strategy_id: Optional[str] = None
    notes: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    # 新增字段
    archetype: Optional[str] = None  # 策略原型: 'trend', 'mean_reversion', etc.
    add_count: int = 0  # 加仓次数
    parent_position_id: Optional[str] = None  # 父仓位 ID（加仓时指向原始仓位）

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                result[key] = value.value
            elif isinstance(value, datetime):
                result[key] = value.isoformat()
            else:
                result[key] = value
        return result


@dataclass
class Order:
    """订单模型"""

    order_id: str
    binance_order_id: Optional[str] = None
    client_order_id: Optional[str] = None
    position_id: Optional[str] = None
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: float = 0.0
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_price: Optional[float] = None
    commission: float = 0.0
    commission_asset: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    canceled_at: Optional[datetime] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                result[key] = value.value
            elif isinstance(value, datetime):
                result[key] = value.isoformat()
            else:
                result[key] = value
        return result


@dataclass
class PositionOperation:
    """仓位操作记录模型"""

    operation_id: str
    position_id: str
    operation_type: OperationType
    operation_time: datetime
    size: float = 0.0
    price: float = 0.0
    pnl: Optional[float] = None
    cumulative_pnl: Optional[float] = None
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    reason: Optional[str] = None
    order_id: Optional[str] = None
    notes: Optional[str] = None
    created_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                result[key] = value.value
            elif isinstance(value, datetime):
                result[key] = value.isoformat()
            else:
                result[key] = value
        return result


@dataclass
class StopLossTrailing:
    """止损上移记录模型"""

    record_id: str
    position_id: str
    old_stop_loss: float
    new_stop_loss: float
    move_time: datetime
    current_price: float
    profit_protected: Optional[float] = None
    reason: Optional[str] = None
    created_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            else:
                result[key] = value
        return result


@dataclass
class PerformanceMetrics:
    """性能指标模型"""

    metric_id: str
    date: datetime
    symbol: Optional[str] = None
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: Optional[float] = None
    total_pnl: float = 0.0
    total_profit: float = 0.0
    total_loss: float = 0.0
    profit_factor: Optional[float] = None
    max_drawdown: Optional[float] = None
    max_drawdown_period: Optional[str] = None
    sharpe_ratio: Optional[float] = None
    average_win: Optional[float] = None
    average_loss: Optional[float] = None
    largest_win: Optional[float] = None
    largest_loss: Optional[float] = None
    created_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            else:
                result[key] = value
        return result
