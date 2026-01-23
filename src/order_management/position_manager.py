"""
仓位管理器
管理仓位生命周期、加仓、减仓、止损止盈
"""
import logging
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from threading import Lock

from .models import (
    Position, PositionSide, PositionStatus,
    PositionOperation, OperationType,
    StopLossTrailing
)
from .storage import Storage
from .binance_api import BinanceAPI

logger = logging.getLogger(__name__)


class PositionManager:
    """仓位管理器"""
    
    def __init__(self, storage: Storage, binance_api: BinanceAPI):
        """
        初始化仓位管理器
        
        Args:
            storage: 存储层实例
            binance_api: Binance API实例
        """
        self.storage = storage
        self.binance_api = binance_api
        self._lock = Lock()
    
    def create_position(
        self,
        symbol: str,
        side: PositionSide,
        entry_price: float,
        size: float,
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None,
        strategy_id: Optional[str] = None,
        notes: Optional[str] = None
    ) -> Position:
        """
        创建新仓位
        
        Args:
            symbol: 交易对符号
            side: 仓位方向
            entry_price: 入场价格
            size: 仓位大小
            stop_loss_price: 止损价格
            take_profit_price: 止盈价格
            strategy_id: 策略ID
            notes: 备注
        
        Returns:
            创建的仓位对象
        """
        with self._lock:
            position_id = f"{symbol}_{side.value}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
            
            position = Position(
                position_id=position_id,
                symbol=symbol,
                side=side,
                entry_time=datetime.now(),
                entry_price=entry_price,
                initial_size=size,
                current_size=size,
                total_cost=entry_price * size,
                status=PositionStatus.OPEN,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
                strategy_id=strategy_id,
                notes=notes
            )
            
            # 计算当前价值和未实现盈亏
            self._update_position_pnl(position)
            
            # 保存到数据库
            if self.storage.create_position(position):
                logger.info(f"创建仓位成功: {position_id}, {symbol}, {side.value}, size={size}")
                return position
            else:
                raise Exception(f"创建仓位失败: {position_id}")
    
    def add_to_position(
        self,
        position_id: str,
        size: float,
        price: float,
        order_id: Optional[str] = None,
        reason: Optional[str] = None
    ) -> Position:
        """
        加仓
        
        Args:
            position_id: 仓位ID
            size: 加仓数量
            price: 加仓价格
            order_id: 订单ID
            reason: 加仓原因
        
        Returns:
            更新后的仓位对象
        """
        with self._lock:
            position = self.storage.get_position(position_id)
            if not position:
                raise ValueError(f"仓位不存在: {position_id}")
            
            if position.status != PositionStatus.OPEN:
                raise ValueError(f"仓位状态不允许加仓: {position.status}")
            
            # 计算新的平均成本价
            old_cost = position.total_cost
            new_cost = price * size
            total_size = position.current_size + size
            new_avg_price = (old_cost + new_cost) / total_size
            
            # 更新仓位
            position.current_size = total_size
            position.total_cost = old_cost + new_cost
            position.entry_price = new_avg_price
            position.status = PositionStatus.OPEN if total_size > 0 else PositionStatus.CLOSED
            
            # 更新盈亏
            self._update_position_pnl(position)
            
            # 保存操作记录
            operation = PositionOperation(
                operation_id=f"op_{uuid.uuid4().hex}",
                position_id=position_id,
                operation_type=OperationType.ADD,
                operation_time=datetime.now(),
                size=size,
                price=price,
                cumulative_pnl=position.unrealized_pnl,
                order_id=order_id,
                reason=reason or "加仓"
            )
            self.storage.create_position_operation(operation)
            
            # 更新数据库
            if self.storage.update_position(position):
                logger.info(f"加仓成功: {position_id}, size={size}, price={price}")
                return position
            else:
                raise Exception(f"加仓失败: {position_id}")
    
    def reduce_position(
        self,
        position_id: str,
        size: float,
        price: float,
        order_id: Optional[str] = None,
        reason: Optional[str] = None
    ) -> Position:
        """
        减仓
        
        Args:
            position_id: 仓位ID
            size: 减仓数量
            price: 减仓价格
            order_id: 订单ID
            reason: 减仓原因
        
        Returns:
            更新后的仓位对象
        """
        with self._lock:
            position = self.storage.get_position(position_id)
            if not position:
                raise ValueError(f"仓位不存在: {position_id}")
            
            if position.current_size < size:
                raise ValueError(f"减仓数量超过当前仓位: {position.current_size} < {size}")
            
            # 计算部分盈亏
            if position.side == PositionSide.LONG:
                pnl = (price - position.entry_price) * size
            else:  # SHORT
                pnl = (position.entry_price - price) * size
            
            # 更新仓位
            position.current_size -= size
            position.total_cost -= position.entry_price * size
            position.realized_pnl += pnl
            
            # 更新状态
            if position.current_size == 0:
                position.status = PositionStatus.CLOSED
                position.exit_time = datetime.now()
                position.exit_price = price
                position.exit_reason = reason or "减仓平仓"
            else:
                position.status = PositionStatus.PARTIAL
            
            # 更新盈亏
            self._update_position_pnl(position)
            
            # 保存操作记录
            operation = PositionOperation(
                operation_id=f"op_{uuid.uuid4().hex}",
                position_id=position_id,
                operation_type=OperationType.REDUCE,
                operation_time=datetime.now(),
                size=size,
                price=price,
                pnl=pnl,
                cumulative_pnl=position.realized_pnl + (position.unrealized_pnl or 0),
                order_id=order_id,
                reason=reason or "减仓"
            )
            self.storage.create_position_operation(operation)
            
            # 更新数据库
            if self.storage.update_position(position):
                logger.info(f"减仓成功: {position_id}, size={size}, price={price}, pnl={pnl}")
                return position
            else:
                raise Exception(f"减仓失败: {position_id}")
    
    def close_position(
        self,
        position_id: str,
        price: float,
        order_id: Optional[str] = None,
        reason: Optional[str] = None
    ) -> Position:
        """
        平仓
        
        Args:
            position_id: 仓位ID
            price: 平仓价格
            order_id: 订单ID
            reason: 平仓原因
        
        Returns:
            更新后的仓位对象
        """
        position = self.storage.get_position(position_id)
        if not position:
            raise ValueError(f"仓位不存在: {position_id}")
        
        return self.reduce_position(
            position_id,
            position.current_size,
            price,
            order_id,
            reason or "平仓"
        )
    
    def update_stop_loss(
        self,
        position_id: str,
        stop_loss_price: float,
        reason: Optional[str] = None
    ) -> Position:
        """
        更新止损
        
        Args:
            position_id: 仓位ID
            stop_loss_price: 新的止损价格
            reason: 更新原因
        
        Returns:
            更新后的仓位对象
        """
        with self._lock:
            position = self.storage.get_position(position_id)
            if not position:
                raise ValueError(f"仓位不存在: {position_id}")
            
            old_stop_loss = position.stop_loss_price
            position.stop_loss_price = stop_loss_price
            
            # 如果是止损上移，记录历史
            if old_stop_loss and stop_loss_price > old_stop_loss:
                # 获取当前价格（从Binance API）
                binance_position = self.binance_api.get_position(position.symbol)
                current_price = binance_position.get('mark_price', position.entry_price) if binance_position else position.entry_price
                
                # 计算保护的利润
                if position.side == PositionSide.LONG:
                    profit_protected = (stop_loss_price - old_stop_loss) * position.current_size
                else:  # SHORT
                    profit_protected = (old_stop_loss - stop_loss_price) * position.current_size
                
                trailing = StopLossTrailing(
                    record_id=f"trail_{uuid.uuid4().hex}",
                    position_id=position_id,
                    old_stop_loss=old_stop_loss,
                    new_stop_loss=stop_loss_price,
                    move_time=datetime.now(),
                    current_price=current_price,
                    profit_protected=profit_protected,
                    reason=reason or "止损上移"
                )
                self.storage.create_stop_loss_trailing(trailing)
                
                # 保存操作记录
                operation = PositionOperation(
                    operation_id=f"op_{uuid.uuid4().hex}",
                    position_id=position_id,
                    operation_type=OperationType.STOP_LOSS_MOVE,
                    operation_time=datetime.now(),
                    size=0,
                    price=current_price,
                    stop_loss_price=stop_loss_price,
                    reason=reason or "止损上移"
                )
                self.storage.create_position_operation(operation)
            
            # 更新数据库
            if self.storage.update_position(position):
                logger.info(f"更新止损成功: {position_id}, stop_loss={stop_loss_price}")
                return position
            else:
                raise Exception(f"更新止损失败: {position_id}")
    
    def update_take_profit(
        self,
        position_id: str,
        take_profit_price: float,
        reason: Optional[str] = None
    ) -> Position:
        """
        更新止盈
        
        Args:
            position_id: 仓位ID
            take_profit_price: 新的止盈价格
            reason: 更新原因
        
        Returns:
            更新后的仓位对象
        """
        with self._lock:
            position = self.storage.get_position(position_id)
            if not position:
                raise ValueError(f"仓位不存在: {position_id}")
            
            position.take_profit_price = take_profit_price
            
            # 保存操作记录
            operation = PositionOperation(
                operation_id=f"op_{uuid.uuid4().hex}",
                position_id=position_id,
                operation_type=OperationType.TAKE_PROFIT_MOVE,
                operation_time=datetime.now(),
                size=0,
                price=position.entry_price,
                take_profit_price=take_profit_price,
                reason=reason or "更新止盈"
            )
            self.storage.create_position_operation(operation)
            
            # 更新数据库
            if self.storage.update_position(position):
                logger.info(f"更新止盈成功: {position_id}, take_profit={take_profit_price}")
                return position
            else:
                raise Exception(f"更新止盈失败: {position_id}")
    
    def move_stop_loss(
        self,
        position_id: str,
        method: str = "fixed",
        value: float = 0.0,
        reason: Optional[str] = None
    ) -> Position:
        """
        止损上移（根据策略自动计算）
        
        Args:
            position_id: 仓位ID
            method: 上移方法 ('fixed', 'percentage', 'atr')
            value: 上移值（固定点数、百分比或ATR倍数）
            reason: 上移原因
        
        Returns:
            更新后的仓位对象
        """
        position = self.storage.get_position(position_id)
        if not position:
            raise ValueError(f"仓位不存在: {position_id}")
        
        # 获取当前价格
        binance_position = self.binance_api.get_position(position.symbol)
        if not binance_position:
            raise ValueError(f"无法获取仓位价格: {position.symbol}")
        
        current_price = binance_position.get('mark_price', position.entry_price)
        old_stop_loss = position.stop_loss_price or position.entry_price
        
        # 计算新的止损价格
        if method == "fixed":
            if position.side == PositionSide.LONG:
                new_stop_loss = old_stop_loss + value
            else:  # SHORT
                new_stop_loss = old_stop_loss - value
        elif method == "percentage":
            if position.side == PositionSide.LONG:
                new_stop_loss = current_price * (1 - value / 100)
            else:  # SHORT
                new_stop_loss = current_price * (1 + value / 100)
        else:
            raise ValueError(f"不支持的止损上移方法: {method}")
        
        # 确保新止损价格合理
        if position.side == PositionSide.LONG:
            if new_stop_loss <= old_stop_loss:
                raise ValueError(f"止损上移价格必须大于当前止损: {new_stop_loss} <= {old_stop_loss}")
        else:  # SHORT
            if new_stop_loss >= old_stop_loss:
                raise ValueError(f"止损上移价格必须小于当前止损: {new_stop_loss} >= {old_stop_loss}")
        
        return self.update_stop_loss(position_id, new_stop_loss, reason)
    
    def get_position(self, position_id: str) -> Optional[Position]:
        """获取仓位信息"""
        return self.storage.get_position(position_id)
    
    def get_open_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """获取所有开仓"""
        return self.storage.get_open_positions(symbol)
    
    def calculate_pnl(self, position_id: str) -> Dict[str, Any]:
        """
        计算盈亏
        
        Returns:
            盈亏信息
        """
        position = self.storage.get_position(position_id)
        if not position:
            raise ValueError(f"仓位不存在: {position_id}")
        
        # 获取当前价格
        binance_position = self.binance_api.get_position(position.symbol)
        current_price = binance_position.get('mark_price', position.entry_price) if binance_position else position.entry_price
        
        # 更新并计算盈亏
        self._update_position_pnl(position, current_price)
        
        return {
            'position_id': position_id,
            'unrealized_pnl': position.unrealized_pnl,
            'realized_pnl': position.realized_pnl,
            'total_pnl': (position.unrealized_pnl or 0) + position.realized_pnl,
            'current_price': current_price,
            'entry_price': position.entry_price
        }
    
    def _update_position_pnl(self, position: Position, current_price: Optional[float] = None):
        """
        更新仓位的盈亏
        
        Args:
            position: 仓位对象
            current_price: 当前价格，如果为None则从Binance API获取
        """
        if current_price is None:
            binance_position = self.binance_api.get_position(position.symbol)
            if binance_position:
                current_price = binance_position.get('mark_price')
            else:
                current_price = position.entry_price
        
        # 计算未实现盈亏
        if position.side == PositionSide.LONG:
            position.unrealized_pnl = (current_price - position.entry_price) * position.current_size
        else:  # SHORT
            position.unrealized_pnl = (position.entry_price - current_price) * position.current_size
        
        # 更新总价值
        position.total_value = current_price * position.current_size
