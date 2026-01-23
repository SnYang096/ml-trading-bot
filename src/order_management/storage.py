"""
SQLite存储层
封装数据库操作，提供CRUD方法
"""
import sqlite3
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path

from .models import (
    Position, PositionSide, PositionStatus,
    Order, OrderSide, OrderType, OrderStatus,
    PositionOperation, OperationType,
    StopLossTrailing,
    PerformanceMetrics
)


class Storage:
    """SQLite存储层"""
    
    def __init__(self, db_path: str):
        """
        初始化存储层
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self._ensure_db_exists()
        self._init_schema()
    
    def _ensure_db_exists(self):
        """确保数据库文件存在"""
        db_file = Path(self.db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
    
    def _init_schema(self):
        """初始化数据库schema"""
        schema_file = Path(__file__).parent / "database" / "schema.sql"
        if schema_file.exists():
            with open(schema_file, 'r', encoding='utf-8') as f:
                schema_sql = f.read()
            
            conn = sqlite3.connect(self.db_path)
            try:
                conn.executescript(schema_sql)
                conn.commit()
            finally:
                conn.close()
    
    def _get_connection(self) -> sqlite3.Connection:
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    # ========== Position CRUD ==========
    
    def create_position(self, position: Position) -> bool:
        """创建仓位"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO positions (
                    position_id, symbol, side, entry_time, exit_time,
                    entry_price, exit_price, initial_size, current_size,
                    total_cost, total_value, unrealized_pnl, realized_pnl,
                    status, stop_loss_price, take_profit_price,
                    trailing_stop_config, exit_reason, strategy_id, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                position.position_id,
                position.symbol,
                position.side.value,
                position.entry_time,
                position.exit_time,
                position.entry_price,
                position.exit_price,
                position.initial_size,
                position.current_size,
                position.total_cost,
                position.total_value,
                position.unrealized_pnl,
                position.realized_pnl,
                position.status.value,
                position.stop_loss_price,
                position.take_profit_price,
                json.dumps(position.trailing_stop_config) if position.trailing_stop_config else None,
                position.exit_reason,
                position.strategy_id,
                position.notes
            ))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
    
    def get_position(self, position_id: str) -> Optional[Position]:
        """获取仓位"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM positions WHERE position_id = ?", (position_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_position(row)
            return None
        finally:
            conn.close()
    
    def get_open_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """获取所有开仓"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            if symbol:
                cursor.execute(
                    "SELECT * FROM positions WHERE status = 'open' AND symbol = ?",
                    (symbol,)
                )
            else:
                cursor.execute("SELECT * FROM positions WHERE status = 'open'")
            
            rows = cursor.fetchall()
            return [self._row_to_position(row) for row in rows]
        finally:
            conn.close()
    
    def update_position(self, position: Position) -> bool:
        """更新仓位"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE positions SET
                    symbol = ?, side = ?, entry_time = ?, exit_time = ?,
                    entry_price = ?, exit_price = ?, initial_size = ?, current_size = ?,
                    total_cost = ?, total_value = ?, unrealized_pnl = ?, realized_pnl = ?,
                    status = ?, stop_loss_price = ?, take_profit_price = ?,
                    trailing_stop_config = ?, exit_reason = ?, strategy_id = ?, notes = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE position_id = ?
            """, (
                position.symbol,
                position.side.value,
                position.entry_time,
                position.exit_time,
                position.entry_price,
                position.exit_price,
                position.initial_size,
                position.current_size,
                position.total_cost,
                position.total_value,
                position.unrealized_pnl,
                position.realized_pnl,
                position.status.value,
                position.stop_loss_price,
                position.take_profit_price,
                json.dumps(position.trailing_stop_config) if position.trailing_stop_config else None,
                position.exit_reason,
                position.strategy_id,
                position.notes,
                position.position_id
            ))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
    
    def _row_to_position(self, row: sqlite3.Row) -> Position:
        """将数据库行转换为Position对象"""
        trailing_stop_config = None
        if row['trailing_stop_config']:
            trailing_stop_config = json.loads(row['trailing_stop_config'])
        
        return Position(
            position_id=row['position_id'],
            symbol=row['symbol'],
            side=PositionSide(row['side']),
            entry_time=datetime.fromisoformat(row['entry_time']) if isinstance(row['entry_time'], str) else row['entry_time'],
            exit_time=datetime.fromisoformat(row['exit_time']) if row['exit_time'] and isinstance(row['exit_time'], str) else row['exit_time'],
            entry_price=row['entry_price'],
            exit_price=row['exit_price'],
            initial_size=row['initial_size'],
            current_size=row['current_size'],
            total_cost=row['total_cost'],
            total_value=row['total_value'],
            unrealized_pnl=row['unrealized_pnl'],
            realized_pnl=row['realized_pnl'],
            status=PositionStatus(row['status']),
            stop_loss_price=row['stop_loss_price'],
            take_profit_price=row['take_profit_price'],
            trailing_stop_config=trailing_stop_config,
            exit_reason=row['exit_reason'],
            strategy_id=row['strategy_id'],
            notes=row['notes'],
            created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] and isinstance(row['created_at'], str) else row['created_at'],
            updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] and isinstance(row['updated_at'], str) else row['updated_at']
        )
    
    # ========== Order CRUD ==========
    
    def create_order(self, order: Order) -> bool:
        """创建订单"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO orders (
                    order_id, binance_order_id, position_id, symbol, side,
                    order_type, quantity, price, stop_price, status,
                    filled_quantity, average_price, commission, commission_asset,
                    created_at, updated_at, filled_at, canceled_at, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                order.order_id,
                order.binance_order_id,
                order.position_id,
                order.symbol,
                order.side.value,
                order.order_type.value,
                order.quantity,
                order.price,
                order.stop_price,
                order.status.value,
                order.filled_quantity,
                order.average_price,
                order.commission,
                order.commission_asset,
                order.created_at or datetime.now(),
                order.updated_at,
                order.filled_at,
                order.canceled_at,
                order.error_message
            ))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """获取订单"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_order(row)
            return None
        finally:
            conn.close()
    
    def get_order_by_binance_id(self, binance_order_id: str) -> Optional[Order]:
        """通过Binance订单ID获取订单"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM orders WHERE binance_order_id = ?", (binance_order_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_order(row)
            return None
        finally:
            conn.close()
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """获取未完成订单"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            if symbol:
                cursor.execute(
                    "SELECT * FROM orders WHERE status = 'pending' AND symbol = ?",
                    (symbol,)
                )
            else:
                cursor.execute("SELECT * FROM orders WHERE status = 'pending'")
            
            rows = cursor.fetchall()
            return [self._row_to_order(row) for row in rows]
        finally:
            conn.close()
    
    def update_order(self, order: Order) -> bool:
        """更新订单"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE orders SET
                    binance_order_id = ?, position_id = ?, symbol = ?, side = ?,
                    order_type = ?, quantity = ?, price = ?, stop_price = ?,
                    status = ?, filled_quantity = ?, average_price = ?,
                    commission = ?, commission_asset = ?,
                    updated_at = CURRENT_TIMESTAMP,
                    filled_at = ?, canceled_at = ?, error_message = ?
                WHERE order_id = ?
            """, (
                order.binance_order_id,
                order.position_id,
                order.symbol,
                order.side.value,
                order.order_type.value,
                order.quantity,
                order.price,
                order.stop_price,
                order.status.value,
                order.filled_quantity,
                order.average_price,
                order.commission,
                order.commission_asset,
                order.filled_at,
                order.canceled_at,
                order.error_message,
                order.order_id
            ))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
    
    def _row_to_order(self, row: sqlite3.Row) -> Order:
        """将数据库行转换为Order对象"""
        return Order(
            order_id=row['order_id'],
            binance_order_id=row['binance_order_id'],
            position_id=row['position_id'],
            symbol=row['symbol'],
            side=OrderSide(row['side']),
            order_type=OrderType(row['order_type']),
            quantity=row['quantity'],
            price=row['price'],
            stop_price=row['stop_price'],
            status=OrderStatus(row['status']),
            filled_quantity=row['filled_quantity'],
            average_price=row['average_price'],
            commission=row['commission'],
            commission_asset=row['commission_asset'],
            created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] and isinstance(row['created_at'], str) else row['created_at'],
            updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] and isinstance(row['updated_at'], str) else row['updated_at'],
            filled_at=datetime.fromisoformat(row['filled_at']) if row['filled_at'] and isinstance(row['filled_at'], str) else row['filled_at'],
            canceled_at=datetime.fromisoformat(row['canceled_at']) if row['canceled_at'] and isinstance(row['canceled_at'], str) else row['canceled_at'],
            error_message=row['error_message']
        )
    
    # ========== PositionOperation CRUD ==========
    
    def create_position_operation(self, operation: PositionOperation) -> bool:
        """创建仓位操作记录"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO position_operations (
                    operation_id, position_id, operation_type, operation_time,
                    size, price, pnl, cumulative_pnl, stop_loss_price,
                    take_profit_price, reason, order_id, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                operation.operation_id,
                operation.position_id,
                operation.operation_type.value,
                operation.operation_time,
                operation.size,
                operation.price,
                operation.pnl,
                operation.cumulative_pnl,
                operation.stop_loss_price,
                operation.take_profit_price,
                operation.reason,
                operation.order_id,
                operation.notes
            ))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
    
    def get_position_operations(self, position_id: str) -> List[PositionOperation]:
        """获取仓位的所有操作记录"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM position_operations WHERE position_id = ? ORDER BY operation_time",
                (position_id,)
            )
            rows = cursor.fetchall()
            return [self._row_to_position_operation(row) for row in rows]
        finally:
            conn.close()
    
    def _row_to_position_operation(self, row: sqlite3.Row) -> PositionOperation:
        """将数据库行转换为PositionOperation对象"""
        return PositionOperation(
            operation_id=row['operation_id'],
            position_id=row['position_id'],
            operation_type=OperationType(row['operation_type']),
            operation_time=datetime.fromisoformat(row['operation_time']) if isinstance(row['operation_time'], str) else row['operation_time'],
            size=row['size'],
            price=row['price'],
            pnl=row['pnl'],
            cumulative_pnl=row['cumulative_pnl'],
            stop_loss_price=row['stop_loss_price'],
            take_profit_price=row['take_profit_price'],
            reason=row['reason'],
            order_id=row['order_id'],
            notes=row['notes'],
            created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] and isinstance(row['created_at'], str) else row['created_at']
        )
    
    # ========== StopLossTrailing CRUD ==========
    
    def create_stop_loss_trailing(self, trailing: StopLossTrailing) -> bool:
        """创建止损上移记录"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO stop_loss_trailing (
                    record_id, position_id, old_stop_loss, new_stop_loss,
                    move_time, current_price, profit_protected, reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trailing.record_id,
                trailing.position_id,
                trailing.old_stop_loss,
                trailing.new_stop_loss,
                trailing.move_time,
                trailing.current_price,
                trailing.profit_protected,
                trailing.reason
            ))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
    
    def get_stop_loss_trailing_history(self, position_id: str) -> List[StopLossTrailing]:
        """获取仓位的止损上移历史"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM stop_loss_trailing WHERE position_id = ? ORDER BY move_time",
                (position_id,)
            )
            rows = cursor.fetchall()
            return [self._row_to_stop_loss_trailing(row) for row in rows]
        finally:
            conn.close()
    
    def _row_to_stop_loss_trailing(self, row: sqlite3.Row) -> StopLossTrailing:
        """将数据库行转换为StopLossTrailing对象"""
        return StopLossTrailing(
            record_id=row['record_id'],
            position_id=row['position_id'],
            old_stop_loss=row['old_stop_loss'],
            new_stop_loss=row['new_stop_loss'],
            move_time=datetime.fromisoformat(row['move_time']) if isinstance(row['move_time'], str) else row['move_time'],
            current_price=row['current_price'],
            profit_protected=row['profit_protected'],
            reason=row['reason'],
            created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] and isinstance(row['created_at'], str) else row['created_at']
        )
    
    # ========== PerformanceMetrics CRUD ==========
    
    def create_performance_metrics(self, metrics: PerformanceMetrics) -> bool:
        """创建性能指标"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO performance_metrics (
                    metric_id, date, symbol, total_trades, winning_trades,
                    losing_trades, win_rate, total_pnl, total_profit, total_loss,
                    profit_factor, max_drawdown, max_drawdown_period, sharpe_ratio,
                    average_win, average_loss, largest_win, largest_loss
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.metric_id,
                metrics.date.date() if isinstance(metrics.date, datetime) else metrics.date,
                metrics.symbol,
                metrics.total_trades,
                metrics.winning_trades,
                metrics.losing_trades,
                metrics.win_rate,
                metrics.total_pnl,
                metrics.total_profit,
                metrics.total_loss,
                metrics.profit_factor,
                metrics.max_drawdown,
                metrics.max_drawdown_period,
                metrics.sharpe_ratio,
                metrics.average_win,
                metrics.average_loss,
                metrics.largest_win,
                metrics.largest_loss
            ))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
    
    def get_performance_metrics(
        self,
        date: Optional[datetime] = None,
        symbol: Optional[str] = None
    ) -> List[PerformanceMetrics]:
        """获取性能指标"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            if date and symbol:
                cursor.execute(
                    "SELECT * FROM performance_metrics WHERE date = ? AND symbol = ?",
                    (date.date() if isinstance(date, datetime) else date, symbol)
                )
            elif date:
                cursor.execute(
                    "SELECT * FROM performance_metrics WHERE date = ?",
                    (date.date() if isinstance(date, datetime) else date,)
                )
            elif symbol:
                cursor.execute(
                    "SELECT * FROM performance_metrics WHERE symbol = ?",
                    (symbol,)
                )
            else:
                cursor.execute("SELECT * FROM performance_metrics")
            
            rows = cursor.fetchall()
            return [self._row_to_performance_metrics(row) for row in rows]
        finally:
            conn.close()
    
    def _row_to_performance_metrics(self, row: sqlite3.Row) -> PerformanceMetrics:
        """将数据库行转换为PerformanceMetrics对象"""
        date_value = row['date']
        if isinstance(date_value, str):
            date_value = datetime.fromisoformat(date_value).date()
        elif isinstance(date_value, datetime):
            date_value = date_value.date()
        
        return PerformanceMetrics(
            metric_id=row['metric_id'],
            date=date_value,
            symbol=row['symbol'],
            total_trades=row['total_trades'],
            winning_trades=row['winning_trades'],
            losing_trades=row['losing_trades'],
            win_rate=row['win_rate'],
            total_pnl=row['total_pnl'],
            total_profit=row['total_profit'],
            total_loss=row['total_loss'],
            profit_factor=row['profit_factor'],
            max_drawdown=row['max_drawdown'],
            max_drawdown_period=row['max_drawdown_period'],
            sharpe_ratio=row['sharpe_ratio'],
            average_win=row['average_win'],
            average_loss=row['average_loss'],
            largest_win=row['largest_win'],
            largest_loss=row['largest_loss'],
            created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] and isinstance(row['created_at'], str) else row['created_at']
        )
