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
                # 先尝试执行 schema，如果失败（表已存在），继续执行列检查
                try:
                    conn.executescript(schema_sql)
                    conn.commit()
                except sqlite3.OperationalError as e:
                    # 如果表已存在，忽略错误，继续执行列检查
                    if "already exists" not in str(e).lower() and "duplicate" not in str(e).lower():
                        # 如果是其他错误，重新抛出
                        raise
                    conn.commit()
                
                # 确保所有必需的表都存在
                self._ensure_order_columns(conn)
                self._ensure_safety_state_table(conn)
                self._ensure_slots_state_table(conn)
                self._ensure_add_position_state_table(conn)
                self._ensure_live_config_table(conn)
            finally:
                conn.close()

    def _ensure_order_columns(self, conn: sqlite3.Connection) -> None:
        """确保orders表包含新增字段（兼容旧数据库）"""
        try:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(orders)")
            existing_cols = {row[1] for row in cursor.fetchall()}

            if "client_order_id" not in existing_cols:
                cursor.execute("ALTER TABLE orders ADD COLUMN client_order_id TEXT")
                conn.commit()
        except sqlite3.Error:
            # 如果表不存在或其他错误，忽略以保持初始化流程
            pass

    def _ensure_safety_state_table(self, conn: sqlite3.Connection) -> None:
        """确保safety_state表存在"""
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS safety_state (
                    state_id TEXT PRIMARY KEY,
                    halted INTEGER DEFAULT 0,
                    halt_reason TEXT,
                    halt_since TIMESTAMP,
                    cooldown_until TIMESTAMP,
                    last_metrics TEXT,
                    last_reset_date DATE,
                    last_daily_halt_date DATE,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()
        except sqlite3.Error:
            # 保持初始化流程稳定
            pass

    def _ensure_slots_state_table(self, conn: sqlite3.Connection) -> None:
        """确保slots_state表存在"""
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS slots_state (
                    position_id TEXT PRIMARY KEY,
                    symbol TEXT,
                    archetype TEXT,
                    opened_at TIMESTAMP,
                    closed_at TIMESTAMP,
                    close_reason TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()
        except sqlite3.Error:
            # 保持初始化流程稳定
            pass

    def _ensure_add_position_state_table(self, conn: sqlite3.Connection) -> None:
        """确保add_position_state表存在"""
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS add_position_state (
                    position_id TEXT PRIMARY KEY,
                    add_count INTEGER DEFAULT 0,
                    locked_profit INTEGER DEFAULT 0,
                    current_r REAL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()
        except sqlite3.Error:
            # 保持初始化流程稳定
            pass

    
    def _get_connection(self) -> sqlite3.Connection:
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        # 启用 WAL 模式以提高并发性能
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    # ========== Safety state ==========

    def get_safety_state(self, *, state_id: str = "global") -> Optional[Dict[str, Any]]:
        """读取安全状态（展开字段结构）"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM safety_state WHERE state_id = ?", (state_id,)
            )
            row = cursor.fetchone()
            if not row:
                return None
            data = dict(row) if isinstance(row, sqlite3.Row) else {}
            try:
                halt_reason = json.loads(data.get("halt_reason") or "[]")
            except Exception:
                halt_reason = []
            try:
                last_metrics = json.loads(data.get("last_metrics") or "{}")
            except Exception:
                last_metrics = {}
            return {
                "halted": bool(data.get("halted", False)),
                "halt_reason": halt_reason,
                "halt_since": data.get("halt_since"),
                "cooldown_until": data.get("cooldown_until"),
                "last_metrics": last_metrics,
                "last_reset_date": data.get("last_reset_date"),
                "last_daily_halt_date": data.get("last_daily_halt_date"),
            }
        finally:
            conn.close()

    def upsert_safety_state(
        self, *, state_id: str = "global", payload: Dict[str, Any]
    ) -> None:
        """写入安全状态（展开字段结构）"""
        conn = self._get_connection()
        try:
            halted = bool(payload.get("halted", False))
            halt_reason = payload.get("halt_reason") or []
            last_metrics = payload.get("last_metrics") or {}
            row_payload = {
                "halted": 1 if halted else 0,
                "halt_reason": json.dumps(halt_reason, ensure_ascii=False),
                "halt_since": payload.get("halt_since"),
                "cooldown_until": payload.get("cooldown_until"),
                "last_metrics": json.dumps(last_metrics, ensure_ascii=False),
                "last_reset_date": payload.get("last_reset_date"),
                "last_daily_halt_date": payload.get("last_daily_halt_date"),
            }
            cursor = conn.cursor()
            columns = ["state_id"] + list(row_payload.keys()) + ["updated_at"]
            values = [state_id] + list(row_payload.values())
            placeholders = ", ".join(["?"] * len(values) + ["CURRENT_TIMESTAMP"])
            update_cols = [f"{c} = excluded.{c}" for c in row_payload.keys()]
            update_cols.append("updated_at = CURRENT_TIMESTAMP")
            cursor.execute(
                f"""
                INSERT INTO safety_state ({", ".join(columns)})
                VALUES ({placeholders})
                ON CONFLICT(state_id) DO UPDATE SET
                    {", ".join(update_cols)}
                """,
                values,
            )
            conn.commit()
        finally:
            conn.close()

    # ========== Slots runtime state ==========

    def get_slots_state(self) -> Dict[str, Any]:
        """读取槽位状态，返回与 SlotsRuntimeState.as_dict() 一致结构"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM slots_state")
            rows = cursor.fetchall()
            active: Dict[str, Any] = {}
            for row in rows:
                data = dict(row) if isinstance(row, sqlite3.Row) else {}
                pid = str(data.get("position_id") or "").strip()
                if not pid:
                    continue
                active[pid] = {
                    "position_id": pid,
                    "symbol": data.get("symbol"),
                    "archetype": data.get("archetype"),
                    "opened_at": data.get("opened_at"),
                    "closed_at": data.get("closed_at"),
                    "close_reason": data.get("close_reason"),
                }
            return {"active": active}
        finally:
            conn.close()

    def upsert_slots_state(self, *, payload: Dict[str, Any]) -> None:
        """写入槽位状态，传入 SlotsRuntimeState.as_dict()"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM slots_state")
            active = (payload or {}).get("active") or {}
            for pid, rec in (active or {}).items():
                r = rec or {}
                cursor.execute(
                    """
                    INSERT INTO slots_state (
                        position_id, symbol, archetype, opened_at, closed_at, close_reason, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                    (
                        str(pid),
                        r.get("symbol"),
                        r.get("archetype"),
                        r.get("opened_at"),
                        r.get("closed_at"),
                        r.get("close_reason"),
                    ),
                )
            conn.commit()
        finally:
            conn.close()

    # ========== Add-position runtime state ==========

    def get_add_position_state(self) -> Dict[str, Any]:
        """读取加仓状态，返回与 AddPositionRuntimeState.as_dict() 一致结构"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM add_position_state")
            rows = cursor.fetchall()
            positions: Dict[str, Any] = {}
            for row in rows:
                data = dict(row) if isinstance(row, sqlite3.Row) else {}
                pid = str(data.get("position_id") or "").strip()
                if not pid:
                    continue
                positions[pid] = {
                    "position_id": pid,
                    "add_count": int(data.get("add_count", 0) or 0),
                    "locked_profit": bool(data.get("locked_profit", 0)),
                    "current_r": data.get("current_r"),
                    "updated_at": data.get("updated_at"),
                }
            return {"positions": positions}
        finally:
            conn.close()

    def upsert_add_position_state(self, *, payload: Dict[str, Any]) -> None:
        """写入加仓状态，传入 AddPositionRuntimeState.as_dict()"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM add_position_state")
            positions = (payload or {}).get("positions") or {}
            for pid, rec in (positions or {}).items():
                r = rec or {}
                cursor.execute(
                    """
                    INSERT INTO add_position_state (
                        position_id, add_count, locked_profit, current_r, updated_at
                    ) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                    (
                        str(pid),
                        int(r.get("add_count", 0) or 0),
                        1 if bool(r.get("locked_profit", False)) else 0,
                        r.get("current_r"),
                    ),
                )
            conn.commit()
        finally:
            conn.close()

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
                    order_id, binance_order_id, client_order_id, position_id, symbol, side,
                    order_type, quantity, price, stop_price, status,
                    filled_quantity, average_price, commission, commission_asset,
                    created_at, updated_at, filled_at, canceled_at, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                order.order_id,
                order.binance_order_id,
                order.client_order_id,
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

    def get_order_by_client_id(self, client_order_id: str) -> Optional[Order]:
        """通过客户端订单ID获取订单"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM orders WHERE client_order_id = ?", (client_order_id,))
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
                    "SELECT * FROM orders WHERE status IN ('pending', 'partially_filled') AND symbol = ?",
                    (symbol,)
                )
            else:
                cursor.execute("SELECT * FROM orders WHERE status IN ('pending', 'partially_filled')")
            
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
                    binance_order_id = ?, client_order_id = ?, position_id = ?, symbol = ?, side = ?,
                    order_type = ?, quantity = ?, price = ?, stop_price = ?,
                    status = ?, filled_quantity = ?, average_price = ?,
                    commission = ?, commission_asset = ?,
                    updated_at = CURRENT_TIMESTAMP,
                    filled_at = ?, canceled_at = ?, error_message = ?
                WHERE order_id = ?
            """, (
                order.binance_order_id,
                order.client_order_id,
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
            client_order_id=row['client_order_id'] if 'client_order_id' in row.keys() else None,
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
    
    # ========== Live Config CRUD ==========
    
    def _ensure_live_config_table(self, conn: sqlite3.Connection) -> None:
        """确保live_config表存在"""
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS live_config (
                    config_id INTEGER PRIMARY KEY CHECK (config_id = 1),
                    enabled_archetypes TEXT NOT NULL,
                    size_multipliers TEXT NOT NULL,
                    window_minutes INTEGER NOT NULL,
                    min_order_interval_minutes INTEGER NOT NULL,
                    nnmultihead_inference TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_by TEXT
                )
                """
            )
            conn.commit()
        except sqlite3.Error:
            pass

    def get_live_config(self) -> Optional[Dict[str, Any]]:
        """获取live配置（解析为结构化字段）"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    enabled_archetypes,
                    size_multipliers,
                    window_minutes,
                    min_order_interval_minutes,
                    nnmultihead_inference
                FROM live_config
                WHERE config_id = 1
                """
            )
            row = cursor.fetchone()
            if not row:
                return None
            return {
                "enabled_archetypes": json.loads(row["enabled_archetypes"]),
                "size_multipliers": json.loads(row["size_multipliers"]),
                "window_minutes": int(row["window_minutes"]),
                "min_order_interval_minutes": int(row["min_order_interval_minutes"]),
                "nnmultihead_inference": json.loads(row["nnmultihead_inference"]),
            }
        finally:
            conn.close()

    def upsert_live_config(
        self,
        *,
        enabled_archetypes: List[str],
        size_multipliers: Dict[str, float],
        window_minutes: int,
        min_order_interval_minutes: int,
        nnmultihead_inference: Dict[str, Any],
        updated_by: Optional[str] = None,
    ) -> None:
        """写入live配置（单行）"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO live_config
                (
                    config_id,
                    enabled_archetypes,
                    size_multipliers,
                    window_minutes,
                    min_order_interval_minutes,
                    nnmultihead_inference,
                    updated_at,
                    updated_by
                )
                VALUES (1, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
                """,
                (
                    json.dumps(list(enabled_archetypes)),
                    json.dumps(dict(size_multipliers)),
                    int(window_minutes),
                    int(min_order_interval_minutes),
                    json.dumps(dict(nnmultihead_inference)),
                    updated_by,
                ),
            )
            conn.commit()
        finally:
            conn.close()
