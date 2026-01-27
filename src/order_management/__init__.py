"""
Binance合约订单管理系统

提供完整的订单管理、仓位管理、风险控制和监控功能。
"""

__version__ = "0.1.0"

# Public adapter for strategy -> order_management integration
try:
    from .signal_bridge import ExecutionSignal, OrderManagementBridge  # noqa: F401
except Exception:  # pragma: no cover
    # Optional dependency chain (e.g., ccxt) might be missing in test env
    ExecutionSignal = None  # type: ignore
    OrderManagementBridge = None  # type: ignore
