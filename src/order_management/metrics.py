"""
Prometheus指标定义
"""
from prometheus_client import Counter, Gauge, Histogram, Summary, start_http_server

# ========== 仓位指标 ==========

position_count = Gauge(
    'order_management_position_count',
    'Current open positions',
    ['symbol', 'side']
)

position_unrealized_pnl = Gauge(
    'order_management_position_unrealized_pnl',
    'Unrealized PnL',
    ['symbol']
)

position_total_value = Gauge(
    'order_management_position_total_value',
    'Total position value',
    ['symbol']
)

position_risk_per_symbol = Gauge(
    'order_management_position_risk_per_symbol',
    'Risk per symbol',
    ['symbol']
)

# ========== 订单指标 ==========

orders_total = Counter(
    'order_management_orders_total',
    'Total orders',
    ['status', 'type', 'symbol']
)

orders_execution_time = Histogram(
    'order_management_orders_execution_time',
    'Order execution time in seconds',
    ['symbol'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

orders_fill_rate = Gauge(
    'order_management_orders_fill_rate',
    'Order fill rate',
    ['symbol']
)

# ========== 风险指标 ==========

daily_pnl = Gauge(
    'order_management_daily_pnl',
    'Daily PnL'
)

daily_loss = Gauge(
    'order_management_daily_loss',
    'Daily loss'
)

margin_usage_ratio = Gauge(
    'order_management_margin_usage_ratio',
    'Margin usage ratio'
)

max_drawdown = Gauge(
    'order_management_max_drawdown',
    'Max drawdown'
)

# ========== 性能指标 ==========

api_request_duration = Histogram(
    'order_management_api_request_duration',
    'API request duration in seconds',
    ['endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

websocket_reconnect_count = Counter(
    'order_management_websocket_reconnect_count',
    'WebSocket reconnects'
)

message_processing_duration = Histogram(
    'order_management_message_processing_duration',
    'Message processing duration in seconds',
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
)

# ========== 辅助函数 ==========

def start_metrics_server(port: int = 8000):
    """
    启动Prometheus metrics HTTP服务器
    
    Args:
        port: 端口号
    """
    start_http_server(port)
    print(f"Prometheus metrics server started on port {port}")
