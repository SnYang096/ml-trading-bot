-- Binance合约订单管理系统数据库Schema

-- 1. 仓位表 (positions)
CREATE TABLE IF NOT EXISTS positions (
    position_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,  -- 'long' or 'short'
    entry_time TIMESTAMP NOT NULL,
    exit_time TIMESTAMP,
    entry_price REAL NOT NULL,
    exit_price REAL,
    initial_size REAL NOT NULL,  -- 初始仓位大小
    current_size REAL NOT NULL,  -- 当前仓位大小
    total_cost REAL NOT NULL,  -- 总成本
    total_value REAL,  -- 当前总价值
    unrealized_pnl REAL,  -- 未实现盈亏
    realized_pnl REAL DEFAULT 0,  -- 已实现盈亏
    status TEXT NOT NULL,  -- 'open', 'closed', 'partial'
    stop_loss_price REAL,
    take_profit_price REAL,
    trailing_stop_config TEXT,  -- JSON格式的追踪止损配置
    exit_reason TEXT,  -- 'stop_loss', 'take_profit', 'manual', 'signal'
    strategy_id TEXT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. 仓位操作记录表 (position_operations)
CREATE TABLE IF NOT EXISTS position_operations (
    operation_id TEXT PRIMARY KEY,
    position_id TEXT NOT NULL,
    operation_type TEXT NOT NULL,  -- 'add', 'reduce', 'stop_loss_move', 'take_profit_move'
    operation_time TIMESTAMP NOT NULL,
    size REAL NOT NULL,  -- 操作数量
    price REAL NOT NULL,  -- 操作价格
    pnl REAL,  -- 本次操作的盈亏
    cumulative_pnl REAL,  -- 累计盈亏
    stop_loss_price REAL,  -- 操作后的止损价
    take_profit_price REAL,  -- 操作后的止盈价
    reason TEXT,  -- 操作原因
    order_id TEXT,  -- Binance订单ID
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (position_id) REFERENCES positions(position_id)
);

-- 3. 订单表 (orders)
CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,  -- 本地订单ID
    binance_order_id TEXT,  -- Binance订单ID
    client_order_id TEXT,  -- 客户端订单ID（用于幂等）
    position_id TEXT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,  -- 'buy' or 'sell'
    order_type TEXT NOT NULL,  -- 'market', 'limit', 'stop', 'stop_market', 'take_profit', 'take_profit_market'
    quantity REAL NOT NULL,
    price REAL,  -- limit订单价格
    stop_price REAL,  -- stop订单触发价
    status TEXT NOT NULL,  -- 'pending', 'filled', 'canceled', 'rejected', 'expired'
    filled_quantity REAL DEFAULT 0,
    average_price REAL,
    commission REAL DEFAULT 0,
    commission_asset TEXT,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP,
    filled_at TIMESTAMP,
    canceled_at TIMESTAMP,
    error_message TEXT,
    FOREIGN KEY (position_id) REFERENCES positions(position_id)
);

-- 4. 止损上移记录表 (stop_loss_trailing)
CREATE TABLE IF NOT EXISTS stop_loss_trailing (
    record_id TEXT PRIMARY KEY,
    position_id TEXT NOT NULL,
    old_stop_loss REAL NOT NULL,
    new_stop_loss REAL NOT NULL,
    move_time TIMESTAMP NOT NULL,
    current_price REAL NOT NULL,  -- 触发时的价格
    profit_protected REAL,  -- 本次移动保护的利润
    reason TEXT,  -- 移动原因
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (position_id) REFERENCES positions(position_id)
);

-- 5. 性能指标表 (performance_metrics)
CREATE TABLE IF NOT EXISTS performance_metrics (
    metric_id TEXT PRIMARY KEY,
    date DATE NOT NULL,
    symbol TEXT,
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    win_rate REAL,
    total_pnl REAL DEFAULT 0,
    total_profit REAL DEFAULT 0,
    total_loss REAL DEFAULT 0,
    profit_factor REAL,  -- 盈亏比
    max_drawdown REAL,
    max_drawdown_period TEXT,
    sharpe_ratio REAL,
    average_win REAL,
    average_loss REAL,
    largest_win REAL,
    largest_loss REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 6. Safety state table (global runtime safety status)
CREATE TABLE IF NOT EXISTS safety_state (
    state_id TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引以提高查询性能
CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
CREATE INDEX IF NOT EXISTS idx_positions_entry_time ON positions(entry_time);
CREATE INDEX IF NOT EXISTS idx_position_operations_position_id ON position_operations(position_id);
CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_position_id ON orders(position_id);
CREATE INDEX IF NOT EXISTS idx_orders_created_at ON orders(created_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_orders_client_order_id ON orders(client_order_id);
CREATE INDEX IF NOT EXISTS idx_stop_loss_trailing_position_id ON stop_loss_trailing(position_id);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_date ON performance_metrics(date);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_symbol ON performance_metrics(symbol);
