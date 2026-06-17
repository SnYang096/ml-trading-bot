-- Binance futures order management: trend / PCM tables only.
-- Multi-leg DDL lives in schema_multi_leg.sql.

-- 1. 仓位表 (positions)
CREATE TABLE IF NOT EXISTS positions (
    position_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,  -- 'long' or 'short'
    entry_time TIMESTAMP NOT NULL,
    exit_time TIMESTAMP,
    entry_price REAL NOT NULL,
    exit_price REAL,
    initial_size REAL NOT NULL,
    current_size REAL NOT NULL,
    total_cost REAL NOT NULL,
    total_value REAL,
    unrealized_pnl REAL,
    realized_pnl REAL DEFAULT 0,
    status TEXT NOT NULL,
    stop_loss_price REAL,
    take_profit_price REAL,
    trailing_stop_config TEXT,
    exit_reason TEXT,
    strategy_id TEXT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. 仓位操作记录表 (position_operations)
CREATE TABLE IF NOT EXISTS position_operations (
    operation_id TEXT PRIMARY KEY,
    position_id TEXT NOT NULL,
    operation_type TEXT NOT NULL,
    operation_time TIMESTAMP NOT NULL,
    size REAL NOT NULL,
    price REAL NOT NULL,
    pnl REAL,
    cumulative_pnl REAL,
    stop_loss_price REAL,
    take_profit_price REAL,
    reason TEXT,
    order_id TEXT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (position_id) REFERENCES positions(position_id)
);

-- 3. 订单表 (orders)
CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    binance_order_id TEXT,
    client_order_id TEXT,
    position_id TEXT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    order_type TEXT NOT NULL,
    quantity REAL NOT NULL,
    price REAL,
    stop_price REAL,
    status TEXT NOT NULL,
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
    current_price REAL NOT NULL,
    profit_protected REAL,
    reason TEXT,
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
    profit_factor REAL,
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
    halted INTEGER DEFAULT 0,
    halt_reason TEXT,
    halt_since TIMESTAMP,
    cooldown_until TIMESTAMP,
    last_metrics TEXT,
    last_reset_date DATE,
    last_daily_halt_date DATE,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 7. Slots runtime state (active slots only)
CREATE TABLE IF NOT EXISTS slots_state (
    position_id TEXT PRIMARY KEY,
    symbol TEXT,
    archetype TEXT,
    opened_at TIMESTAMP,
    closed_at TIMESTAMP,
    close_reason TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 8. Add-position runtime state (per position)
CREATE TABLE IF NOT EXISTS add_position_state (
    position_id TEXT PRIMARY KEY,
    add_count INTEGER DEFAULT 0,
    locked_profit INTEGER DEFAULT 0,
    current_r REAL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

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

-- Additional indexes (2026-06-17 review)
CREATE INDEX IF NOT EXISTS idx_orders_binance_order_id ON orders(binance_order_id);
CREATE INDEX IF NOT EXISTS idx_positions_symbol_status ON positions(symbol, status);
CREATE INDEX IF NOT EXISTS idx_orders_position_id_status ON orders(position_id, status);
