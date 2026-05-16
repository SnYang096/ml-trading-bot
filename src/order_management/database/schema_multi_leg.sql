-- Multi-leg runtime schema (SQLite) — hedge / chop_grid / dual_add_trend storages only.
-- Do not apply this file to classical trend PCM order_management.db.

CREATE TABLE IF NOT EXISTS multi_leg_runs (
    run_id TEXT PRIMARY KEY,
    mode TEXT NOT NULL,
    strategies TEXT,
    symbols TEXT,
    account_label TEXT,
    config_json TEXT,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP,
    status TEXT DEFAULT 'running',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS multi_leg_orders (
    local_order_id TEXT PRIMARY KEY,
    run_id TEXT,
    strategy TEXT NOT NULL,
    symbol TEXT NOT NULL,
    leg_id TEXT,
    side TEXT NOT NULL,
    position_side TEXT,
    order_type TEXT NOT NULL,
    purpose TEXT,
    quantity REAL NOT NULL,
    price REAL,
    stop_price REAL,
    client_order_id TEXT,
    exchange_order_id TEXT,
    status TEXT NOT NULL,
    filled_quantity REAL DEFAULT 0,
    average_price REAL,
    commission REAL DEFAULT 0,
    commission_asset TEXT,
    filled_at TIMESTAMP,
    canceled_at TIMESTAMP,
    error_message TEXT,
    raw_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES multi_leg_runs(run_id)
);

CREATE TABLE IF NOT EXISTS multi_leg_positions (
    leg_id TEXT PRIMARY KEY,
    run_id TEXT,
    strategy TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    entry_price REAL NOT NULL,
    quantity REAL NOT NULL,
    status TEXT NOT NULL,
    parent_leg_id TEXT,
    protection_order_ids TEXT,
    raw_json TEXT,
    opened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    closed_at TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES multi_leg_runs(run_id)
);

CREATE TABLE IF NOT EXISTS multi_leg_execution_reports (
    event_id TEXT PRIMARY KEY,
    run_id TEXT,
    strategy TEXT,
    symbol TEXT,
    order_id TEXT,
    client_order_id TEXT,
    status TEXT,
    execution_type TEXT,
    raw_json TEXT NOT NULL,
    event_time TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES multi_leg_runs(run_id)
);

CREATE TABLE IF NOT EXISTS multi_leg_reconciliation_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    run_id TEXT,
    strategy TEXT,
    symbol TEXT,
    ok INTEGER NOT NULL,
    raw_json TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES multi_leg_runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_multi_leg_orders_run ON multi_leg_orders(run_id);
CREATE INDEX IF NOT EXISTS idx_multi_leg_orders_strategy_symbol ON multi_leg_orders(strategy, symbol);
CREATE UNIQUE INDEX IF NOT EXISTS idx_multi_leg_orders_client_order_id ON multi_leg_orders(client_order_id);
CREATE INDEX IF NOT EXISTS idx_multi_leg_positions_run ON multi_leg_positions(run_id);
CREATE INDEX IF NOT EXISTS idx_multi_leg_positions_strategy_symbol ON multi_leg_positions(strategy, symbol);
CREATE INDEX IF NOT EXISTS idx_multi_leg_execution_reports_run ON multi_leg_execution_reports(run_id);
CREATE INDEX IF NOT EXISTS idx_multi_leg_reconciliation_run ON multi_leg_reconciliation_snapshots(run_id);
