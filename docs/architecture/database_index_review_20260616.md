# 全部数据库设计与索引审查（2026-06-16）

> **范围**：项目全部 9 个数据库 + 引擎状态 JSON 文件
> **代码**：schema SQL 文件 + Python inline DDL

---

## 0. 数据库全景

| #   | 数据库                          | 默认路径                                              | Schema 来源                              | 用途                         |
| --- | ------------------------------- | ----------------------------------------------------- | ---------------------------------------- | ---------------------------- |
| 1   | `order_management.db`           | `data/order_management.db`                            | `schema_trend.sql` + `storage.py` inline | 经典 trend/PCM 订单/仓位     |
| 2   | `multi_leg_order_management.db` | `data/multi_leg_order_management.db`                  | `schema_multi_leg.sql`                   | 多腿 (chop_grid/trend_scalp) |
| 3   | `live_monitor.db`               | `live/highcap/data/db/live_monitor.db`                | `stats_collector.py` inline              | 15min 信号漏斗               |
| 4   | `account_equity.db`             | `live/highcap/data/db/account_equity.db`              | `account_equity_snapshots.py` inline     | 每日权益快照                 |
| 5   | `spot_order_management.db`      | `live/highcap/data/spot_order_management.db`          | `spot_order_manager.py` inline           | 现货订单追踪                 |
| 6   | `spot_accum_live.db`            | `data/spot_accum_live.db`                             | `run_spot_accum_live.py` inline          | 现货定投运行时状态           |
| 7   | `spot_accum_ledger.db`          | `live/highcap/data/spot_accum_ledger.db`              | `run_spot_accum_live.py` (state_kv)      | 现货持仓账本                 |
| 8   | `rd_registry.sqlite`            | `results/rd_registry.sqlite`                          | `store.py` inline                        | 监控事件注册                 |
| 9   | `kill_switch_state.json`        | `data/multi_leg_live/state/kill_switch_state.json`    | JSON 文件                                | C 层熔断状态                 |
| -   | `trend_scalp_{SYM}.json`        | `data/multi_leg_live/state/trend_scalp_{SYMBOL}.json` | JSON 文件                                | trend_scalp 引擎状态         |
| -   | `chop_grid_{SYM}.json`          | `data/multi_leg_live/state/chop_grid_{SYMBOL}.json`   | JSON 文件                                | chop_grid 引擎状态           |

---

## 1. `order_management.db` — 经典 trend/PCM

**Schema**: `src/order_management/database/schema_trend.sql` + `src/order_management/storage.py` (inline safety/slots/add_position)

### 表清单

| 表名                  | 行数趋势     | 清理 |
| --------------------- | ------------ | ---- |
| `positions`           | 持续增长     | 无   |
| `position_operations` | 持续增长     | 无   |
| `orders`              | 持续增长     | 无   |
| `stop_loss_trailing`  | 低速增长     | 无   |
| `performance_metrics` | 每日增长     | 无   |
| `safety_state`        | 1 行         | —    |
| `slots_state`         | 与持仓数相当 | 无   |
| `add_position_state`  | 与持仓数相当 | 无   |

### Schema

```sql
CREATE TABLE IF NOT EXISTS positions (
    position_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
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

CREATE TABLE IF NOT EXISTS slots_state (
    position_id TEXT PRIMARY KEY,
    symbol TEXT,
    archetype TEXT,
    opened_at TIMESTAMP,
    closed_at TIMESTAMP,
    close_reason TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS add_position_state (
    position_id TEXT PRIMARY KEY,
    add_count INTEGER DEFAULT 0,
    locked_profit INTEGER DEFAULT 0,
    current_r REAL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 现有索引

```sql
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
```

### 缺失索引

```sql
-- orders.binance_order_id (exchange API 查单)
CREATE INDEX IF NOT EXISTS idx_orders_binance_order_id ON orders(binance_order_id);
-- positions (symbol, status) 覆盖索引
CREATE INDEX IF NOT EXISTS idx_positions_symbol_status ON positions(symbol, status);
-- orders (position_id, status) JOIN+过滤
CREATE INDEX IF NOT EXISTS idx_orders_position_id_status ON orders(position_id, status);
```

---

## 2. `multi_leg_order_management.db` — 多腿运行时

**Schema**: `src/order_management/database/schema_multi_leg.sql`

### 表清单

| 表名                                 | 行数趋势         | 清理 |
| ------------------------------------ | ---------------- | ---- |
| `multi_leg_runs`                     | 很低             | 无   |
| `multi_leg_orders`                   | 持续增长         | ⚠️ 无 |
| `multi_leg_positions`                | 持续增长（不删） | ⚠️ 无 |
| `multi_leg_execution_reports`        | 持续增长         | ⚠️ 无 |
| `multi_leg_reconciliation_snapshots` | 持续增长         | ⚠️ 无 |

### Schema

```sql
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
```

### 现有索引

```sql
-- multi_leg_orders
CREATE INDEX IF NOT EXISTS idx_multi_leg_orders_run ON multi_leg_orders(run_id);
CREATE INDEX IF NOT EXISTS idx_multi_leg_orders_strategy_symbol ON multi_leg_orders(strategy, symbol);
CREATE UNIQUE INDEX IF NOT EXISTS idx_multi_leg_orders_client_order_id ON multi_leg_orders(client_order_id);

-- multi_leg_positions
CREATE INDEX IF NOT EXISTS idx_multi_leg_positions_run ON multi_leg_positions(run_id);
CREATE INDEX IF NOT EXISTS idx_multi_leg_positions_strategy_symbol ON multi_leg_positions(strategy, symbol);

-- multi_leg_execution_reports
CREATE INDEX IF NOT EXISTS idx_multi_leg_execution_reports_run ON multi_leg_execution_reports(run_id);

-- multi_leg_reconciliation_snapshots
CREATE INDEX IF NOT EXISTS idx_multi_leg_reconciliation_run ON multi_leg_reconciliation_snapshots(run_id);
```

### 🔴 缺失索引（按优先级）

```sql
-- P0: 每次 CMS/Reconcile 全表扫描 — 最关键！
CREATE INDEX IF NOT EXISTS idx_multi_leg_positions_status ON multi_leg_positions(status);

-- P1: phantom cleanup 级联更新
CREATE INDEX IF NOT EXISTS idx_multi_leg_orders_leg_id ON multi_leg_orders(leg_id);

-- P1: reconcile 每 60s 查询 open exchange orders
CREATE INDEX IF NOT EXISTS idx_multi_leg_orders_exchange_status ON multi_leg_orders(exchange_order_id, status);

-- P1: 诊断/Console 查询
CREATE INDEX IF NOT EXISTS idx_multi_leg_execution_reports_symbol_time ON multi_leg_execution_reports(symbol, event_time);

-- P2: 时间范围查询
CREATE INDEX IF NOT EXISTS idx_multi_leg_orders_created_at ON multi_leg_orders(created_at);
```

---

## 3. `live_monitor.db` — 15min 信号漏斗

**Schema**: `src/time_series_model/live/stats_collector.py` inline

```sql
CREATE TABLE IF NOT EXISTS stats_15min (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp             TEXT    NOT NULL,
    symbol                TEXT    NOT NULL DEFAULT '',
    window                TEXT    NOT NULL DEFAULT '15min',
    bars_processed        INTEGER DEFAULT 0,
    direction_assigned    INTEGER DEFAULT 0,
    gate_passed           INTEGER DEFAULT 0,
    entry_filter_passed   INTEGER DEFAULT 0,
    evidence_passed       INTEGER DEFAULT 0,
    pcm_selected          INTEGER DEFAULT 0,
    orders_placed         INTEGER DEFAULT 0,
    by_strategy           TEXT    DEFAULT '{}',
    positions             TEXT    DEFAULT '{}',
    system_health         TEXT    DEFAULT '{}',
    regime                TEXT    DEFAULT 'NORMAL'
);

CREATE INDEX IF NOT EXISTS idx_stats_15min_ts ON stats_15min(timestamp);
```

✅ 索引充足。有定期清理 (`DELETE WHERE timestamp < ?`)。

---

## 4. `account_equity.db` — 每日权益快照

**Schema**: `src/mlbot_console/services/account_equity_snapshots.py` inline

```sql
CREATE TABLE IF NOT EXISTS account_equity_daily (
    snapshot_date TEXT NOT NULL,
    scope TEXT NOT NULL,
    wallet_balance_usdt REAL,
    equity_usdt REAL,
    unrealized_pnl_usdt REAL,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (snapshot_date, scope)
);

CREATE INDEX IF NOT EXISTS idx_account_equity_daily_date ON account_equity_daily(snapshot_date);
```

✅ 索引充足。复合主键天然索引，每日 N 行。

---

## 5. `spot_order_management.db` — 现货订单

**Schema**: `src/order_management/spot_order_manager.py` inline

```sql
CREATE TABLE IF NOT EXISTS spot_orders (
    order_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    order_type TEXT NOT NULL,
    quantity REAL NOT NULL,
    price REAL,
    status TEXT NOT NULL,
    exchange_order_id TEXT,
    client_order_id TEXT,
    raw_json TEXT,
    filled_quantity REAL DEFAULT 0,
    filled_quote_usdt REAL DEFAULT 0,
    updated_at TEXT
);
```

### 缺失索引

```sql
-- P0: 按币种+状态查询 (CMS 页面)
CREATE INDEX IF NOT EXISTS idx_spot_orders_symbol_status ON spot_orders(symbol, status);
-- P1: 按交易所 ID 查单
CREATE INDEX IF NOT EXISTS idx_spot_orders_exchange_order_id ON spot_orders(exchange_order_id);
-- P1: 按 client_order_id 查单 (uuid 生成，逻辑唯一)
CREATE UNIQUE INDEX IF NOT EXISTS idx_spot_orders_client_order_id ON spot_orders(client_order_id);
-- P2: 低速增长表，时间排序优先级低
CREATE INDEX IF NOT EXISTS idx_spot_orders_created_at ON spot_orders(created_at);
```

---

## 6. `spot_accum_live.db` — 现货定投运行时

**Schema**: `scripts/run_spot_accum_live.py` inline

```sql
CREATE TABLE IF NOT EXISTS state_kv (
    k TEXT PRIMARY KEY,
    v TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS daily_counters (
    day_key TEXT NOT NULL,
    symbol TEXT NOT NULL,
    buy_entries INTEGER NOT NULL DEFAULT 0,
    deploy_usdt REAL NOT NULL DEFAULT 0.0,
    PRIMARY KEY (day_key, symbol)
);
```

✅ KV 表 + 复合主键即索引。无需额外索引。

---

## 7. `spot_accum_ledger.db` — 现货持仓账本

```sql
CREATE TABLE IF NOT EXISTS state_kv (
    k TEXT PRIMARY KEY,
    v TEXT NOT NULL
);
```

✅ 同 #6。KV 表，`k='positions'` 存 JSON。无需索引。

---

## 8. `rd_registry.sqlite` — 监控事件注册

**Schema**: `src/monitoring/store.py` inline

```sql
CREATE TABLE IF NOT EXISTS monitor_event (
    id            TEXT PRIMARY KEY,
    cadence       TEXT,
    source        TEXT,
    strategy      TEXT,
    status        TEXT,
    detail_json   TEXT,
    report_path   TEXT,
    run_ts        TEXT,
    output_dir    TEXT,
    ts            TEXT
);
```

### 缺失索引

```sql
-- 按时间范围查询 (dashboard 加载最近 N 条)
CREATE INDEX IF NOT EXISTS idx_monitor_event_ts ON monitor_event(ts);
-- 按 cadence source + 时间查最新事件 (upsert_monitor_event 查询模式)
CREATE INDEX IF NOT EXISTS idx_monitor_event_source_ts ON monitor_event(source, ts);
-- 按策略+状态过滤
CREATE INDEX IF NOT EXISTS idx_monitor_event_strategy_status ON monitor_event(strategy, status);
```

---

## 9. JSON 状态文件

### 9.1 kill_switch_state.json

> 来源：`MultiLegKillSwitchTracker.save()` → `SafetyRuntimeState.as_dict()`

```jsonc
{
  "current_day": "2026-06-16",
  "current_month": "2026-06",
  "current_week": "2026-W25",
  "day_start_equity": 50800.0,
  "last_equity": 51200.0,
  "month_start_equity": 49800.0,
  "peak_equity": 52100.0,
  "safety": {
    "cooldown_until": null,
    "halt_reason": [],
    "halt_since": null,
    "halted": false,
    "last_daily_halt_date": null,
    "last_metrics": {},
    "last_reset_date": null
  },
  "week_start_equity": 50100.0
}
```

### 9.2 trend_scalp_{SYMBOL}.json（开仓状态）

> 来源：`DualAddTrendState` dataclass，`json.dumps(asdict(state))` 序列化

```jsonc
{
  "active": true,
  "add_long_count": 2,
  "add_short_count": 1,
  "atr": 420.5,
  "bar_index": 5,
  "block_reseed_after_flip": false,
  "center": 68350.0,
  "inventory": [
    {
      "entry_price": 68350.0,
      "entry_time": "2026-06-16T08:00:00Z",
      "exit_limit_ex_id": "",
      "exit_repeg_bar": 0,
      "exit_reason": "",
      "exiting": false,
      "leg_id": "BTCUSDT_..._fill0",
      "protection_order_ids": ["1234567890", "1234567891"],
      "quantity": 0.1314,
      "seq": 0,
      "side": "LONG",
      "symbol": "BTCUSDT"
    }
  ],
  "last_add_long": 68350.0,
  "last_add_short": 68350.0,
  "last_entry_signal_ts": "2026-06-16T08:00:00Z",
  "last_reconciliation_issues": [],
  "last_reconciliation_ok": true,
  "last_timestamp": "2026-06-16T10:00:00Z",
  "pending_orders": [
    {
      "client_order_id": "",
      "created_at": "",
      "created_bar": 3,
      "exchange_order_id": "",
      "filled_quantity": 0.0,
      "max_slippage_bps": 5.0,
      "order_id": "BTCUSDT_..._trend_add_BUY_1_3",
      "post_only": false,
      "price": 68500.0,
      "quantity": 0.0657,
      "reason": "trend_add",
      "reference_price": 68450.0,
      "seq": 1,
      "side": "BUY",
      "status": "pending",
      "symbol": "BTCUSDT"
    }
  ],
  "realized_pnl": 12.34,
  "segment_id": "BTCUSDT_2026-06-16T08:00:00Z",
  "segment_state": "active",
  "symbol": "BTCUSDT",
  "trend_side": "LONG"
}
```

### 9.3 chop_grid_{SYMBOL}.json

> 来源：`GridState` dataclass，`json.dumps(asdict(state))` 序列化

```jsonc
{
  "active": true,
  "center": 68350.0,
  "current_regime": "idle",
  "grid_id": "BTCUSDT_2026-06-16T08:00:00Z",
  "inventory": [
    {
      "entry_price": 68100.0,
      "entry_quantity": 0.0587,
      "entry_time": "2026-06-16T08:00:00Z",
      "leg_id": "cg_BTCUSDT_..._BUY_l2_fill0",
      "level": 2,
      "protection_order_ids": ["9876543210"],
      "quantity": 0.0587,
      "side": "LONG",
      "symbol": "BTCUSDT"
    }
  ],
  "last_entry_signal_ts": "2026-06-16T08:00:00Z",
  "last_reconciliation_issues": [],
  "last_reconciliation_ok": true,
  "last_timestamp": "2026-06-16T10:00:00Z",
  "level_replenish_count": {},
  "pending_dust_exits": [],
  "pending_orders": [
    {
      "client_order_id": "",
      "created_at": "",
      "exchange_order_id": "",
      "filled_quantity": 0.0,
      "level": 1,
      "order_id": "cg_BTCUSDT_..._BUY_l1",
      "price": 67900.0,
      "quantity": 0.0587,
      "side": "BUY",
      "status": "pending",
      "symbol": "BTCUSDT"
    }
  ],
  "realized_pnl": 0.0,
  "segment_state": "active",
  "spacing": 210.5,
  "symbol": "BTCUSDT"
}
```

**与 trend_scalp 的关键区别**：
- 顶层用 `grid_id` 而非 `segment_id`；用 `spacing` 而非 `atr`
- 无 `trend_side` / `add_long_count` 等趋势字段
- 多了 `current_regime`、`level_replenish_count`、`pending_dust_exits`
- `GridPosition` 多了 `entry_quantity`（原始成交 qty，exchange sync 用）和 `entry_time`
- `GridOrder` 比 `DualAddOrder` 少了 `reason` / `seq` / `reference_price` / `max_slippage_bps` / `post_only`

---

## 10. 汇总：全部缺失索引 DDL

```sql
-- ===== order_management.db (classical trend) =====
CREATE INDEX IF NOT EXISTS idx_orders_binance_order_id ON orders(binance_order_id);
CREATE INDEX IF NOT EXISTS idx_positions_symbol_status ON positions(symbol, status);
CREATE INDEX IF NOT EXISTS idx_orders_position_id_status ON orders(position_id, status);

-- ===== multi_leg_order_management.db =====
-- P0: 最关键
CREATE INDEX IF NOT EXISTS idx_multi_leg_positions_status ON multi_leg_positions(status);
-- P1
CREATE INDEX IF NOT EXISTS idx_multi_leg_orders_leg_id ON multi_leg_orders(leg_id);
CREATE INDEX IF NOT EXISTS idx_multi_leg_orders_exchange_status ON multi_leg_orders(exchange_order_id, status);
CREATE INDEX IF NOT EXISTS idx_multi_leg_execution_reports_symbol_time ON multi_leg_execution_reports(symbol, event_time);
-- P2
CREATE INDEX IF NOT EXISTS idx_multi_leg_orders_created_at ON multi_leg_orders(created_at);

-- ===== spot_order_management.db =====
-- P0
CREATE INDEX IF NOT EXISTS idx_spot_orders_symbol_status ON spot_orders(symbol, status);
-- P1
CREATE INDEX IF NOT EXISTS idx_spot_orders_exchange_order_id ON spot_orders(exchange_order_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_spot_orders_client_order_id ON spot_orders(client_order_id);
-- P2 (低速增长表)
CREATE INDEX IF NOT EXISTS idx_spot_orders_created_at ON spot_orders(created_at);

-- ===== rd_registry.sqlite =====
CREATE INDEX IF NOT EXISTS idx_monitor_event_ts ON monitor_event(ts);
CREATE INDEX IF NOT EXISTS idx_monitor_event_source_ts ON monitor_event(source, ts);
CREATE INDEX IF NOT EXISTS idx_monitor_event_strategy_status ON monitor_event(strategy, status);
```

---

## 11. 数据生命周期总览

| 数据库                          | 写入频率                | 清理策略                       | 风险                 |
| ------------------------------- | ----------------------- | ------------------------------ | -------------------- |
| `order_management.db`           | 每次下单/成交           | **无**                         | ⚠️ 8 表持续增长       |
| `multi_leg_order_management.db` | 每次下单/成交/reconcile | **无**                         | ⚠️ 5 表持续增长       |
| `live_monitor.db`               | 每 15min                | ✅ `DELETE WHERE timestamp < ?` | 低                   |
| `account_equity.db`             | 每日 1 次               | **无**                         | 低 (每日 ~5 行)      |
| `spot_order_management.db`      | 每次现货下单            | **无**                         | ⚠️ 低速增长           |
| `spot_accum_live.db`            | 每次定投                | **无**                         | 低 (KV + 每日计数器) |
| `spot_accum_ledger.db`          | 每次定投完成            | **无**                         | 低 (KV 单行)         |
| `rd_registry.sqlite`            | 每次监控运行            | **无**                         | ⚠️ 持续增长           |

### 清理建议

| 数据库                               | 建议策略                                     | SQL 示例                                                                                                          |
| ------------------------------------ | -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `multi_leg_orders`                   | status='filled'/'cancelled' 超过 30 天可归档 | `DELETE FROM multi_leg_orders WHERE status IN ('filled','cancelled') AND created_at < datetime('now','-30 days')` |
| `multi_leg_execution_reports`        | 超过 30 天可清理                             | `DELETE FROM multi_leg_execution_reports WHERE created_at < datetime('now','-30 days')`                           |
| `multi_leg_reconciliation_snapshots` | 超过 7 天可清理（仅诊断用）                  | `DELETE FROM multi_leg_reconciliation_snapshots WHERE created_at < datetime('now','-7 days')`                     |
| `multi_leg_positions`                | status='closed' 超过 90 天可归档             | `DELETE FROM multi_leg_positions WHERE status='closed' AND closed_at < datetime('now','-90 days')`                |
| `rd_registry.sqlite`                 | 超过 30 天可清理                             | `DELETE FROM monitor_event WHERE ts < datetime('now','-30 days')`                                                 |
| `order_management.db`                | 与 multi_leg 类似，按业务需要归档            | —                                                                                                                 |

> ⚠️ 清理前务必备份数据库文件。建议在 orchestrator 启动时或每日 cron 中执行。

### JSON 状态文件并发安全

各 JSON 文件写入方式（经代码验证，2026-06-17）：

| 文件 | 写入方式 | 是否原子 |
|---|---|---|
| `kill_switch_state.json` (`multi_leg_kill_switch.py:155`) | `write_text()` 直接写入 | ❌ |
| `trend_scalp_{SYM}.json` (`position_tracker.py:1169`) | `tmp.write_text()` + `tmp.replace()` | ✅ |
| `chop_grid_{SYM}.json` (`chop_grid_concurrency.py:129`) | `write_text()` 直接写入 | ❌ |

`position_tracker.py` 已实现原子写入。其余两个文件在进程崩溃时可能截断/损坏。

**建议**：将 `multi_leg_kill_switch.py` 和 `chop_grid_concurrency.py` 改为 write-to-temp + `os.replace()` 原子重命名：
```python
import os
tmp = path.with_suffix('.tmp')
tmp.write_text(json.dumps(data), encoding='utf-8')
os.replace(tmp, path)  # POSIX 原子操作
```
