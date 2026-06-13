# C 系统多腿 Timeline 回测 — 设计文档

**日期：** 2026-06-13
**参考：** B 系统 `event_backtest` (`docs/architecture/backtest_vs_live_execution.md`)

## 1. B 系统架构回顾

```
event_backtest
├── backtester.py          — 主循环：逐 bar 驱动 PositionSimulator
├── PositionSimulator      — 复用 live position_logic（enforce/add_position）
├── OMBridge               — MockBinanceAPI + OrderManager + PositionManager
├── AccountLedger          — 追踪权益曲线
├── Constitution           — add_position_rules, kill_switch
└── 1min bar               — 每根 1min bar 检查 SL/TP
```

**关键设计**：回测与实盘共用同一份 rules（`position_logic.py`），通过 `MockBinanceAPI` 替换交易所 API。Sizing 随权益复利（`compound_sizing`）。

## 2. C 系统当前 vs 目标

| 能力 | 当前 (per-symbol) | 目标 (timeline) |
|------|-------------------|-----------------|
| 多 symbol 共享账户 | ❌ 独立 | ✅ 同一 equity pool |
| 1min bar 执行 | ⚠️ 简化网格模型 | ✅ 同 live adapter |
| Order 生命周期 | ❌ 无 | ✅ MockBinanceAPI |
| Constitution | ❌ 无 | ✅ Governor + kill_switch |
| PnL 追踪 | per-symbol 独立 | ✅ 共享账户 |
| 引擎代码 | ⚠️ 简化引擎 | ✅ 同 live engine |

## 3. 架构设计

```
backtest_multileg_timeline.py
├── 1min bar 时间线        — 多 symbol 交织，按时间排序
├── Feature provider       — 预计算 2h 特征，1min bar 触发 on_bar()
├── Engines                — ChopGridLiveEngine + DualAddTrendLiveEngine（复刻 live）
├── MultiLegExecutionAdapter  — shadow=False, MockBinanceAPI
├── MultiLegPortfolioRiskGovernor — 宪法 gross/net/dd 上限
├── MultiLegConcurrencyGate     — chop↔trend cooldown, symbol cap
└── Account tracker        — equity, peak, daily_pnl, kill_switch
```

### 3.1 与 B 系统的对应

| B 系统 | C 系统 | 复用状态 |
|--------|--------|----------|
| `PositionSimulator` | `ChopGridLiveEngine.on_bar()` + `DualAddTrendLiveEngine.on_bar()` | ✅ 同 live engine |
| `OMBridge` | `MultiLegExecutionAdapter` + `MockBinanceAPI` | ✅ 同 live adapter |
| `AccountLedger` | `Account` dataclass | ✅ 新写 |
| `Constitution` | `MultiLegPortfolioRiskGovernor` + kill_switch | ✅ 已有 |
| `compound_sizing` | 每 symbol 独立 equity | ✅ 与 per-symbol 一致 |

### 3.2 关键时序：两阶段 action 处理

**问题**：同一 bar 内 engine 同时生成 `place` + `cancel`，`cancel` 需要 `place` 的 exchange_order_id。

**Live flow**：
```
Bar1: on_bar() → [place] → adapter → exchange_order_id=mock_123
       → on_execution_results() → order.exchange_order_id=mock_123
Bar2: on_bar() → [cancel(exchange_order_id=mock_123)] → adapter → ok
```

**修复：两阶段**：
```
Phase 1: adapter.execute_actions([place, cancel])  → 得到 exchange_order_id
Phase 2: engine.on_execution_results(results)       → 更新 order 的 exchange_id
Phase 3: adapter.execute_actions([market_exit])     → 用 exchange_id 平仓
Phase 4: engine.on_execution_results(results)       → 更新状态
```

### 3.3 PnL 追踪

从 `engine.state.realized_pnl` 逐 bar diff → `Account.cum_pnl`

## 4. MockBinanceAPI 补全

| 方法 | 用途 |
|------|------|
| `get_symbol_info(symbol)` | 返回 minQty/minNotional，避免 adapter 拒绝订单 |
| `cancel_algo_order` | mock 撤单 |
| `get_open_orders_for_sl_cleanup` | 返回 [] |

## 5. 实现状态

| 步骤 | 状态 |
|------|------|
| 1min bar 加载 + 2h 特征 | ✅ |
| Engines 初始化 | ✅ |
| MockBinanceAPI 补全 | ✅ |
| Adapter + Governor 集成 | ✅ |
| 两阶段 action 处理 | ✅ |
| PnL tracking (realized_pnl diff) | ✅ |
| Constitution gates | ✅ |
| 验证 | ⏳ |

## 6. 验证标准

- [ ] 1 周 dry-run：无 "local-only" cancel 警告
- [ ] 1 周 live-run：PnL ≠ 0，engine state 正确更新
- [ ] 6 月 backtest：与 per-symbol ±20% 一致
- [ ] Kill switch：在 20% 峰值 DD 触发
