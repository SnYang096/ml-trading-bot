# Multi-leg User-stream 架构说明（现状 + 待改进）

**最后更新**: 2026-06-15  
**Canonical 入口**: [multi_leg_live_daemon.md](live_stream/multi_leg_live_daemon.md) · [HEDGE_RECONCILIATION_CN.md](../deployment/HEDGE_RECONCILIATION_CN.md) · [LIVE_CADENCE_AND_STORAGE_CN.md](../deployment/LIVE_CADENCE_AND_STORAGE_CN.md)

> v1.x 曾误写「Multi-leg 未接入 User-stream」。**v2.0 以代码为准**：接入已在 `scripts/run_multi_leg_live.py` 完成；本文描述现状、三层兜底、与 B·Trend 的差异，以及后续改进 backlog。

---

## 1. 背景

### 1.1 B·Trend — User-stream ✅

B·Trend 通过 `run_live.py` → `MultiSymbolManager` 集成 `BinanceUserStream`：

```text
run_live.py
  → MultiSymbolManager
    → BinanceUserStream (ORDER_TRADE_UPDATE + ACCOUNT_UPDATE)
      → OrderFlowListener.on_execution_report(report)
        → PositionTracker 状态更新 + SL 同步
```

TP/SL 不依赖引擎 action 队列：`PositionTracker.enforce_all()` 直接调 Binance API。

### 1.2 C·Multi-leg — User-stream ✅（runner 层）

Multi-leg **不在** `multi_leg_daemon.py` 里建 WebSocket；由入口脚本 `scripts/run_multi_leg_live.py` 在 `testnet` / `mainnet` 且 API 为真实 `BinanceAPI` 时启动 User-stream，路由到各 runtime 的 orchestrator。

```text
run_multi_leg_live.py (async_main)
  → BinanceUserStream(binance_api, on_execution_report)
       → MultiLegLiveOrchestrator.on_execution_report(report)
            → engine.on_execution_report → pop_pending_actions → adapter 挂 TP/SL
            → storage.record_execution_report (若启用 SQLite)
  → MultiLegLiveDaemon.run_forever()   ← bar 慢路径，与 WS 同进程、同 asyncio loop
  → periodic_multi_leg_order_backfill  ← REST 补洞（默认 60s）
```

**职责划分**

| 模块 | 职责 |
|------|------|
| `scripts/run_multi_leg_live.py` | 生命周期：User-stream、backfill、metrics、storage、hedge mode 探针 |
| `src/order_management/multi_leg_daemon.py` | 慢路径：bar poll → `sync_live_exchange_state` → `on_bar` → `run_actions` + reconcile |
| `src/order_management/multi_leg_orchestrator.py` | 快路径：`on_execution_report`；慢路径：`run_actions` / `reconcile` |

### 1.3 两进程关系

```text
┌─────────────────────────────────────────────┐
│  run_live.py (B·Trend)                      │
│  ✅ BinanceUserStream → OrderFlowListener    │
│  ✅ ACCOUNT_UPDATE → 持仓推送                │
│  ✅ PositionTracker 独立管理 SL              │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  run_multi_leg_live.py (C·Multi-leg)      │
│  ✅ BinanceUserStream → orchestrator         │
│  ✅ periodic_multi_leg_order_backfill        │
│  ✅ MultiLegLiveDaemon：bar + reconcile      │
│  ⚠️ 无 ACCOUNT_UPDATE 订阅（见 §5.2）       │
└─────────────────────────────────────────────┘
```

推荐 **独立子账户**（`MULTI_LEG_BINANCE_FUTURES_*`），与 B·Trend 的 `BINANCE_FUTURES_TESTNET_*` / `BINANCE_API_KEY` 隔离。详见 `run_multi_leg_live.py` 文件头注释。

---

## 2. 三层 fill / TP 链路

Fill 后挂 TP/SL 不是单一路径，而是 **主路径 + 两条兜底**：

```text
Order fill (Binance)
  │
  ├─ [主] User-stream (<1s)
  │     run_multi_leg_live.on_execution_report
  │     → orchestrator.on_execution_report → adapter 挂 TP/SL
  │
  ├─ [中] 新 bar（feature-store 下 poll 默认 5s；--poll-seconds 默认 60s）
  │     multi_leg_daemon.run_once
  │     → sync_live_exchange_state（REST 快照对齐持仓，不走 on_execution_report）
  │     → on_bar + pop_pending_actions（Path ④）
  │     → orchestrator.run_actions + reconcile
  │
  └─ [慢] periodic_multi_leg_order_backfill（默认 60s，MLBOT_MULTI_LEG_ORDER_BACKFILL_*）
        → on_new_fill → orchestrator.on_execution_report（WS 漏报补洞）
        → multi_leg_orders 表 REST 对齐
```

| 路径 | 典型延迟 | 是否调用 `on_execution_report` | 主要作用 |
|------|---------|-------------------------------|----------|
| User-stream | <1s | ✅ | Fill 后立即生成并提交 TP/SL |
| Bar + Path ④ | 下一根 bar | 间接（`on_bar` 合并 `_pending_actions`） | 引擎逻辑 + 未 drain 的 pending |
| Backfill | 默认 60s | ✅ | WS 断线/漏事件补洞 |
| Reconcile | 默认 60s | ❌（`actions_ensure_protection` 等） | 保护单缺失、孤儿单 |

**WebSocket 是主路径；bar sync、backfill、reconcile 互补，不互斥。**

### 2.1 模式矩阵

| `--mode` | User-stream | 说明 |
|----------|-------------|------|
| `shadow` | ❌ | `MockBinanceAPI`，无 listenKey |
| `testnet` | ✅ | `isinstance(api, BinanceAPI)` 时启动 |
| `mainnet` | ✅ | 同上 |

### 2.2 `orchestrator.on_execution_report` 行为（已实现）

代码：`src/order_management/multi_leg_orchestrator.py`（约 L381）。

1. `_enrich_execution_report`（DB purpose / order_type → `protection_type`）
2. `engine.on_execution_report(report)` → 更新库存、写入 `_pending_actions`
3. `pop_pending_actions()` → `adapter.execute_actions()` → `on_execution_results`
4. `_persist_positions()` + `_persist_execution_report`

单测：`tests/order_management/test_multi_leg_orchestrator.py`（`test_user_stream_execution_report_is_forwarded_to_engine` 等）。

---

## 3. 设计原则（仍适用）

| 原则 | 说明 |
|------|------|
| **向后兼容** | 保留 bar sync + backfill + reconcile 兜底 |
| **单一职责** | User-stream 只做事件分发与持久化触发，业务在 engine/orchestrator |
| **解耦** | `BinanceUserStream` 与策略无关，通过 callback 路由 |
| **容错** | WS 断线自动重连；backfill/reconcile 补洞 |
| **可观测** | 已有 `mlbot_multi_leg_user_stream_events_total`；连接态/latency 待补（§6） |

### 3.1 为什么不用 SQLite 事件总线 / IPC

1. `BinanceUserStream` 已在 B·Trend 与 Multi-leg 生产路径验证
2. `run_multi_leg_live.py` 单进程管理所有 symbol/runtime — 无需跨进程总线
3. `orchestrator.on_execution_report` 已实现 — runner 层路由即可
4. 相对「新表 + migration + 消费进程」复杂度低一个数量级

---

## 4. 生产代码参考

实际接入（非伪代码）：

```python
# scripts/run_multi_leg_live.py — async_main
user_stream: BinanceUserStream | None = None
if isinstance(exchange_api, BinanceAPI):

    def on_execution_report(exec_report: Dict[str, Any]) -> None:
        sym = str(exec_report.get("symbol") or "").upper().strip()
        if not sym:
            return
        for rt in daemon.runtimes:
            if rt.symbol.upper() == sym:
                rt.orchestrator.on_execution_report(exec_report)
                # storage.record_execution_report + metrics ...

    user_stream = BinanceUserStream(exchange_api, on_execution_report)
    await user_stream.start()
```

`BinanceUserStream` 构造函数签名：`BinanceUserStream(binance_api, on_execution_report, on_account_update=None, keepalive_interval=1800)`。listenKey keepalive 默认 30 分钟。

Backfill 桥接（WS 漏报）：

```python
# 同一文件 — periodic_multi_leg_order_backfill(..., on_new_fill=_route_backfill_fill)
# _route_backfill_fill 构造归一化 report → rt.orchestrator.on_execution_report
```

---

## 5. 已知缺口与风险

### 5.1 并发（⚠️ 待评估）

User-stream 回调与 `MultiLegLiveDaemon.run_once()` 共享 **同一 asyncio 事件循环**，可交错修改 engine 状态。Multi-leg engine **没有** `OrderManager` 那样的 per-process `Lock`。

**现有缓解**

- Engine：`max(filled_qty)`、FILLED 状态判断、segment/inventory 持久化
- Orchestrator：异常不向外抛出阻塞 WS 循环（`BinanceUserStream._handle_message` 内 catch）

**待决**

- 是否 per-runtime 串行化（asyncio.Lock 或把 WS 事件 queue 到 daemon tick）
- 压测：fill 与 bar 同 tick 是否重复 TP（依赖 adapter / exchange idempotency）

### 5.2 Symbol 路由

当前实现对 **所有** `rt.symbol == sym` 的 runtime 调用 `on_execution_report`（无 `break`）。Daemon 层有 `symbol_owner` 限制**新开仓**冲突，但 WS 事件仍可能进入多个 engine。

**运行契约**：同一 symbol 只应有一个 active 多腿策略；多策略同 symbol 需人工收紧 governor + 文档化 owner。

### 5.3 与 B·Trend 的差异

| 能力 | B·Trend | Multi-leg |
|------|---------|-----------|
| `ORDER_TRADE_UPDATE` | ✅ | ✅ |
| `ACCOUNT_UPDATE` | ✅ → OrderFlowListener | ❌ 未订阅 |
| 持仓真相 | User-stream + TruthSync | reconcile + `sync_live_exchange_state` + engine JSON |

Multi-leg 未订阅 `ACCOUNT_UPDATE` 是**当前实现选择**（对账以 REST + reconcile 为主）。若要做「推送级持仓真相」，需在 runner 层加 callback 并评估与 engine 库存模型的一致性。

### 5.4 幂等 / 重复 TP

同一次 fill 可能经 **User-stream + backfill** 各触发一次 `on_execution_report`。依赖：

- Engine 侧订单/leg 查找与 filled_qty 单调更新
- Protection 动作是否带稳定 `client_order_id`（`derive_multileg_client_order_id`）
- Reconcile `actions_ensure_protection` 最后一道补挂

验收时需专门测 **WS 恢复后 backfill 不重复挂 TP**。

---

## 6. 监控

### 6.1 已有

| Metric | 说明 |
|--------|------|
| `mlbot_multi_leg_user_stream_events_total{strategy,symbol}` | 路由到 orchestrator 的 execution report 计数 |
| `mlbot_multi_leg_daemon_polls_total` | Daemon poll 次数 |
| `mlbot_reconciliation_*{scope="hedge"}` | 对账 OK / issue（见 HEDGE_RECONCILIATION_CN） |

### 6.2 建议新增（backlog）

命名遵循 repo 惯例 `mlbot_*` 前缀：

| Metric | Type | 说明 |
|--------|------|------|
| `mlbot_multi_leg_user_stream_connected` | Gauge | Multi-leg User-stream 连接态（需在 `BinanceUserStream` 连接/重连时 set） |
| `mlbot_multi_leg_user_stream_reconnect_total` | Counter | 重连次数 |
| `mlbot_multi_leg_fill_to_tp_latency_ms` | Histogram | fill `trade_time` → protection submit |
| `mlbot_multi_leg_execution_report_source_total` | Counter | label: `user_stream` / `backfill` / `bar`（可选） |

**告警建议**：`mlbot_multi_leg_user_stream_connected == 0` 持续 2min → **warning**（非 critical；backfill + reconcile 仍可用）。

---

## 7. 验收清单（替代原 Phase 1「待集成」）

| 项 | 方法 |
|----|------|
| WS 主路径 | testnet 24h：`mlbot_multi_leg_user_stream_events_total` 与成交大致一致 |
| 断 WS | 断网 2min 后 backfill/reconcile 补 TP，无重复 protection |
| 重启 | inventory JSON + `multi_leg_orders` 恢复 leg ↔ protection 映射 |
| 单测回归 | `pytest tests/order_management/test_multi_leg_orchestrator.py -k execution_report` |
| Path ④ | 无 WS 的 shadow 模式仍可通过 `on_bar` + reconcile 跑通 |

---

## 8. 后续改进优先级

| 优先级 | 项 |
|--------|-----|
| P1 | `mlbot_multi_leg_user_stream_connected` + warning 告警 |
| P1 | testnet 断 WS + 重复 TP 集成场景 |
| P2 | `fill_to_tp_latency_ms` histogram |
| P2 | WS vs daemon 并发串行化方案（若压测发现问题） |
| P3 | 评估 Multi-leg 是否需要 `ACCOUNT_UPDATE`（与 Trend TruthSync 对齐程度） |

---

## 9. 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v1.0 | 2026-06-15 | 初始版（错误前提：Multi-leg 未接入） |
| v1.1 | 2026-06-15 | 方案改为直接集成（仍误写接入点在 daemon） |
| v2.0 | 2026-06-15 | **以代码为准**：接入在 `run_multi_leg_live.py`；三层链路、缺口、监控 backlog、验收清单 |
