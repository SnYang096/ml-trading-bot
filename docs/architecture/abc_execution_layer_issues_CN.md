# ABC 执行层：近期修复、结构性问题与建议

> 日期：2026-06-14  
> 范围：A/B/C 战略分层下的 **live 执行与对账**（非 R&D Phase 1 scan）  
> 状态：**Review 稿** — §5–§9 为详细实现计划，待确认后按 Phase 0 起执行  
> 相关：[segment-lifecycle.md](segment-lifecycle.md) · [ABC三层收益结构_战略框架_CN.md](../strategy/ABC三层收益结构_战略框架_CN.md) · [漂移监控_mlbot_monitor_CN.md](../strategy/漂移监控_mlbot_monitor_CN.md)

---

## 1. 近期重大修复（执行相关）

### 1.1 C 层 multileg（chop_grid / trend_scalp）— 最高优先级

| 问题 | 影响 | 修复 |
| ---- | ---- | ---- |
| **Ghost segment**：TP/SL 后 `active=True` 占并发 slot | 6 symbol 卡死、无法开新段 | P0 auto-deactivate → P1–P4 `SegmentLifecycleMixin`（`643deb9c`） |
| **Orphan exchange SL** | 交易所残单 vs 本地 state 脱节 | `per_leg_stop_loss: false`（`1d8d9e1c`） |
| **chop ↔ trend 互斥** | 同 symbol 双引擎抢 slot | symbol mutex + trend SL 清理（`3bf16d9a`） |
| **Stale pending** | Binance 已 cancel，SQLite 仍 `pending` | terminal backfill 调用 `reconcile_open_orders`（`fb4e6b85`） |
| **并发门控 ghost slot** | `active` 无仓仍占 cap | `holds_real_grid_slot` / gate cooldown（2026-06-12） |

### 1.2 CMS / 对账可见性

| 问题 | 修复 commit |
| ---- | ----------- |
| hedge 共享 SL 一对多 exit 配对 | `e52fcbbd` |
| dust `market_exit` link PnL 夸大 | `3ad16941` |
| trend `_fillN` / `skipped_no_position` 回合配对 | `8731c4be` / `041a590e` |
| SQLite INT vs TEXT → HYPE markers 缺失 | `cde59b8c` |
| 条件单 vs 限价 pending 混淆 | `fb4e6b85` + `displayOrderKind` |

### 1.3 共用基础设施

| 问题 | 修复 |
| ---- | ---- |
| feature-bus 缺 multileg alias/gate | `a93148be` / `35e2a237` |
| `trend_direction` 类型不一致 | `dd918924` |
| WS queue overflow | `d7dd239f` |
| 宪法 sizing vs trend gate | `2a0ce2f5` |

### 1.4 B 层（信号/研究，执行间接）

TPC direction band 对齐、fast_scalp dual-head、ME CompressionBreakout rework 等——影响 **是否触发**，不替代 C 的 multileg 段生命周期。

---

## 2. ABC 是「分账户」，为什么还要谈 ExecutionTruthSync？

### 2.1 两个不同维度

| 维度 | ABC 分层 | ExecutionTruthSync（本文用语） |
| ---- | -------- | ------------------------------ |
| **解决什么** | payoff 错配、风控预算、KPI 混账本 | **同一 runtime 内** 本地 state / DB / 交易所 三者 drift |
| **隔离单位** | 子账户 / 宪法 bucket / 策略 KPI | 单账户内的 engine JSON、SQLite、`openOrders` |
| **典型事故** | 用 B 的 holding 扛 A 的 beta | ghost segment、stale pending、orphan SL |
| **目标态** | A/B/C 物理子账户（见 A 层扩展规划） | **每个** 子账户仍需要 truth sync |

**结论**：ABC 分账户 **不能替代** truth sync。分账户后，C 账户里仍会有 engine state + orders 表 + Binance open orders；ghost/stale 仍可能发生，只是不会污染 A/B 的 PnL 归因。

### 2.2 当前 prod 与 ABC 目标态的差距

战略文档要求 A/B/C **账本隔离**；**今天** 更接近：

| 层 | 执行栈 | 账户（现状） |
| -- | ------ | ------------ |
| **C** | `multi_leg_daemon` + chop/trend live engine + hedge SQLite | **同一 hedge 账户**，chop 与 trend 共享并发 gate |
| **B** | PCM / event live / 单腿 trend | 常与 C 同所、同 bus，路径不同 |
| **A** | spot / rolling（研究+部分 live） | USDT-M；dapi 未做 |

因此近期 ghost/orphan/stale 事故都出在 **C 栈 + 共用 hedge 账户**，修复也集中在此——这与「ABC 应分账户」**不矛盾**：分账户是 **下一步**，truth sync 是 **每一步都要的内功**。

### 2.3 ExecutionTruthSync 指什么（不是合并 ABC，也不是新进程）

> **术语说明**：`ExecutionTruthSync` 是本文用的**工程概念名**，指「单账户 runtime 内的 truth sync helper / 模块契约」。**不是** systemd 第四进程，**不是**跨 A/B/C 的中央同步服务。

不是把 A/B/C 合成一个系统，而是 **在单账户、现有进程边界内**：

1. **统一 reconcile 调度**：谁、何时调用  
   - `reconcile_open_orders`（open ↔ local pending）  
   - `reconcile_recent_terminal_orders`（终态回填）  
   - multileg daemon `reconcile`（engine state ↔ exchange，60s 或 on action）  
   - `MonitoringService.reconcile_open_orders`（legacy 路径）

2. **统一 metrics / 告警字段**：同一 issue 类型不因入口不同而丢失（见 §3）。

3. **明确三源优先级**：exchange truth 用于 replenishment guard；段 slot 以本地 inventory/pending 为主（见 [segment-lifecycle.md §4.4](segment-lifecycle.md)）。

**落地形态（Phase 4）**：共享 Python 模块（候选路径 `src/order_management/execution_truth_sync.py`），由 `quant-hedge-multileg`、`quant-trend-fattail` 等**现有进程 import 调用**——不新增 daemon。

即使 chop 与 trend 永远分账户，**每个账户** 仍需要上述 1–3；ExecutionTruthSync 是 **单账户内部的工程债名称**，不是否定 ABC 分账。

---

## 3. 监控缺口（修正版）

### 3.1 `fb4e6b85` 与 metrics 的真实状态

`terminal_order_backfill.py` 在调用 `reconcile_open_orders` 后 **有** 调用：

```python
METRICS.update_reconciliation_metrics(
    issue_counts={
        "stale_local_order": stale_marked,
        "api_error": api_error,
        "open_reconcile_updated": len(open_updated),  # ← 传入
    },
)
```

但 `metrics_exporter.update_reconciliation_metrics` **只写入固定 bucket**：

```python
for issue in (
    "missing_exchange_order",
    "orphan_exchange_order",
    "stale_local_order",
    "position_mismatch",
    "api_error",
):
    ...
```

**`open_reconcile_updated` 不在 allowlist 内，会被静默丢弃。** 因此：

- ❌ 不能说「已暴露 Prometheus 指标」  
- ✅ 仅有 log + `issue_counts` 传参意图；**要告警需补 bucket 或 `record_strategy_event`**

### 3.2 其他未覆盖项

| 信号 | 现状 | 告警可用？ |
| ---- | ---- | ---------- |
| `open_reconcile_updated` | 传入但被丢弃 | ❌ |
| `ghost_cleared`（segment lifecycle） | `_deactivate` 仅 logger | ❌ |
| multileg daemon reconcile issues | `multi_leg_reconciliation_issues_total` | ⚠️ 部分 |
| slot 占用 / ghost 检测 | `update_slot_metrics`（PCM 路径） | ⚠️ C multileg 未统一 |

### 3.3 与 `mlbot monitor` manifest 的关系

[漂移监控_mlbot_monitor_CN.md](../strategy/漂移监控_mlbot_monitor_CN.md) 中 C 执行层仍以 **`multileg monitor` 月报** 为主，未纳入：

- reconcile issue 时间序列  
- ghost clear 计数  
- open reconcile 更新行数  

**待办（非阻塞，但要准确）**：

1. `update_reconciliation_metrics` 增加 bucket `open_reconcile_updated`（或改用 counter + `inc`）  
2. `SegmentLifecycleMixin._deactivate("ghost_cleared")` → `METRICS.record_strategy_event(...)`  
3. monitor manifest / Grafana 面板引用上述 series  
4. §6 segment-lifecycle live 验证项与 prod 观察挂钩  

---

## 4. 执行层结构性问题（按 ABC）

### 4.1 跨层（今天仍共用所级基础设施）

| 问题 | 说明 |
| ---- | ---- |
| 三源 drift | engine JSON ↔ SQLite ↔ exchange；分账户后每账户仍存在 |
| reconcile 多入口 | terminal backfill、daemon、MonitoringService、orchestrator on action — 行为/频率不一致 |
| 监控链断裂 | B Regime 有 parquet verb；C Regime 未接；C 执行无实时 reconcile 告警 |
| 宪法 bucket | ABC 应用 constitution 分 gross/net cap，而非仅 strategy slug |

### 4.2 C 层（优先）

| 优先级 | 项 |
| ------ | -- |
| P0 | Live 验证：TP fill 后 slot 在 `on_execution_report` 释放（segment-lifecycle §6） |
| P1 | 补 §3 metrics + manifest |
| P1 | `sync_live_exchange_state` chop/trend 抽共享 helper，防下一处 drift |
| P2 | timeline backtest 与 live 共用 segment lifecycle 语义 |
| P2 | PnL `math.isclose` + trade-level audit（multileg_sizing DECISION §6） |

### 4.3 B 层

- 不套用 C 的 segment lifecycle；需要 **position-level** lifecycle + **`ledger` realized-R verb**（监控文档 T5）  
- 审计 backtest tier / noise_penalty 在 live 是否生效（[backtest_vs_live_execution.md](backtest_vs_live_execution.md)）

### 4.4 A 层

- `abc_macro_regime_score` 与 C 的 2h router **特征可共用、决策链分离**  
- 执行重点是 **低 churn、慢出场**；dapi 暂缓（全栈重写）  
- 子账户隔离见 [A层多子账户扩展规划_CN.md](../strategy/A层多子账户扩展规划_CN.md)

---

## 5. Review 结论（2026-06-14）

近期重大修复集中在 **C 层 live 执行**：ghost segment、orphan 保护单、stale pending、symbol 冲突、CMS 对账可见性。代码侧 P0–P4（segment lifecycle）与 `fb4e6b85`（open reconcile 调用）已合并；**监控与文档仍落后**。

| 维度 | 状态 |
| ---- | ---- |
| 段生命周期 refactor | ✅ 已合并（`643deb9c` 等） |
| stale pending reconcile | ✅ 已调用 `reconcile_open_orders` |
| `open_reconcile_updated` Prometheus | ❌ 传参后被 allowlist 丢弃 |
| `ghost_cleared` 可观测 | ❌ 仅 logger |
| ExecutionTruthSync 表述 | ⚠️ 易误解为新进程 → 本文 §2.3 / §8 已澄清 |
| segment-lifecycle §4.3 代码片段 | ⚠️ `_maybe_deactivate_if_fully_closed` 缺 exchange guard → Phase 0 修 doc |

**第一实施目标**：C hedge runtime（`quant-hedge-multileg`）+ trend backfill（`quant-trend-fattail`）的 metrics 与 issue 命名统一；**不**在本迭代新增进程。

---

## 6. 目标架构（进程内 helper，非新服务）

```mermaid
flowchart LR
  subgraph cProc [quant-hedge-multileg]
    cEngine["ChopTrendEngines"] --> cState["EngineJSON"]
    cEngine --> cDb["SQLiteOrders"]
    cEngine --> cExchange["BinanceOpenOrders"]
    cSync["TruthSyncHelper"] --> cState
    cSync --> cDb
    cSync --> cExchange
    cSync --> cMetrics["Prometheus"]
  end

  subgraph bProc [quant-trend-fattail]
    bBackfill["TerminalOrderBackfill"] --> bMetrics["Prometheus"]
  end

  noteNode["ExecutionTruthSync=模块契约,非systemd服务"]
  noteNode --> cSync
```

| 进程 | 职责 | truth sync 相关入口 |
| ---- | ---- | ------------------- |
| `quant-hedge-multileg` | C 层 chop/trend 多腿 | daemon reconcile 60s / on action；`sync_live_exchange_state` |
| `quant-trend-fattail` | B 层 PCM/单腿 | `terminal_order_backfill` → `reconcile_open_orders` + 终态回填 |
| `quant-spot-accum` | A 层 spot | 独立 reconcile metrics（scope=spot） |

---

## 7. 详细实现计划

### Phase 0：文档对齐（本文件 + segment-lifecycle）

**目的**：避免后续代码 review 时再次误解「新进程 / 中央服务」。

| 任务 | 文件 | 内容 |
| ---- | ---- | ---- |
| 0.1 | 本文 §2.3 | 已写明：非 systemd 服务、非跨账户同步器 |
| 0.2 | [segment-lifecycle.md](segment-lifecycle.md) §4.3 | `_maybe_deactivate_if_fully_closed`  snippet 补上 `if self._exchange_has_open_activity(): return` |
| 0.3 | [segment-lifecycle.md](segment-lifecycle.md) §1.2 | `auto-deactivate` 行与 ghost 行统一：均含 exchange guard |
| 0.4 | 本文 §7 变更记录 | 实施完成后勾选各 Phase |

**验收**：文档自洽；无「纯本地空即 ghost」与 §4.4 实现矛盾的表述。

**工作量**：~0.5d

---

### Phase 1：Metrics 缺口修复（`open_reconcile_updated`）

**问题**：`terminal_order_backfill.py` 传入 `open_reconcile_updated`，但 `metrics_exporter.update_reconciliation_metrics` 固定 5 个 bucket，该值被静默丢弃。

**改动**：

| 文件 | 改动 |
| ---- | ---- |
| `src/time_series_model/live/metrics_exporter.py` | 增加常量 `RECONCILIATION_ISSUE_BUCKETS`；allowlist 加入 `open_reconcile_updated` |
| `src/live_data_stream/terminal_order_backfill.py` | 无需改逻辑；确认 `ok=` 语义：`open_reconcile_updated > 0` **不应**单独置 `reconciliation_ok=0`（这是修复动作，不是持续错误） |
| 新/扩测试 | 断言 `mlbot_reconciliation_issue_count{issue="open_reconcile_updated"}` 被写入 |

**预期 PromQL**：

```text
mlbot_reconciliation_issue_count{scope="trend", issue="open_reconcile_updated"}
```

**不做**：把 `ghost_cleared` 放进 `reconciliation_issue_count`——成功清 ghost 是**事件**，不是当前对账错误（见 Phase 2）。

**验收**：

- 单元测试：`update_reconciliation_metrics(issue_counts={"open_reconcile_updated": 3})` → gauge 为 3  
- prod scrape 后 Trend dashboard 可见该 series

**工作量**：~0.5d

---

### Phase 2：段生命周期事件 metrics（`ghost_cleared` 等）

**问题**：`SegmentLifecycleMixin._deactivate` 仅 logger；prod 无法回答「是否发生过 ghost 清理 / 段是否正常收工」。

**改动**：

| 文件 | 改动 |
| ---- | ---- |
| `src/time_series_model/live/segment_lifecycle.py` | 在 `_deactivate(reason)` 内 best-effort 调用 `METRICS.record_strategy_event` |
| 事件名 | `event=f"segment_{reason}"`，如 `segment_ghost_cleared`、`segment_fully_closed`、`segment_regime_exit` |
| labels | `scope="hedge"`, `strategy=self._engine_name`, `symbol=...`, `side="na"` |
| 约束 | `try/except` 包裹；**metrics 失败不得影响** `_deactivate` / gate 通知 |

**不做**：让 `ghost_cleared` 把 `reconciliation_ok` 打红——系统自愈后告警应绿。

**验收**：

- 单元测试：mock METRICS，触发 `_deactivate("ghost_cleared")` → `record_strategy_event` 被调用一次  
- Grafana：`increase(mlbot_strategy_event_total{event="segment_ghost_cleared"}[1h])`

**工作量**：~0.5d

---

### Phase 3：Grafana / 告警接线

**原则**：复用现有 `mlbot_reconciliation_*` 与 `mlbot_strategy_event_total`；**不**为 `open_reconcile_updated` 立即加 hard alert（先面板观察阈值）。

| 文件 | 改动 |
| ---- | ---- |
| `deploy/monitoring/grafana-provisioning/dashboards/quant_strategy_map_trend.json` | Trend 对账区增加 `open_reconcile_updated` 分 issue 展示 |
| `deploy/monitoring/grafana-provisioning/dashboards/quant_strategy_map_hedge.json` | 增加 `segment_ghost_cleared` / `segment_fully_closed` event rate 或累计 |
| `deploy/monitoring/grafana-provisioning/alerting/quant_ops.yaml` | **保持**现有 `QuantTrendReconciliationManualCheck` / `QuantHedgeReconciliationManualCheck`；Phase 3 不新增 alert，除非 prod 观察后定阈值 |
| `tests/deploy/test_monitoring_provisioning.py` | 断言新 PromQL 出现在 dashboard JSON；alert YAML 仍合法 |

**验收**：`pytest tests/deploy/test_monitoring_provisioning.py` 通过；CMS/Grafana 手动看一眼新 panel。

**工作量**：~1d

---

### Phase 4：进程内 Truth Sync Helper（代码模块，非新进程）

**前置**：Phase 1–3 稳定后再做，避免 helper 封装错误 metrics。

**候选模块**：`src/order_management/execution_truth_sync.py`

| 职责 | 非职责 |
| ---- | ------ |
| issue bucket 命名常量（与 `RECONCILIATION_ISSUE_BUCKETS` 对齐） | 下单 / 撤单 |
| reconcile 周期 bookkeeping（last run ts、source tag） | 持有 strategy state |
| 统一调用 `METRICS.update_reconciliation_metrics` / 摘要 log | 跨 A/B/C 账户 |
| | 替代 `MultiLegReconciler` |

**迁移顺序**：

1. `terminal_order_backfill.py` metrics 发布 → 走 helper  
2. `multi_leg_daemon.py` / orchestrator reconcile metrics → 走 helper  
3. legacy `order_management/monitoring.py` → 最后或标记 deprecated  

**验收**：各入口 issue 名称一致；无 duplicate/conflicting gauge 写入；**仍只有两个（或三个）现有 systemd 进程**。

**工作量**：~1–2d

---

### Phase 5：测试与 Live 验证

**自动化**（CI / 本地）：

```bash
pytest tests/unit/test_segment_lifecycle.py \
       tests/unit/test_dual_add_trend_live_engine.py \
       tests/order_management/test_order_manager.py::test_reconcile_open_orders_syncs_canceled_pending
# + 新增 metrics_exporter 测试
pytest tests/deploy/test_monitoring_provisioning.py
```

**Live / paper（segment-lifecycle §6 最后一项）**：

| 观察项 | 通过标准 |
| ------ | -------- |
| TP/SL fill 后 slot | `on_execution_report` 后 `holds_real_grid_slot()==False`，不必等下一根 bar |
| ghost 清理 | 若发生，`segment_ghost_cleared` event 有计数；并发 cap 释放 |
| stale pending | `open_reconcile_updated` 偶发 >0 可接受；持续 `stale_local_order` + `reconciliation_ok=0` 需人工 |

**工作量**：Live 观察 1–3 个交易日；自动化 ~0.5d

---

### Phase 6（中期，本文仅登记不展开）

| 项 | 说明 |
| -- | ---- |
| C Regime/Prefilter 接 parquet verb | 见 [漂移监控_mlbot_monitor_CN.md](../strategy/漂移监控_mlbot_monitor_CN.md) |
| B `ledger` realized-R verb | 执行层监控 T5 |
| `sync_live_exchange_state` 抽共享 helper | chop/trend 防 drift |
| timeline backtest 共用 segment lifecycle | 见 `c_timeline_backtest_design.md` |
| A/B/C 子账户 + constitution per-layer bucket | 战略；见 A 层扩展规划 |

---

## 8. 风险与护栏

1. **Metrics label 低基数**：禁止 order_id / client_id / 异常原文进 label。  
2. **自愈事件不当告警**：`open_reconcile_updated`、`segment_ghost_cleared` 先面板、后阈值 alert。  
3. **不新增 daemon**：本迭代禁止为 truth sync 单独起 systemd unit。  
4. **不合并 ABC 账户职责**：helper 是单 runtime  plumbing，不是策略路由。  
5. **metrics 不得阻塞交易**：`_deactivate`、reconcile、on_bar 主路径 best-effort only。  
6. **`reconciliation_ok` 语义**：仅反映**当前** unresolved issue；一次 successful reconcile 修复不应永久留红。

---

## 9. 实施顺序总览

```text
Phase 0  文档对齐（segment-lifecycle snippet + 本文术语）     ~0.5d
Phase 1  open_reconcile_updated → Prometheus allowlist        ~0.5d
Phase 2  segment_* events → strategy_event_total               ~0.5d
Phase 3  Grafana panels + provisioning tests                    ~1d
Phase 4  execution_truth_sync.py（进程内 helper）             ~1–2d
Phase 5  自动化测试 + Live 观察                                 ~0.5d + 1–3 交易日
Phase 6  Regime verb / ledger / 子账户（中期，单独立项）
```

---

## 10. Out of scope

- R&D Phase 1 scan / promotion 流程（见 experiments README）  
- CMS PnL 配对细节（已单独系列 commit）  
- 币本位 / dapi（ABC 文档已标记远期）  
- 新建 `quant-truth-sync` 或类似 systemd 服务  

---

## 11. 变更记录

| 日期 | 说明 |
| ---- | ---- |
| 2026-06-14 | 初版：近期修复清单、ABC vs ExecutionTruthSync 澄清、metrics 修正 |
| 2026-06-14 | 增补 §5–§9：Review 结论、目标架构、Phase 0–6 详细实现计划（中文） |
