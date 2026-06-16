# 幻影仓位事故复盘与修复方案（2026-06-16）

> **范围**：multi-leg 引擎 `trend_scalp` · CMS 仓位显示  
> **前置事故**：[Late-fill 无限循环复盘](20260616_late_fill_infinite_loop_postmortem_CN.md)  
> **状态**：✅ Bug #8 已修复（方案 A：reconcile 对账清理 `_sync_phantom_positions`）+ CMS 已正确过滤

---

## 1. 现象

CMS 多腿仓位页面显示 **0 个多腿仓位**，但引擎 DB 里有 **5 行 `status='open'`** 的 `multi_leg_positions` 记录。

| 币种     | 交易所实际 (Binance) | DB `multi_leg_positions` | CMS 显示                 |
| -------- | -------------------- | ------------------------ | ------------------------ |
| HYPEUSDT | ✅ LONG 41.01         | ✅ LONG 29.71             | ✅ (trend scope, qty=5.0) |
| BTCUSDT  | ❌ 无仓位             | ⚠️ SHORT 0.031 open       | ❌ 被过滤                 |
| ETHUSDT  | ❌ 无仓位             | ⚠️ SHORT 1.154 open       | ❌ 被过滤                 |
| SOLUSDT  | ❌ 无仓位             | ⚠️ LONG 27.46 open        | ❌ 被过滤                 |
| BNBUSDT  | ❌ 无仓位             | ⚠️ SHORT 1.47 open        | ❌ 被过滤                 |

CMS 最终只显示 11 个仓位：全是 spot + 1 个 trend HYPEUSDT。**0 个 multi_leg**。

---

## 2. Bug 链条全景

共发现 **8 个相互关联的 Bug**，按因果顺序排列：

### Bug #1：`multileg_symbol_owner.py` 未提交 → crash loop

- **文件**：`src/order_management/multileg_symbol_owner.py`
- **根因**：新文件写完后忘记 `git add`，Docker 构建时缺少该模块
- **后果**：引擎每分钟 crash 重启一次，每次重启都重新读取 DB、重新处理 fill 事件
- **修复**：已提交 (`09139287`)

### Bug #2：`close_absent_positions()` 无条件清空 DB

- **文件**：`src/order_management/multi_leg_storage.py` + `src/order_management/multi_leg_orchestrator.py`
- **根因**：引擎重启后 inventory 为空 → `_persist_positions()` 调用 `close_absent_positions()` → 把 DB 中所有 open position 标记为 closed
- **后果**：重启一次清空一次 DB 记录；但 crash loop 导致反复清空又重建
- **修复**：`_inventory_synced` 标志 + 空列表守卫 (`df89dc09`, `ce2c7e57`)

### Bug #3：CMS `closed` 状态被当作终态过滤

- **文件**：`src/mlbot_console/services/open_positions_list.py` L171-173
- **根因**：`_MULTILEG_TERMINAL_ORDER_STATUSES` 包含了 `'closed'`
- **后果**：正常已平仓的订单也被过滤，导致 CMS 无法正确判断仓位状态
- **修复**：从终态集合中移除 `'closed'` (`68f6a8fa`)

### Bug #4：854 行 crash-loop 残留脏数据

- **文件**：生产 DB `multi_leg_order_management.db`
- **根因**：Bug #1 crash loop 期间产生的无效订单/仓位记录
- **后果**：干扰正常仓位判断
- **修复**：标记 `error_message='bug'`，CMS 查询加 `AND (error_message IS NULL OR error_message != 'bug')` (`5550ff31`)

### Bug #5：BTCUSDT 有效 entry 误标为 bug

- **文件**：生产 DB
- **根因**：Bug #4 清理脚本过于激进，误伤了一条正常 BTCUSDT entry
- **后果**：BTCUSDT 仓位短暂不可见
- **修复**：精确清除该行的 bug 标记

### Bug #6：Grafana 监控 crash

- **文件**：`docker/` Grafana 配置
- **根因**：telegram 告警配置引用了未注入的环境变量
- **后果**：Grafana 容器无法启动
- **修复**：配置移到 `.templates/` 目录

### Bug #7：`_fill{N}` leg_id 不匹配

- **文件**：`src/time_series_model/live/dual_add_trend_live_engine.py` L795
- **根因**：
  - 引擎每次 fill 创建 position 时用 `{order_id}_fill{N}` 作为 leg_id
  - `multi_leg_orders` 表存的是裸 `order_id`
  - CMS 按 leg_id 匹配 TP/SL 订单和 ghost 检测时，两边对不上
- **后果**：TP/SL 信息无法正确关联到 position 显示
- **修复**：CMS 增加 `_fill` 后缀匹配逻辑 (`68f6a8fa`)
  ```python
  # open_positions_list.py — leg_key_matches_open_position_legs()
  fill_prefix = key + "_fill"
  return any(str(al).startswith(fill_prefix) for al in active_leg_ids)
  ```

### Bug #8（当前活跃）：引擎创建幻影 DB 仓位

- **文件**：`src/time_series_model/live/dual_add_trend_live_engine.py` L780-800
- **根因**：
  1. 引擎从 fill 事件创建 `DualAddPosition` 并写入 DB `multi_leg_positions`
  2. 但交易所侧的仓位已经不存在（被强平、TP 触发、或 crash-loop 期间的反复开平）
  3. 引擎没有用交易所 API 校验 DB 里的 position 是否真实存在
  4. SL 挂单状态仍是 `new`（未成交），说明仓位是被交易所侧平掉的，但引擎没收到对应的 execution report
- **后果**：DB 有 4 个 open position，交易所实际为 0
- **CMS 行为**：exchange_ledger 交叉验证正确过滤了这些幻影仓位（`open_positions_list.py` L668-705）
- **修复方案**：见下文 §3

---

## 3. Bug #8 根因深挖

### 3.1 幻影仓位产生路径

```
引擎重启 (crash loop)
  → 重新初始化，从 DB 加载历史订单
  → REST 对账发现 fill 事件
  → on_execution_results() 处理 fill
  → 创建 DualAddPosition(leg_id="{order_id}_fill{N}")
  → _persist_positions() 写入 DB multi_leg_positions (status=open)
  → 但交易所仓位早已被以下方式平掉：
      a) TP/SL 已成交（但 execution report 丢失或未处理）
      b) crash-loop 期间反复 market_exit 平仓
      c) 交易所强平（资金不足）
```

### 3.2 为什么 reconcile 没发现？

`multi_leg_orchestrator.py` 的 `reconcile()` 方法（L244-310）确实做了交易所对账，但：

1. **reconcile 比较的是 engine.inventory vs exchange positions**
2. 如果 engine 刚从 fill 事件重建了 inventory → engine 认为自己有仓位
3. 交易所说没有 → 产生 `position_mismatch`
4. 但 `position_mismatch` 只是**发 TG 告警**（L315-325），**不会自动清理 DB**
5. `_persist_positions()` 随后把 engine 的 inventory 写入 DB → 幻影永存

### 3.3 关键代码证据

**`dual_add_trend_live_engine.py` L789-795** — 无条件创建 position：
```python
# 每次 fill 都创建新 position，不校验交易所
new_position = DualAddPosition(
    leg_id=f"{order.order_id}_fill{len(self.state.inventory)}",
    symbol=order.symbol,
    side=pos_side,
    entry_price=last_px if last_px > 0 else order.price,
    quantity=fill_delta,
    ...
)
self.state.inventory.append(new_position)
```

**`multi_leg_orchestrator.py` L529-560** — 无条件写 DB：
```python
def _persist_positions(self) -> None:
    for idx, pos in enumerate(inventory):
        self.storage.upsert_position({
            ...
            "status": "open",  # 永远写 open
            ...
        })
```

---

## 4. 修复方案

### 4.1 方案 A：引擎启动时交易所对账清理（推荐，短期）— ✅ 已实现

> **实现状态**：已落地（见 `MultiLegLiveOrchestrator._sync_phantom_positions`）。实际实现相对下面草案有三点强化：
> 1. **数据来源**：直接比对 *引擎 inventory*（leg 级）与交易所快照，而非只读 DB；这样引擎与 DB 同时被清理，且对 `skip_position_reconciliation`（chop+trend 共享同一 symbol）配置仍然成立。
> 2. **只处理「交易所完全平仓」的明确场景**：仅当某 `(symbol, side)` 交易所 qty ≈ 0 时才判定为幻影。交易所 qty>0 的部分错配交给常规 reconcile / 保护单逻辑，不在此清理（避免误删共享 symbol 的另一引擎持仓）。
> 3. **双层误删防护**（应对 Binance API 抖动）：
>    - **可疑快照门禁（Layer 1）**：若交易所持仓总数一步从 N>0 塌缩为 0，本周期直接跳过（且不累加确认计数）——「全部仓位同时消失」远比「同一周期全平」更可能是 API 故障。
>    - **连续确认计数器（Layer 2）**：`MLBOT_MULTI_LEG_PHANTOM_CONFIRM_CYCLES`（默认 2），一个 `(symbol, side)` 需连续 N 个 reconcile 周期都为幻影才清理。
>
>    残余风险：Binance 连续多周期返回空（非报错）仍可能误平，此场景与「账户确实已全平」无法区分；硬错误会让 `sync_positions` 抛异常、reconcile 中断，不触发清理。每周期运行（非一次性），并复用 `_inventory_synced` 门禁。
>
> 清理动作：`engine.remove_inventory_legs(leg_ids)` + `storage.close_positions_by_leg_ids(...)`（即使所有 leg 都是幻影、active 列表为空也能精确平账，弥补 `close_absent_positions` 的空列表保护盲区）+ Telegram 告警 `hedge:phantom:<symbol>`。
>
> 回归测试：`tests/order_management/test_live_safety_regressions.py`（确认计数、交易所有仓不误删、单次空快照不清理、reset 计数、DB allow-list 关闭、TG 告警）。

原始草案（保留作设计记录）——在 `multi_leg_orchestrator.py` 的 `check_actions()` 或首次 `_persist_positions()` 之前，增加一步：

```python
def _sync_positions_with_exchange(self, exchange_positions: list[Mapping]) -> None:
    """启动后首次 reconcile 时，清理交易所已不存在的 DB open positions。"""
    if not self._inventory_synced:
        return  # 等 reconcile 完成
    
    if self._positions_cleaned:
        return  # 只执行一次
    
    exchange_syms = {
        (str(p.get("symbol","")).upper(), _position_side(p))
        for p in exchange_positions
        if _position_quantity(p) > 0
    }
    
    # 获取 DB 中所有 open positions
    db_open = self.storage.get_open_positions(
        strategy=self.strategy_name, symbol=self.symbol
    )
    
    phantom_leg_ids = []
    for row in db_open:
        sym = str(row.get("symbol","")).upper()
        side = str(row.get("side","")).upper()
        if (sym, side) not in exchange_syms:
            phantom_leg_ids.append(row["leg_id"])
    
    if phantom_leg_ids:
        logger.warning(
            "Cleaning %d phantom positions not on exchange: %s",
            len(phantom_leg_ids), phantom_leg_ids
        )
        self.storage.close_positions_by_leg_ids(phantom_leg_ids, reason="exchange_sync")
        # 同时清理 engine inventory 中对应的 position
        self.engine.remove_inventory_by_leg_ids(phantom_leg_ids)
    
    self._positions_cleaned = True
```

**调用时机**：在 `reconcile()` 成功后、`_persist_positions()` 之前：
```python
# multi_leg_orchestrator.py check_actions() L178-210
if reconcile:
    reconciliation, reconciliation_results = self.reconcile(...)
    self._inventory_synced = True
    # ← 新增：首次对账后清理幻影
    self._sync_positions_with_exchange(positions)
self._persist_positions()
```

### 4.2 方案 B：CMS 侧标记幻影（短期 workaround）

在 CMS `collect_open_positions()` 中，对被 exchange_ledger 过滤掉的 multi_leg 仓位，不只 drop，而是：

1. 自动调用 engine API 标记该 position 为 `phantom`
2. 或在 CMS 界面上显示 "⚠️ 幻影仓位（交易所已平）" 状态

这不需要改引擎代码，但只是治标。

### 4.3 方案 C：execution report 可靠性提升（长期）

根因是引擎没收到交易所的平仓 execution report。需要：

1. **启用 Binance User Data Stream**（WebSocket）实时接收 execution reports
2. **定期 REST 轮询**订单状态：对所有 `multi_leg_orders` 中 status 为 `new` 的 SL/TP 挂单，每 N 分钟查一次交易所实际状态
3. 如果发现 SL/TP 已成交/取消但 DB 未更新 → 触发 position 关闭

```python
# 在 reconcile 流程中增加
def _reconcile_stale_protection_orders(self, exchange_orders: list) -> None:
    """检查 DB 中 new 状态的 SL/TP 挂单是否在交易所还存在。"""
    stale = self.storage.get_open_protection_orders(
        strategy=self.strategy_name, symbol=self.symbol
    )
    exchange_ids = {str(o.get("orderId","")) for o in exchange_orders}
    for order in stale:
        ex_id = str(order.get("exchange_order_id",""))
        if ex_id and ex_id not in exchange_ids:
            # 交易所没有这个单了 → 查询最终状态
            final = self.adapter.query_order(order["symbol"], ex_id)
            if final and final["status"] in ("FILLED", "CANCELED", "EXPIRED"):
                self.storage.update_order_status(ex_id, final["status"], final)
                # 如果关联的 position 没有其他 active protection → 标记 position closed
```

---

## 5. 临时数据清理脚本

在修复部署前，可以用以下 SQL 清理幻影 DB 记录：

```sql
-- 在生产服务器上执行（需 sudo）
-- 先备份
cp /opt/quant-engine/data/multi_leg_order_management.db \
   /opt/quant-engine/data/multi_leg_order_management.db.bak_20260616

-- 将交易所不存在的 open positions 标记为 phantom_closed
UPDATE multi_leg_positions
SET status = 'phantom_closed',
    closed_at = datetime('now'),
    close_reason = 'exchange_sync_phantom'
WHERE status = 'open'
  AND symbol IN ('BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT')
  AND leg_id NOT IN (
    SELECT leg_id FROM multi_leg_positions 
    WHERE status = 'open' AND symbol = 'HYPEUSDT'
  );
```

---

## 6. 当前状态总结

| Bug                  | 状态       | 已推 commit            |
| -------------------- | ---------- | ---------------------- |
| #1 未提交文件        | ✅ 已修复   | `09139287`             |
| #2 close_absent 清空 | ✅ 已修复   | `df89dc09`, `ce2c7e57` |
| #3 closed 状态误过滤 | ✅ 已修复   | `68f6a8fa`             |
| #4 854 行脏数据      | ✅ 已标记   | `5550ff31`             |
| #5 BTCUSDT 误标 bug  | ✅ 已清除   | 手动 SQL               |
| #6 Grafana crash     | ✅ 已修复   | 已推送                 |
| #7 _fill{N} 不匹配   | ✅ 已修复   | `68f6a8fa`             |
| #8 幻影仓位          | ✅ 已修复（方案 A） | 待推送                 |

**Bug #8 的直接后果**：CMS 多腿显示 0 个仓位。4 个幻影记录在 DB 中持续存在。

**短期行动**：✅ 方案 A（reconcile 对账清理 `_sync_phantom_positions`，连续 2 周期确认 + 引擎/DB 双清 + TG 告警）已实现。历史脏数据仍需一次性临时 SQL 清理。  
**长期行动**：方案 C（execution report 可靠性 + protection order 对账）

---

## 7. 架构改进方向

```
┌─────────────────────────────────────────────────────────┐
│                   改进后的对账流程                         │
│                                                          │
│  on_bar() → check_actions() → execute_actions()         │
│           ↓                                              │
│  reconcile(exchange_orders, exchange_positions)          │
│           ↓                                              │
│  ┌─────────────────────────────────┐                    │
│  │ NEW: _sync_phantom_positions()  │ ← 方案 A           │
│  │  对比 DB open vs exchange       │                    │
│  │  交易所没有的 → phantom_closed  │                    │
│  └─────────────────────────────────┘                    │
│           ↓                                              │
│  ┌─────────────────────────────────┐                    │
│  │ NEW: _reconcile_protection()    │ ← 方案 C           │
│  │  SL/TP 挂单是否还在交易所       │                    │
│  │  已平仓 → 关闭关联 position    │                    │
│  └─────────────────────────────────┘                    │
│           ↓                                              │
│  _persist_positions()                                    │
│  _notify_orphan_positions()                              │
└─────────────────────────────────────────────────────────┘
```

---

## 8. 教训

1. **fill 事件不等于交易所仓位存在**：引擎从 fill 重建 inventory 时，必须用交易所 API 校验
2. **reconcile 发现 mismatch 后必须有 action**：不能只发告警不修正
3. **crash loop 的二阶效应**：一个 bug（未提交文件）→ crash loop → 10+ 个衍生问题
4. **`close_absent_positions` 是危险操作**：必须有严格前置条件
5. **CMS 的 exchange_ledger 过滤是正确的防御层**：即使引擎 DB 有幻影，CMS 也不会展示错误仓位
