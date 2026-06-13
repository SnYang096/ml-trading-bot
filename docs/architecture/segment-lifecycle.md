# Segment Lifecycle 架构改进方案

> 日期：2026-06-13  
> 范围：`chop_grid_live_engine.py` + `dual_add_trend_live_engine.py` + `segment_lifecycle.py`  
> 状态：**P0–P4 已完成**（2026-06-13）

---

## 1. 问题根因

### 1.1 核心缺陷：段生命周期没有显式建模

`active` 是一个 `bool`，却承担了三个语义：

| 语义              | 判断位置                     | 问题                  |
| ----------------- | ---------------------------- | --------------------- |
| "段是否在运行"    | `on_bar` 入口 `should_enter` | 正确                  |
| "是否占并发 slot" | `holds_real_grid_slot()`     | 需要二次解读 `active` |
| "是否允许开新段"  | `should_enter = not active`  | 一个 bool 不够        |

### 1.2 `active=False` 的 4 条路径各自为政

| 路径                            | 触发条件                   | 清 inventory? | 清 pending?  | 通知 gate? | `save_state`?            |
| ------------------------------- | -------------------------- | ------------- | ------------ | ---------- | ------------------------ |
| `_exit_grid()`                  | regime exit / risk stop    | ✅ market_exit | ✅ cancel all | ✅          | 否（靠 `on_bar` 尾调用） |
| `clear_stale_active_if_ghost()` | 4 个条件全满足             | ❌             | ❌            | ✅          | ✅ 立即                   |
| `auto-deactivate` (新)          | inventory=[] && pending=[] | ❌             | ❌            | ✅          | 否（靠 `on_bar` 尾调用） |
| trend: `_exit_all()`            | regime 不满足              | ✅             | ✅ cancel     | ✅          | 否（靠 `on_bar` 尾调用） |

**真正的 inconsistency**：`save_state()` 调用时机不一致——`clear_stale_active_if_ghost` 立即持久化，其他三条依赖 `on_bar` 尾部的 `save_state()`。

**没有统一的 `_deactivate()` 入口。** 状态转换散落在 4 处，行为不一致。

### 1.3 保护单（TP/SL）生命周期独立于段生命周期

```
单腿 TP/SL 成交 → _handle_protection_fill → 更新 inventory
    trend: inventory.remove(pos)
    chop:  _after_level_tp_closed（移除该腿，可能同步 replenish → pending_orders 非空）
                                                 ↓
                                         ❌ 没人检查 "全空了没"
                                                 ↓
                                        active 一直 True → GHOST
```

这是 chop_grid 6 个 ghost + trend_scalp 同类问题的根本原因。

### 1.4 `_live_exchange_has_activity` 是跨 tick 的全局可变状态

- 在 `sync_live_exchange_state` 设一次
- `is_stale_active_ghost` / `holds_real_grid_slot` / `on_bar` 都依赖它
- 网络抖动、API 限流、临时残留单都可能导致误判
- DB 的 `status=new` 记录 ≠ 交易所实有挂单，但容易被混淆

---

## 2. 影响范围

### 2.1 chop_grid (live) — **严重**

- 已出 6 个 ghost（已手动清理）
- `per_leg_stop_loss: false` 已部署，不再产生新 orphan SL
- **临时修复**：`on_bar` 末尾 auto-deactivate（已合并 `baf3ba14`）

### 2.2 trend_scalp (live) — **中危**

- 同样存在 `_handle_protection_fill` 不清 `active` 的问题
- 目前 prod 6 个标的都是 `active=False`（段正常结束了）
- 但如果 TP/SL 在单段中成交，可能产生 ghost
- **临时修复**：同上 auto-deactivate（已合并 `4b90c235`）

### 2.3 回测 — 不受影响

| 引擎                                      | 为什么不受影响                                                                                                                 |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `ChopGridEngine`                          | 纯函数式：接收一段 bar → 处理 → 返回结果。无持久 state、无 `active` flag                                                       |
| `ChopGridLiveEngine(bar_simulation=True)` | 复用 live 引擎但 `sync_live_exchange_state` 跳过、`_live_exchange_has_activity=False`。新加的 auto-deactivate 在仿真模式也生效 |

---

## 3. 已完成的紧急修复

| Commit     | 内容                                                                              |
| ---------- | --------------------------------------------------------------------------------- |
| `baf3ba14` | chop_grid: `on_bar` 末尾 auto-deactivate（inventory+pending 皆空 → active=False） |
| `4b90c235` | trend_scalp: 同上                                                                 |
| 此前       | `per_leg_stop_loss: false` 配置部署，不再产生新 orphan SL                         |
| 手动运维   | 清理 6 个 ghost state、标记 8 条 orphan SL 为 canceled、重启 multileg             |

**当前状态**：6 个标的网格正常运行，无 ghost。

---

## 4. 架构重构方案

### 4.1 统一 `_deactivate(reason)` 入口

```python
def _deactivate(self, reason: str) -> None:
    """Single entry point for all deactivation paths."""
    logger.info(
        "%s deactivate: symbol=%s reason=%s",
        self._engine_name,
        self.state.symbol,
        reason,
    )
    self.state.active = False
    if hasattr(self.state, "current_regime"):
        self.state.current_regime = "idle"
    self.save_state()
    gate = getattr(self, "_concurrency_gate", None)
    if gate is not None:
        gate.notify_deactivation(self.state.symbol, self._engine_name)
```

所有调用点统一走这里：

```python
_exit_grid()          → self._deactivate("regime_exit")
clear_stale_active()  → self._deactivate("ghost_cleared")
auto-deactivate       → self._deactivate("fully_closed")
```

| Reason          | Source                                            |
| --------------- | ------------------------------------------------- |
| `regime_exit`   | `_exit_grid` / `_exit_all`                        |
| `ghost_cleared` | `clear_stale_active_if_ghost`                     |
| `fully_closed`  | auto-deactivate / post-fill check                 |
| `risk_stop`     | chop `_risk_stop` → `_exit_grid`（可选子 reason） |

> **注意**：两个引擎重复了 ~80 行 ghost + slot + auto-deactivate 逻辑。考虑用 mixin 或基类抽取公共代码。
>
> 从 `on_bar` 路径（`_exit_grid` / auto-deactivate）调用 `_deactivate()` 时，`on_bar` 尾部仍会再 `save_state()` 一次——重复 persist 无害，也可后续加 `persist=` 参数优化。

### 4.2 段状态机（可选，风险更低但更大改动）

```python
class SegmentState(Enum):
    IDLE = "idle"           # 无活动段
    ENTERING = "entering"   # 正在挂单
    ACTIVE = "active"       # 有仓/挂单
    CLOSING = "closing"     # 正在平仓
    CLOSED = "closed"       # 已平完，待清理

# 状态转换：
# IDLE → ENTERING (start_grid)
# ENTERING → ACTIVE (first fill)
# ACTIVE → CLOSING (exit_grid / last leg closed)
# CLOSING → CLOSED (all fills confirmed)
# CLOSED → IDLE (cleanup done)
# ACTIVE → IDLE (ghost detected + no pending)
```

### 4.3 post-fill 即时 deactivate（替代 `on_bar` 末尾延迟）

**正确 hook**：保护单成交通过 **`on_execution_report`**（user stream）到达，不是 `on_execution_results`：

```python
# chop_grid_live_engine.py:619
def on_execution_report(self, report):
    if self._handle_protection_fill(report):
        self.save_state()
        # ← 应该在这里加：self._maybe_deactivate_if_fully_closed()
        return
```

```python
# dual_add_trend_live_engine.py:650
def on_execution_report(self, report):
    if self._handle_protection_fill(report):
        # ← 同上
        return
```

**P0 的 timing window**：auto-deactivate 在 `on_bar` **末尾**运行，但 TP/SL fill 在 `on_execution_report` **mid-cycle** 到达。fill 到 next bar 之间，`active=True` + inventory=[] 的 ghost 状态可能持续数秒到一分钟。P0 减少了 ghost **持续**时间（最多一个 bar），P2 关闭的是 **延迟**窗口（fill 后立即清理）。

```python
def _maybe_deactivate_if_fully_closed(self) -> None:
    if self.state.active and not self.state.inventory and not self.state.pending_orders:
        self._deactivate("fully_closed")
```

调用点：`on_execution_report` 尾部 + `on_bar` 尾部（双保险）。

**chop_grid replenish**：单腿 TP 后若触发 replenish，`_replenish_actions_for_level` 会同步写入 `pending_orders`，此时 inventory 可能已空但段仍在运行，不应 deactivate。仅当 grid 真正收工（inventory + pending 皆空）时才应 `_deactivate("fully_closed")`。

### 4.4 消除 `_live_exchange_has_activity` 全局变量

改为每次查询时**直接问 exchange adapter**，不做跨 tick 缓存：

```python
def is_stale_active_ghost(self) -> bool:
    return (
        self.state.active
        and not self.state.pending_orders
        and not self.state.inventory
        # ← 不再依赖 _live_exchange_has_activity
    )
```

简化 ghost 条件：只要本地无仓无挂单就是 ghost。交易所残留由 reconciliation 机制处理，不阻塞 ghost 清理。

**⚠️ Replenishment 交互**：`_maybe_replenish_empty_levels` 使用 `_live_exchange_has_activity` 来避免交易所仍有挂单时的虚假 replenishment：

```python
# chop_grid_live_engine.py:1651-1658
if (
    self._live_exchange_has_activity
    and not self.state.pending_orders
    and not self.state.inventory
):
    return []
```

P3 需要额外处理这个路径——否则消除 `_live_exchange_has_activity` 后，exchange 仍有 orphan 单时可能触发不当 replenishment。替代方案：在 replenishment 前直接查询 exchange adapter，而非依赖缓存标记。

---

## 5. 实施优先级

| 优先级   | 改动                                      | 风险 | 工作量         | 备注                         |
| -------- | ----------------------------------------- | ---- | -------------- | ---------------------------- |
| **P0** ✅ | auto-deactivate（chop + trend）           | 低   | 0.5d（已完成） | 减少 ghost 持续时间至 ≤1 bar |
| **P1** ✅ | 统一 `_deactivate()`（`SegmentLifecycleMixin`） | 低   | 0.5d（已完成） | 消除 4 处重复代码            |
| **P2** ✅ | `on_execution_report` 尾部 post-fill 检查 | 中   | 0.5d（已完成） | 关闭 fill→bar 延迟窗口       |
| **P3** ✅ | 消除 `_live_exchange_has_activity`（ghost/slot） | 中   | 1d（已完成）   | replenish 用 `_exchange_open_orders` |
| **P4** ✅ | `SegmentState` 状态机 + trend CLOSING   | 高   | 3d（已完成）   | `segment_state` 持久化 + 旧 JSON 迁移 |

---

## 6. 测试清单（`tests/unit/test_segment_lifecycle.py`）

- [x] 单腿 TP 平仓后验证 `active=False`
- [x] 多腿逐腿 TP 平完后验证 `active=False`
- [x] SL 全部扫完后验证 `active=False`
- [x] regime exit 后验证 `active=False` 且 inventory/pending 已清
- [x] 并发 gate 验证 ghost 不占 slot
- [x] 仿真模式 (`bar_simulation=True`) 回归
- [x] 重启后 state 加载正确（`segment_state=idle`）
- [x] TP fill 后 `holds_real_grid_slot()` 返回 `False`（P2 post-fill deactivate）
- [x] replenish 后再次全平 → `active=False`

---

## 7. Out of scope

本方案只覆盖 `chop_grid_live_engine.py` 和 `dual_add_trend_live_engine.py` 的段生命周期问题。

**不在范围内**：
- 其他 live engine（如 PCM、spot 策略）
- CMS PnL 配对逻辑（已单独修复）
- 回测 `ChopGridEngine`（不受影响，见 §2.3）
- feature bus / regime 信号质量
- `MultiLegConcurrencyGate` 核心逻辑（仅在 deactivate 时调用其 `notify_deactivation`）
