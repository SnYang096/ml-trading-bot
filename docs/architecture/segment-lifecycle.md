# Segment Lifecycle 架构改进方案

> 日期：2026-06-13  
> 范围：`chop_grid_live_engine.py` + `dual_add_trend_live_engine.py`  
> 状态：紧急修复已完成，架构重构待排期

---

## 1. 问题根因

### 1.1 核心缺陷：段生命周期没有显式建模

`active` 是一个 `bool`，却承担了三个语义：

| 语义 | 判断位置 | 问题 |
|------|---------|------|
| "段是否在运行" | `on_bar` 入口 `should_enter` | 正确 |
| "是否占并发 slot" | `holds_real_grid_slot()` | 需要二次解读 `active` |
| "是否允许开新段" | `should_enter = not active` | 一个 bool 不够 |

### 1.2 `active=False` 的 4 条路径各自为政

| 路径 | 触发条件 | 清 inventory? | 清 pending? | 通知 gate? |
|------|---------|-------------|------------|-----------|
| `_exit_grid()` | regime exit / risk stop | ✅ market_exit | ✅ cancel all | ❌ |
| `clear_stale_active_if_ghost()` | 4 个条件全满足 | ❌ | ❌ | ✅ |
| `auto-deactivate` (新) | inventory=[] && pending=[] | ❌ | ❌ | ✅ |
| trend: 退出逻辑 | regime 不满足 | ✅ | ✅ cancel | ✅ |

**没有统一的 `_deactivate()` 入口。** 状态转换散落在 3+ 处，行为不一致。

### 1.3 保护单（TP/SL）生命周期独立于段生命周期

```
单腿 TP/SL 成交 → _handle_protection_fill → inventory.remove(pos)
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

| 引擎 | 为什么不受影响 |
|------|--------------|
| `ChopGridEngine` | 纯函数式：接收一段 bar → 处理 → 返回结果。无持久 state、无 `active` flag |
| `ChopGridLiveEngine(bar_simulation=True)` | 复用 live 引擎但 `sync_live_exchange_state` 跳过、`_live_exchange_has_activity=False`。新加的 auto-deactivate 在仿真模式也生效 |

---

## 3. 已完成的紧急修复

| Commit | 内容 |
|--------|------|
| `baf3ba14` | chop_grid: `on_bar` 末尾 auto-deactivate（inventory+pending 皆空 → active=False） |
| `4b90c235` | trend_scalp: 同上 |
| 此前 | `per_leg_stop_loss: false` 配置部署，不再产生新 orphan SL |
| 手动运维 | 清理 6 个 ghost state、标记 8 条 orphan SL 为 canceled、重启 multileg |

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
    self.state.current_regime = "idle"
    self.save_state()
    gate = getattr(self, "_concurrency_gate", None)
    if gate is not None:
        gate.notify_deactivation(self.state.symbol, self._engine_name)
```

所有 3 个调用点统一走这里：

```python
_exit_grid()          → self._deactivate("regime_exit")
clear_stale_active()  → self._deactivate("ghost_cleared")
auto-deactivate       → self._deactivate("fully_closed")
```

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

### 4.3 保护单与段生命周期绑定

```python
def on_execution_results(self, results):
    for r in results:
        ...
        if r.action == "protection_fill":
            self._handle_protection_fill(r)
    
    # ← 关键：处理完所有成交后统一检查
    if self.state.active and not self.state.inventory and not self.state.pending_orders:
        self._deactivate("fully_closed")
```

把检查放在 `on_execution_results` 尾部（而非 `on_bar` 尾部），因为 `on_bar` 只产生 actions，实际成交在 `on_execution_results` 才反映到 inventory。

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

---

## 5. 实施优先级

| 优先级 | 改动 | 风险 | 工作量 |
|--------|------|------|--------|
| **P0** ✅ | auto-deactivate（chop + trend） | 低 | 0.5d（已完成） |
| **P1** | 统一 `_deactivate()` | 低 | 0.5d |
| **P2** | `on_execution_results` 尾部检查 | 中 | 0.5d |
| **P3** | 消除 `_live_exchange_has_activity` | 中 | 1d |
| **P4** | 状态机重构 | 高 | 3d |

---

## 6. 测试建议

- [ ] 单腿 TP 平仓后验证 `active=False`
- [ ] 多腿逐腿 TP 平完后验证 `active=False`
- [ ] SL 全部扫完后验证 `active=False`
- [ ] regime exit 后验证 `active=False` 且 inventory/pending 已清
- [ ] 并发 gate 验证 ghost 不占 slot
- [ ] 仿真模式 (`bar_simulation=True`) 回归
- [ ] 重启后 state 加载正确（`active` 从文件恢复）
