# chop_grid Emergency SL 方案：黑天鹅保护设计

> 最后更新：2026-06-15
> 相关策略：chop_grid
> 相关代码：`src/time_series_model/grid/chop_grid_engine.py`、`src/time_series_model/live/chop_grid_live_engine.py`

---

## 1. 背景与问题

chop_grid 当前设计为 **无 per-leg SL 保护**（`per_leg_stop_loss: false`），策略通过三层机制控制回撤：

1. `max_levels_per_side: 3` —— 限制单边最大开仓层数
2. `max_loss_per_grid: 0.03` —— 整个 grid 最大亏损 3%
3. `force_exit_on_regime_loss: true` —— regime 失效时强制退出

这些保护在**正常波动**下表现良好（回撤非常小），但面对**黑天鹅事件**（如交易所宕机、极端插针、流动性枯竭）时，现有保护是否足够？

---

## 2. 现有保护机制详解

### 2.1 三层保护的分级

| 保护级别 | 机制 | 触发条件 | 覆盖范围 | 有效性 |
|---------|------|---------|---------|--------|
| **Level 1** | `max_levels_per_side` | 固定：最多 3 层 | 单 symbol 单边 | ✅ 正常市场始终生效 |
| **Level 2** | `max_loss_per_grid` | 浮动盈亏 ≤ -3% | 单 symbol 整个 grid | ✅ 正常市场始终生效 |
| **Level 3** | `force_exit_on_regime_loss` | regime 条件不再满足 | 单 symbol 整个 grid | ✅ 正常市场始终生效 |
| **——** | —— | **极端情况** | —— | —— |
| **Level 4（缺失）** | `emergency_stop_loss_pct` | 价格暴跌/暴涨超出正常范围 | 单 symbol 整个 grid | ❌ 未实现 |

### 2.2 关键区别

- **Level 1~3**：基于**策略逻辑**的保护（grid 内部机制）
- **Level 4（Emergency SL）**：基于**价格绝对水平**的保护（不依赖策略逻辑）

---

## 3. Emergency SL 与现有保护的区别

### 3.1 本质区别

| 维度 | 现有保护（Level 1~3） | Emergency SL（Level 4） |
|------|---------------------|------------------------|
| **触发逻辑** | 基于 grid 盈亏和 regime 状态 | 基于价格绝对跌幅/涨幅 |
| **触发速度** | 需要新 bar 计算 | 交易所 STOP_MARKET 实时触发 |
| **独立性** | 依赖策略引擎运行 | 独立于策略引擎 |
| **保护范围** | 单个 grid | 单个 symbol 所有持仓 |
| **适用场景** | 正常波动、趋势转换 | 黑天鹅、极端插针、闪崩 |
| **对策略影响** | 无（策略逻辑的一部分） | 可能提前止损、错过反弹 |

### 3.2 为什么现有保护不够

**场景 1：极端插针（Flash Crash）**

```
价格：65.00 → 45.00（-30%，瞬间插针）→ 64.00（恢复）

现有保护：
  - max_loss_per_grid=3%：在插针瞬间已触发（如果引擎来得及计算）
  - 但 chop_grid 是每 bar 计算，如果插针发生在 bar 内，
    下一根 bar 计算时价格已恢复，保护可能来不及触发
  
Emergency SL：
  - 交易所 STOP_MARKET 在 45.00 时自动触发平仓
  - 即使价格瞬间恢复，也已保护住本金
```

**场景 2：交易所宕机**

```
币安宕机时：
  - 策略引擎无法连接交易所 → 无法计算盈亏
  - 本地 state 可能 stale（已平仓但本地不知道）
  - max_loss_per_grid 无法触发（引擎停止运行）
  
Emergency SL：
  - 如果宕机前已挂好 STOP_MARKET 单 → ❌ 同样失效
    （交易所宕机 = 订单无法执行）
  - 如果宕机前价格已触发 SL → ✅ 可能已平仓
  - **结论：交易所宕机时，SL 和策略保护同时失效**
```

---

## 4. 币安 SL 机制与宕机风险

### 4.1 币安 SL 的运作机制

币安的 STOP_MARKET（止损市价单）是**交易所侧订单**：

1. 挂单后存在币安服务器
2. 当 mark price 触及 stop price 时，自动以 MARKET 单执行
3. 不需要客户端（机器人）在线

### 4.2 宕机时的表现

| 场景 | 币安状态 | SL 是否有效 | 说明 |
|------|---------|-----------|------|
| 机器人宕机，币安正常 | ✅ 币安正常 | ✅ 有效 | SL 在交易所侧，机器人宕机不影响 |
| 币安宕机 | ❌ 交易所不可用 | ❌ 失效 | 订单无法执行 |
| 币安闪断后恢复 | ⚠️ 恢复中 | ⚠️ 不确定 | 取决于宕机期间价格是否触发 SL |
| 币安被黑客攻击 | ❌ 极端情况 | ❌ 失效 | 最坏情况 |

### 4.3 关键结论

> **币安 SL 只能防止"价格极端波动但交易所正常运行"的情况，无法防止"交易所本身故障"的情况。**

---

## 5. 是否值得加 Emergency SL

### 5.1 支持加的论据

1. **"保险"效应**：即使概率低，一旦发生黑天鹅，损失巨大
2. **心理安慰**：有兜底机制，策略运行时更安心
3. **极端行情保护**：如 LUNA/UST 崩盘（-99%）、战争导致市场恐慌
4. **与其他策略对齐**：trend_scalp 已有 SL 保护

### 5.2 反对加的论据

1. **违背策略逻辑**：chop_grid 是均值回归，价格下跌加仓是核心逻辑
2. **假止损风险**：正常波动中可能误触发，然后价格反弹
3. **增加复杂度**：需要设计合理的触发距离（太近会误伤，太远没意义）
4. **币安宕机时无效**：如果真的担心交易所风险，SL 帮不上忙

### 5.3 量化思考

```
假设：
  - chop_grid 年化收益：假设 30%
  - chop_grid 最大回撤（正常）：3%
  - 黑天鹅频率：假设每 5 年一次
  - 黑天鹅损失（无 SL）：假设 -50%
  - Emergency SL 触发损失：假设 -15%

期望损失（5 年）：
  - 无 SL：5 × 30% - 0.5 × 本金（一次性 -50%）= 灾难性
  - 有 SL：5 × 30% - 0.15 × 本金（一次性 -15%）= 可接受

但实际上：
  - 黑天鹅可能 10 年才遇到一次
  - 如果假止损每年发生 1 次（损失 15%），10 年损失 150%
  - 这比黑天鹅损失更大！
```

---

## 6. 实验方案建议

如果决定加 Emergency SL，建议先做实验验证影响。

### 6.1 实验设计

**实验组 A：无 Emergency SL（基线）**
- 配置：`per_leg_stop_loss: false`
- 运行：完整历史回测

**实验组 B：弱 Emergency SL（-10%）**
- 配置：`emergency_stop_loss_pct: 0.10`
- 触发条件：价格偏离 entry 价格 ≥ 10%

**实验组 C：中 Emergency SL（-15%）**
- 配置：`emergency_stop_loss_pct: 0.15`
- 触发条件：价格偏离 entry 价格 ≥ 15%

**实验组 D：强 Emergency SL（-20%）**
- 配置：`emergency_stop_loss_pct: 0.20`
- 触发条件：价格偏离 entry 价格 ≥ 20%

### 6.2 评估指标

| 指标 | 基线（无 SL） | 弱 | 中 | 强 |
|------|-------------|---|---|---|
| 总收益 | — | | | |
| 夏普比率 | — | | | |
| 最大回撤 | — | | | |
| 交易次数 | — | | | |
| SL 触发次数 | 0 | ? | ? | ? |
| 假止损占比 | 0% | ? | ? | ? |
| 黑天鹅保护次数 | 0 | ? | ? | ? |

### 6.3 关键问题

1. **假止损率**：多少次 Emergency SL 触发后价格反弹？
2. **对夏普比率的影响**：SL 保护是否抵消了策略收益？
3. **极端行情回测**：在历史极端行情（如 2020 年 3 月、2022 年 5 月）中的表现

---

## 7. 技术实现建议（如果实验通过）

### 7.1 配置设计

```yaml
# config/strategies/chop_grid/archetypes/execution.yaml
risk:
  # 现有保护
  per_leg_stop_loss: false           # 保持 false（策略逻辑不依赖 SL）
  max_loss_per_grid: 0.03            # Level 2 保护
  force_exit_on_regime_loss: true    # Level 3 保护
  
  # 新增：Emergency SL（Level 4 保护）
  emergency_stop_loss:
    enabled: false                    # 默认关闭，实验通过后开启
    trigger_pct: 0.15                 # 触发阈值：-15%（以 entry price 为基准）
    # 可选：以 ATR 为基准的 dynamic trigger
    # trigger_atr_mult: 6.0
    # trigger_atr_window: 120
```

### 7.2 实现位置

在 `chop_grid_live_engine.py` 的 `_enforce_position` 或新 bar 处理逻辑中添加：

```python
def _check_emergency_stop_loss(self, position, current_price):
    """Check if emergency SL should be triggered.
    
    Called independently from strategy logic.
    """
    if not self.cfg.emergency_stop_loss_enabled:
        return False
    
    entry_price = position.entry_price
    sl_pct = self.cfg.emergency_stop_loss_trigger_pct
    
    if position.side == "LONG":
        if current_price <= entry_price * (1 - sl_pct):
            logger.warning(
                "Emergency SL triggered for %s LONG: "
                "entry=%.4f current=%.4f loss_pct=%.2f%%",
                position.symbol, entry_price, current_price,
                (current_price - entry_price) / entry_price * 100
            )
            return True
    else:  # SHORT
        if current_price >= entry_price * (1 + sl_pct):
            logger.warning(
                "Emergency SL triggered for %s SHORT: "
                "entry=%.4f current=%.4f loss_pct=%.2f%%",
                position.symbol, entry_price, current_price,
                (entry_price - current_price) / entry_price * 100
            )
            return True
    
    return False
```

### 7.3 与交易所 SL 的关系

| 保护类型 | 实现位置 | 触发时机 | 宕机时 |
|---------|---------|---------|--------|
| `max_loss_per_grid` | 策略引擎 | 每 bar 计算 | ❌ 失效 |
| `force_exit_on_regime_loss` | 策略引擎 | regime 变化时 | ❌ 失效 |
| **Emergency SL** | **交易所 STOP_MARKET** | **实时** | ❌ 失效 |

> **注意**：Emergency SL 也应通过交易所 STOP_MARKET 实现（而不是本地判断），这样才能在机器人宕机时仍然有效。

---

## 8. 最终建议

### 8.1 短期（不改动）

- **当前配置保持不变**（`per_leg_stop_loss: false`）
- 原因：
  - 现有三层保护在正常市场下已足够
  - chop_grid 回撤非常小（已验证）
  - Emergency SL 对策略逻辑有潜在负面影响

### 8.2 中期（实验验证）

- 在历史数据上做实验（见第 6 节）
- 重点关注：
  - 假止损率
  - 对夏普比率的影响
  - 极端行情下的保护效果

### 8.3 长期（条件性开启）

- 如果实验表明 **假止损率 < 5%** 且 **对夏普比率影响 < 10%**：
  - 开启 `emergency_stop_loss.enabled: true`
  - 建议参数：`trigger_pct: 0.15`（-15%）
- 如果实验表明假止损率过高或对策略伤害大：
  - 保持 `enabled: false`
  - 考虑其他替代方案（如多交易所分散、期权对冲等）

### 8.4 替代方案（如果担心交易所风险）

如果核心担忧是**交易所宕机**而非**价格极端波动**，Emergency SL 帮不上忙。可以考虑：

1. **多交易所部署**：同时在币安、OKX 运行，分散风险
2. **定期 reconcile**：已实现的 `sync_live_exchange_state` + 定期 reconcile
3. **外部监控 + 报警**：独立的监控服务，异常时人工介入
4. **期权对冲**：购买 far OTM put/call 作为保险（成本较高）

---

## 9. 总结

| 问题 | 答案 |
|------|------|
| chop_grid 需要 Emergency SL 吗？ | **不一定需要**，现有保护已足够应对正常市场 |
| 现有保护和 Emergency SL 的区别？ | 现有保护是**策略逻辑内**的，Emergency SL 是**策略逻辑外**的兜底 |
| 币安 SL 在宕机时有用吗？ | ❌ **没用**，SL 也在交易所，宕机时无法执行 |
| 什么时候需要加？ | 黑天鹅频发、或者你对极端风险极度厌恶时 |
| 建议怎么做？ | **先做实验**，验证假止损率和收益影响后再决定 |

---

## 附录：风险层级对照

```
┌─────────────────────────────────────────────────────────────┐
│                    风险保护层级金字塔                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│    Level 5: 多交易所分散（终极方案）                           │
│       ↑ 交易所级风险                                          │
│                                                             │
│    Level 4: Emergency SL（交易所 STOP_MARKET）               │
│       ↑ 极端价格波动（交易所正常运行）                         │
│                                                             │
│    Level 3: force_exit_on_regime_loss                        │
│       ↑ 趋势转换、regime 失效                                 │
│                                                             │
│    Level 2: max_loss_per_grid (3%)                          │
│       ↑ 正常波动中的超额亏损                                  │
│                                                             │
│    Level 1: max_levels_per_side (3)                          │
│       ↑ 基础仓位控制                                          │
│                                                             │
│    基础：账户总资金分配、杠杆控制                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```
