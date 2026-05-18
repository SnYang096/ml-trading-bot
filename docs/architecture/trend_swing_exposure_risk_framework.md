# Trend-Swing 暴露与宪法对齐框架

> 目标：让「最坏情况下的同时止损」不超过宪法缓冲区，而不是单纯减少交易次数。

---

## 一、结构性风险（落地前的隐含上界）

宪法与 PCM 当前关键参数：

| 参数 | 当前值 | 位置 |
|------|--------|------|
| `risk_per_slot` | 1% | `config/constitution/constitution.yaml` → `slots` |
| `max_unprotected_symbols` | 1 | `resource_allocation.slot_policy.trend_pool_guard` |
| `max_symbols_after_unlock` | **3** | 同上 |
| `max_add_times` | 2（SRB 宪法写 3，execution 阶梯仅 2 档） | `per_strategy_limits` + `execution.yaml` |
| `require_locked_profit` | true（trend 四兄弟） | `per_strategy_limits` |
| `max_risk_per_trade` | 1%（每条腿相同） | `per_strategy_limits` |
| `add_size_multipliers` | **[1, 2]**（递增，非递减） | 各 archetype `execution.yaml` |

### 最坏同时止损（理论）

单 symbol 三条风险腿（主仓 + 2 次加仓），`size_multiplier` 作用于 `max_risk_per_trade`：

- 主仓：1.0 × 1% = **1%**
- 第一加仓：1.0 × 1% = **1%**
- 第二加仓：2.0 × 1% = **2%**
- **单 symbol 合计 ≈ 4%**（不是「3×1%=3%」）

unlock 后最多 3 个 symbol 同时持仓：

- **3 × 4% = 12%** → 与 `monthly_loss_limit: 0.12` 同量级，几乎无缓冲
- BTC/ETH/SOL 等同向高相关时，等价于单一宏观因子上的 12% 赌注

趋势反转或相关性挤兑时，月度亏损很容易顶满宪法 **12%** 硬停，表现为 maxDD 直接撞宪法。

---

## 二、稳定季度盈利需解决的三层

### 第一层：同时暴露的 symbol 数量

- `max_unprotected_symbols: 1` 方向正确：在至少一笔达到 breakeven lock 之前，只允许 1 个 symbol 承担未保护趋势风险。
- `max_symbols_after_unlock: 2` 是当前落地值；在趋势行情里保留扩展，但避免 3 个高 beta symbol 叠满。
- **建议**：已有仓位与候选 symbol 收益相关性 > **0.8**（同向）时，PCM 拒绝新开（例如 `reject_pcm_symbol_correlation_cap`）。
- 高相关组合（BTC/ETH/SOL）≈ 同一方向开三倍，不是分散。

### 第二层：加仓腿的 risk 预算

- `require_locked_profit: true` 已对齐：加仓前要求母仓利润锁定（breakeven lock）。
- 落地后 `add_size_multipliers: [0.5, 0.25]`，加仓腿按主仓 1% 的 50% / 25% 递减。
- **建议 risk 阶梯**（占 equity）：
  - 主仓：**1%**（TPC 触发为主入口语义）
  - 第一加仓：**0.5%**（BPC + `require_locked_profit`）
  - 第二加仓：**0.25%**（BPC/ME + `require_locked_profit`）
- 实现路径（二选一或组合）：
  1. `per_strategy_limits` 分腿 `max_risk_per_trade`（若执行层支持按 `add_position_seq` 解析）；
  2. `add_size_multipliers` 改为 `[0.5, 0.25]`（相对主仓 1% 的分数），并确认回测/实盘 sizing 用 **risk fraction** 而非杠杆倍数放大。

单 symbol 上界：**1.75%**（非 4%）。

### 第三层：月度亏损的节奏控制

宪法硬停（当前）：

| 窗口 | 限制 | 触发后 |
|------|------|--------|
| daily | 6% | 停止新开 |
| weekly | 8% | 停止新开 |
| monthly | 12% | 停止新开 |

对 swing 系统，**月度 12% 很紧**：震荡市连续小亏易累积触顶，系统在需要参与时被硬停。

**建议软限制**（策略层，非替代宪法）：

- 当月已实现亏损 > **8%** 时，将 `risk_per_slot`（或有效 `max_risk_per_trade`）降至 **0.5%**；
- 仍允许开仓，但降低单笔伤害；
- 月亏回落至阈值以下后恢复 1%。

PCM 回测风险曲线已支持 `monthly_soft_loss_limit` + `derated_risk_per_slot` 软降风险；实盘硬停仍由 constitution runtime 负责。

---

## 三、目标配置（建议落地）

```
主仓：1% risk，TPC 触发（PCM 优先级已把 tpc 列在 enabled_archetypes 首位）
第一加仓：0.5% risk，BPC 触发 + require_locked_profit
第二加仓：0.25% risk，BPC/ME 触发 + require_locked_profit

max_unprotected_symbols: 1          # 不变
max_symbols_after_unlock: 2         # 从 3 降到 2
相关性过滤：同向 symbol 相关性 > 0.8 → 不新开

月度软限制：月亏 > 8% → risk_per_slot 降为 0.5%（硬停仍 12%）
```

**目标最坏暴露**：2 symbol × 1.75% = **3.5%**，相对 monthly 12% 留有缓冲。

---

## 四、一句话

控制暴露的核心不是减少交易次数，而是让 **最坏情况下的同时止损** 不超过宪法缓冲区。**加仓腿递减 risk + 降低并发 symbol 上限 + 相关性过滤**，是最直接、可配置验证的解法。

---

## 五、与代码/配置的映射（实施 checklist）

| 建议项 | 配置/代码落点 | 状态（见下节审计） |
|--------|----------------|-------------------|
| `max_symbols_after_unlock: 2` | `constitution.yaml` → `trend_pool_guard` | 已落地 |
| 相关性 > 0.8 拒开 | `live_pcm.py` + `symbol_correlation_guard` | 已落地 |
| 加仓 risk 递减 | `execution.yaml` `add_size_multipliers` | 已落地（[0.5, 0.25]） |
| 月亏 8% 软降 risk | `kill_switch.monthly_soft_loss_limit` + 回测风险曲线 | 已落地（PCM/回测） |
| TPC 主仓 / BPC 加仓分工 | PCM `enabled_archetypes` 顺序 + 各策略 `allow_add_on` | 部分（优先级有，无强制分工） |

相关已有文档：

- Slot guard 实验矩阵：`docs/z实验_trend_slot_guard/trend_slot_guard_validation_matrix.md`
- Kill switch 入场节流（ME 聚集问题）：`docs/z实验_005_统一研究/实施文档_05_kill_switch_入场节流改进.md`
- 三兄弟语义：`docs/architecture/strategies/trend_trio_tpc_bpc_me.md`

---

*文档创建：2026-05-18。配置审计以 `config/constitution/constitution.yaml` 与同内容的 `live/highcap/config/constitution/constitution.yaml` 为准。*
