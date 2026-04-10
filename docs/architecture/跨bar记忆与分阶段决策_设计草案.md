# 跨 bar 记忆与分阶段决策 — 设计草案（Review 用）

> 状态：草案，未实现  
> 日期：2026-04-10  
> 背景：Prefilter / Gate / Entry / Direction 常在**相邻 bar** 分别成立，当前实现要求**同一 bar 全 true**，导致「双底+吸收」等结构上明明有信号却不开仓。滚动阈值调优**不能**自动解决这种**时序错位**。

---

## 1. 问题定义

### 1.1 现状

- 决策入口：`GenericLiveStrategy.decide()`（`src/time_series_model/live/generic_live_strategy.py`）。
- 每一根 bar 输入一个 `features: Dict`，**无显式「上一根是否通过 prefilter」** 的状态参与 AND。
- 顺序：**Prefilter → Direction → Gate → Entry → Evidence → Execution**。

### 1.2 期望

在**不引入未来函数**的前提下，允许例如：

- 「最近 `K` 根内曾通过 Prefilter，且**当前** Direction + Gate + Entry 成立」→ 可成交；或  
- 「Prefilter + Direction 成立后进入 **armed** 状态，后续 `K` 根内仅检查 Gate + Entry」→ 分阶段触发。

目标策略：**FER 优先**；机制应为 **策略无关**，ME / BPC **可选用**（只要存在多层串联且存在错层现象）。

---

## 2. 设计原则

| 原则 | 说明 |
|------|------|
| **无未来** | 状态只依赖 **当前及过去** bar 的漏斗结果或特征；不用「下一根」信息。 |
| **按 symbol 隔离** | 状态键必须包含 `symbol`（多标的并行回测/实盘）。 |
| **可关闭** | 默认行为与现网一致（`alignment_window_bars: 0` 或 `staged: false`）。 |
| **可审计** | 漏斗或日志中增加 `alignment_used`、`armed_until_ts` 等字段，便于对照图表。 |
| **事件回测一致** | `event_backtest` / 实盘流式同一条状态机逻辑，避免「回测能开、实盘不能开」。 |

---

## 3. 方案 A：滑动窗口「近期曾通过」标记（推荐先做）

### 3.1 语义

维护每个 `(strategy, symbol)` 的位掩码或计数器：

- `prefilter_recent[K]`：当前 bar 及前 `K-1` 根中，**至少一根** `prefilter == True`。
- 最终放行条件（示例，可配置）：

```text
prefilter_recent[K] AND direction_now AND gate_now AND entry_now
```

或放宽为：

```text
(prefilter_now OR prefilter_recent[K]) AND direction_now AND gate_now AND entry_now
```

### 3.2 状态变量（建议）

- `last_prefilter_pass_bar_index` 或 `prefilter_pass_streak`（按 bar 序号或 timestamp）。
- `K`：`alignment_window_bars`（整数，如 2～5，对应 120T 下约 0.3～2.5 天）。

### 3.3 配置草案（YAML，挂在策略 meta 或 archetype 根）

```yaml
decision_alignment:
  enabled: true
  mode: prefilter_recent_window   # 可扩展 staged_armed
  window_bars: 3
  layers_required_same_bar: [gate, entry]   # 仍要求同 bar
  layers_allow_recent_window: [prefilter]   # 允许窗口内任一 bar 为真
```

### 3.4 实现落点

1. 在 `GenericLiveStrategy` 内增加小型 **`DecisionAlignmentState`**（dict 或 per-symbol 对象）。  
2. 每 bar `decide()` 开头：根据**上一根结束时的 funnel 结果**更新状态（或在本 bar 算完 prefilter 后更新 `recent`）。  
3. 在 prefilter 失败分支**之前或之后**插入逻辑：若 `enabled` 且 `prefilter_recent` 为真，则**不立即 return**，而是进入「降级路径」仅重算 direction/gate/entry（需避免重复计算 direction 两次 —— 实现时建议先算 prefilter，更新 state，再统一走后续链）。

**注意**：若 prefilter 失败直接 `return []`，则必须在失败前更新「上一根 prefilter 结果」到 state；若本根 prefilter 失败但窗口内曾通过，应 **skip prefilter hard return**，继续 direction（需严格定义：是否要求本根仍满足部分弱条件，避免完全放开）。

### 3.5 风险

- 窗口过大 → 噪声上升；需与 **gate / entry** 收紧联动。  
- 与 **PCM 多策略** 并发时，状态必须 **per (strategy, symbol)**。

---

## 4. 方案 B：分阶段有限状态机（FSM）

### 4.1 状态

- `IDLE`  
- `ARMED`（例：Prefilter ∧ Direction 已成立）  
- `FIRED` / 回到 `IDLE`（成交或超时）

### 4.2 转移（示例）

- `IDLE` → `ARMED`：本 bar `prefilter && direction != 0`。  
- `ARMED` → 成交：后续 `T` 根内 `gate && entry`（direction 是否重算可配置：固定为 armed 时方向 vs 每 bar 重算）。  
- `ARMED` → `IDLE`：超时 `T` bars 或 direction 变 0 或硬性 invalidate。

### 4.3 适用场景

比方案 A **更贴「先定性、再择时」**，但 **参数多**（`T`、是否锁定方向、是否允许 re-arm），建议第二阶段再做。

---

## 5. 方案 C：特征层「记忆」（不推荐作为第一步）

在特征工程里做 `ewm` / `max` over last K bars 的「曾触发」标志，例如 `prefilter_proxy_rolling_max`。

**缺点**：与 YAML 规则重复，且易与 **真实 prefilter 逻辑** 漂移；调试难度大。更适合作为 **实验分支**，不作为主路径。

---

## 6. 与 ME / BPC 的关系

- **机制层**：方案 A/B 写在 `GenericLiveStrategy`（或共享 mixin）→ **FER / ME / BPC 均可开关**。  
- **需求层**：FER 因 **direction 稀疏 + 多层 AND**，错层问题最显眼；ME/BPC 若也出现「蓝线/紫线不同步」，同样可受益。  
- **配置层**：各策略 `meta.yaml` 或 `archetypes/*.yaml` 增加 `decision_alignment` 块即可差异化。

---

## 7. 测试与验收

| 用例 | 预期 |
|------|------|
| `window_bars=0` | 与现有单测 / 回测结果 **逐笔一致**（回归） |
| 合成序列：bar1 prefilter 过、bar2 仅 direction+gate+entry | 方案 A 下 bar2 **可成交**；现状 **不可** |
| 多 symbol 交替 | 状态 **不串单** |
| `event_backtest` 七月 FER 样本 | 红框内成交是否增加 + `sl` 占比是否恶化（产品决策） |

---

## 8. 建议实施顺序

1. **方案 A + `window_bars` 默认 0**（纯回归）  
2. FER-only 开 `window_bars=2~3` 做 fast_month 对比  
3. 若仍不足，再 **方案 B** 或 **独立吸收型策略**（见 `6种对称策略的启发式规则.md` §11.4）

---

## 9. 参考代码位置

- `src/time_series_model/live/generic_live_strategy.py` — `decide()`  
- `src/time_series_model/archetype/loader.py` — prefilter / gate 求值  
- `src/time_series_model/execution/entry_filter.py` — entry OR 分支  
- 事件回测：`scripts/event_backtest.py`（需确认每 bar 调用 `decide` 的路径与状态生命周期一致）

---

## 10. Review 问题清单（请反馈）

1. 能否接受 **「prefilter 在窗口内曾通过即可」** 带来的假阳性上升？是否必须加 **本 bar 最低条件**（如 `fer_impulse_failure_score > 低阈值`）？  
2. **Direction** 在窗口内是否允许变化？还是 **ARMED 时锁定**？  
3. 优先 **通用 GenericLiveStrategy** 还是 **仅 FER 子类** 试点（更快但技术债）？
