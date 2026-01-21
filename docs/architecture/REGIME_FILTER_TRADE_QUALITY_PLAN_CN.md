# Regime Filter + Trade Quality 方案摘要（v0）

> 目标：解决“用 trade quality (mfe/mae/ttm/dir_conf) 反推 Regime”导致的阈值难调问题。
> 本文总结三种改法，并给出推荐路径与 TODO。

---

## 问题陈述

当前 `compute_mode_3action` 的 3-action 输出仍会聚合为 `MEAN/TREND/NO_TRADE`，但**内部 Regime 已拆分为 `TC/TE/ER`**，
且 `mfe/mae/ttm/dir_conf` 只应视作 **Outcome Path（交易质量预测）**，**不等价于市场结构/Regime**，因此：

**命名约定（消除旧术语冲突）**：
- 旧文档里的 **TREND ≈ TC/TE**  
- 旧文档里的 **MEAN ≈ ER（Extreme Reversion）**

- 阈值难解释，稳定性差
- 预测偏差会直接污染 Regime 判断
- 调参时难区分“结构错”还是“质量错”

---

## Path 角色对齐（Price Trajectory vs Outcome vs p-path）

为避免“Path 命名冲突”，本方案统一以下三类对象：

| 名称 | 层级 | 是否预测 | 核心对象 |
| --- | --- | --- | --- |
| **Outcome Path** | 预测层 | 是 | `(dir, mfe, mae, mtt)` |
| **Price Trajectory Path** | 状态层 | 否 | 价格如何实际走 |
| **p-path** | 事后验证层 | 否 | 轨迹是否兑现承诺 |

**关系（因果顺序）**：
```
Price history
  ↓
NN → Outcome Path (dir/mfe/mae/mtt)
  ↓
Execution
  ↓
Price Trajectory Path (realized)
  ↓
p-path KPI（兑现/风险释放/Regime 对齐）
```

**关键结论**：
- `p-path` ≈ “Trajectory + 兑现 / 风险释放 / Regime 对齐”  
- World 只做 **可交易性 veto**，不该承担 p-path 判断  
- Regime/Archetype 应围绕 **Outcome Path → Execution 映射**  
- p-path 只用于 **事后验证与校准**（不反向污染 World）

---

## 工程级总表（输入 / 输出 / KPI）

```
┌────────────┬──────────────────────────┬──────────────────────────┬──────────────────────────────┐
│ Layer      │ 输入                     │ 输出                     │ KPI                          │
├────────────┼──────────────────────────┼──────────────────────────┼──────────────────────────────┤
│ NN Path    │ 原始特征                  │ dir, mfe, mae, mtt       │ IC(dir), IC(mfe/mae),        │
│ (Root)     │ (price/vol/flow/htf)      │ + calib stats            │ calibration, monotonicity    │
├────────────┼──────────────────────────┼──────────────────────────┼──────────────────────────────┤
│ World      │ 市场统计 + NN summary     │ World label              │ 覆盖率, 存活率,              │
│            │ (vol, jump, cont.)        │ TC_WORLD/TE_WORLD/       │ world-conditioned entropy    │
│            │                            │ MEAN_WORLD/NO_TRADE      │                              │
├────────────┼──────────────────────────┼──────────────────────────┼──────────────────────────────┤
│ Regime     │ World + Path moments      │ Regime label             │ Conditional IC,              │
│            │ (dir,mfe/mae ratio,mtt)   │ TC / TE / ER / NONE      │ regime separation (JS / KS)  │
├────────────┼──────────────────────────┼──────────────────────────┼──────────────────────────────┤
│ Gate       │ Regime + Path score       │ TRADE / NO_TRADE         │ ΔSharpe, ΔWinRate,           │
│            │ (semantic / confidence)   │ + trade_strength         │ precision@trade              │
├────────────┼──────────────────────────┼──────────────────────────┼──────────────────────────────┤
│ Archetype  │ Regime + Path geometry    │ Archetype ID             │ Archetype stability,         │
│            │ (dir,mfe,mae,mtt)         │ TC_EXEC/TE_EXEC/         │ MFE capture %, variance      │
│            │                            │ FR_TREND/FR_MEAN/ET_EXEC │                              │
├────────────┼──────────────────────────┼──────────────────────────┼──────────────────────────────┤
│ Execution  │ Archetype + Path params   │ Trades (orders, exits)   │ R-multiple, MAE control,     │
│            │ (SL/TP/hold/trail)        │                          │ slippage-adjusted PnL        │
├────────────┼──────────────────────────┼──────────────────────────┼──────────────────────────────┤
│ p-path     │ All realized outcomes     │ success / fail           │ Attribution: world/regime/   │
│ (Outcome)  │ (PnL, MAE, MFE, TTE)      │ + failure cause          │ archetype / execution        │
└────────────┴──────────────────────────┴──────────────────────────┴──────────────────────────────┘
```

**命名映射（实现对齐）**：
- `TC_EXEC` → `TrendContinuationTC`
- `TE_EXEC` → `TrendExpansionTE`
- `FR_TREND/FR_MEAN` → `FailureReversionFR`（建议后续拆分）
- `ET_EXEC` → `ExhaustionTurnET`

**边界规则（必须遵守）**：
- IC 到 Regime，Sharpe 从 Gate，钱在 Execution，尸体在 p-path  
- World 只负责可交易性，不能背 Sharpe  
- Regime 只负责结构分离，不替 Gate 赚收益  

---

## Physics vs Regime（不再一一对应）

**Physics / World**：只管“是否可交易 + 风险带宽”，输出是**硬 veto + band**。  
**Regime**：在可交易世界内判断**执行相关状态**（TC/TE/ER/NONE）。  
因此两者**不是同层**，只是 Regime 受 World 约束时看起来一一对应。

---

## 分桶逻辑归属（Regime/Gate）

**分桶不是 World 逻辑**，属于 **Regime/Gate 诊断与执行选择**：
- 对 `semantic_score / regime_score` 分桶  
- 观察甜点区/毒区是否稳定  
- 结论只用于 **execution 选择或切换**（如最高桶 → MEAN-A）

---

## Sharpe 的“挖掘”路径（架构内体现）

Sharpe **不能在 World/Regime 出现**，必须从 Gate 之后开始：

```
Regime/Gate 分桶 → 选择 Execution family
          ↓
Execution 产出交易 → E2E KPI 统计 Sharpe
          ↓
归因到 bucket / world / symbol
```

**关键原则**：
- Gate 的 KPI 是 ΔSharpe（开 vs 不开）  
- Execution 才对 Sharpe 负责  
- E2E 只做汇总与归因，不回调 World  

---

## World vs Regime：为什么不能合并（划分算法视角）

**结论**：World 与 Regime 是在**不同不变性假设**下的两次划分，不能合并。  
World 追求 **低频稳定**，Regime 追求 **高频状态敏感**；二者合并会导致边界漂移与失去可复用性。

**形式化**：
```
W_t = f_world(X_{t-L:t})          # 长窗口、低频、分布级别
R_t = f_regime(X_{t-k:t} | W_t)   # 短窗口、条件状态
```

**合并的统计问题**：
- 同时满足“稳定”和“敏感”的单一边界在统计上不可兼容  
- World 会被短期噪声拖着跑（高频抖动）  
- 条件期望不可分解：`E[PnL | W, R, A]` 退化为 `E[PnL | Z, A]`

**直观例子**：
同一 **Trend World** 内会出现 pullback/chop，  
若把它当作 World 切换，TC 会被系统性错杀。

**工程口径一句话**：
> World 是“在哪个物理宇宙”，Regime 是“该宇宙当前天气”。

---

## Regime vs Archetype：为什么不能合并

**结论**：Regime 是“什么时候可以做”，Archetype 是“怎么做”。  
合并会导致归因与扩展能力丢失。

**条件期望角度**：
```
E[PnL | W, R, A]
```
Regime 改变 **分布条件**，Archetype 定义 **执行模板**。

**反例（关键）**：
同一个 Regime（如 Pullback）可以支持多个 Archetype：  
TC（顺势回调入）、TE（回调后放量突破）、FR（假回调失败反转）。  

**什么时候“看起来可以合并”**（但只是退化）：  
- 只有一个 Archetype 在赚钱（如 TC-only）  
- Regime 定义过粗（变成 World）  

**工程口径一句话**：
> Regime 是状态标签，Archetype 是执行模板。

---

## Mean World 的 FR ≠ Trend Failure（必须显式区分）

**关键纠偏**：
- **Trend World 下的 FR**：趋势尝试失败（continuation/expansion 失败）  
- **Mean World 下的 FR**：极端偏离 → 均值回归套利（不是趋势失败）

**冻结规则**：
- Mean World 中 **TC/TE 永远不允许**  
- FR 在 Mean World 必须走 **mean-style execution**（宽 SL / 慢 exit）

**建议 Archetype 拆分**：
```
FR_TREND   # trend physics 下的失败结构
FR_MEAN    # mean physics 下的反转套利
```

---

## 方案对比

### 方案 A：最小改动（Gate 增加 Regime 约束）

保留现有 Router 阈值，**在 Gate 层加结构约束**（如 ADX、SR 距离、vol regime）。

优点：
- 不改 Router 代码
- 快速验证“结构过滤 + 质量打分”的组合

风险：
- 结构规则与 Router 仍有重叠
- 规则位置分散，语义可能模糊

---

### 方案 B：新增 Regime Filter（中改）

在 Router 前新增 `compute_regime()`（启发式结构判断），Router 只做 “质量门”。

流程：

```
features -> Regime Filter -> allowed_modes
features -> trade_quality (mfe/mae/ttm/dir_conf)
if allowed and quality>theta -> trade
```

优点：
- 结构与质量解耦，阈值更好解释
- 调参分离：Regime 阈值 vs Quality 阈值

风险：
- 引入新模块，需要新 KPI

---

### 方案 C：完整重构（Router 只输出 trade_quality）

Router 不再输出 MEAN/TREND，仅输出连续质量分数。
执行模式完全由 Regime Filter 决定。

优点：
- 职责边界最清晰
- 训练更稳定，校准更简单

风险：
- 需要调整现有评估/调参协议
- 改动面最大

---

## 推荐路径（v0）

优先采用 **方案 A** 验证：

1) 在 Gate 里加入 Regime 结构约束（ADX / SR / vol regime）
2) 观察 Router KPI 与 Gate 通过率变化
3) 若效果稳定，再推进到方案 B

---

## TODO（准备给 review）

### A. Gate 结构约束最小集
- TREND: `adx > 25` + `sr_distance_normalized > 0.5` + `bb_width_percentile > 0.6`
- MEAN: `adx < 20` + `sr_distance_normalized < 0.3` + `bb_width_percentile < 0.4`

### B. Regime 诊断与 KPI
- 输出 `regime` 分布与稳定性（switch_rate, entropy）
- 统计 “regime_allow 后的 trade_quality 分布”

### C. 质量阈值与执行效果
- `trade_quality` 分层统计（P50/P75/P90）
- `quality>theta` 后的 win_rate/hold_time

### D. 文档与流程
- 在 `ARCHITECTURE.md` 增加 Regime Filter 说明入口
- 在 `THRESHOLD_PLATEAU_TUNING_PROTOCOL_CN.md` 追加 Regime 阈值与 Quality 阈值分离说明

---

## A/B/C KPI 对齐清单（用于对比）

为避免“只是放松信号数量”的错觉，A/B/C 必须在同一 KPI 体系下对比：

### 1) 分布与稀疏度
- allow_rate / veto_rate
- mean_rate / trend_rate
- by_symbol allow_rate（避免单币驱动）

### 2) 稳定性
- switch_rate
- entropy
- 极端行情窗口的收缩率（Top 1% 波动期 allow_rate）

### 3) 执行后质量（Execution KPI）
- win_rate / mean_ret / median_ret / p25 / p75
- avg_pos_ret / avg_neg_ret
- by_symbol win_rate

### 4) 可解释性审计
- 随机抽取 50 个 veto 样本，列出命中规则
- 随机抽取 50 个 allow 样本，列出最强证据

> A/B/C 只有在上述 KPI 同时成立时才有“更靠谱”的结论。

---

## 补充：为什么把 Regime 判断从多头模型换成规则模型

这一步是从“炼丹师”到“系统架构师”的分水岭：  
在真实市场里，**生存优先于预测**。关键不是“预测对了多少次”，而是“错的时候怎么死”。

### 1) 错觉的本质：拟合噪声的“勤奋”

- **多头模型**擅长平滑插值，会在模糊区间也给出“看起来合理”的判断。  
- **规则模型**承认“未知”，在结构不清时保持沉默。  

这不是胆小，而是对账户的保护。

### 2) 确定性 vs 概率性风险（类比）

- **多头模型（实习机长）**：天气不好也想起飞，说“我有 51% 把握”。  
- **规则模型（老机长）**：侧风超过阈值就拒飞。  

工业级系统宁要“残缺的确定性”，也不要“全面的不确定性”。

### 3) 系统工程三个维度对比

| 维度 | 多头模型 (Pure NN) | 规则模型 (Heuristic Gate) | 结论 |
| --- | --- | --- | --- |
| 可观测性 | 隐藏在权重/激活 | 显式逻辑 | 规则胜 |
| 失效模式 | 静默失效 | 显性阻断 | 规则胜 |
| 迭代成本 | 需重训 | 局部调参 | 规则胜 |

### 4) 视角切换：把 NN 关进“笼子”

多头模型不该当决策者，而应当评分员：

```
Regime Classifier (硬逻辑) -> Veto Gate (硬逻辑) -> NN Head (评分/排序)
```

NN 的价值是“区分更优机会”，而不是“决定做不做”。

### 5) 如何量化“谁更强”

不要只看 Sharpe，看执行后的稳定性：

1. **压力测试**：极端行情（Top 1%）亏损曲线是否水平。  
2. **滑点模拟**：对“似是而非”信号加倍滑点后还剩多少收益。  
3. **可解释性审计**：随机抽样不交易样本，理由是否符合交易直觉。  

如果规则模型能让系统更可控、更敢上线，它就是更强的组件。

---

## 补充：TE（趋势启动）为何会被硬规则压死，以及如何处理

### 1) 关键矛盾点

如果用硬规则定义 Regime，**早期趋势（TE）会被永久性压死**。  
因为 TE 的本质是“趋势刚启动、确认指标尚未到位”，而不是已确认趋势。

### 2) 为什么 ADX > 25 会错杀 TE？

TE 通常满足：

- ADX 低但上升中  
- 方向性刚建立  
- 波动开始扩张但未确认  
- SR 刚突破或刚脱离

因此如果写：

```
TREND if adx > 25
```

你其实定义的是 **Confirmed Trend**，不是 **Trend Expansion**。

### 3) 这并不意味着 Regime 要交给 NN

❌ 不能用“放松 ADX + 交给 NN”解决 TE  
✅ TE 必须被当作独立 archetype/过渡态处理

否则 NN 会在噪声中追趋势，系统稳定性崩掉。

### 4) 正确结构：把 TE 当成“受限 TREND”

#### Regime 从二值变为三态（或四态）

```
Regime ∈ { NO_TRADE, MEAN, TE, TREND }
```

| Regime | 本质 |
| --- | --- |
| MEAN | 已确认区间结构 |
| TE | 趋势启动 / 扩张 |
| TREND | 已确认趋势 |
| NO_TRADE | 不交易 |

#### Regime 规则是“资格”不是“确认”

**TREND（已确认）**  
```
adx > 25
price > ma200
sr_distance > 0.4
```

**TE（趋势启动）**  
```
adx > 15 and adx_slope > 0
price_cross_ma200_recent
vol_expanding or sqs_rising
```

关键差异：  
- TE 用“变化率/斜率”  
- TREND 用“绝对值/确认”

---

## NN 分数在正确架构下能做什么

多头模型不应决定 Regime，但可以影响 **已允许 Regime 内的执行强度**：

1) **仓位缩放**  
```
size = base * f(nn_score)
```
TE 应强制小仓位区间（如 0.1–0.5）。

2) **入场延迟 / 确认强度**  
```
if nn_score < 0.4: wait
```
TE 场景尤为重要。

3) **执行 aggressiveness（吃单 vs 挂单）**

4) **止损宽度 / 时间止损**  
低分 → 更短 TTM，更快止损。

5) **Regime 升级（高级用法）**  
```
if regime == TE and nn_score > 0.7 and adx > 20:
    upgrade to TREND
```

---

## 系统语言总结

> 规则负责“世界是否允许这种交易存在”  
> NN 负责“在允许的世界里，哪些更值得下注”
