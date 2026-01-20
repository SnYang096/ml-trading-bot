# Regime Filter + Trade Quality 方案摘要（v0）

> 目标：解决“用 trade quality (mfe/mae/ttm/dir_conf) 反推 Regime”导致的阈值难调问题。
> 本文总结三种改法，并给出推荐路径与 TODO。

---

## 问题陈述

当前 `compute_mode_3action` 以 `mfe/mae/ttm/dir_conf` 为输入，直接划分 `MEAN/TREND`。
这些是“交易质量预测”，不等价于“市场结构/Regime”，因此：

- 阈值难解释，稳定性差
- 预测偏差会直接污染 Regime 判断
- 调参时难区分“结构错”还是“质量错”

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
