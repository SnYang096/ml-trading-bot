你现在问的其实是一个**组合层（Portfolio Construction & Capital Management）级别的问题**：

> 当多个 archetype / 多个 symbol 同时发信号，优先级怎么排？

这已经不是 gate / evidence 的问题了，而是 **PCM 层的资源分配问题**。

而且你问得非常对：
优先级要分三种冲突场景。

---

# 一、三类冲突，本质不同

### ① 同一 symbol + 不同 archetype

例如：

* BTC 同时触发 BPC 和 ME
* 或 ME 和 Shallow

本质是：

> 这是同一段行情的不同解释。

这里不能当成两个独立机会。

处理原则：

✅ 选择一个
❌ 不能叠加仓位

否则你是在对同一个风险暴露加杠杆。

---

### ② 不同 symbol + 同一 archetype

例如：

* BTC 出现 BPC
* ETH 也出现 BPC

本质：

> 同一种市场结构在不同资产同时出现

这通常代表一个 **regime 级别结构机会**

可以分仓，但要考虑相关性。

---

### ③ 不同 symbol + 不同 archetype

例如：

* BTC = BPC
* SOL = Reversal

本质：

> 不同结构逻辑的风险暴露

理论上最“正交”，最应该同时持有。

---

# 二、优先级不是人工规则，而是公式

优先级不能写成：

```
ME > Shallow > BPC > Reversal
```

那太主观。

应该变成：

```
Score = ExpectedReturn × Confidence × Uniqueness / RiskCost
```

---

# 三、我建议的统一评分框架

你可以定义一个 Archetype Opportunity Score：

```
AOS = 
    w1 * EV_estimate
  + w2 * Regime_strength
  + w3 * Evidence_score
  - w4 * Correlation_penalty
  - w5 * Drawdown_overlap_penalty
```

解释：

### 1️⃣ EV_estimate

来自你历史 backtest：

* Sharpe
* MFE/MAE profile
* Winrate × RR

这是“盈利归因”的统计化表达。

---

### 2️⃣ Regime_strength

例如：

* HTF trend strength
* Volume expansion
* Vol regime

这决定是不是顺势风口。

---

### 3️⃣ Evidence_score

你 router 的综合强度。

---

### 4️⃣ Correlation_penalty

如果：

* BTC BPC
* ETH BPC

相关性高 → 分数衰减。

---

### 5️⃣ Drawdown_overlap_penalty

如果：

* BTC ME
* BTC BPC

这俩 MAE profile 高度重合 → 只选一个。

---

# 四、Same Symbol Archetype 处理建议

对同一 symbol：

不要 priority。

改成：

```
选择 AOS 最高的那个 archetype
```

本质是：

> 让它们竞争。

而不是写死顺序。

---

# 五、Reversal 三个 Entry Filter 是否共用 Gate？

你问：

> reversal 公用 gate 和 evidence 合理吗？

答案：

✅ Gate 可以公用（例如 exhaustion / liquidity condition）

❌ Evidence 不一定相同

因为：

* 缠论 中枢离开段失败
* Liquidity sweep
* Auction exhaustion

触发逻辑不同。

但你现在合并成一个大 Reversal archetype 是合理的。

等盈利归因清晰后再拆。

---

# 六、你现在最重要的选择

你现在的系统已经很完整：

```
features
gate
evidence
entry
execution
holding
pcm
```

我建议你：

---

## 阶段 1

只保留：

* BPC
* ME
* Reversal（合并）

打磨到：

* 盈利归因清晰
* MAE profile 稳定
* Sharpe 稳定

---

## 阶段 2

再加 Shallow

因为 Shallow 非常容易塌缩进 ME 或 BPC。

---

# 七、关于你说的“是不是该存成文件？”

这个属于：

**PCM 设计原则**

你应该单独建一个：

```
docs/PCM_PRIORITY_FRAMEWORK.md
```

把：

* 同symbol冲突解决公式
* 跨symbol分配规则
* 相关性惩罚
* 总风险预算

写清楚。

这会是你系统的“资本调度大脑”。

---

# 八、一个很重要的认知

你说：

> 哪怕只有一个 archetype，多 symbol 也很多机会

这个认知非常成熟。

真正稳定的系统：

* archetype 少
* 资产多
* 资金分散

而不是：

* archetype 多
* 资产少
* 逻辑重叠

---

# 九、最后给你一句结构性判断

你现在已经不缺 archetype。

你缺的是：

> 清晰的盈利归因 + 冲突调度机制

如果你愿意，我可以：

* 帮你设计一个完整的 AOS 公式
* 或帮你把三类冲突写成一套数学分配规则
* 或帮你设计一个“组合层评分路由器”

你想往哪一块深挖？

# Archetype Opportunity Score 是可以根据历史数据统计出来的，不需要训练模型是吧，还有就是用以前的多头模型做，mfe/(mae * mtt)，但是reversal的可能不友好

可以。我们把它压到**极限简化版**。

你现在已经有：

* Archetype（bpc / shallow / me / reversal）
* Gate
* Evidence
* Execution

你只缺一个：

> 当多个机会出现时怎么选？

我们做一个**不训练、不统计复杂分布、不算协方差**的版本。

---

# ✅ 极简版 AOS（3 行公式）

只用两个东西：

1. 该 archetype 的历史 Expectancy
2. 当前 evidence 强度

---

## 第一步：离线统计一次（每个 archetype）

对每个 archetype 统计：

```
Edge_archetype = AvgR_multiple
```

例如：

| Archetype | Avg R |
| --------- | ----- |
| ME        | 0.85  |
| BPC       | 0.62  |
| Shallow   | 0.48  |
| Reversal  | 0.55  |

不用滚动，不用模型。

固定一个半年或一年窗口即可。

---

## 第二步：当前信号评分

你本来就有 evidence 强度：

```
E ∈ [0,1]
```

那就定义：

```
AOS = Edge_archetype × E
```

结束。

---

# ✅ 同 symbol 冲突

如果：

* ME 和 BPC 同时触发

直接选：

```
AOS 最大的
```

不用 router。

---

# ✅ 跨 symbol 冲突（你只有 2 slots）

所有候选排序：

```
按 AOS 从大到小
选前 2 个
```

就完了。

---

# 为什么这够？

因为：

* Edge 代表长期统计优势
* Evidence 代表当前强度
* 两者相乘 = 长期胜率 × 当前置信

这是最小闭环。

---

# 你甚至可以更极端

如果你觉得 Edge 差异不大：

```
AOS = Evidence
```

然后优先级：

```
ME > Shallow > BPC > Reversal
```

就完了。

---

# 真正的问题

你现在是不是：

> 想用复杂评分去弥补 Archetype 设计不够清晰？

如果 archetype 足够正交：

* 优先级 + Evidence
* 就够了

你现在系统规模不大，
slot 只有 2，
symbol 不多，

**复杂 PCM 反而是噪音。**

---

# 我给你的结论

现在阶段：

> 用优先级 + Evidence × 固定 Edge
> 不要 percentile
> 不要 rolling 分布
> 不要协方差矩阵

等你规模扩大，
再升级。

---

如果你愿意，我可以帮你：

* 把 4 个 archetype 排成一个“稳定优先级结构”
* 或帮你推导一个“绝不会自我冲突的极简风险规则”

你现在更在意的是简洁，还是理论完备？

---

# 十、实施决策记录 (2026-02-15)

## 决策 1: AOS = Evidence Score（不加 Edge_archetype）

**理由**：
- Edge_archetype 来自回测交易统计的平均 R-multiple，本质是“近期偏好”
- 回测期赚钱的 archetype ≠ 未来赚钱的，overfitting 风险高
- Evidence 本身已是逐 bar 的执行质量评估，足够做 Tier 分配
- 当前只有 BPC 一个 archetype，不需要跨 archetype 排序
- `compute_archetype_edges.py` 这一步可以先跳过

## 决策 2: Evidence Quantiles 统一研发与实盘

**研发回测** (`backtest_execution_layer.py`):
- `--quantile-train-end DATE`：指定 OOS 开始日期，用其前 6 个月数据算 quantiles
- `precomputed_quantiles` 必须传入，不允许 None（禁止 look-ahead）
- 不足 6 个月数据 → 报错退出

**实盘** (`run_live.py`):
- `_compute_initial_quantiles()` 从 180 天 warmup 数据计算
- `MLBOT_QUANTILE_LOOKBACK_DAYS` 环境变量可覆盖

两者逻辑一致：都用历史数据算 quantiles，应用到未来数据。

## 决策 3: Quantiles 范围

仅包含 **9 个 evidence 特征**（见 evidence.yaml）。
Gate 用绝对阈值，Entry Filter 用布尔条件，均不需要 quantiles。
