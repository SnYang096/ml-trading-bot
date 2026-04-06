# Plateau 优化方法论：为什么慢 + 怎么改

## 核心结论（两句话）

1. **Plateau 搜索慢的核心原因不是算力，而是可变参数太多 + 每个参数都被当成"可塑形变量"**
2. **正确顺序：Optuna 找锚点（可行最优解） → Plateau 验证稳健区间**

---

## 详细说明

**Plateau 搜索慢的核心原因不是算力，而是：**

> **可变参数太多 + 每个参数都被当成"可塑形变量"，而 plateau 搜索本质是在找"不敏感区间"。**

这两者在数学上是**冲突的**。

👉 **平坦高原 ≠ 高维自由搜索**  
👉 **平坦高原 = 低维稳定子空间**

**正确的优化顺序：**

👉 **Optuna 用来"找点"（找可行锚点解）**  
👉 **Plateau 用来"画面"（验证稳健区间）**

---

## 为什么"全量一个个找"在 plateau 里一定慢（而且不稳）

当前做法实际上是：

```text
在 d 维参数空间里
寻找一个区域
使 score(x) 在 ε 内变化
```

**现实问题：**

- **d > 6 时，高原体积指数级变小**
- 加入的很多参数不是同一"物理层级"：
  - adx_min
  - sr_dist
  - sqs
  - ma_pos
  - extreme
  - te/tc split
  - switch_rate
  - rate_min / max

👉 **这些不是同一"物理层级"的参数，却一起被放进搜索空间。**

---

## 正确做法：分层冻结 + 子空间找 plateau

### 核心原则

> **只有"连续、可微、同语义"的参数才一起找 plateau。**

---

## 三阶段策略（可直接执行）

### 🧱 Stage 0：物理层冻结

**这些参数不该进 plateau 搜索：**

❌ 是否启用某条规则  
❌ 是否拆 TE / TC  
❌ 是否启用 extreme contraction  
❌ 是否启用某类 veto

👉 **这些是结构参数，不是阈值参数**

**做法：**

- 固定一套结构
- 写清楚「这是本轮 plateau 的物理前提」

---

### 🧱 Stage 1：锚点参数一次性定死

**这些参数只需要粗粒度 sweep 一次：**

- adx_min（20 / 25 / 30）
- ma200_pos（0 / 0.1）
- extreme_percentile（0.9 / 0.95）

**目标不是找最优值，而是问：**

> **"这个维度会不会彻底毁掉 plateau？"**

👉 只要找到一个：
- plateau_frac 不为 0
- trade_rate 不塌

就**冻结它**。

---

### 🧱 Stage 2：真正找 plateau 的子空间（2~3 维）

**通常只剩这些值得调：**

- sr_distance（MEAN）
- sqs_min（MEAN）
- adx_slope / adx_delta（TE）

**最多 2～3 个。**

这时做的才是：

> "是否存在对阈值不敏感的稳定区间？"

---

## 工程判断标准

> **如果某个参数的 ±10% 改动导致 trade_rate 或 regime_rate 变化 > 30%，它不是 plateau 参数，是 gate 参数，应冻结。**

---

## 为什么"固定大部分，跑完看看"反而更对

**Plateau 不是用来找"最优"，而是用来证明"不脆弱"。**

**正确流程：**

1. 固定 80% 参数
2. 证明剩下 20% 有宽区间
3. 再考虑是否值得放松冻结

---

## 协议规则（重要）

**Plateau 搜索协议：**

> **Plateau 搜索不允许同时调节超过 3 个连续阈值参数。  
> 结构性开关与语义层参数必须先冻结。  
> 否则 plateau 结果不具工程意义。**

---

## 参数分类标准

### 结构参数（Stage 0 冻结）
- 规则启用/禁用开关
- Archetype 拆分策略
- 执行模式选择

### Gate 参数（Stage 1 粗扫后冻结）
- 离散阈值（adx_min, ma200_pos）
- 极端值阈值（extreme_percentile）
- 如果 ±10% 改动导致 trade_rate 变化 > 30%，应冻结

### Plateau 参数（Stage 2 精细搜索）
- 连续、可微的阈值参数
- 同语义层级的参数
- 最多 2~3 个同时搜索

---

## 实施建议

### 1. 参数标注
标注现有所有参数：
- 「结构 / gate / plateau」

### 2. Plateau 参数白名单
设计一个 plateau 参数白名单 + 冻结模板

### 3. Plateau-friendly 配置示例
提供一份"plateau-friendly regime config 示例"

---

## 优化器选择：Optuna vs 启发式

### 核心判断

> **不用 Optuna，完全可以；关键不是"用什么优化器"，而是：Plateau 只能动"可调旋钮"，不能同时重构世界观。**

👉 **Plateau 的前提不是"参数空间完整"，而是"基本假设已冻结"。**

---

## Optuna → Plateau 的正确顺序

### 简短结论

**是的，而且这是更"工程正确"的顺序。**

👉 **Optuna 找锚点（可行最优解） → Plateau 验证稳健区间**，而不是反过来。

**或者：启发式找锚点 → Plateau 验证稳健区间**（同样有效）

---

### 为什么先 Optuna、后 Plateau 是对的

#### 1️⃣ Plateau 不是用来"找解"的

Plateau 在做的是：

> 在某个解**附近**，验证"参数小幅变化，系统是否不塌"

如果你一开始就不知道"解在哪里"，那 plateau 就是在**高维空间里找不存在的平坦面**——一定慢、而且多半失败。

Optuna 的职责正好相反：

- 快速定位**可行区域中心**
- 接受一定脆弱性
- 不要求可解释

👉 **它们是互补工具，不是替代关系。**

#### 2️⃣ 当前问题，Optuna 正好擅长

当前症状是：

- 参数多
- 规则叠加
- trade_rate 容易被压到 0
- 高原很窄 / 不明显

这说明缺的不是"稳健性"，而是：

> **一个"还活着"的中心解**

Optuna 在这里非常合适：

- 它会自动避开"全杀区"
- 会偏向 trade_rate ≠ 0 的解
- 可以很快给你 5–10 个可行点

---

### 正确的两阶段流程

#### 🧭 Phase 1：Optuna 找"锚点解"（Anchor）

**目标不是 Sharpe 最大，而是：**

- trade_rate > 下限（比如 0.15）
- regime 分布不退化
- 没有明显爆仓 / 极端不稳定

**Objective 示例：**

```python
score = (
    pnl_score
    - 5 * max(0, min_trade_rate - trade_rate)
    - 3 * regime_entropy_penalty
    - 2 * extreme_drawdown_penalty
)
```

**关键点：**

- **强制 trade_rate 下限**
- 不追求极致稳健
- 参数空间可以稍大

👉 Optuna 输出：**1～3 个"能跑、能交易、不离谱"的配置**

#### 🧱 Phase 2：Plateau 验证"这个解稳不稳"

现在做 plateau，才是**合理的**：

- 冻结 70–80% 参数（结构 + gate）
- 只对 2～3 个连续阈值跑 plateau
- KPI 不看 pnl，只看：
  - regime 稳定性
  - switch_rate
  - extreme 收缩是否一致

**当前 plateau 跑不出来的原因：**

> **80% 的原因是你在 Phase 1 还没完成。**

---

### 工程化判断标准

**决定"是否进入 plateau 阶段"的规则：**

> 如果某组参数：
>
> - trade_rate ∈ [0.15, 0.35]
> - regime_rate 没有单一 > 70%
> - 轻微扰动 ±5% 不立即塌
>
> 👉 才允许进入 plateau 搜索
>
> 否则，继续 Optuna。

---

### 反例：不要这样做

❌ **不要**这样做：

> 用 Optuna 找最优 → 再让 Optuna + robustness penalty 一起优化

**为什么？**

- Optuna 会为了 robustness 把解推到边界
- 最终又回到 "trade_rate → 0"
- 你已经见过这个现象了（max_robustness 优先）

👉 **稳健性必须后验验证，不能当主目标。**

---

### 核心原则（一句话）

> **Optuna 用来"找点"，Plateau 用来"画面"。**

**或者：启发式用来"找点"，Plateau 用来"画面"。**

---

## 为什么"跑得慢 + 高原很窄"：自由度错位

### 根因不是算法，而是**自由度错位**

在 plateau 里动的参数，实际上包含三类：

| 层级 | 参数类型 | 是否该进 Plateau |
|------|---------|----------------|
| **世界观** | regime 是否拆 TE/TC、是否需要 FR | ❌ 不该 |
| **结构** | 哪些规则存在、是否 veto | ❌ 不该 |
| **阈值** | adx_min、sr_max、sqs_min | ✅ 该 |

一旦**前两类**混进来：

- plateau 必然变"尖峰"
- 参数组合数爆炸
- 会出现："稍微一动，全系统 trade_rate → 0"

**这不是 plateau 失败，是输入不合法。**

---

## 启发式方法完全 OK，而且更适合有经验的开发者

### 为什么启发式可能更好

对于有经验的开发者：

- 对 TE / TC / FR 的语义理解已经稳定
- 有大量 rule-based 直觉（sr breakout / reversal / compression）
- 知道哪些参数范围是"合理的"

👉 **在这种情况下，启发式比 Optuna 更快、更干净。**

---

## 正确的"启发式 → Plateau"工作流

### 🧭 Step 0：冻结"不可动的东西"（最重要）

明确写进文档 / 代码注释的那种冻结：

- Regime 架构：`{TC, TE, FR, ET}`
- 每个 regime **存在的规则集合**
- 哪些是 **hard veto**，哪些不是

> 这一步不跑任何优化，只做设计决策。

---

### 🧱 Step 1：启发式给一个"能跑的 anchor"

**目标只有一个：**

> **"系统在多数 symbol 上不会全杀"**

**可以用非常朴素的方式：**

- adx：
  - trend_min = 25
  - mean_max = 30
- sr_distance：
  - mean_max = 0.4
  - trend_min = 0.4
- sqs：
  - veto only：`< 0.2 → deny`

**只要满足：**

- trade_rate ≈ 0.2～0.4
- 各 regime 都有样本

👉 **这就是 anchor**，不需要最优。

---

### 🧪 Step 2：只挑 2～3 个"连续阈值"进 Plateau

**关键原则：**

> Plateau 只能作用在"连续可微的旋钮"上。

**典型可 plateau 的参数：**

- adx_min / adx_max
- sr_distance threshold
- sqs_min

**不该进的：**

- 是否启用某规则
- 是否拆 TE/TC
- 是否用某个 feature

**当前 plateau 跑得慢的原因：**

> 很可能是动了 6–10 个旋钮

👉 **建议一次只动 2 个。**

---

### 🧠 Step 3：Plateau 的 KPI 也要"冻结语义"

**Plateau 不看：**

- trade_rate
- pnl

**Plateau 只看：**

- switch_rate
- entropy
- extreme_contraction

这一步是 **plateau 的正宗用途**。

---

## 工程准则（一句话总结）

> **Plateau 不负责回答"这个世界该怎么建"，  
> 它只负责回答"这个世界对参数是否宽容"。**

**Plateau 不应该同时找：**

- regime 定义
- 规则是否存在
- 哪些 feature 有用

**Plateau 只应该找：**

- **这些规则的阈值有没有"宽容区间"**

这和用不用 Optuna **没有本质关系**。

---

## 总结

**当前问题：**
- 搜索空间维度设错
- 把结构参数和 gate 参数都放进 plateau 搜索
- **顺序错误**：在 Phase 1（找锚点）未完成时就进入 Phase 2（验证稳健性）

**解决方案：**
1. **Phase 1（Optuna 或启发式）**：找可行锚点解（trade_rate > 下限，能跑、能交易）
   - **Optuna**：适合参数空间大、不确定的情况
   - **启发式**：适合有经验、语义理解稳定的情况（更快、更干净）
2. **Phase 2（Plateau）**：验证锚点解的稳健区间（只对 2~3 个参数）
3. **冻结不可动的东西**：
   - 世界观层：regime 架构、规则存在性
   - 结构层：哪些规则存在、是否 veto
   - 只对阈值层（连续可微参数）做 plateau
4. **参数分类**：
   - 冻结：世界观 + 结构参数
   - Plateau：2~3 个连续阈值参数
   - 永不优化：hard veto 规则

**效果：**
- 速度提升（搜索空间指数级缩小）
- 质量提升（找到真正稳定的区间）
- 避免在不存在的高原上浪费时间

---

## 相关文档

- [平坦高原优化工作流程](./PLATEAU_OPTIMIZATION_WORKFLOW.md)
- [Plateau vs Optuna 对比](./PLATEAU_VS_OPTUNA_COMPARISON.md)
- [Hard-Gate System](../architecture/HARD_GATE_SYSTEM.md)
