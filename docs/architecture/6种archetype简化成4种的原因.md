# 6种Archetype简化成4种的原因

> **文档创建时间**: 2026-01-11  
> **来源**: 基于 `docs/architecture/archetype灭绝级回测.md` 和 `docs/architecture/6种对称策略的启发式规则.md` 的总结

---

## 📋 总纲

### 核心结论

> **简化成4种的根本原因不是"减少数量"，而是"失败模式互斥"。**

文档（`archetype灭绝级回测.md:1234`）明确指出：

> **不是减少数量，而是消除"同一失败模式的多次下注"**

---

## 🎯 三种原因的优先级排序

### 原因1：失败模式相同（⭐⭐⭐⭐⭐ 最重要）

#### 6种策略的失败模式分析

| 原6种策略 | 失败原因 | → 合并到 | 失败模式 |
|-----------|----------|----------|----------|
| ① Breakout → Pullback → Continuation | **假结构、假突破** | **TC** | 结构失效 |
| ③ HTF Bias + LTF Entry | **假结构（HTF判断错误）** | **TC** | 结构失效 |
| **合并原因**：两者都因"假结构"失败 | | | |
| ② Momentum Expansion | **假加速、假趋势** | **TE** | 动能衰竭 |
| **独立原因**：失败模式与其他不同 | | | |
| ④ Failed Breakout Fade | **世界没回归（趋势继续）** | **FR** | 回归假设错误 |
| ⑤ Liquidity Sweep → Rejection | **世界没回归（趋势继续）** | **FR** | 回归假设错误 |
| **合并原因**：两者都因"回归假设错误"失败 | | | |
| ⑥ Auction Exhaustion Reversal | **趋势续命（衰竭反转失败）** | **ET** | 衰竭判断错误 |
| **独立原因**：失败模式与其他不同 | | | |

#### 关键发现

**6种策略中有4种共享失败模式：**

- **策略①和③** → 都因"假结构"失败 → 合并为 **TC**
- **策略④和⑤** → 都因"回归假设错误"失败 → 合并为 **FR**

**为什么必须合并？**

如果同时运行失败模式相同的多个策略：
- ❌ 会在同一市场条件下同时失败
- ❌ 风险叠加，而非分散
- ❌ 账户在单次失败中承受多倍损失

#### 物种隔离原则

文档（`archetype灭绝级回测.md:1775`）的《交易宪法》第一条：

```yaml
### 第一条：物种隔离原则

> 不允许两个 **失败方式相同** 的 archetype 同时高权重运行。

- TE 与 FR 不得同时 active
- TE 与 ET 不得同时 active
```

**核心逻辑**：
- Router 的任务不是"选最强"，而是"避免同死"
- 如果多个 archetype 失败模式相同，必须合并或隔离

---

### 原因2：便于灭绝分析（⭐⭐⭐⭐ 重要）

#### 灭绝分析需要"失败模式互斥"

文档（`archetype灭绝级回测.md:1355-1364`）：

```text
为什么这4个互不内耗？

| Archetype | 失败原因 | 不会和谁同时失败 |
|-----------|----------|------------------|
| TC        | 假结构   | 不和 TE 同时重仓 |
| TE        | 假加速   | 不和 FR 同时活跃 |
| FR        | 世界没回归 | 不和 TC 同时高权重 |
| ET        | 趋势续命 | 不和 TE 同时触发 |

👉 Router 的任务不是"选最强"，而是"避免同死"
```

#### 灭绝分析的实际需求

如果 archetype 失败模式互斥，灭绝分析可以：

1. **识别灭绝路径**（哪些 archetype 组合会同时失败）
2. **设计灭绝 replay**（每种 archetype 独立的失败场景）
3. **制定隔离规则**（避免同时触发）

**复杂度对比**：

```yaml
# 6种策略：C(6,2) = 15种组合需要分析
# → 失败模式重叠，分析被混淆

# 4种archetype：C(4,2) = 6种组合需要分析
# → 失败模式互斥，分析清晰
```

**灭绝场景示例**：

```yaml
extinction_scenarios:
  - scenario_1:
      archetypes: [TC, FR]  # 如果这两个同时失败
      probability: 0.05     # 灭绝概率
      cause: "假结构 + 回归假设错误"
  
  - scenario_2:
      archetypes: [TE, ET]  # 如果这两个同时失败
      probability: 0.02     # 灭绝概率（更低，因为已经隔离）
      cause: "假加速 + 衰竭判断错误"
```

**结论**：4种 archetype 失败模式互斥，使得灭绝分析清晰可行。

---

### 原因3：Safety 控制仓位（⭐⭐⭐ 直接应用）

#### 每个 Archetype 需要独立的 Size Cap

文档（`archetype灭绝级回测.md:1790`）的《交易宪法》第二条：

```yaml
### 第二条：趋势优先原则

Router 强制偏置：

longevity_bias:
  TC: 0.9  # 最稳，size可以大
  FR: 0.5  # 中等风险
  TE: 0.4  # 高风险
  ET: 0.3  # 极端风险，size最小
```

#### 为什么需要独立 Size Cap？

1. **不同 archetype 有不同的失败风险**
2. **如果多个 archetype 共享失败模式，它们的 size cap 应该关联**
3. **4种 archetype 失败模式互斥，可以独立设置 size cap**

**Size Cap 配置示例**：

```yaml
# 每个archetype独立的size cap（基于失败模式）
archetype_size_caps:
  TC:
    base_size: 1.0
    max_size: 1.2
    risk_factor: 0.1  # 假结构风险低
    
  TE:
    base_size: 0.8
    max_size: 1.0
    risk_factor: 0.3  # 假加速风险高
    
  FR:
    base_size: 0.6
    max_size: 0.8
    risk_factor: 0.5  # 回归假设错误风险最高
    
  ET:
    base_size: 0.4
    max_size: 0.6
    risk_factor: 0.7  # 衰竭判断错误风险极高
```

**对比6种策略的问题**：

如果6种策略，需要设置6个 size cap，但其中：
- 策略①和③失败模式相同 → 应该共享 size cap
- 策略④和⑤失败模式相同 → 应该共享 size cap

**问题**：容易设置错误，导致风险放大。

**4种 archetype 的解决方案**：

- 每种 archetype 失败模式互斥 → 可以独立设置 size cap
- 风险控制更精确 → 避免同一失败模式的多次下注

---

## 📊 最终映射关系

### 6种策略 → 4种Archetype

```text
原6种策略：
TREND族（3个）：
  ① Breakout → Pullback → Continuation  → TC (Trend Continuation)
  ② Momentum Expansion                  → TE (Trend Expansion)  
  ③ HTF Bias + LTF Entry               → TC的一部分（execution层）

MEAN族（3个）：
  ④ Failed Breakout Fade              → FR (Failure Reversion)
  ⑤ Liquidity Sweep → Rejection      → FR的一部分
  ⑥ Auction Exhaustion Reversal       → ET (Exhaustion Turn)
```

### 4种Archetype的最终定义

文档（`archetype灭绝级回测.md:1752-1758`）：

```text
A. Trend Continuation (TC)   ← 主力、养老金
B. Trend Expansion (TE)     ← 火箭、一次只来一个
C. Failure Reversion (FR)   ← 均值里唯一值得活的
D. Exhaustion Turn (ET)     ← 极端条件、医生

这个划分是**终局版本**，不要再动了。
```

---

## 🔄 三种原因的闭环关系

### 为什么是"闭环"？

```text
失败模式互斥
    ↓
灭绝分析可行
    ↓
Safety仓位可控
    ↓
避免同死风险
    ↓
系统长期稳定
    ↓
（回到失败模式互斥的验证）
```

**关键点**：

1. **失败模式互斥** → 使得灭绝分析清晰可行
2. **灭绝分析清晰** → 使得 Safety 仓位控制准确
3. **Safety 仓位可控** → 避免同死风险，系统稳定
4. **系统稳定** → 验证了失败模式互斥的正确性

---

## ✅ 总结

### 核心答案

**为什么要简化成4种？**

1. **失败模式相同**（最重要）：6种策略中有4种共享失败模式，必须合并避免"同一失败模式的多次下注"
2. **便于灭绝分析**（重要）：4种 archetype 失败模式互斥，使得灭绝分析清晰可行
3. **Safety 控制仓位**（直接应用）：每种 archetype 可以独立设置 size cap，风险控制更精确

### 关键结论

> **简化成4种的根本原因是"失败模式互斥"，而不是"减少数量"。**

- 这是**工程优化**（降低复杂度）
- 也是**风险管理**的必然（避免同死风险）
- 更是**系统长期稳定**的保证（灭绝分析可行，Safety可控）

---

## 📚 相关文档

- `docs/architecture/archetype灭绝级回测.md` - 灭绝分析详细说明
- `docs/architecture/6种对称策略的启发式规则.md` - 原6种策略的订单流语义
- `config/nnmultihead/execution_archetypes.yaml` - 最终4种archetype的配置

---

*文档自动生成，请勿手动编辑*
