# Failure-Residual → Return-Amplification Pipeline

> **从「避免失败」到「放大成功」的完整工作流**

---

## 核心判断标准

> **"我现在是在试图少亏，还是在让已经对的东西赚更多？"**
>
> - 前者 → Failure-first（已完成）
> - 后者 → RR / Plateau（当前阶段）

---

## Pipeline 总览

```
Phase 0: 宪法层（Constitution）
    ↓
Phase 1: Failure Audit（封口）
    ↓
Phase 2: 样本空间重构（GOOD）
    ↓
Phase 3: Conditional Return Shaping
    ↓
Phase 4: Plateau 参数搜索
    ↓
Phase 5: 闭环更新
```

---

## Phase 0: 宪法层（Constitution Layer）

> **定义系统绝不接受的失败**

| 项目 | 说明 |
|------|------|
| **输入** | path metrics, execution outcome, domain intuition |
| **操作** | 定义封顶型 failure 标签（不追求穷举） |
| **输出** | `failure_rr_extreme`, `failure_no_opportunity` |
| **规则** | 只定义"愿意永久接受或拒绝的失败"，不为"可能可优化的"失败建宪法 |

**Checkpoint**: 宪法一旦稳定，长期不动

---

## Phase 1: Failure-First Audit（最后一次）

> **确认 failure 是否还有可学习空间**

| 项目 | 说明 |
|------|------|
| **输入** | model_selected, failure labels |
| **操作** | 计算 `lift = P(failure|selected) / P(failure|all)` |
| **判断** | lift ≈ 1 → 不可学; lift ≪ 1 → 已被治理 |
| **输出** | Failure 可治理性结论 |

**判断结果**:
- `failure_no_opportunity` → 已治理 → ❌ 不再拆
- `failure_rr_extreme` → 接近不可学 → ❌ 不再深挖

**Checkpoint**: Failure-first 正式封口

---

## Phase 2: 样本空间重构

> **从「避免失败」切换到「放大成功」**

| 项目 | 说明 |
|------|------|
| **输入** | 所有 trades, failure_any |
| **操作** | 构造 `GOOD = ~failure_any`，按 confidence/regime/direction 切片 |
| **输出** | Conditional Return Dataset（每行都是系统愿意做的 trade） |

---

## Phase 3: Conditional Return Shaping（核心金矿）

> **在"已经决定要做"的前提下，什么决定 RR 能跑多远？**

### 3A: Return 结构拆解

| 切片 | 问题 | 可能来源 |
|------|------|----------|
| high_confidence & low_rr | 为什么没跑？ | entry 太保守 / TP 太近 / regime 误判 |
| same dir/regime → RR 差异 | 哪类结构能跑 fat tail？ | 结构质量 / 订单流强度 |
| plateau 不明显 | execution 还是 entry？ | 参数待优化 |

### 3B: Return Tree

```python
target = realized_rr (clipped)
sample = GOOD only
purpose = 找 RR 的"上限条件"
```

**输出**: RR 关键条件、plateau 触发结构、return ceiling evidence

---

## Phase 4: Plateau 参数搜索

> **让"已经好的结构"跑得更久**

| 项目 | 说明 |
|------|------|
| **输入** | RR-evidence clusters, execution params |
| **操作** | 在子集里调: TP 拉宽、trail 延迟、scale-out 曲线 |
| **禁止** | 在全样本上调 execution、破坏宪法条款 |
| **输出** | regime × execution 参数表 |

---

## Phase 5: 闭环更新

> **只对"系统行为还能被改变"的 suboptimality 训练**

| 项目 | 说明 |
|------|------|
| **输入** | RR 未达潜力的 GOOD trades |
| **任务** | 不是"是否做"，而是"做了以后怎么做得更好" |
| **产出** | TP multiplier head / trail aggressiveness head / holding bias score |
| **约束** | 不触碰 failure gate |

---

## 树模型的角色（唯一合法身份）

> **树 = Research Instrument，不是生产组件**

### 只做 3 件事

| 职责 | 产出 |
|------|------|
| ① 定位可避免的 failure 区域 | Hard Gate 宪法条款 |
| ② 在 GOOD 中定位 RR/plateau 证据 | Soft Evidence 维度 |
| ③ 给出可被反证的阈值区间 | Evidence 的 score shaping 形状 |

### 绝不做

- ❌ 直接进实盘
- ❌ 替换 gate/evidence
- ❌ 直接预测 TP/SL

---

## 分层模型的 4 层职责

### Layer 1: Gate
**只接收 Hard Failure 宪法**，回答"是否被宪法禁止"
- 形式: `deny | heavy_downweight`
- 输入: 树模型导出的负规则

### Layer 2: Evidence
**从 if-then 规则 → 连续评分函数**
- 不是原样 copy 树模型阈值
- 而是转成 sigmoid / piecewise 连续函数

### Layer 3: Execution
**只消费 Evidence，不看树规则**
- 输入: archetype_score, evidence_strength, path_primitives
- 决定: TP 放多远、trail 多慢、是否 scale

### Layer 4: PCM
**只做剩余风险管理**
- 不需要树规则、bpc label、regime
- 只管: exposure, correlation, drawdown budget

---

## 关键洞见

### 为什么树模型的 if-then 不能原样进 Evidence？

1. **树模型阈值 ≠ 市场物理边界**
   - `vpin <= 0.2369` 是样本分布下的 impurity reduction 最优，不是 VPIN 的物理拐点

2. **硬规则杀死 Execution 自由度**
   - 如果 evidence 是 require/deny，execution 退化成 entry filter
   - Return amplification 消失

### Failure-first 的终极目标

> **不是让 failure = 0，而是把 failure 压缩到一个稳定、可接受、不可再治理的残差集合**

### 系统能否 scale 的决定因素

> **不取决于 failure 少不少，而取决于：好样本能不能被放大**

---

## 工程落地顺序

```
Step 1: 从树模型导出
        - 10-20 条 failure hard rules
        - 5-10 个 RR/plateau 证据维度
            ↓
Step 2: 转换为
        - gate.yaml (deny/downweight only)
        - evidence.py (continuous score functions)
            ↓
Step 3: Evidence 聚合为
        - archetype_confidence_score
            ↓
Step 4: Execution 只用
        - (archetype, confidence_score, path_primitives)
            ↓
Step 5: 回到树模型
        - 只研究 confidence 高但 RR 低的样本
        - 形成闭环
```

---

## 一句话总结

> **Failure-first 封口后，进入 Return-Amplification 阶段：
> 用树模型发现证据，用分层架构消费证据，用 Execution 放大收益。**
