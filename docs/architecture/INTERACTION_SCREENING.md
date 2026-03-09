# Interaction Screening + Bell Partition：规则组合交互判定机制

> **核心问题**：多条规则组合时，应该 AND 还是 OR？不能盲目假设。

---

## 一、为什么不能直接用 SHAP Interaction

Step 4 的 SHAP∩Gain 已经计算了 `interaction_pairs`，但它**不能**直接指导规则组合。
三层根本性不匹配：

### 1.1 目标函数不同

| 方法                        | 对齐目标                                         | 衡量的是                                 |
| --------------------------- | ------------------------------------------------ | ---------------------------------------- |
| **SHAP interaction**        | LightGBM 的 loss（logloss / MSE）                | 特征 A 和 B 在**预测值**上的联合贡献     |
| **Uplift Interaction Test** | 层的实际目标函数（return uplift / tail capture） | 规则 A 和 B 组合后**目标指标的实际变化** |

树模型认为"两个特征一起用能更好地预测 label" ≠ "两条规则 AND 在一起能更好地过滤差 bar"。

SHAP interaction 高可能只是说模型内部分裂路径上这俩特征经常一起出现，但 AND 后 pass_rate 可能只剩 0.3%。

### 1.2 连续值 vs 二值化后的交互结构不同

- **SHAP interaction**：在**连续特征值**上计算 — `dist_to_nearest_sr` 全值域 × `oi_exhaustion_score` 全值域
- **规则组合**：在**二值化 deny_mask** 上组合 — `dist_to_nearest_sr < -3.41` 切成 0/1

阈值切割后交互结构可能完全翻转：
- 连续空间：synergistic（联合非线性效应）
- 切到具体阈值后：deny_mask 高度重叠 → substitutive → 应该 OR 而非 AND

**SHAP interaction 发生在阈值选择之前，规则组合发生在阈值选择之后。**

### 1.3 SHAP interaction 不告诉你 AND 还是 OR

SHAP interaction 只给一个数值（importance），说"这俩有交互"，但不区分组合方向：

| 交互类型     | 含义                   | 组合建议             |
| ------------ | ---------------------- | -------------------- |
| synergistic  | 两条都 deny 时目标最差 | AND deny（协同拦截） |
| substitutive | 任一条 deny 就够了     | OR deny（任一即可）  |
| independent  | 互不影响               | 默认 OR（更安全）    |
| antagonistic | 组合后反而更差         | 不要组合             |

Uplift Interaction Test 的 2×2 表直接给出这四种分类。

### 1.4 正确的分工

| 阶段             | 工具                    | 用途                       |
| ---------------- | ----------------------- | -------------------------- |
| Step 4 特征筛选  | SHAP∩Gain interaction   | 发现"哪些特征值得一起考虑" |
| Step 5+ 规则组合 | Uplift Interaction Test | 决定"已有规则后，怎么组合" |

两者互补，不是重复。

---

## 二、Uplift Interaction Test 方法

### 2.1 2×2 表

对每对规则 (A, B)，将 holdout 样本分为 4 组：

```
          pass B    deny B
pass A     g00       g01
deny A     g10       g11
```

计算每组的平均目标值（return/tail_capture）：`r00, r10, r01, r11`。

### 2.2 交互度量

```
additive = r10 + r01 - r00   # 无交互期望
delta    = r11 - additive     # 交互强度
```

### 2.3 分类逻辑

```python
if |delta| < 0.05 * max(|r10 - r00|, |r01 - r00|):
    → independent
elif r11 > max(r10, r01) and delta > 0:
    → synergistic    # AND 有协同
elif |r11 - max(r10, r01)| < 0.05:
    → substitutive   # OR 更合理
elif r11 < min(r10, r01):
    → antagonistic   # 组合是伪信号
else:
    → substitutive   # 默认 OR 更安全
```

---

## 三、Bell Partition 搜索

### 3.1 原理

N 条规则的所有分组方式（Bell 数：N=2→2, N=3→5, N=4→15）：
- **组内 = OR**（任一 deny → 组 deny）
- **组间 = AND**（所有组都 pass → 最终 pass）

### 3.2 评分函数

```
score = 0.40 × KS
      + 0.30 × max(uplift, 0)
      + 0.15 × (log(pass_rate) + 3) / 3
      - interaction_penalty
```

- `KS`：pass/deny 分布分离度
- `uplift`：pass 组收益提升
- `log(pass_rate)`：惩罚极端过滤（AND 太多 → pass_rate 极低 → 重罚）
- `interaction_penalty`：违反 Interaction Screening 建议的结构扣分

### 3.3 Interaction Penalty 规则

| 违规                                  | 惩罚  |
| ------------------------------------- | ----- |
| OR 了 synergistic pair（应该 AND）    | +0.15 |
| AND 了 substitutive pair（应该 OR）   | +0.10 |
| AND 了 independent pair（没必要 AND） | +0.05 |
| 包含 antagonistic pair                | +0.30 |

### 3.4 pass_rate 硬约束

```python
BELL_MIN_PASS_RATE = max(min_pass_rate, 0.02)  # 绝对下限 2%
if pr < BELL_MIN_PASS_RATE:
    return None  # 直接淘汰
```

---

## 四、各层适用性分析

### 现状总结

| 层               | 规则组合方式                  | 有 Interaction Screening? | 有 Bell Partition? |
| ---------------- | ----------------------------- | ------------------------- | ------------------ |
| **Prefilter**    | Bell Partition（AND/OR 混合） | ✅ 已实现                  | ✅ 已实现           |
| **Entry Filter** | OR only                       | ❌                         | ❌                  |
| **Gate**         | AND only + compound(all_of)   | ❌                         | ❌                  |
| **Evidence**     | 加权和（连续）                | N/A                       | N/A                |
| **Execution**    | Tier 分档（离散）             | N/A                       | N/A                |

### 4.1 Prefilter — 已实现 ✅

- **规则数**：≤4 条
- **现状**：Interaction Screening + Bell Partition 已在 `analyze_archetype_feature_stratification.py` Step 5c 实现
- **评分目标**：KS + return_uplift + log(pass_rate) - interaction_penalty

### 4.2 Entry Filter — 不需要 ❌

- **规则数**：≤5 条，OR only
- **信号强度**：弱（timing 信号）
- **数据量**：小（gate-passed 子集）
- **已有约束**：`entry_filter_layer.yaml` line 10 明确 "禁止 2D interaction surface"
- **结论**：OR-only 已经是正确选择，不需要 Interaction Screening（没有 AND 决策需要做），不需要 Bell Partition（结构已固定为全 OR）

### 4.3 Gate — 应该加 ✅

**当前问题**：

Gate 层的规则在 `optimize_gate_unified.py` 中全部 AND pass。当规则数 ≥3 时容易出现：
1. **pass_rate 全杀**：3+ 条 AND → holdout 0% pass（FER 已踩坑）
2. **盲目裁剪**：Phase 2 按 lift 弱到强移除，不知道哪些规则是 substitutive（移除损失小）vs synergistic（移除损失大）
3. **Compound 规则与单规则的交互未验证**：Lift Surface 生成的 compound gate 和单规则 gate 之间的交互关系不明

**应该加的改造**：

#### A. Interaction Screening（优先级高）

在 `optimize_gate_unified.py` 的累积 AND pass rate 模拟之前，对所有 gate 规则对做 2×2 表：

```
目标值 = forward_rr（与 prefilter 一致）
deny_mask = 每条 gate 的 deny mask
分类 → synergistic / substitutive / independent / antagonistic
```

**价值**：
- 裁剪时优先移除 substitutive 规则（移除后几乎不影响拦截效果）
- 保留 synergistic 规则（移除后拦截效果显著下降）
- 发现 antagonistic 对（组合后反而放过了差 bar）

#### B. Bell Partition（优先级中）

Gate 的 AND pass = OR deny。Bell Partition 在 Gate 语义下：

```
原始：deny_1 OR deny_2 OR deny_3（全 AND pass）
Bell:  (deny_1 AND deny_2) OR deny_3
      = 组{1,2}内两条都 deny 才生效 + 组{3}单独 deny 就生效
      → pass 语义：(pass_1 OR pass_2) AND pass_3
```

**适用场景**：当两条 gate 规则是 substitutive（拦截相同类型的差 bar），将它们放入同组（AND deny）不会漏掉风险，但显著提升 pass_rate。

**约束**：Gate 是安全层，Bell Partition 后必须验证 tail_capture 不降（deny 组的 bad_rate 不降）。

### 4.4 Evidence — 不适用

连续评分（0-1），加权和聚合，没有布尔组合逻辑。

### 4.5 Execution — 不适用

Tier 分档映射，不涉及规则组合。

---

## 五、Gate 层改造方案

### 5.1 改造位置

`scripts/optimize_gate_unified.py` 的累积 AND pass rate 模拟段（~L1480）。

### 5.2 改造步骤

```
Step 1: 收集所有 gate 规则的 deny_mask
Step 2: Interaction Screening（2×2 表）
        → 输出 interaction_map: (i,j) → type
Step 3: Bell Partition 搜索（同 prefilter）
        → 评分 = 0.40×tail_capture + 0.30×effect_size + 0.15×log(pass_rate) - penalty
        → 约束: pass_rate ≥ min_combined_pass_rate
Step 4: 如果最优结构 ≠ pure-AND → 生成 OR 组 gate（新 YAML 格式）
Step 5: 替代当前的"盲裁剪"逻辑
```

### 5.3 评分函数适配

Gate 的目标不是 return_uplift，是 tail risk interception：

```
score = 0.40 × tail_capture     # deny 组 bad_rate / baseline bad_rate
      + 0.30 × effect_size       # mean_rr(allow) - mean_rr(deny)
      + 0.15 × (log(pass_rate) + 3) / 3
      - interaction_penalty
```

### 5.4 安全约束

- `tail_capture` 不得低于单独最强 gate 的 80%（防止 OR 组合放过太多差 bar）
- `pass_rate ≥ min_combined_pass_rate`（现有约束保留）
- 最终 gate 数量 ≤ max_rules（现有约束保留）

---

## 六、实现优先级

| 优先级   | 层                   | 改造                                   | 理由                              |
| -------- | -------------------- | -------------------------------------- | --------------------------------- |
| ✅ 已完成 | Prefilter            | Interaction Screening + Bell Partition | FER/ME 管线已验证                 |
| 🔴 高     | Gate                 | Interaction Screening                  | 替代盲裁剪，信息量大且实现简单    |
| 🟡 中     | Gate                 | Bell Partition                         | 替代 Phase 2 裁剪，需要验证安全性 |
| ⚪ 不需要 | Entry Filter         | —                                      | 1D OR 已足够，信号弱+数据小       |
| ⚪ 不适用 | Evidence / Execution | —                                      | 连续/离散评分，非布尔组合         |
