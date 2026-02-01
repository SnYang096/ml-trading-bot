# Evidence 架构 V2：抽象对 · 实现稳

> **硬 Gate + 结构化软 Evidence + 分档 Execution**

---

## 一、整体架构

```
Raw Features
   ↓
Gate（硬规则 / 安全过滤）         → allow | deny
   ↓
Evidence Axes（结构化连续评分）   → [0, 1] per axis
   ↓
Evidence Aggregation（加权总分）  → overall_score ∈ [0, 1]
   ↓
Execution Tiers（分档执行）       → TP/SL/Size 参数
   ↓
PCM / Risk / Logging
```

**这个层级关系冻结，3-5 年不动。**

---

## 二、每一层职责

### Layer 1: Gate — 是否允许参与博弈

| 属性 | 说明 |
|------|------|
| **性质** | 硬规则 / deny-first |
| **输出** | `allow \| deny` |
| **职责** | 剔除必亏 / 不可控 / 反身性风险 |

**典型 Gate 规则**：
- 反身性过高（shd_pct > 0.85）
- regime 完全不匹配
- 极端跳空 / 波动异常
- 流动性不可交易（vpin < 0.15）

> **Gate 永远是硬的，这是底线。不做"好不好"判断，只做"能不能玩"。**

---

### Layer 2: Evidence Axes — 好坏程度的来源

#### 抽象定义

```python
Evidence Axis:
  输入: 原始特征子集
  输出: score ∈ [0, 1]
  语义: 单一"质量维度"
```

#### 强制约束

- 每个 axis：**语义单一、可解释、可单独回测**
- Axis 之间：**不直接交叉、不互相调用**

#### 最小 Axis 集合（3-4 个）

| Axis | 语义 | 典型特征 | 输出 |
|------|------|----------|------|
| **Structure** | 路径质量 | path_efficiency, dir_consistency | 0-1 |
| **Orderflow** | 真实参与 | vpin, cvd_change, delta_imbalance | 0-1 |
| **Regime** | 环境适配 | atr_percentile, jump_risk | 0-1 |
| **Timing** (可选) | 时机窗口 | compression_score | 0-1 |

#### 实现原则（稳）

- ❌ 不用黑盒 NN
- ❌ 不端到端学习
- ✅ 用 quantile mapping / piecewise linear / monotonic sigmoid
- ✅ 树模型**指导**轴设计，不直接预测 score

---

### Layer 3: Evidence Aggregation — 统一尺度

**唯一职责：加权求和**

```python
overall_score = (
    w1 * structure_score +
    w2 * orderflow_score +
    w3 * regime_score
)
```

**约束**：
- 权重初始人为设定，后续平坦高原微调
- 不允许非线性组合、if-else 逻辑、axis 间条件依赖

> **这是为了稳定性和可归因性。**

---

### Layer 4: Execution Tiers — 把"好"转成"赚"

#### 分档，不连续函数

```yaml
Tier 1 (score ≥ 0.70):  # 强证据
  tp_r: 3.0
  sl_r: 0.8
  size: 1.2
  trailing: activate_at_rr=1.5

Tier 2 (score ≥ 0.50):  # 中等证据
  tp_r: 2.5
  sl_r: 1.0
  size: 1.0

Tier 3 (score ≥ 0.35):  # 弱证据
  tp_r: 2.0
  sl_r: 1.2
  size: 0.8
  max_holding_bars: 24

< 0.35 → deny
```

#### 为什么是分档？

| 连续函数 | 分档规则 |
|----------|----------|
| 难校准 | 易校准（每档独立回测） |
| 难解释 | 易解释（3 档语义清晰） |
| 行为跳变 | 行为稳定 |
| 参数空间大 | plateau 搜索空间小 |

---

## 三、树模型的角色

> **树 = 研究工具，不是生产组件**

### 树模型只做三件事

1. **发现重要特征**
2. **发现稳定分裂区间（quantile）**
3. **指导 Axis 的 scoring mapping**

### 禁止

- ❌ 树直接下单
- ❌ 树直接输出 allow/deny
- ❌ 树直接预测 TP/SL

---

## 四、长期维护宪法

### ✅ 允许变的

- Axis 内 scoring 参数
- Axis 权重
- Tier 的阈值 / TP / SL / size
- 新增 Axis（不删旧）

### ❌ 永远不允许变的

- Execution 直接看原始特征
- Evidence 变回硬规则
- Gate 变成软评分
- 为修 bug 改层级结构

---

## 五、对比总结

| 维度 | 全硬规则 | 全软评分 | **路径2.5（推荐）** |
|------|----------|----------|---------------------|
| Gate | ✅ 硬 | ✅ 硬 | ✅ 硬 |
| Evidence | ❌ 硬（规则膨胀） | ✅ 软（难解释） | ✅ **结构化软**（可解释） |
| Execution | ❌ 固定RR | ✅ 动态RR（难校准） | ✅ **分档RR**（易校准） |
| 可维护性 | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 可解释性 | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| 收益上限 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 落地难度 | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

---

## 六、与现有系统的关系

### 研究轨（树模型）

```
config/strategies/{strategy}/
├── risk_gate.yaml        # Gate 规则（硬）
├── evidence_axes.yaml    # Axis 定义（软，结构化）
└── execution_tiers.yaml  # 分档参数
```

### 实盘轨（分层架构）

```
config/nnmultihead/
└── execution_archetypes.yaml  # 聚合所有 archetype 的配置
```

**数据流**：
```
树模型发现规则 → risk_gate.yaml / evidence_axes.yaml
                        ↓
              合并到 execution_archetypes.yaml
                        ↓
              tree_gate.py 执行
```

---

## 七、核心洞见

1. **Gate 和 Evidence 的本质区别**
   - Gate = "有/无"（二元）→ 硬
   - Evidence = "好/差"（程度）→ 软

2. **Evidence 必须结构化**
   - 每个 axis 语义独立
   - 可追溯、可归因
   - 避免黑盒

3. **Execution 必须分档**
   - 连续函数难校准
   - 分档易回测、易优化
   - 每档独立验证

4. **树模型是望远镜，不是方向盘**
   - 用树发现规律
   - 不用树直接决策

> **抽象一次选对，实现永远只调参数，不动结构。**
