# Archetype 特征语义约束规范

> 核心原则：每个 archetype 的特征配置必须与自身因果语义对齐，防止树模型利用跨语义统计噪声导致过拟合。

## 一、各层级语义约束强度

| 层级 | 约束强度 | 允许的特征范围 | 理由 |
|------|---------|---------------|------|
| **Gate (树模型特征)** | **中等** | 本 archetype 核心语义 + 通用市场状态特征 | 树模型自选有区分度的列过滤 extreme rr，但不应跨 archetype 引入无因果关系的特征 |
| **Gate (guardrail 硬规则)** | **严格** | 仅本 archetype 核心语义 | YAML 配置的 when/then 规则（从树模型结果提取），直接决定是否拦截交易，必须有因果支撑 |
| **Evidence** | **严格** | 仅本 archetype 核心语义 | 衡量"当前场景多大程度符合本 archetype"，语义不对齐的特征即使当前有效也不稳定 |
| **Entry** | **严格** | 仅本 archetype 核心语义 | 入场时机判断，必须符合本 archetype 的触发条件语义 |

## 二、决策依据

### 为什么 Gate 可以比 Evidence/Entry 宽松？

- Gate 的目标是**过滤坏时机**（not-to-trade），本质是"排除法"
- 排除法容忍更多候选信号，因为错过一个好特征的代价（漏掉坏交易）大于引入一个弱相关特征的代价（稍微多过滤一些）
- 但仍然不应引入**其他 archetype 的核心特征**，因为跨 archetype 的统计相关没有因果基础

### 为什么 Gate 也不能无限放宽（不能放所有 archetype 核心特征）？

- 树模型会无差别使用任何有区分度的列
- 例：BPC gate 放入 `dual_ignition_f`(ME 核心)，如果它恰好对过滤 BPC 坏时机有统计区分度，模型就会用
- 但 ME 点火信号和 BPC 坏时机之间没有因果关系 → 不稳定 → 过拟合风险
- **结论**：Gate 放宽 ≠ 放所有，而是"本 archetype 语义 + 通用状态"

### Guardrail 的实现方式

- Guardrail 是 YAML 配置的 `when/then` 规则（gate.yaml 中 `phase: guardrail`），**不是硬编码的 if 条件**
- 来源：先用树模型在大量特征上训练 → 从模型结果中提取有因果意义的规则 → 写入 gate.yaml 作为 guardrail
- Guardrail 不参与后续优化（`frozen: true`），保证策略语义底线

### 为什么 Evidence 和 Entry 必须严格？

- Evidence 回答：**"当前有多符合本 archetype？"** → 必须只看本 archetype 的因果信号
- Entry 回答：**"现在是不是该进场？"** → 必须基于本 archetype 的触发逻辑
- 引入跨语义特征 = 让模型用错误的因果解释做决策 → 即使回测有效，实盘也会衰减

## 三、各 Archetype 允许的特征语义域

| Archetype | 核心语义 | 允许的 Gate 通用特征 | 不允许的特征 |
|-----------|---------|-------------------|-------------|
| **BPC** (延续) | compression → breakout → continuation | vol_regime, trend_strength, atr, market_cap_rank | ME 的 ignition/expansion 特征, FER 的 exhaustion/reversal 特征 |
| **ME** (扩张) | ignition → expansion → momentum | vol_regime, trend_strength, atr, market_cap_rank | BPC 的 compression 特征, FER 的 exhaustion/reversal 特征 |
| **FER** (反转) | exhaustion → failure → reversal | vol_regime, trend_strength, atr, market_cap_rank | BPC 的 compression 特征, ME 的 ignition/expansion 特征 |
| **LV** (杠杆脆弱) | leverage_buildup → vulnerability → liquidation | vol_regime, trend_strength, atr, market_cap_rank, funding_rate | BPC 的 compression 特征, ME 的 ignition 特征 |

## 四、配置文件设计原则

```
config/strategies/{archetype}/
├── features.yaml             # FS build 用：gate + evidence + entry 特征并集
├── features_gate.yaml        # Gate 训练用：本 archetype 语义 + 通用市场状态（较宽松）
├── features_evidence.yaml    # Evidence 训练用：仅本 archetype 核心语义（严格）
├── features_entry.yaml       # Entry 训练用：仅本 archetype 核心语义（严格）
├── labels_rr_extreme.yaml    # Gate 标签
└── labels_return_tree.yaml   # Evidence 标签
```

- `features.yaml` 是所有子模型特征的**并集**，确保 Feature Store 一次 build 包含所有需要的列
- 各子模型 yaml 从 FS 宽表中**选列**训练，不需要额外 build

## 五、审查清单

新增或修改任何 archetype 的特征配置时，必须检查：

- [ ] Evidence/Entry yaml 中**不含**其他 archetype 核心语义特征
- [ ] Gate yaml 中**不含**其他 archetype 核心特征（通用市场状态除外）
- [ ] `features.yaml` 是 gate + evidence + entry 的**超集**
- [ ] 所有 `requested_features` 都已在 `feature_dependencies.yaml` 中注册
