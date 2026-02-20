# Archetype 特征语义约束规范

> 核心原则：每个 archetype 的特征配置必须与自身因果语义对齐，防止树模型利用跨语义统计噪声导致过拟合。

## 一、特征分类

特征分为两类：

| 类别 | 定义 | 示例 | 适用层级 |
|------|------|------|----------|
| **Archetype 核心语义** | 与本 archetype 因果链直接相关的特征 | BPC: bpc_pullback_depth, bpc_volume_compression | 所有层级 |
| **通用市场质量特征** | 描述市场环境/噪声/质量的通用特征，不属于任何 archetype 的因果链 | 数学特征, vol_regime, trend_strength | 所有层级（作为 quality modifier） |

### 通用市场质量特征清单

以下特征描述的是市场环境质量，对所有 archetype 均有效：

| 特征 | 描述 | 用途 |
|------|------|------|
| `hurst_price_f` | Hurst 指数 — 市场是趋势型还是噪声型 | 噪声环境评估 |
| `spectrum_features_f` | 频谱特征 — 有无主导周期/结构 | 市场结构性评估 |
| `hilbert_phase_f` | Hilbert 变换 — 包络与相位 | 周期相位/时机评估 |
| `wpt_price_fluctuation_f` | 小波包变换 — 多尺度噪声/信号比 | 多尺度质量评估 |
| `evt_features_f` | EVT 极值理论 — 尾部风险/黑天鹅概率 | 极端风险评估 |
| `vol_regime_features_f` | 波动率状态 | 环境适配 |
| `trend_r2_20_f` 等 | 趋势强度/效率 | 趋势质量 |
| `atr_percentile_f` | ATR 百分位 | 波动水平 |

> **设计变更记录 (2026-02-20)**:
> 数学特征（Hurst/Spectrum/Hilbert/WPT/EVT）原计划仅用于 Execution 层的 Noise Penalty 模块。
> 但该模块未实现，且实践中发现这些特征在 Gate 训练中有显著区分力。
> 决策：将数学特征归类为「通用市场质量特征」，Gate/Evidence/Entry 均可使用。
> 决策依据：(1) 这些特征描述市场环境质量而非 archetype 因果链，对所有 archetype 正交且有效；
> (2) 树模型会自动选择有区分度的特征，无用特征不会被选中，风险可控；
> (3) 概念上等价于 `score = archetype_conformity × market_quality_modifier`，两者正交可分离。

## 二、各层级语义约束强度

| 层级 | 约束强度 | 允许的特征范围 | 理由 |
|------|---------|---------------|------|
| **Gate (树模型特征)** | **中等** | 本 archetype 核心语义 + 通用市场质量特征 | 树模型自选有区分度的列过滤 extreme rr，但不应跨 archetype 引入无因果关系的特征 |
| **Gate (guardrail 硬规则)** | **严格** | 仅本 archetype 核心语义 | YAML 配置的 when/then 规则（从树模型结果提取），直接决定是否拦截交易，必须有因果支撑 |
| **Evidence** | **中等** | 本 archetype 核心语义 + 通用市场质量特征 | 核心语义衡量 archetype 符合度，市场质量特征作为 quality modifier 调节评分 |
| **Entry** | **中等** | 本 archetype 核心语义 + 通用市场质量特征 | 核心语义决定触发逻辑，市场质量特征辅助判断入场时机质量 |

## 三、决策依据

### 跨 archetype 核心特征：所有层级禁止

- 树模型会无差别使用任何有区分度的列
- 例：BPC gate 放入 `dual_ignition_f`(ME 核心)，如果它恰好对过滤 BPC 坏时机有统计区分度，模型就会用
- 但 ME 点火信号和 BPC 坏时机之间没有因果关系 → 不稳定 → 过拟合风险
- **结论**：所有层级都不应引入其他 archetype 的核心特征

### 通用市场质量特征：所有层级允许

- 市场质量（噪声水平、尾部风险、周期相位）对所有 archetype 都有效
- 概念上：`score = archetype_conformity × market_quality_modifier`
- archetype 核心特征决定 conformity，市场质量特征决定 modifier，两者正交
- Gate: 高噪声/高尾部风险 = 不该做 → 有因果逻辑
- Evidence: 同样的形态在低噪声市场质量更高 → 合理的质量调节因子
- Entry: 高噪声时机不好，周期相位影响入场时机 → 与时机判断相关

### Guardrail 硬规则：仅限 archetype 核心语义

- Guardrail 是 YAML 配置的 `when/then` 规则（gate.yaml 中 `phase: guardrail`），**不是硬编码的 if 条件**
- 来源：先用树模型在大量特征上训练 → 从模型结果中提取有因果意义的规则 → 写入 gate.yaml 作为 guardrail
- Guardrail 不参与后续优化（`frozen: true`），保证策略语义底线
- 通用市场质量特征不应出现在 guardrail 中（guardrail 需要明确的因果解释）

## 四、各 Archetype 允许的特征语义域

| Archetype | 核心语义 | 通用市场质量特征 (所有层级可用) | 不允许的特征 |
|-----------|---------|-------------------------------|-------------|
| **BPC** (延续) | compression → breakout → continuation | 数学特征, vol_regime, trend_strength, atr, market_cap_rank | ME 的 ignition/expansion 特征, FER 的 exhaustion/reversal 特征 |
| **ME** (扩张) | ignition → expansion → momentum | 数学特征, vol_regime, trend_strength, atr, market_cap_rank | BPC 的 compression 特征, FER 的 exhaustion/reversal 特征 |
| **FER** (反转) | exhaustion → failure → reversal | 数学特征, vol_regime, trend_strength, atr, market_cap_rank | BPC 的 compression 特征, ME 的 ignition/expansion 特征 |
| **LV** (杠杆脆弱) | leverage_buildup → vulnerability → liquidation | 数学特征, vol_regime, trend_strength, atr, market_cap_rank, funding_rate | BPC 的 compression 特征, ME 的 ignition 特征 |

## 五、配置文件设计原则

```
config/strategies/{archetype}/
├── features.yaml             # FS build 用：gate + evidence + entry 特征并集
├── features_gate.yaml        # Gate 训练用：archetype 核心语义 + 通用市场质量特征
├── features_evidence.yaml    # Evidence 训练用：archetype 核心语义 + 通用市场质量特征
├── features_entry.yaml       # Entry 训练用：archetype 核心语义 + 通用市场质量特征
├── labels_rr_extreme.yaml    # Gate 标签
└── labels_return_tree.yaml   # Evidence 标签
```

- `features.yaml` 是所有子模型特征的**并集**，确保 Feature Store 一次 build 包含所有需要的列
- 各子模型 yaml 从 FS 宽表中**选列**训练，不需要额外 build
- 通用市场质量特征可在 Gate/Evidence/Entry 的 features yaml 中共享，树模型自行选择有区分度的列

## 六、审查清单

新增或修改任何 archetype 的特征配置时，必须检查：

- [ ] **所有层级** 的 yaml 中**不含**其他 archetype 核心语义特征
- [ ] 通用市场质量特征可在任意层级使用（不受 archetype 语义约束）
- [ ] Guardrail 硬规则中**不含**通用市场质量特征（仅限 archetype 核心语义）
- [ ] `features.yaml` 是 gate + evidence + entry 的**超集**
- [ ] 所有 `requested_features` 都已在 `feature_dependencies.yaml` 中注册
