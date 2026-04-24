# 统一建模方法论 (Meta-Algorithm)

## 设计哲学

**因果假设 → 构建语义特征 → 用统计验证有效性。**

系统从结构表象 (SR breakout) → 博弈结果 (BPC/ME/FER) → 市场机制 (OI+FR+LV) 逐层进化。
每一层的认知进步都遵循同一原则: ML 发现规律，规则实盘执行。

> ML → 发现规律  
> 规则 → 实盘执行  
> 因果是方向，统计是裁判。

---

## 核心思想

所有 per-trade 决策模块共用同一套「特征发现 → 统计验证」流程，只在 **标签、数据范围、输出格式** 三个维度做模块级参数化。

```
   ┌─────────────────────────────────────────────────────┐
   │                Meta-Algorithm (固定)                 │
   │                                                     │
   │  Step 1  LightGBM 建模 (label 由模块决定)           │
   │  Step 2  SHAP importance ∩ Gain importance → top N  │
   │  Step 3  SHAP interaction → 交互对 → 2D Lift Surface│
   │  Step 4  统计验证 (lift / robustness / time-fold)   │
   │  Step 5  规则/候选输出 (格式由模块决定)             │
   │                                                     │
   └────────┬──────────┬──────────┬──────────┬───────────┘
            │          │          │          │
       Prefilter     Gate    Evidence  Entry Filter
       (archetype   (尾部    (交易     (入场
        是否成立)    风险)    多好)     时机)
```

---

## Step-by-Step 详解

### Step 1: LightGBM 建模

在模块对应的数据集上训练一棵 LightGBM，获得 booster 对象。

- 模型不直接上线，仅作为「特征重要性提取器」
- 目标：产出 Gain importance + SHAP importance

### Step 2: SHAP ∩ Gain 特征发现

共用函数: `_compute_shap_gain_features()` (位于 `scripts/export_lightgbm_rules_to_readme.py`)

| 方法       | 本质                       | 偏向                           | Gate 场景                                        |
| ---------- | -------------------------- | ------------------------------ | ------------------------------------------------ |
| Gain       | 训练时分裂降 loss 的统计   | 高频 split、连续特征、噪声特征 | 选出 trend_slope, volatility 等高频特征          |
| SHAP       | 预测时每个特征对输出的贡献 | 真实预测影响                   | 能选出 crowding, direction_flip 等低频高冲击特征 |
| **∩ 交集** | 两者 top-10 取交集         | 既影响预测又频繁使用           | **最稳定候选**                                   |

```
Fallback: 交集 < 3 → 用 SHAP 排序; SHAP 失败 → 用 Gain
```

参数:
| 参数                   | 含义                      | 默认值          |
| ---------------------- | ------------------------- | --------------- |
| `top_n`                | 输出候选特征数            | 模块决定 (8~15) |
| `compute_interactions` | 是否计算 SHAP interaction | True            |

返回: `(top_features, shap_importance_map, interaction_pairs)`

#### Walk-Forward SHAP 稳定性筛选 (Step 2.5)

在管线中位于 Step 2 (Prepare) 和 Step 3 (Prefilter) 之间，用于自动裁剪不稳定特征:

```
1. 按时间切 N 个 fold (默认 4 个半年窗口)
2. 每个 fold: 训练 LightGBM → TreeExplainer (sample=2000) → mean |SHAP| 排名
3. 聚合: 稳定特征 = 在 >= 75% 的 fold 中排名 top-K (默认 K=20)
4. 输出: shap_stable_features.json → 写回 features_gate.yaml / features_evidence.yaml
```

安全约束:
- 稳定特征 >= 8 个才裁剪 (否则 fallback 跳过)
- `atr_f` 永远保留 (执行层 SL/TP 必需)
- 裁剪前后都跑 backtest 确认不降 Sharpe

脚本: `scripts/shap_feature_selection.py`，管线参数: `--skip-shap` 可跳过。

### Step 3: SHAP Interaction → 2D/3D Surface

对 Step 2 返回的 `interaction_pairs` (top 交互对)，做二维精细网格扫描:

```
1. 取交互对 (feat_a, feat_b)
2. 分位数 bin (Q10~Q90, 10 bins) → ~100 cells
3. 每个 cell 计算 lift = cell_bad_rate / overall_bad_rate
4. 找 lift > 1.3 的高危区域
5. 贪心合并相邻高 lift cells → 矩形区域 → 复合规则
6. Robustness: time-fold stability × 0.6 + cross-sample stability × 0.4
```

#### 2D/3D Surface 是 target-agnostic 的

Surface 的本质是数据统计: `Z = E[target | featureA, featureB]`

target 可以是任何东西:

| target                                         | surface 含义             | 对应模块     |
| ---------------------------------------------- | ------------------------ | ------------ |
| `P(tail_loss \| ATR, funding)`                 | tail risk surface        | Gate         |
| `P(success \| structure_f1, structure_f2)`     | archetype 成功率 surface | Prefilter    |
| `E[RR \| evidence_f1, evidence_f2]`            | 收益 surface             | Evidence     |
| `P(entry_success \| pullback_depth, momentum)` | entry quality surface    | Entry Filter |

每个模块只需换 target (标签)，同一套 surface 扫描方法自动适用。

#### Lift Surface 算法

```
        volume_compression
        ↑
   3.0  |      ███████      ← 高 lift 区域 = Gate deny 区域
   2.0  |   ██████████
   1.0  | ████████████
        +--------------
          pullback_depth →
```

1. SHAP interaction 选 top 5 特征对
2. 每对特征: 10 个分位数 bin → ~100 cells
3. 每个 cell 计算 lift = cell_bad_rate / overall_bad_rate
4. 筛选 lift > 1.3 的 hot cells → 贪心合并 → 矩形区域
5. 区域 → 规则条件 (最多 2 个条件, 避免过拟合)
6. Robustness 验证: time-fold + cross-sample stability

比 tree split 强: tree split 只能沿贪心路径找单条件; Lift Surface 全局扫描 2D 空间，能发现 non-linear alpha region。

### Step 4: 统计验证

对 Step 2 (单特征) + Step 3 (交互对) 产出的候选做统计验证:

| 指标            | 计算方式                                       | 用途                            | 配置文件                                                           |
| --------------- | ---------------------------------------------- | ------------------------------- | ------------------------------------------------------------------ |
| lift            | deny组 bad_rate / 整体 bad_rate                | 区分力 (>1.0 单, >1.3 复合)     | `kpi_gates/{prefilter,gate,entry_filter}_layer.yaml`               |
| effect_size     | mean_rr(allow) - mean_rr(deny)                 | 经济意义 (>0.10 单, >0.15 复合) | `kpi_gates/{prefilter,gate,entry_filter}_layer.yaml`               |
| robustness      | time-fold stability × 0.6 + cross-sample × 0.4 | 泛化能力 (>0.4 单, >0.35 复合)  | `kpi_gates/{prefilter,gate,entry_filter}_layer.yaml`               |
| gate_score      | tail_capture - good_deny_rate (Youden's J)     | Gate 专用: 净信息量             | `kpi_gates/gate_layer.yaml` → `thresholds.min_gate_score`          |
| bad_suppression | P(low_score\|bad) - P(low_score\|good)         | Evidence 专用: 坏交易压制力     | `kpi_gates/evidence_layer.yaml` → `validation.min_bad_suppression` |
| snotio          | mean(R-multiples) + plateau CV + z-test        | Entry Filter 专用: 执行质量     | `kpi_gates/entry_filter_layer.yaml` → `plateau.snotio_cv_max`      |

> **配置路径总览**: 所有统计门槛集中在 `config/kpi_gates/` 下, 脚本运行时自动读取。
> 修改门槛无需改代码, 只需编辑对应 YAML。

### Step 5: 规则/候选输出

每个模块的文件流: **候选/草稿 → 优化 → --promote → archetypes/**

| Step         | 候选/草稿位置                                     | --promote 产出                     | 脚本                                          |
| ------------ | ------------------------------------------------- | ---------------------------------- | --------------------------------------------- |
| Prefilter    | `{strategy}/features_prefilter.yaml` (特征池)     | `archetypes/prefilter.yaml`        | `analyze_archetype_feature_stratification.py` |
| Direction    | `{strategy}/direction.yaml` (规则定义)            | `archetypes/direction.yaml` (copy) | `direction_strict_validation.py`              |
| Gate         | `{strategy}/gate_draft.yaml` (训练产出)           | `archetypes/gate.yaml`             | `optimize_gate_unified.py`                    |
| Evidence     | `results/.../evidence_candidates.yaml` (训练产出) | `archetypes/evidence.yaml`         | `optimize_evidence_plateau.py`                |
| Entry Filter | 无候选 (SHAP∩Gain 内联发现)                       | `archetypes/entry_filters.yaml`    | `optimize_entry_filter_plateau.py`            |
| Execution    | 无候选 (Grid Search 内联)                         | `archetypes/execution.yaml`        | `optimize_execution_grid.py`                  |

**目录约定**:
- `config/strategies/{strategy}/` — 候选特征、标签、模型配置 (训练输入)
- `config/strategies/{strategy}/archetypes/` — 生产配置 (实盘读取)
- `results/.../{strategy}/` — 训练产物 (predictions/logs_gated/candidates)

**数据流规范** (见 `research_pipeline.yaml` `data_flow` 段):
- Gate 之前: 读 `predictions.parquet`
- Gate 之后 (Evidence/Entry Filter/Execution/Backtest): 读 `logs_gated.parquet`

---

## 模块实例化

### 总览

| 维度         | Prefilter                        | Gate                                                | Evidence                        | Entry Filter                        |
| ------------ | -------------------------------- | --------------------------------------------------- | ------------------------------- | ----------------------------------- |
| **问的问题** | archetype 是否成立?              | 尾部风险高不高?                                     | 这笔交易有多好?                 | 现在入场时机好不好?                 |
| **标签**     | `success_no_rr_extreme < 0.5`    | `success_no_rr_extreme < 0.5` 或 `forward_rr < Q30` | `forward_rr` (连续)             | exec R-multiple (二值化 or snotio)  |
| **数据范围** | ALL `features_labeled`           | prefiltered data                                    | gate-passed data                | gate-passed + direction             |
| **top_n**    | 8                                | 8                                                   | 10~15                           | 8                                   |
| **选择标准** | bad_rate_diff + holdout          | Gate Score (Youden's J > 0)                         | bad_suppression                 | snotio 显著性 (z-test p<0.05)       |
| **输出格式** | AND deny rules                   | AND deny rules                                      | 候选特征 + bins                 | OR timing conditions                |
| **输出文件** | `prefilter.yaml`                 | `gate.yaml`                                         | `evidence_candidates.yaml`      | `entry_filters.yaml`                |
| **实现状态** | ✅ 已完成                         | ✅ 已完成                                            | ✅ 已完成                        | ✅ 已完成                            |
| **KPI 配置** | `kpi_gates/prefilter_layer.yaml` | `kpi_gates/gate_layer.yaml`                         | `kpi_gates/evidence_layer.yaml` | `kpi_gates/entry_filter_layer.yaml` |

### Prefilter

**职责**: 定义 archetype 的语义边界。不是"优化 CV 的技巧"，而是"这个 bar 满不满足 archetype 的因果前提"。

**标签定义**:
- `success_no_rr_extreme`: 1 = 好交易, 0 = 坏交易 (二值)
- bad = `success_no_rr_extreme < 0.5`
- 在 ALL `features_labeled.parquet` 上计算 (不经过任何预过滤)

**数据范围**: ALL data — Prefilter 是第一道关，必须在全量数据上训练，不能依赖自身过滤后的子集

**特征范围**: archetype 专属特征 (非全量通用特征)

Prefilter 回答的是"这个 bar 是不是 BPC/ME/FER"，如果用通用特征 (RSI, MACD)，所有 archetype 会选到相同特征，失去区分度。每个 archetype 需要独立的 `features_prefilter.yaml`:

```yaml
# config/strategies/bpc/features_prefilter.yaml
feature_pipeline:
  requested_features:
    - bpc_volume_compression_pct_f    # BPC 专属
    - bpc_bb_compression_f            # BPC 专属
    - bpc_cvd_z_f                     # BPC 专属
    - dual_compression_f              # BPC 交叉
    - funding_compression_score_f     # 场景语义
```

**SHAP∩Gain 参数**: `top_n=8, compute_interactions=True`

**规则输出**: AND 前置条件

```yaml
rules:
  - feature: bpc_volume_compression_pct
    operator: "<="
    value: 0.1503
    rationale: "统计验证 lift=X.XX, robustness=X.XX, holdout验证通过"
```

**当前问题**: 使用分位数 bad_rate_diff 方法 (无 ML, 无 SHAP, 无 holdout) → 过拟合

**改进方向**:
1. 在 ALL data 上训 LightGBM → SHAP∩Gain 发现边界特征
2. 统计验证时加入 train/holdout split
3. Prefilter 模型独立于 Gate 模型 (数据范围不同: ALL vs prefiltered)

### Gate

**职责**: 在 prefiltered 数据上判断「这笔交易尾部风险是否过高」。高风险 → deny。

#### 方案演进

| 版本   | 方法                  | 失败原因                                           |
| ------ | --------------------- | -------------------------------------------------- |
| v1     | tree_split 提取分裂点 | 同特征多阈值→矛盾区间; 10条AND→veto爆炸            |
| v2     | 多seed看稳定          | 多seed取交集淘汰了有效特征，不是无效特征           |
| v3     | SHAP+imodels蒸馏      | teacher pred全在0.52-0.59，蒸馏质量不足            |
| **v4** | **统计验证法**        | SHAP∩Gain + SHAP interaction + 统计验证 (最终方案) |

v1~v3 都在"从模型里抄规则"。正确思路: **SHAP∩Gain 发现候选 → 统计验证规则 → YAML 落地**。

#### 标签定义

- 优先用 `success_no_rr_extreme`: bad = value < 0.5
- Fallback `forward_rr`: bad = value < Q30 (底部30分位)

#### Gate Score (Youden's J)

```
tail_capture   = P(deny|bad)  → 规则拦截了多少比例的坏交易
good_deny_rate = P(deny|good) → 规则误杀了多少比例的好交易
gate_score     = tail_capture - good_deny_rate
```

| 场景     | tail_cap | good_deny | gate_score | 判断     |
| -------- | -------- | --------- | ---------- | -------- |
| 优秀     | 50%      | 5%        | 0.45       | 精准拦截 |
| 良好     | 30%      | 10%       | 0.20       | 可用     |
| 无区分力 | 20%      | 20%       | 0.00       | 淘汰     |
| 误杀型   | 10%      | 30%       | -0.20      | 淘汰     |

规则选择: gate_score > 0 保留，按 gate_score 降序，条数由数据决定 (受 max_rules 上限约束)。

#### SHAP Interaction 找复合规则

单看 SHAP importance 只能发现重要单特征。SHAP interaction 能发现交互效应:

```
例: sma_200_position  SHAP=0.08 (一般)
    dir_flip_count    SHAP=0.06 (一般)
    interaction       = 0.21 (极强!)
→ 价格在均线附近 AND 方向频繁翻转 → breakout 极易失败
```

经验: **80% 稳定 Gate 都是 2D interaction**。复合规则最多 2 个条件 (1=不够强, 2=最优, 3+=过拟合)。

#### 规则输出

```yaml
hard_gates:
  - id: gate_feature_name
    tag: HARD_FEATURE_NAME
    phase: hard_gate
    when:
      feature_name: {value_gt: 0.1234}
    then: {action: deny}
```

**实现状态**: 已完成 (`_generate_gate_rules_statistical()`)

### Evidence

**职责**: 在 gate-passed 数据上评估「这笔交易有多好」。

Evidence 最终定位:
```
Gate → Evidence(入场门槛 + 仓位控制) → 开仓

1. 入场门槛: evidence < min_score → 拒绝开仓 (极端不利不让进)
2. 仓位控制: size = f(evidence_score) (根据条件好坏调整仓位)
```

Evidence 不回答: "这个信号是否比已有仓位更值得持有"（slot 竞争已删除）。

#### 标签定义

- `forward_rr` (连续 R-multiple)
- bad 定义同 Gate (用于 preliminary bad_suppression)

#### Preliminary Bad Suppression

```
1. 用 good 样本的分位数 (Q20/Q40/Q60/Q80) 作参考分布
2. 将特征值映射为 percentile rank (0.1~0.9 五档)
3. 两个方向都试:
   - higher_is_better: bs = P(low_pct|bad) - P(low_pct|good)
   - lower_is_better:  bs = P(high_pct|bad) - P(high_pct|good)
4. 取 bad_suppression 更大的方向
```

#### 规则输出

```yaml
evidence_candidates:
  - rank: 1
    feature: feature_name
    discovery_method: shap_gain
    shap_importance: 0.1234
    prelim_bad_suppression: 0.15
    direction_hint: higher_is_better
    quantile_mapping:
      bins: [0.2, 0.4, 0.6, 0.8]
      labels: [suppress, downweight, neutral, favor, amplify]
    interaction_partners:
      - feature: partner_feature
        interaction_score: 0.05
```

#### Evidence Score 三组件结构 (升级路径)

单模型 `score = model(X)` 需要同时学习 alpha/regime/risk 三种任务，困难且不稳定。工业标准拆分:

```
Evidence Score = α × model_score + β × regime_score - γ × risk_score
```

| 组件   | 预测目标         | 典型特征                              | 是否 ML  |
| ------ | ---------------- | ------------------------------------- | -------- |
| Model  | E[RR \| signal]  | structure, momentum, orderflow        | 是       |
| Regime | 策略当前是否有效 | volatility_regime, trend_strength     | 通常不是 |
| Risk   | tail risk / 拥挤 | funding_crowding, liquidation, spread | 通常不是 |

核心原则: **Model finds alpha, Rules protect capital。**

升级路径:
```
阶段 1 (当前): 手工特征 → quantile bin → score (feature independent)
阶段 2: SHAP interaction + Lift Surface → 发现 interaction → 组合 bin
阶段 3: Evidence LightGBM → score → percentile bin (自动学习 interaction)
阶段 4: Model + Regime + Risk → 合并 score → percentile bin
```

#### Score Monotonicity (质量判断)

Evidence 系统好坏看各 level 的 Sharpe 是否单调递增:

```
amplify    → Sharpe 最高
favor      → Sharpe 次高
neutral    → Sharpe 中等
downweight → Sharpe 低
suppress   → Sharpe 最低 (或负)
```

满足 monotonic Sharpe = Evidence 有效。违反单调性 = 某些 bin 映射有误。

**实现状态**: 已完成 (`_generate_evidence_candidates_yaml()`)

### Entry Filter

**职责**: 在 gate-passed + direction 确定后判断「现在是不是好的入场时机」。

**标签定义** (待设计):
- 候选 1: exec R-multiple > 0 = good entry (二值)
- 候选 2: 进场后 N 根 K 线最大回撤 < X (执行质量)

**选择标准**: snotio 显著性

```
snotio = mean(R-multiples) 每笔的平均风险调整收益

准入分级 (OR 组合):
  Tier A (PLATEAU): plateau + z-test 显著 (p<0.05)
  Tier B (SNOTIO):  snotio > baseline + z-test 显著

Plateau 判定: 滑动窗口(size=5)
  - snotio CV < 0.3 (收益稳定性)
  - Trades CV < 0.4 (执行节奏稳定性)
  - recommended = plateau 偏宽容侧
```

**当前问题**: 仅扫描 4 个手工定义的 `shallow_pullback_*` 条件，搜索空间太小 → 全部不显著

**改进方向**:
1. SHAP 在 50+ 特征上自动发现 entry timing 特征
2. 设计 entry quality 专属标签
3. 保留 plateau scanning + snotio 作为 Step 4 验证手段

---

## 职责分层: Gate vs Evidence

两者预测目标完全不同，不能合并:

| 模块     | 预测目标     | 作用         | 输出       |
| -------- | ------------ | ------------ | ---------- |
| Gate     | P(tail loss) | 删除坏交易   | deny (0/1) |
| Evidence | E(return)    | 决定仓位大小 | score      |

仓位公式:
```python
if gate_deny:
    pass  # Hard Gate: 不交易
else:
    size = evidence_score  # Evidence 直接决定仓位
```

为什么不能只用 Evidence: Evidence 对 tail risk 不敏感。

```
交易A: E(return) = 0.5R
  分布: 80% → +0.6R, 20% → -2R
Evidence 喜欢 (均值正) → Gate 拒绝 (20% 概率 -2R = tail risk)
```

Gate = tail risk protection, Evidence = alpha ranking + position sizing。

Soft Filter 不再独立存在 (避免 double counting: soft filter 特征已在 Evidence 训练集中，再惩罚 = `model_penalty × rule_penalty`，过度惩罚)。

---

## PCM 为什么不适配

PCM (Portfolio Condition Monitor) 是 **control 问题**，不是 prediction 问题:

| 维度   | Per-trade 模块 (prediction) | PCM (control)             |
| ------ | --------------------------- | ------------------------- |
| 粒度   | 每笔交易                    | 整个市场环境 (持续数周)   |
| 样本量 | 数千~数万笔                 | 数十个 regime 切换        |
| 方法   | LightGBM + SHAP → 统计验证  | grid search / simulation  |
| 输出   | deny/allow/score            | priority + position_scale |

PCM 要解决 size/tp/trail/scale_in 等 **policy optimization**，不适合 SHAP∩Gain 方法。保持 validate + snapshot 流程。

---

## 共用代码架构

```
scripts/export_lightgbm_rules_to_readme.py
├── _compute_shap_gain_features()          ← Gate/Evidence 共用 (已实现)
│   ├── Gain importance (booster.feature_importance)
│   ├── SHAP importance (TreeExplainer, 2000 samples)
│   ├── SHAP ∩ Gain 交集 (top10 ∩ top10, fallback: SHAP-only)
│   └── SHAP interaction values (500 samples, top 10 pairs)
│
├── _generate_gate_rules_statistical()     ← Gate Step 3-5 (已实现)
│   ├── 分位数 threshold sweep (Q15~Q85)
│   ├── lift + effect_size + robustness
│   ├── 相关性剪枝 (|corr| > 0.80)
│   ├── Lift Surface 2D (SHAP interaction pairs)
│   └── Gate Score (Youden's J) 选择
│
└── _generate_evidence_candidates_yaml()   ← Evidence Step 3-5 (已实现)
    ├── preliminary bad_suppression (quantile-based)
    ├── direction auto-detection
    ├── interaction partners
    └── YAML 输出 (→ optimize_evidence_plateau.py 优化)

scripts/shap_feature_selection.py           ← Step 2.5 SHAP 稳定性筛选
scripts/analyze_archetype_feature_stratification.py  ← Prefilter (待改造)
scripts/optimize_entry_filter_plateau.py             ← Entry Filter (待改造)
```

后续 Prefilter 和 Entry Filter 改造时，复用 `_compute_shap_gain_features()`。

---

## 管线中的位置

```
Step 1:   Feature Store
Step 2:   Prepare (features_labeled.parquet)
Step 2.5: SHAP Feature Selection (Walk-Forward 稳定性筛选)
Step 3:   Prefilter  ← meta-algorithm
Step 4:   Direction
Step 5:   Gate Train ← meta-algorithm
Step 6:   Evidence   ← meta-algorithm
Step 7:   Entry Filter ← meta-algorithm
Step 8:   Execution
Step 9:   Backtest
```

---

## 实施优先级

| 优先级 | 模块             | 状态       | 理由                                                               |
| ------ | ---------------- | ---------- | ------------------------------------------------------------------ |
| P0     | Gate             | 已完成     | `_generate_gate_rules_statistical()`                               |
| P0     | Evidence         | 已完成     | `_generate_evidence_candidates_yaml()`                             |
| P1     | **Prefilter**    | **待改造** | 当前 bad_rate_diff 无 holdout → 过拟合; 直接影响 Gate 训练数据质量 |
| P2     | **Entry Filter** | **待改造** | 当前全部不显著 (搜索空间太小); 非核心路径                          |
| —      | PCM              | 不适配     | Regime 级别，保持 validate + snapshot                              |

---

## 方法论约束

1. **每个模块用自己的标签**: 标签必须精确回答模块的问题 (archetype成立? 尾部风险? 执行质量?)
2. **数据范围逐级收窄**: ALL → prefiltered → gate-passed → direction-confirmed
3. **模型独立**: 每个模块可以训自己的 LightGBM (标签/数据不同 → 重要性排序不同)
4. **统计验证必须包含 holdout**: 不做 holdout 验证的规则不可上线
5. **共用函数不可分叉**: `_compute_shap_gain_features()` 只有一份代码，所有模块调用同一个
6. **Prefilter 定义语义边界**: 不是优化 CV 的技巧，而是定义 archetype 因果前提的机制
7. **Prefilter 特征 archetype 专属**: 每策略独立 `features_prefilter.yaml`，仅含该 archetype 的专属 + 交叉 + 场景语义特征
8. **向量回测 vs 事件回测分工**: 向量回测=快速验证训练效果(研究侧); 事件回测=上线前把关+验证线上交易符合算法(执行侧); 不追求数值一致

## 我最喜欢你设计的一点

你写的这一句非常好：

因果是方向，统计是裁判。

这句话其实是 量化系统最核心的方法论。

完整版本应该是：

因果 → 生成特征
ML → 发现规律
统计 → 验证稳定
规则 → 实盘执行

你现在基本就是这套。
