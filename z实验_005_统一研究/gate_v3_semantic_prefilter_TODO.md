# Gate v3: 语义预筛选 + 数学特征 + 方向分离

> 创建时间: 2026-02-20
> 背景: Gate v1/v2 训练在全量 bar 上效果差 (ME CV=0.018, LV CV=0.018)
> 根因: 训练样本包含大量 archetype 不相关的 bar，专属特征被通用特征淹没

---

## 核心问题诊断

### 特征重要性分析 (2026-02-20)

| 策略 | CV | 专属特征重要性 | 通用特征重要性 | 结论 |
|---|---|---|---|---|
| BPC (4H) | **0.079** | bpc_dir_consistency(484), bpc_score_pullback(43) | fp_max_imbalance(1050), evt_es_99_right(482) | 专属+通用都有用 |
| ME (1H) | 0.018 | me_atr_pct(92), me_vol_regime(33), **其余全=0** | shd_pct(537), sma_200(471), fp_val(448) | 专属完全无用 |
| FER (4H) | 0.045 | fer_trapped_longs/shorts (经 optimizer 验证有效) | 通用特征主导 | 仅 trapped 有效 |
| LV (15min) | 0.018 | 无专属特征 | trade_cluster_max_buy_run(1276!), fp_hvn(442) | 通用统治 |

### 根因分析

ME v2 训练集: 26,058 bars（全量），其中：
- 大部分 bar 没有动量扩张（me_atr_pct < 0.4）
- 在这些 bar 上，me_accel_5k / me_cvd_alignment 等特征是**纯噪声**
- LightGBM 自然忽略噪声特征 → importance=0
- 模型退化为"通用市场环境 vs 踩坑"分类器，失去 archetype 语义

**类比**：这相当于把所有年龄段的人混在一起训练"老年病预测模型"——年龄<30 的样本只是噪声。

---

## TODO Phase A: 语义预筛选 (Semantic Pre-filter)

### A.1 核心思想

```
训练样本 = 只保留 archetype 语义匹配的 bar
目标: 让模型只回答 "这个 archetype 信号会不会失败？"
而不是 "这个随机时刻会不会亏钱？"
```

### A.2 预筛选条件定义

每个 archetype 的预筛选条件来自其 **guardrail 规则**（gate.yaml 中已定义的策略前提）：

#### ME (动量扩张)
```yaml
# 来源: config/strategies/me/archetypes/gate.yaml → guardrails
# 语义: "市场正在扩张" 才算 ME 相关样本
pre_filter:
  me_atr_pct:         # Energy > 0.40 (guardrail: me_guardrail_energy_insufficient)
    value_gte: 0.40
  me_cvd_alignment:   # Flow > 0.40 (guardrail: me_guardrail_flow_inconsistent)
    value_gte: 0.40
  me_volume_surge:    # Volume > 0.30 (guardrail: me_guardrail_volume_insufficient)
    value_gte: 0.30
```

**预期效果**:
- 样本从 ~26K 降到 ~5-8K (只保留"有扩张"的 bar)
- me_accel_5k, me_cvd_strength 等特征重新获得区分力
- 模型回答: "扩张已发生，但会不会踩坑？"

#### FER (失败力竭反转)
```yaml
# 来源: config/strategies/fer/archetypes/gate.yaml → hard_gates (frozen)
# 语义: "有人被套住了" 才算 FER 相关样本
pre_filter:
  fer_trapped_longs_score:   # > 2.75 (hard_gate: fer_hg_no_trapped_longs)
    value_gte: 2.75
  # 或
  fer_trapped_shorts_score:  # > 3.75 (hard_gate: fer_hg_no_trapped_shorts)
    value_gte: 3.75
  # 注意: 两者满足其一即可 (OR 逻辑)
```

**预期效果**:
- 仅保留"有 trapped 燃料"的 bar
- 模型聚焦回答: "有人被套了，但反转会成功吗？"

#### LV (清算脆弱性)
```yaml
# 来源: config/strategies/lv/archetypes/gate.yaml → hard_gates
# 语义: "杠杆已积累" 才算 LV 相关样本
pre_filter:
  oi_zscore:                    # > -1.0 (hard_gate: gate_oi_too_low)
    value_gte: 0.5              # 放宽一点: 不仅"不低"而且"偏高"
  funding_rate_abs_zscore_50:   # > 0.0 (hard_gate: gate_funding_neutral)
    value_gte: 0.5              # 同理: Funding 有一定偏离
```

**预期效果**:
- 仅保留"杠杆有积累"的 bar
- 模型聚焦: "杠杆积累了，但清算会真正发生吗？"

#### BPC (压缩突破)
```yaml
# 来源: config/strategies/bpc/archetypes/gate.yaml → guardrails
# 语义: "有压缩结构" 才算 BPC 相关样本
pre_filter:
  bpc_volume_compression_pct:  # > 0.30 (guardrail: guardrail_bpc_volume_compression_missing)
    value_gte: 0.30
  price_position:              # < 0.90 (guardrail: guardrail_price_position_extreme_high)
    value_lte: 0.90
```

**预期效果**:
- BPC 已有 CV=0.079，预筛选可能进一步提升

### A.3 Guardrail 特征的完整来源链路

四策略的预筛选条件不是拍脑袋得出的，而是经过三步验证的完整链路：

```
Step 1: 启发式因果设计 → 基于 archetype 语义写特征计算代码
  ↓
Step 2: 分位数分层验证 → 用百分位阈值切数据, 对比 bad rate / median RR
  ↓
Step 3: Plateau 稳健性搜索 → 用 optimize_gate_unified.py 找稳定阈值区间
  ↓
输出: gate.yaml 中的 guardrail / hard_gate 规则
  ↓
复用: 这些规则的条件直接作为训练时的 pre_filter
```

---

#### Step 1: 启发式因果设计（特征从哪来）

每个 archetype 的专属特征基于**因果机制**手工设计，不是从数据中挖掘的统计相关性：

| Archetype | 因果公式 | 特征计算模块 | 核心特征及物理含义 |
|---|---|---|---|
| **ME** | `Energy × Acceleration × Participation` | `momentum_expansion_features.py` | `me_atr_pct`=波动百分位(能量), `me_cvd_alignment`=CVD方向对齐(参与), `me_volume_surge`=成交量爆发百分位(参与) |
| **FER** | `Trapped × Drawdown × CVD` | `fer_features.py` | `fer_trapped_longs_score`=多头被套(drawdown×CVD百分位), `fer_trapped_shorts_score`=空头被套(反弹×反向CVD) |
| **BPC** | `Compression → Breakout → Continuation` | `bpc_features.py` | `bpc_volume_compression_pct`=成交量压缩百分位, `bpc_dir_consistency_long`=50bar方向一致性 |
| **LV** | `Leverage × Funding × OI` | `open_interest_features.py` | `oi_zscore`=OI z-score(杠杆积累), `funding_rate_abs_zscore_50`=资金费率偏离(单边偏向) |

**设计原则** (参考 `archetype特征语义约束规范.md`):
- 每个特征必须有因果解释："谁在亏钱？为什么必须亏？"
- 不允许跨 archetype 引入核心特征（防止统计噪声过拟合）
- 特征只描述"环境是否允许"，不描述"方向"（方向由 Direction 层独立负责）

**ME 特征算法示例**:
```python
# Energy: ATR 在过去 100 根 K 线中的百分位
me_atr_pct = rolling_percentile(atr, window=100)  # [0,1]

# Participation: CVD 方向与价格方向的一致性
cvd_dir = sign(cvd_change_5)
price_dir = sign(close.diff())
me_cvd_alignment = (price_dir * cvd_dir + 1) / 2  # [0,1]

# Participation: 成交量 / MA 的百分位
me_volume_surge = rolling_percentile(volume / vol_ma20, window=100)  # [0,1]
```

**FER trapped 算法示例**:
```python
# 多头被套 = 从高点回撤(ATR归一化) × CVD在高位强度(多头在冲)
drawdown_from_high = clip((rolling_high - close) / atr, 0, ∞)
cvd_lookback_pct = rank(cvd_change_lookback, window=lookback*4)  # [0,1]
fer_trapped_longs_score = clip(drawdown_from_high * cvd_lookback_pct * 2, 0, 5)
```

---

#### Step 2: 分位数分层验证（特征有没有区分力）

**验证脚本**: `scripts/analyze_archetype_feature_stratification.py`
**实验报告**: `z实验_005_统一研究/gate_semantic_prefilter_design.md`

**算法**: 百分位阈值分层 (Percentile Stratification)
```
对每个 archetype 专属特征:
  1. 取 Feature Store 全量数据 (如 ME: 25674 rows, 6 symbols, 1H)
  2. 按 P5 / P10 / P80 / P90 / P95 阈值切分数据为两组
  3. 计算两组的:
     - bad rate = extreme_rr_failure 占比 (forward_rr < -0.8R)
     - median RR = 中位数风险回报
  4. 差异越大 → 该特征在此阈值处有区分力
```

**实验结果摘要**:

| 策略 | 有效特征 | 最强信号 | 差异 | 结论 |
|---|---|---|---|---|
| BPC | 5+ 双端有效 | `impulse_return_atr` P90: bad 36.7% vs 49.6% | **12.9%** | 特征多且强，Gate 正常工作 |
| FER | **仅 trapped (2个)** | `trapped_longs_score` P90: bad 37.4% vs 49.4% | **12.0%** | 10个噪声淹没2个信号 → Gate全放行 |
| ME | 多个反信号(high=bad) | `accel_5k` P5: bad 55.0% vs 44.2% | **10.8%** | 反信号+共线 → Gate极弱 |
| LV | 待评估 | — | — | 15min 天然正交 |

---

#### Step 3: Plateau 稳健性搜索（阈值选在哪里）

**验证脚本**: `scripts/optimize_gate_unified.py`

**核心算法: Lift + Stable Plateau + Robustness Score**

```
1. Threshold Scan（阈值扫描）:
   对特征 [P5, P95] 范围，以步长 0.05 逐点扫描
   每个阈值 θ 计算:
     pass_rate_good = good 样本中通过率
     pass_rate_bad  = bad 样本中通过率
     Lift = pass_rate_good / pass_rate_bad - 1
     (Lift > 0 表示该阈值选择性保留 good, 淘汰 bad)

2. Stable Plateau Detection（稳定平台搜索）:
   找连续阈值区间，满足:
     - Lift 波动 < ±30% of anchor
     - PassRate 波动 < ±15%
     - Coverage 波动 < ±10%
     - 区间宽度 ≥ min_plateau_width (默认 0.05)
   语义: 阈值轻微偏移不会炸

3. Robustness Score（稳健性评分）:
   在 plateau 内选最佳点:
     - param_stability: 参数扰动稳定性
     - temporal_stability: 时间稳定性（不同时期表现一致）
     - overall_score: 综合评分
```

**各策略优化结果**:

| 策略 | 规则 | 阈值 | Lift | PassRate | 来源 |
|---|---|---|---|---|---|
| BPC | `bpc_dir_consistency_long > 0.55` → deny | 0.55 | 11.4% | 74.3% | plateau_mid |
| BPC | `wpt_ignition_score > 0.209` → deny | 0.209 | 21.2% | 35.8% | plateau_mid |
| BPC | `wpt_exhaustion_score < 0.309` → deny | 0.309 | 562% | 44.4% | plateau_mid |
| FER | `fer_trapped_longs_score < 2.75` → deny | 2.75 | 31.0% | 30.3% | frozen (stable plateau) |
| FER | `fer_trapped_shorts_score < 3.75` → deny | 3.75 | 13.8% | 23.0% | frozen (stable plateau) |
| ME | `me_atr_pct < 0.65` → deny | 0.65 | 18.1% | 29.9% | frozen (stable plateau) |

---

#### 从 guardrail/hard_gate 到 pre_filter 的复用逻辑

```
guardrail 定义 ≡ "archetype 的必要前提条件"
  ↓
如果 bar 不满足 guardrail → 该 bar 不是 archetype 的有效场景
  ↓  
训练时包含这些 bar = 引入噪声
  ↓
pre_filter = guardrail 条件 → 只保留有效场景的 bar
```

**关键区分**:
- **Guardrail**: 实盘执行时阻止不合理的信号（运行时保护）
- **Pre-filter**: 训练时排除不相关的样本（数据质量控制）
- 两者使用**相同的条件**，但作用在不同阶段

### A.4 与旧数据划分的对比：从"拍脑袋"到"因果语义驱动"

**旧方案（v1/v2）：无语义筛选**

当前所有 `labels_rr_extreme.yaml` 的 `filters` 段只做一件事：

```yaml
# 旧方案: 只剥离 NaN，不做语义筛选
filters:
  - column: success_no_rr_extreme
    notna: true     # ← 仅移除标签为空的行，全量 bar 都参与训练
```

结果：ME 训练集 26,058 bars，其中大部分 bar 没有动量扩张 → 专属特征 importance=0。

**新方案（v3）：基于 archetype 因果语义筛选**

```yaml
# 新方案: archetype 语义划分 + 标签 NaN 剥离
filters:
  # 🆕 语义预筛选: 只在 ME 相关的 bar 上训练
  - column: me_atr_pct           # Energy > 0.40
    min: 0.40
  - column: me_cvd_alignment     # Flow > 0.40
    min: 0.40
  - column: me_volume_surge      # Volume > 0.30
    min: 0.30
  # 原有标签过滤
  - column: success_no_rr_extreme
    notna: true
```

**关键发现：`apply_filters()` 已原生支持 `min`/`max` 操作符**，不需要新建 `pre_filter` 机制：

```python
# train_strategy_pipeline.py 已有代码 (L616-L632)
def apply_filters(df, filters):
    for filt in filters:
        if filt.get("notna"): result = result[result[column].notna()]
        if "min" in filt:     result = result[result[column] >= filt["min"]]   # ← 已支持!
        if "max" in filt:     result = result[result[column] <= filt["max"]]   # ← 已支持!
    return result
```

→ **零代码修改**，只需改 YAML 配置即可实现语义预筛选。

**对比总结**:

| 维度 | 旧方案 (v1/v2) | 新方案 (v3 semantic pre-filter) |
|---|---|---|
| 数据划分依据 | 无（全量 bar） | archetype 因果语义 |
| 划分算法 | 拍脑袋 / 仅 notna | 启发式设计 → 分位数验证 → Plateau 稳健性 |
| 训练样本数 | ~26K (ME) | 预计 5-8K (只保留扩张 bar) |
| archetype 语义 | 无（模型自学） | 显式定义"哪些 bar 属于此 archetype" |
| 一举两得 | — | ✅ 解决训练噪声 + 解决 archetype 分配 |

### A.5 一举两得：语义划分同时解决两个问题

```
问题 1: 训练噪声
  "模型应该只在 archetype 相关的 bar 上训练"
  → pre_filter 剥离不相关样本

问题 2: archetype 分配
  "这个 bar 属于哪个 archetype?"
  → pre_filter 条件 ≡ archetype 语义边界

两者用的是同一组条件：
  ME:  me_atr_pct ≥ 0.40 AND me_cvd_alignment ≥ 0.40 AND me_volume_surge ≥ 0.30
  FER: fer_trapped_longs_score ≥ 2.75 OR fer_trapped_shorts_score ≥ 3.75
  LV:  oi_zscore ≥ 0.5 AND funding_rate_abs_zscore_50 ≥ 0.5
  BPC: bpc_volume_compression_pct ≥ 0.30 AND price_position ≤ 0.90

这组条件为每个 bar 赋予 archetype 语义标签，同时决定:
  - 训练时: 只在此 archetype 相关样本上训练 Gate
  - 实盘时: guardrail 拦截不符合语义的信号
  - PCM 时: 可以基于此做 archetype slot 分配
```

### A.6 Plateau 应用于 pre_filter 阈值优化

**核心思想**: 当前 pre_filter 阈值来自 guardrail（人工设定），但可以用 Plateau 算法自动搜索最优语义边界。

**方法**: 复用 `optimize_gate_unified.py` 的 Lift + Plateau 算法，但目标函数不同：

```
Gate hard_gate 优化:    Lift = "这个阈值能够选择性地保留 good / 淘汰 bad"
pre_filter 阈值优化: Lift = "这个阈值上，专属特征的区分力最大化"
```

**具体算法**:

```
对 pre_filter 的每个语义特征 (如 me_atr_pct):

  1. Threshold Scan:
     对阈值 θ ∈ [0.1, 0.2, 0.3, ..., 0.9] 逐点扫描:
     a) 取 sub_df = df[feature >= θ]  (语义子集)
     b) 在 sub_df 上训练 LightGBM Gate
     c) 计算专属特征的平均 importance
     d) 计算 CV
     e) KPI = CV * mean_archetype_feature_importance

  2. Stable Plateau Detection:
     找 KPI 稳定的 θ 区间（同 Gate Plateau 逻辑）

  3. 选择 Plateau 中点作为最终 pre_filter 阈值
```

**简化方案（推荐先做）**:

不重新训练模型，而是在现有 Gate 训练数据上用统计方法扫描：

```
对每个语义特征 (如 me_atr_pct) 和每个阈值 θ:
  sub_df = df[feature >= θ]
  KPI = (1 - bad_rate(sub_df)) * coverage(sub_df)
  # KPI 平衡两个目标:
  #   - 降低 bad_rate (语义筛选效果)
  #   - 保留足够样本 (coverage, 避免过度筛选)
找 Stable Plateau → 取中点
```

**实现路径**:

```bash
# 复用现有脚本，用 guardrail 特征作为候选规则扫描
python scripts/optimize_gate_unified.py \
  --logs results/train_final_xxx/me/predictions.parquet \
  --strategy me \
  --label-col success_no_rr_extreme \
  --output results/me_prefilter_optimization.json
  # guardrail 规则已在 gate.yaml 中定义，optimizer 会自动扫描
```

**优势**: 阈值不再是人工设定的 0.40/0.30，而是经过 Plateau 稳健性搜索的结构性解。

### A.7 关键脚本和文件索引

| 文件 | 作用 | 备注 |
|---|---|---|
| `src/features/time_series/momentum_expansion_features.py` | ME 因果特征计算 | Energy × Acceleration × Participation |
| `src/features/time_series/fer_features.py` | FER 因果特征计算 | trapped + efficiency + absorption |
| `src/features/time_series/bpc_features.py` | BPC 因果特征计算 | compression + breakout + continuation |
| `src/features/time_series/open_interest_features.py` | LV OI特征计算 | zscore + divergence + scene |
| `scripts/analyze_archetype_feature_stratification.py` | 分位数分层分析脚本 | Step 2: 按 P5/P10/P80/P90/P95 切分, 对比 bad rate + median RR |
| `z实验_005_统一研究/gate_semantic_prefilter_design.md` | 分位数分层实验报告 | BPC/FER/ME 三策略分析结果 |
| `scripts/optimize_gate_unified.py` | Lift + Plateau 优化器 | 可复用于 pre_filter 阈值优化 |
| `scripts/train_strategy_pipeline.py` L616-L632 | 训练管线 `apply_filters()` | 已支持 min/max，零代码修改 |
| `config/strategies/{arch}/labels_rr_extreme.yaml` | Gate 标签配置 | 在 filters 段加 min 条件即可 |
| `config/strategies/{arch}/archetypes/gate.yaml` | guardrail 规则定义 | pre_filter 条件的原始来源 |

### A.7.1 分位数分层分析脚本使用指南

**脚本**: `scripts/analyze_archetype_feature_stratification.py`

#### 数据来源链路

```
① 特征配置文件
   config/strategies/{arch}/features_gate.yaml    # Gate 特征列表
   config/strategies/{arch}/features.yaml          # 全量特征列表(gate+evidence)
      ↓
② Feature Store 构建
   mlbot fs build --strategy {arch} --freq {4H/1H/15min}
   → 读取 features_gate.yaml 中的 requested_features
   → 调用各特征计算模块 (bpc_features.py / me_features.py / fer_features.py / open_interest_features.py)
   → 生成 Feature Store parquet 缓存
      ↓
③ Gate 训练
   mlbot train final --strategy {arch} --label rr_extreme
   → 从 Feature Store 加载特征 (含 archetype 专属特征)
   → 从 labels_rr_extreme.yaml 加载标签配置
   → 通过 apply_filters() 筛选训练样本
   → 训练 LightGBM 模型
      ↓
④ 输出 predictions.parquet          ←←← 本脚本的输入
   位置: results/{run_name}/{arch}/predictions.parquet
   内容: holdout 测试集的所有特征列 + 标签列 + 预测列
   关键列:
     - bpc_*/me_*/fer_*/oi_*: archetype 专属特征 (来自各策略的特征计算模块)
     - success_no_rr_extreme: 二元标签 (1=好, 0=踩大坑, forward_rr < -0.8R)
     - forward_rr: 连续值, 前向风险回报比 (部分旧训练可能缺失此列)
```

**重要**: archetype 专属特征不是脚本自己计算的，而是在 Feature Store 构建时就已经计算好，并通过训练管线写入 predictions.parquet。

#### 特征来源配置文件

| 策略 | 特征配置文件 | 特征计算模块 | 特征前缀 | 说明 |
|---|---|---|---|---|
| BPC | `config/strategies/bpc/features_gate.yaml` | `src/features/time_series/bpc_features.py` | `bpc_` | 38 个特征 (compression + breakout + pullback) |
| ME | `config/strategies/me/features_gate.yaml` | `src/features/time_series/momentum_expansion_features.py` | `me_` | 18 个特征 (energy × acceleration × participation) |
| FER | `config/strategies/fer/features_gate.yaml` | `src/features/time_series/fer_features.py` | `fer_` | 12 个特征 (trapped + efficiency + absorption) |
| LV | `config/strategies/lv/features_gate.yaml` | `src/features/time_series/open_interest_features.py` | `oi_`, `funding_rate_*` | OI zscore + funding rate zscore |

#### 命令行参数详解

| 参数 | 必须 | 默认值 | 说明 |
|---|---|---|---|
| `--logs` | ✅ | — | predictions.parquet 路径。例如: `results/bpc_gate/bpc/predictions.parquet` |
| `--strategy` | ✅ | — | 策略名称: `bpc` / `me` / `fer` / `lv`。决定自动推断的特征前缀 |
| `--prefix` | ❌ | 自动推断 | 手动指定特征前缀，逗号分隔。例: `--prefix me_,me_accel` |
| `--percentiles` | ❌ | `5,10,20,80,90,95` | 百分位列表。低端(P5/P10/P20)检测“缺失信号”，高端(P80/P90/P95)检测“强信号” |
| `--rr-col` | ❌ | `forward_rr` | 连续 RR 列名。缺失时自动 fallback 到 `bpc_impulse_return_atr` |
| `--label-col` | ❌ | `success_no_rr_extreme` | 二元标签列名。缺失时自动从 forward_rr 生成 (threshold=-0.8R) |
| `--output` | ❌ | — | JSON 报告输出路径。包含所有分层结果 + 分类摘要 |
| `--min-samples` | ❌ | `30` | 每组最小样本量，低于此值跳过该分层（避免统计噪声） |

#### 实际用法示例

```bash
# ① BPC 分层分析 (最新训练结果)
python scripts/analyze_archetype_feature_stratification.py \
    --logs results/bpc_gate/bpc/predictions.parquet \
    --strategy bpc
# 输出: 80 正信号 / 20 反信号 / 34 低端信号

# ② FER 分层分析
python scripts/analyze_archetype_feature_stratification.py \
    --logs results/fer_gate/fer/predictions.parquet \
    --strategy fer
# 输出: 29 正信号 (trapped 最强, -20.6%)

# ③ ME 分层分析
python scripts/analyze_archetype_feature_stratification.py \
    --logs results/me_gate_v2/me/predictions.parquet \
    --strategy me
# 输出: 13 正信号 / 14 反信号 (me_atr_pct 最强, -8.5%)

# ④ 导出 JSON 报告并指定特定百分位
python scripts/analyze_archetype_feature_stratification.py \
    --logs results/bpc_gate/bpc/predictions.parquet \
    --strategy bpc \
    --percentiles 10,90,95 \
    --output results/bpc_stratification.json
```

#### 输出解读

脚本自动将结果分为三类，每类的判定标准是 bad_rate 差异 > 2%:

| 分类 | 含义 | 应用场景 | 示例 |
|---|---|---|---|
| **正信号** | 特征高值时 bad_rate 降低 | 可做 guardrail 正向条件 / pre_filter | `bpc_dir_consistency_long P95: bad 35.6% vs 49.8% (-14.2%)` |
| **反信号** | 特征高值时 bad_rate 升高 | 可做反向 hard_gate (deny) | `bpc_bb_compression P95: bad 72.3% vs 47.5% (+24.8%)` |
| **低端信号** | 特征缺失时 bad_rate 升高 | 可做最低门槛 (min 条件) | `bpc_volume_compression_pct P10: bad 58.0% vs 47.7% (+10.3%)` |

---

### A.7.2 `apply_filters()` 的 `min`/`max` 操作详解

**位置**: `scripts/train_strategy_pipeline.py` L616-L632

#### 函数实现

```python
def apply_filters(df: pd.DataFrame, filters: List[Dict[str, Any]]) -> pd.DataFrame:
    result = df
    for filt in filters:
        column = filt.get("column")
        if not column or column not in result.columns:
            continue
        if filt.get("notna"):
            result = result[result[column].notna()]    # 删除 NaN 行
        if "include" in filt:
            result = result[result[column].isin(filt["include"])]   # 白名单
        if "exclude" in filt:
            result = result[~result[column].isin(filt["exclude"])]  # 黑名单
        if "min" in filt:
            result = result[result[column] >= filt["min"]]   # ← 最小值筛选
        if "max" in filt:
            result = result[result[column] <= filt["max"]]   # ← 最大值筛选
    return result
```

#### 五种操作符说明

| 操作符 | YAML 语法 | SQL 等价 | 语义 |
|---|---|---|---|
| `notna: true` | `column: xxx` + `notna: true` | `WHERE xxx IS NOT NULL` | 删除 NaN 行 |
| `include: [a,b]` | `include: [a, b]` | `WHERE xxx IN (a, b)` | 白名单筛选 |
| `exclude: [c]` | `exclude: [c]` | `WHERE xxx NOT IN (c)` | 黑名单排除 |
| **`min: 0.40`** | `min: 0.40` | **`WHERE xxx >= 0.40`** | 保留 >= 此值的行（最小门槛） |
| **`max: 0.90`** | `max: 0.90` | **`WHERE xxx <= 0.90`** | 保留 <= 此值的行（最大上限） |

#### 当前 YAML 配置（旧方案，无语义筛选）

```yaml
# config/strategies/me/labels_rr_extreme.yaml 当前状态
filters:
  - column: success_no_rr_extreme
    notna: true     # 仅删除标签为空的行，全量 bar 都参与训练
```

#### 新方案 YAML 配置（加入 min/max 语义筛选）

```yaml
# config/strategies/me/labels_rr_extreme.yaml 新方案
filters:
  # 🆕 语义预筛选: 只在 ME 相关的 bar 上训练
  - column: me_atr_pct           # Energy > 0.40
    min: 0.40                    # ← apply_filters 执行: df = df[df['me_atr_pct'] >= 0.40]
  - column: me_cvd_alignment     # Flow > 0.40
    min: 0.40                    # ← apply_filters 执行: df = df[df['me_cvd_alignment'] >= 0.40]
  - column: me_volume_surge      # Volume > 0.30
    min: 0.30                    # ← apply_filters 执行: df = df[df['me_volume_surge'] >= 0.30]
  # 原有标签过滤
  - column: success_no_rr_extreme
    notna: true
```

```yaml
# config/strategies/bpc/labels_rr_extreme.yaml 新方案 (同时用 min + max)
filters:
  - column: bpc_volume_compression_pct
    min: 0.30                    # ← df = df[df['bpc_volume_compression_pct'] >= 0.30]
  - column: price_position
    max: 0.90                    # ← df = df[df['price_position'] <= 0.90]
  - column: success_no_rr_extreme
    notna: true
```

#### 执行流程图

```
训练流程:
  Feature Store (全量 bar)
      ↓
  apply_filters(df, labels_rr_extreme.yaml['filters'])
      ↓ min: 0.40 → 删除 me_atr_pct < 0.40 的行
      ↓ min: 0.40 → 删除 me_cvd_alignment < 0.40 的行
      ↓ min: 0.30 → 删除 me_volume_surge < 0.30 的行
      ↓ notna: true → 删除标签为空的行
      ↓
  筛选后训练集 (~5K-8K bars, 只含扩张场景)
      ↓
  LightGBM 训练 Gate 模型
```

**关键点**: 零代码修改。整个语义预筛选机制完全通过 YAML 配置实现，
`apply_filters()` 函数已经原生支持 `min`/`max` 操作符，无需修改任何 Python 代码。

### A.8 执行清单

- [ ] ME `labels_rr_extreme.yaml` 的 filters 段加 min 条件（零代码修改）
- [ ] FER `labels_rr_extreme.yaml` 加 min 条件 (OR 逻辑需要新增 `any_of` 支持)
- [ ] LV `labels_rr_extreme.yaml` 加 min 条件
- [ ] BPC `labels_rr_extreme.yaml` 加 min + max 条件
- [ ] (可选) 用 Plateau 搜索优化 pre_filter 阈值，替代 guardrail 固定值
- [ ] 重建 FS (如有新特征) + 重训 Gate v3
- [ ] 对比 CV 和 feature importance 变化

### A.9 `prefilter.yaml` 配置化特征识别方案

#### A.9.1 问题：硬编码前缀的局限性

旧方案 (`analyze_archetype_feature_stratification.py`) 通过硬编码前缀识别 archetype 特征：

```python
# 旧方案: 硬编码前缀映射
STRATEGY_PREFIX_MAP = {
    "bpc": ["bpc_"],
    "me": ["me_"],
    "fer": ["fer_"],
    "lv": ["oi_", "funding_rate_abs_zscore", "funding_rate_zscore"],
}
```

**局限性**:
- 只匹配 archetype 专属前缀的列（BPC ~10个，ME ~5个，FER ~3个）
- 遗漏大量通用特征（`trend_r2_20`, `vol_zscore`, `atr_percentile`, `vpin_*`, `shd_pct`, 交叉特征等）
- 通用特征在不同策略下可能有不同的区分力，漏掉就不完整
- 前缀硬编码，改名或新增特征时需要改代码

#### A.9.2 新方案：`prefilter.yaml` 配置 + `feature_dependencies.yaml` 解析

**设计决策依据**：

| 方案 | 做法 | 优点 | 缺点 |
|------|------|------|------|
| 硬编码前缀 | `STRATEGY_PREFIX_MAP` | 简单 | 遗漏通用特征，改名需改代码 |
| 读 features_gate.yaml | 解析 requested_features | 对齐配置源 | `_f` 是管线工厂名 ≠ 实际列名，需额外映射 |
| **prefilter.yaml（选定）** | 独立配置 + 依赖解析 | 精准聚焦、可扩展、不改代码 | 需维护额外文件 |
| 全量 parquet 列 | 排除 meta 列后分析所有 | 最完整 | 太多无关特征，信噪比低 |

选定 `prefilter.yaml` 方案的理由：
1. **精准聚焦**：pre-filter 候选特征本就不多（每策略 3-8 个 `_f`），不需要全量扫描
2. **依赖解析可靠**：`feature_dependencies.yaml` 的 `output_columns` 字段提供 `_f` → 实际列名的精确映射
3. **职责分离**：`prefilter.yaml` 只关注"哪些特征值得做前置筛选"，与 `features_gate.yaml`（Gate 训练特征）不重叠
4. **改配置不改代码**：新增候选特征只需编辑 YAML

#### A.9.3 `prefilter.yaml` 格式规范

```yaml
# config/strategies/{arch}/prefilter.yaml
# 职责: 声明待验证的 pre-filter 候选特征 (Step 2 分位数分层分析的输入)
# 脚本通过 feature_dependencies.yaml 的 output_columns 解析出实际列名

description: "ME 语义预筛选候选特征"

candidates:
  # archetype 核心特征
  - me_soft_phase_f       # → me_atr_pct, me_vol_regime, me_accel_2k, ...
  - me_failure_f          # → me_false_expansion, me_vol_divergence, me_flow_exhaustion
  - me_context_f          # → me_cvd_alignment, me_volume_surge, ...
  # 想额外验证的通用特征 (可选)
  - vol_regime_features_f  # → vol_zscore, vol_percentile_approx
  - atr_percentile_f       # → atr_percentile
```

**解析链路**：
```
prefilter.yaml (candidates 列表)
    ↓ 读取 _f 名
feature_dependencies.yaml (output_columns 字段)
    ↓ 解析实际列名
predictions.parquet (匹配存在的列)
    ↓ 逐列分位数分层
输出报告
```

#### A.9.4 反信号与 Gate 训练的职责边界

**核心原则：pre-filter 定义语义边界，Gate 学习失败模式**

| 信号类型 | 含义 | 归属 | 理由 |
|----------|------|------|------|
| **正信号** (high=good) | 高值 = 属于此 archetype | **pre-filter** | 定义"这个 bar 属不属于此 archetype" |
| **低端信号** (absence=bad) | 低值 = 缺失 archetype 信号 | **pre-filter** | 等价于正信号的反面，设 min 阈值 |
| **反信号** (high=bad) | 高值 = bad rate 升高 | **Gate** | Gate 的核心能力就是学"有信号但会失败"的模式 |
| **反信号 (语义边界)** | 高值 = 不是此 archetype | **pre-filter** | 少数例外，如 BPC price_position max: 0.90 |

**不重复的决策依据**：
- pre-filter 在 Gate 训练**之前**执行，定义训练数据范围
- Gate 在 pre-filter 后的数据上学习，两者职责不重叠
- 反信号仍然由脚本分析并输出（用于参考），但总结中标注 `→ 建议留给 Gate`

**计算开销评估**：每策略 ~15-30 个实际列 × 6 分位 = ~90-180 次 groupby，predictions.parquet 仅几万行，总计算 < 1 秒，不构成性能瓶颈。

#### A.9.5 四策略 `prefilter.yaml` 配置

| 策略 | 配置路径 | 核心候选 _f | 通用候选 _f |
|------|---------|-------------|-------------|
| BPC | `config/strategies/bpc/prefilter.yaml` | bpc_soft_phase_f, bpc_pullback_depth_pct_f, bpc_dir_consistency_multi_f, bpc_volume_compression_pct_f | vol_regime_features_f, atr_percentile_f |
| ME | `config/strategies/me/prefilter.yaml` | me_soft_phase_f, me_failure_f, me_context_f | vol_regime_features_f, atr_percentile_f |
| FER | `config/strategies/fer/prefilter.yaml` | fer_failure_signals_f | vol_regime_features_f, atr_percentile_f |
| LV | `config/strategies/lv/prefilter.yaml` | oi_features_f, funding_rate_features_f | vol_regime_features_f, atr_percentile_f, garch_features_f |

#### A.9.6 脚本改造要点

`analyze_archetype_feature_stratification.py` 改造：

1. **新增 `--config` 参数**：指定 `prefilter.yaml` 路径
2. **新增 `--deps` 参数**：指定 `feature_dependencies.yaml` 路径（默认: `config/feature_dependencies.yaml`）
3. **特征解析链路**：prefilter.yaml → feature_dependencies.yaml → output_columns → 匹配 parquet 列
4. **保留 `--prefix` fallback**：未指定 `--config` 时仍支持旧的前缀模式
5. **输出标注**：每个信号标注建议用途 (pre-filter / leave-to-gate)

```bash
# 新用法 (配置驱动)
python scripts/analyze_archetype_feature_stratification.py \
    --logs results/train_final_xxx/bpc/predictions.parquet \
    --strategy bpc \
    --config config/strategies/bpc/prefilter.yaml

# 旧用法 (前缀模式, 仍支持)
python scripts/analyze_archetype_feature_stratification.py \
    --logs results/train_final_xxx/bpc/predictions.parquet \
    --strategy bpc --prefix bpc_
```

---

## Phase B: 数学特征加入 ME/LV

### B.1 决策依据

| 特征组 | 物理含义 | BPC 实证 | 是否加入 ME | 是否加入 LV |
|---|---|---|---|---|
| EVT | 尾部风险分布形态 | imp=482 (Top3) | ✅ | ✅ (已有) |
| Spectrum | 频域能量分布 | imp=97 (Top15) | ✅ | ✅ (已有) |
| Hurst | 时序持续性/记忆性 | 未使用 | ✅ | ✅ (已有) |
| WPT | 多尺度能量分解 | 通过 scene 使用 | ✅ | 未验证 |

**判断标准**: 这些不是"纯统计特征"，每个都有明确的物理语义：
- EVT: "当前极端事件概率有多大？" → 直接关联"踩大坑"标签
- Spectrum: "动量是真趋势(低频)还是噪音(高频)？" → ME 核心区分
- Hurst: "行情会持续还是反转？" → H>0.5=趋势延续，H<0.5=均值回归
- WPT: "能量集中在哪个时间尺度？" → 结构稳定性

### B.2 已执行

- [x] ME `features_gate.yaml`: 取消注释 evt_features_f, spectrum_features_f, hurst_price_f, hurst_cvd_f, wpt_price_fluctuation_f, wpt_volume_energy_f, wpt_cvd_fluctuation_f

### B.3 待执行

- [ ] 重建 ME 1H Feature Store (含 EVT/Spectrum/Hurst/WPT)
- [ ] LV `features_gate.yaml`: 补充 Trade Cluster zscore 系列 (当前 FS 有但 yaml 缺)、shd_pct_f、cvd 系列
- [ ] 重建 LV 15min Feature Store (修复 trade cluster freq 问题)

---

## TODO Phase C: 方向分离 (Long/Short)

### C.1 当前缺陷

所有 4 个策略的 `labels_rr_extreme.yaml` 均硬编码 `direction: long`：

```yaml
label_generator:
  params:
    direction: long  # ← 所有策略都是 long
```

**问题**:
- `forward_rr(long) = (max_high - entry - (entry - min_low)) / ATR`
- 模型只学"做多会不会踩坑"
- 对做空信号: 做多踩坑的条件 ≈ 做空赚钱的条件 → Gate 可能产生反向 veto

### C.2 解决方案

**方案 A (推荐): 双向标签 + 方向特征**

为每个 bar 计算两个标签：
```python
forward_rr_long  = (MFE_long - MAE_long) / ATR
forward_rr_short = (MFE_short - MAE_short) / ATR

# 实际训练时，结合信号方向选择对应标签：
# - 如果该 bar 有 long 信号 → 用 forward_rr_long
# - 如果该 bar 有 short 信号 → 用 forward_rr_short
# - 如果没有信号 → 两个都用（data augmentation）
```

**依赖**: 需要 Phase A (pre_filter) 先落地提供信号方向信息

**方案 B (简单): 只训练 long，实盘对 short 信号不做 Gate veto**

暂时 skip short veto，等 Phase A 落地后再做方向分离。

### C.3 执行清单

- [ ] (依赖 Phase A) 确认训练数据中有信号方向列
- [ ] `failure_first_label.py` 添加双向标签计算
- [ ] `labels_rr_extreme.yaml` 支持 `direction: both` 配置
- [ ] Gate apply 时按信号方向选择对应的 veto 判断

---

## 执行优先级

```
Phase A (语义预筛选) ──→ 最高优先级, 根因修复
  ↓
Phase B (数学特征)   ──→ 同步做, 改配置即可
  ↓
Phase C (方向分离)   ──→ 依赖 A 的信号列基础设施
```

**预期收益**:
- Phase A: ME CV 从 0.018 → 预期 0.05+ (专属特征重获区分力)
- Phase B: ME 增加 EVT/Spectrum 特征 → 预期 CV 进一步提升
- Phase C: 消除 long/short 方向偏差 → 实盘表现改善

---

## 进度追踪

| 阶段 | 状态 | 备注 |
|---|---|---|
| Phase A: 语义预筛选 | ⏳ 设计完成 | 待实现 pre_filter 逻辑 |
| Phase B: 数学特征 | 🔨 ME 已改配置 | LV 待改 + 两者待重建 FS |
| Phase C: 方向分离 | 📋 方案已定 | 依赖 Phase A |
