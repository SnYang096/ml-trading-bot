# Evidence 单调性验证算法

## 目标

验证 evidence_score 是否具有**单调性**：即 score 越高，交易结果越好。
这是 evidence 仓位缩放（`size = 0.5 + 0.5 × evidence`）的**前提条件**。

## 算法步骤

### 1. 数据准备

从 `trade_details`（向量回测逐笔交易记录）提取两个数组：

- `evidence_score[]` — 每笔交易的 evidence 综合评分（∈ [0, 1]）
- `realized_rr[]` — 每笔交易的原始 R-multiple（**缩放前**，反映真实交易质量）

最少需要 10 笔交易，否则跳过。

### 2. 等宽分箱

将 evidence_score 按 5 个等宽区间分箱：

| Bin | 范围       |
| --- | ---------- |
| 0   | [0.0, 0.2) |
| 1   | [0.2, 0.4) |
| 2   | [0.4, 0.6) |
| 3   | [0.6, 0.8) |
| 4   | [0.8, 1.0] |

每个 bin 统计：
- **count** — 落入该 bin 的交易数
- **mean_rr** — 该 bin 内所有交易的平均 R-multiple
- **win_rate** — 该 bin 内 R > 0 的比例

### 3. Spearman 秩相关

在**逐笔交易**粒度（非 bin 级别）计算：

```
spearman_r, p_value = spearmanr(evidence_score[], realized_rr[])
```

- `spearman_r > 0` 且 `p < 0.05` → evidence 与交易质量**正相关**
- `spearman_r ≤ 0` 或 `p ≥ 0.05` → evidence 无效或反向

选择 Spearman 而非 Pearson 的原因：
- 不假设线性关系，只要求**单调**
- 对 R-multiple 的极端值（大赚/大亏）鲁棒

### 4. 单调性判定

在 **bin 级别** 判定严格单调递增：

```
is_monotonic = all(bin_means[i] <= bin_means[i+1] for i in range(len - 1))
```

- 跳过空 bin（count = 0 的 bin 不参与判定）
- 允许相等（弱单调），不要求严格递增

### 5. 综合结论

| 条件                                              | 结论                 | 后续动作                                        |
| ------------------------------------------------- | -------------------- | ----------------------------------------------- |
| `is_monotonic=True` 且 `spearman_r > 0, p < 0.05` | ✅ Evidence 可信      | 可启用 `evidence_position_scale: true`          |
| `spearman_r > 0` 但 `is_monotonic=False`          | ⚠️ 整体正相关但非单调 | 检查哪个 bin 拐头，可能某个 evidence 特征质量差 |
| `spearman_r ≤ 0`                                  | ❌ Evidence 无效      | 不应启用仓位缩放，需重新筛选 evidence 特征      |

---

## min_score 验证

### min_score 是什么

`min_score` 是 evidence 的**入场门槛**：composite evidence_score < min_score 的交易会被直接拒绝。
每个 archetype 有自己的 min_score，写在 `archetypes/evidence.yaml` 中。

### min_score 如何计算

在 `optimize_evidence_plateau.py` 中**纯数学推导**，不基于任何实证：

```python
# 原则: 只拒绝所有 evidence axis 都给出 "suppress" (0.0) 的信号
weights = [1.0 / max(1, ef.rank) for ef in optimized_features]
total_weight = sum(weights)
min_weight = min(weights)
auto_min_score = min_weight * 0.25 / total_weight
```

含义：当最轻权重的 evidence 特征给出 "downweight" (0.25)、其余全部给 "suppress" (0.0) 时的加权平均。

**问题：这是一个几何构造的阈值，没有任何数据验证**——不检查被拒绝交易是否真的更差。

### min_score 有效的前提

`min_score` 有效**完全依赖 evidence_score 本身有效**：

1. 如果 evidence_score 与 R 正相关 → 低分交易确实更差 → min_score 过滤有效
2. 如果 evidence_score 与 R 无关 → 低分交易不比高分差 → **min_score 随机拒绝好坏交易**
3. 如果 evidence_score 与 R 负相关 → 低分交易反而更好 → **min_score 反向过滤，丢弃好交易**

**没有独立的 min_score 验证机制**——它不是一个自洽的过滤器，只是 evidence 假设链上的下游产物。

---

## 实证案例：PCM 回测验证结果

### 回测配置

```bash
python scripts/backtest_execution_layer.py \
  --pcm bpc fer me --from-raw \
  --test-start 2025-01-01 --test-end 2025-07-31 \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT --use-1min
```

### Evidence 单调性结果

```
📊 Evidence Monotonicity Analysis (109 trades):
   Bin             Count    Mean R    Win%
   --------------------------------------
   [0.0,0.2)          30    0.1635   63.3%   ← 最低分反而最好
   [0.2,0.4)          28   -0.0064   42.9%
   [0.4,0.6)          19    0.0780   47.4%
   [0.6,0.8)          21    0.0445   38.1%
   [0.8,1.0)          11   -0.2514   45.5%   ← 最高分反而最差
   Spearman r=-0.0508 (p=0.6001)  ⚠️  非单调
```

**结论：❌ Evidence 无效**
- Spearman r = -0.05，接近零，p = 0.60 完全不显著
- 方向**反转**：最低分 bin mean_R = +0.16，最高分 bin mean_R = -0.25

### min_score 拒绝统计

```
🔒 per-strategy evidence_min_score: {'bpc': 0.1, 'fer': 0.0111, 'me': 0.0294}
🔍 Reject reasons: evidence_min=34, per_strat=367, global=0, both=0
```

- 34 笔交易被 evidence_min_score 拒绝
- **没有 counterfactual 追踪**——被拒绝交易的 R-multiple 没有单独统计
- 但 evidence_score 本身与 R 无关 (r=-0.05, p=0.60)，所以 min_score 过滤**等同于随机丢弃**

### Counterfactual（被丢弃信号的后续 R）

```
bpc rejected: 6 trades, mean_R=0.7350, win=83.33%
fer rejected: 16 trades, mean_R=0.2713, win=50.00%
me rejected: 10 trades, mean_R=-0.0922, win=70.00%
```

注：这是所有原因（evidence_min + per_strat + global）拒绝的交易汇总（共 401 笔被拒，34 笔因 evidence_min）。
被拒交易的 mean_R 普遍 **优于** 已执行交易（mean_R=0.0208），说明过滤器在丢弃好交易。

### 各 Archetype 表现

```
Archetype       Trades   Mean R   Sharpe    Win%
----------------------------------------------------------
bpc                  8   0.0051   0.0070   62.5%
fer                 75   0.0461   0.0567   40.0%
me                  26  -0.0475  -0.1297   69.2%
```

### 根因分析：Evidence 为何在 Archetype 内部失效

**数据人口剧变（Population Shift）**：

| 阶段                                       | 数据量  | 留存率   |
| ------------------------------------------ | ------- | -------- |
| Evidence 训练：predictions.parquet (全量)  | ~15000+ | 100%     |
| Evidence 优化：logs_gated.parquet (RR分层) | ~15000+ | ~100%    |
| Gate 过滤                                  | 1006    | 6.6%     |
| Entry Filter                               | 167     | 1.1%     |
| PCM 执行                                   | **109** | **0.7%** |

Evidence 在 100% 数据上训练，在 0.7% 的高度筛选子集上评估。

**三层原因**：

1. **Gate 消耗了 Evidence 的预测力**：Gate 和 Evidence 从同一个 LightGBM 模型（SHAP∩Gain）发现特征。Gate 过滤掉 93.4% 后，Evidence 特征在剩余子集上的区分力衰减为零。
2. **Good/Bad 标签不匹配**：Evidence 优化用全量 RR 的 Q80/Q20 定义 Good/Bad，但 Gate 放行子集的 RR 分布已右移，全量 Q80 在放行子集中可能覆盖 60-70% 样本。
3. **条件独立性破坏（Simpson's Paradox）**：特征在全量数据上预测 RR，但条件在 Gate=allow 下，特征-RR 关系可能消失甚至反转。

---

## 输出示例（理想情况）

```
📊 Evidence Monotonicity Analysis (342 trades):
   Bin            Count   Mean R    Win%
   --------------------------------------
   [0.0,0.2)          8  -0.4521   25.0%
   [0.2,0.4)         47  -0.1233   40.4%
   [0.4,0.6)        189   0.0312   51.3%
   [0.6,0.8)         82   0.1847   58.5%
   [0.8,1.0]         16   0.3102   68.8%
   Spearman r=0.1823 (p=0.0007)  ✅ 单调递增
```

## 代码位置

- 单调性验证：`scripts/backtest_execution_layer.py` → `_report_evidence_monotonicity()`
- min_score 计算：`scripts/optimize_evidence_plateau.py` → `_promote_evidence_yaml()` L237-L249
- min_score 拒绝：`scripts/backtest_execution_layer.py` → `simulate_rr_execution()` L1205-L1210
- 实盘拒绝：`src/time_series_model/live/generic_live_strategy.py` L484-L497

## 设计要点

1. **用原始 R，不用缩放后的 R** — 避免循环论证（缩放本身就让高 evidence 的 R 更大）
2. **等宽分箱而非等频** — evidence_score 的分布可能偏态，等宽能暴露空 bin 问题
3. **Spearman + bin 递增双重验证** — Spearman 看整体趋势，bin 递增看局部一致性
4. **min_score 无独立验证** — 依赖 evidence 假设成立；evidence 无效则 min_score 必然无效
5. **需要 counterfactual 追踪** — 被 min_score 拒绝交易的 R 应该被记录，用于验证过滤有效性

---

## 改进方案：在 Gate 放行子集上重新训练 Evidence

> **实现状态：✅ 全部 4 个措施已实现并集成到 auto_research_pipeline**
>
> | 措施 | 状态 | 代码位置 |
> |------|------|----------|
> | 措施1: Gate 放行子集过滤 | ✅ 已实现 | `optimize_evidence_plateau.py` L1428-L1447, `--gate-yaml` 参数 |
> | 措施2: Good/Bad 标签在放行子集内计算 | ✅ 已实现 | `optimize_evidence_plateau.py` L1458-L1467 |
> | 措施3: 排除 Gate 特征+高相关特征 | ✅ 已实现 | `_extract_gate_features()` + `_find_gate_correlated()` |
> | 措施4: Spearman 预筛 (D路径) | ✅ 已实现 | `_spearman_prescreen()` L87-L112 |
> | 管线集成 | ✅ 已实现 | `auto_research_pipeline.py` L1016-L1041 自动传 `--gate-yaml` |
> | 措施4建议A: top_n=5 | ⚠️ 未落地 | 仍用默认 top_n=15, Spearman 预筛做等价过滤 |

### 问题诊断

当前 Evidence 管线有三个结构性缺陷：

1. **训练-评估人群不一致**：在全量数据上训练，在 Gate 放行的 0.7% 子集上评估
2. **Good/Bad 标签基于全量分布**：Q80/Q20 在全量 RR 上计算，放行子集的 RR 分布已右移
3. **Gate-Evidence 特征家族重叠**：Gate 和 Evidence 从同一个 LightGBM 模型发现特征，存在 double counting

### Gate 与 Evidence 特征重叠分析

| Archetype | Gate 特征                                                                                                     | Evidence 特征                                                                                                          | 重叠风险                                  |
| --------- | ------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| BPC       | `bpc_volume_compression_pct`, `bpc_bb_compression`, `bpc_cvd_z`                                               | `shd_pct`, `macd_signal_atr`                                                                                           | 低（不同家族）                            |
| FER       | `fer_trapped_longs_score`, `fer_trapped_shorts_score`, `fer_volume_price_divergence`                          | `fer_signed_efficiency`, `fer_impulse_failure_direction`, `fer_momentum_efficiency_decay`, `fer_signed_efficiency_pct` | **高（同一 fer_failure_signals_f 家族）** |
| ME        | `atr_percentile`, `me_atr_pct`, `me_cvd_alignment`, `sma_200_position`, `evt_var_99_right`, `evt_es_99_right` | `vpin_volatility_20`, `vpin_ma20`, `evt_scale_right`                                                                   | **中（evt 家族重叠）**                    |

FER 的 4/6 evidence 特征与 gate 特征属于同一 `fer_failure_signals_f` 家族，是 double counting 的高风险区。

### 改进措施

#### 措施 1: 在 Gate 放行子集上优化 Evidence ✅

**已实现**：`optimize_evidence_plateau.py` 新增 `--gate-yaml` 参数 (L1369-L1374)。

当 `--gate-yaml` 提供时 (L1428-L1447)：
1. 用 `_extract_gate_features()` 解析 gate.yaml 中的特征名
2. 过滤 `gate_decision == "allow"` 行
3. 后续 Q80/Q20 和 plateau 优化都在放行子集上进行

管线集成：`auto_research_pipeline.py` L1016-L1041 自动传入 `--gate-yaml config_dir/archetypes/gate.yaml`。

#### 措施 2: Good/Bad 标签在放行子集内计算 ✅

**已实现**：`optimize_evidence_plateau.py` L1458-L1467。

因为 Gate 过滤在 Q80/Q20 计算之前执行（L1428-L1447 先过滤，L1459 再算分位数），
所以 Q80/Q20 自动基于放行子集计算。日志中会显示 `(放行子集内)` 标注 (L1467)。

这样 Good/Bad 反映的是"在已通过 Gate 的交易中，哪些特别好/特别差"。

#### 措施 3: 排除与 Gate 特征高相关的候选 ✅

**已实现**，两层排除：

**候选生成层** (`export_lightgbm_rules_to_readme.py` L1171-L1181)：
- `_generate_evidence_candidates_yaml()` 接受 `exclude_features` 参数
- `train_strategy_pipeline.py` L4131-L4163 加载 gate.yaml，提取 gate 特征传入

**优化层** (`optimize_evidence_plateau.py` L1521-L1554)：
- `_find_gate_correlated()`: gate 特征本身 + Pearson |r| > 0.7 的高相关特征一并排除
- 排除列表打印到日志

#### 措施 4: 解决 Gate 过滤后数据量不足问题 ✅ (D路径已实现)

**问题**：Gate 放行子集可能只有 ~1000 行，Q80/Q20 分层后 Good ~200, Bad ~200。

**方案对比**：

| 方案                     | 做法                                           | 状态     | 优劣                                   |
| ------------------------ | ---------------------------------------------- | -------- | -------------------------------------- |
| A. 减少候选特征数        | 从 top-15 降到 top-5                           | ⚠️ 未落地 | 降低多重检验风险，n/p 比从 70 升到 200 |
| B. 简化分箱              | 从 5 bin 降到 3 bin (suppress/neutral/amplify) | 未实现   | 每 bin 样本更多，但分辨率下降          |
| C. 放宽 plateau 邻域要求 | 语义邻居从 ≥3 降到 ≥2                          | 未实现   | 降低 plateau 门槛                      |
| D. Spearman 直接验证     | 直接算 Spearman(feature, RR)，不经过 SHAP∩Gain | ✅ 已实现 | 最简单，但可能漏掉非线性信号           |

**实际实现：D 路径 + SHAP∩Gain 保留（双轨制）**

`_spearman_prescreen()` (L87-L112)：
1. SHAP∩Gain 发现路径保留（`top_n=15` 默认，未缩减到 5）
2. 在放行子集上对每个候选特征计算 `spearman(feature, RR)`
3. 只保留 `spearman_r > 0 且 p < 0.1` 的特征进入 plateau 优化
4. 无特征通过时 fallback 到全部候选（L1548-L1549）

> 注: 方案 A (top_n=5) 尚未落地。当前 Spearman 预筛已等价实现了"放行子集实证过滤"的效果，
> 即使 top_n 仍为 15，被 Spearman 排除的候选不会进入 plateau 优化。

### 实现位置（已完成）

| 改动                        | 文件                                 | 函数 / 行号                                        | 状态 |
| --------------------------- | ------------------------------------ | -------------------------------------------------- | ---- |
| 新增 `--gate-yaml` 参数     | `optimize_evidence_plateau.py`       | `main()` L1369-L1374                               | ✅    |
| 提取 gate 特征              | `optimize_evidence_plateau.py`       | `_extract_gate_features()` L56-L69                 | ✅    |
| 过滤放行子集 + 重算标签     | `optimize_evidence_plateau.py`       | `main()` L1428-L1476                               | ✅    |
| 排除 gate 相关特征          | `optimize_evidence_plateau.py`       | `_find_gate_correlated()` L115-L140                | ✅    |
| 放行子集 Spearman 验证      | `optimize_evidence_plateau.py`       | `_spearman_prescreen()` L87-L112                   | ✅    |
| Evidence 候选排除 gate 特征 | `export_lightgbm_rules_to_readme.py` | `_generate_evidence_candidates_yaml()` L1171-L1181 | ✅    |
| 训练管线传入 gate 排除列表  | `train_strategy_pipeline.py`         | L4131-L4163                                        | ✅    |
| 管线集成 (auto_research)    | `auto_research_pipeline.py`          | L1016-L1041 传 `--gate-yaml`                       | ✅    |

---

## 最终结论：Evidence 模块应当删除

### Gate 放行子集重训结果（2025-01-01 ~ 2025-07-31）

在完成全部 4 个改进措施后，用 `--gate-yaml` 重训 3 个策略的 evidence，回测结果：

```
📊 Evidence Monotonicity Analysis (112 trades):
   Bin             Count    Mean R    Win%
   --------------------------------------
   [0.0,0.2)          16    0.5179   50.0%   ← 最低分仍然最好
   [0.2,0.4)          21   -0.1206   33.3%
   [0.4,0.6)          42   -0.0004   66.7%
   [0.6,0.8)          18    0.2255   44.4%
   [0.8,1.0)          15    0.0248   33.3%
   Spearman r=-0.0677 (p=0.4782)  ⚠️  非单调
```

**改善幅度：零。** 旧 r=-0.048 → 新 r=-0.068，反而更差。

### Spearman 预筛淘汰情况

| 策略 | 候选 | Gate排除 | Spearman排除 | 通过plateau                              | 最终evidence       |
| ---- | ---- | -------- | ------------ | ---------------------------------------- | ------------------ |
| BPC  | 9    | 0        | 7            | 1 (macd_signal_atr)                      | 1 特征             |
| FER  | 10   | 3        | 4            | 2 (roc_5, fer_momentum_efficiency_decay) | 2 特征             |
| ME   | 9    | 2        | 5            | **0**                                    | **无有效evidence** |

ME 的 9 个候选特征在 gate 放行子集 (788/12065 行, 6.5%) 上全部失效：
- `vpin_volatility_20`: spearman_r = **-0.327** (方向反转)
- `vpin_ma20`: spearman_r = **-0.312** (方向反转)
- 其余 5 个 p > 0.1 (不显著)

### 前后对比

| 指标          | 旧 evidence (全量训练) | 新 evidence (gate放行子集训练) | 变化   |
| ------------- | ---------------------- | ------------------------------ | ------ |
| Spearman r    | -0.048                 | -0.068                         | 更差   |
| p-value       | 0.620                  | 0.478                          | 无改善 |
| 单调性        | ❌                      | ❌                              | 不变   |
| Trades        | 108                    | 112                            | +4     |
| Mean R        | 0.0555                 | 0.0577                         | +0.002 |
| Sharpe (ann.) | 0.56                   | 0.58                           | +0.02  |
| Win Rate      | 48.15%                 | 50.00%                         | +1.85% |

Evidence 仓位缩放对系统收益的贡献 **≈ 0**。

### 根因定论

**Prefilter + Gate 已经耗尽了特征的区分力。**

信号管线的漏斗结构决定了 evidence 必然失效：

```
全量数据 (~15000 bars)
  │
  ├─ Prefilter 训练 → 保留满足前置条件的 bar
  ├─ LightGBM 模型 → 生成预测方向
  ├─ Gate 过滤 → 淘汰 80-95% (BPC 18.5%, FER 15.3%, ME 6.5% 放行)
  ├─ Entry Filter → 再淘汰 ~80%
  │
  └─ 最终执行: 112 笔 (0.7% 留存)
      ↑
      Evidence 试图在这里区分好坏
      但 Gate 已经用同源特征淘汰了所有"明显差"的信号
      剩余信号在 evidence 特征维度上是同质的
```

这不是实现 bug，而是**架构性矛盾**：

1. Gate 和 Evidence 共享 LightGBM 同一特征池 (SHAP∩Gain)
2. Gate 的职责是移除尾部风险 → 移除了特征空间中的"坏"区域
3. Evidence 的职责是在剩余信号中区分好坏 → 但"坏"区域已被 Gate 移除
4. 结果：Gate 放行子集在 evidence 特征维度上近乎均匀分布，无法区分

### 决策

**删除 Evidence 模块。** 理由：

1. **Evidence 仓位缩放无效** — Spearman r ≈ 0, p > 0.4，仓位缩放等同于随机噪声
2. **min_score 过滤有害** — 在 evidence 无效的前提下，min_score 随机丢弃信号（包括好信号）
3. **改进措施已穷尽** — gate 放行子集训练 + 特征排除 + Spearman 预筛，全部无效
4. **系统收益来源明确** — 模型方向 + Gate + Entry Filter 贡献了全部 alpha，evidence 贡献 ≈ 0

### 删除后的系统行为

| 组件        | 删除前                                | 删除后                               |
| ----------- | ------------------------------------- | ------------------------------------ |
| 仓位大小    | `0.5 + 0.5 × evidence` (∈ [0.5, 1.0]) | 固定 1.0                             |
| 入场过滤    | min_score 可拒绝信号                  | 无 evidence 过滤                     |
| Regime缩放  | `regime_scale × evidence_scale`       | 仅 `regime_scale`                    |
| 回测 Sharpe | 0.58                                  | 预期不变或微升（不再随机丢弃好信号） |

### 保留的替代方案（备忘）

如果未来希望恢复置信度缩放，需满足：
- Evidence 特征池**完全独立于 Gate 特征池**（不同特征家族）
- 或引入 Gate 无法触及的外部信号源（如链上数据、情绪指标、跨市场因子）
- 训练数据必须是 Gate 放行子集，而非全量
