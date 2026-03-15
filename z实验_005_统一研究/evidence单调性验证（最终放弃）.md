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

---

## 附录：Evidence Backtest → Live Sharpe 衰减的结构性分析

> 这是量化系统中非常常见且隐蔽的问题。大量 crypto desk 在 early research 阶段都会遇到：
> backtest Sharpe ≈ 2，live Sharpe ≈ 0.7–1。Evidence 模块往往是衰减最严重的部分。

原因不是单一的，而是 **7 类结构性问题**（按从最常见到最隐蔽排列）。

---

### 问题 1：条件样本偏差（Conditional Sample Bias）

Evidence 本质是 signal 条件下的再筛选。例如 BPC breakout + compression duration > X。

Backtest 会观察到 "compression long → expectancy ↑"，但代价是样本数量急剧下降：

| 条件        | 样本数 | Sharpe |
| ----------- | ------ | ------ |
| signal only | 2000   | 0.6    |
| + evidence  | 120    | 2.0    |

120 个样本的 variance 非常大，live trading 时 Sharpe 必然回归。

---

### 问题 2：Evidence 在学"环境"，不是学"结构"

很多 evidence 特征实际在学习某一阶段的市场环境（regime dependent），而非持久的市场结构。

例如 `funding extreme`：某些年份 funding extreme → squeeze，另一些时期 funding extreme → trend continuation。
Backtest 看起来稳定，实则 regime change → edge 消失。

---

### 问题 3：特征共线性（Feature Collinearity）

很多 evidence 特征是同一信息的不同表达，例如：

- compression duration
- ATR contraction
- range tightness

本质都是 volatility compression。多个共线特征叠加后 backtest confidence ↑，但实际只是重复同一信息。Live trading 时信号没有增加，噪声增加。

---

### 问题 4：隐性 Lookahead Bias

Evidence 很容易不小心使用未来信息。典型案例：

- 用 breakout candle volume 做 evidence
- 但 entry 发生在 breakout candle close

此时 volume 在 entry 时 candle 尚未结束，已构成 lookahead。Backtest 中 volume spike → good trades，live 中 volume 未知，Sharpe 下降。

---

### 问题 5：优化 Trade Ranking 而非 Trade Edge

这是最常见但最难发现的问题。很多 evidence 实际做的是 trade ranking（例如每天 20 个 signal，只做 top 5 evidence），而非提升单笔 trade 的 edge。

Backtest 时排名结构固定，Sharpe ↑；live 时信号数量和 regime 变化，排名结构改变，Sharpe 回落。

---

### 问题 6：单调性幻觉（Distribution Shift）

常见误区是期望 evidence score ↑ → winrate ↑。但现实中常见：

| evidence | winrate | avg R |
| -------- | ------- | ----- |
| low      | 60%     | 1R    |
| mid      | 50%     | 2R    |
| high     | 35%     | 6R    |

只看 winrate 会误以为 evidence 无效；只看 tail 会误以为 evidence 非常强。
真实情况是 **distribution shift**：evidence 改变的是收益分布形状，不是简单胜率。

---

### 问题 7：Crypto 特有 — 流动性与执行拖累

Token 市场的额外问题：backtest 假设 price = mid，live 存在 slippage / spread / impact。
很多 evidence 偏好的 setup（volume spike、liquidation、fast move）恰好是最难成交的场景。
Execution drag 会系统性吃掉 Sharpe。

---

### 三条缓解策略

#### 策略 1：Evidence 不做 Hard Filter，改用 Position Scaling

不做 `if score > threshold` 的二元决策，而是：

```text
size = base_size × evidence_scale
```

这样 edge 不会被 binary decision 放大或完全丢弃。

#### 策略 2：Evidence 特征数量保持极少

经验值：**3–5 个特征**通常最稳定。更多特征 → 共线性 + 过拟合。

#### 策略 3：评估看 Distribution，不看 Winrate

评估 evidence 有效性时应关注：

- **Expectancy**（期望收益）
- **Tail**（尾部收益分布）
- **Drawdown**（最大回撤）

而不是 winrate。

---

### 核心结论

> 很多 evidence 在 backtest 看起来很好，其实只是在历史样本中成功地"排序了运气"，
> 而不是捕捉了稳定结构。

Crypto quant 的经验表明：大多数 token 策略的 Evidence 最终都会收敛到 3 类特征（而非几十个），
这背后是市场 microstructure 的根本限制 — 可被稳定捕捉的独立信息维度极为有限。

---

## 附录：为什么 Evidence 最终收敛到 3 类特征

> 这是量化系统的典型演化规律。很多 crypto desk 一开始有 20–50 个 evidence 特征，
> 但 live trading 1–2 年后通常收敛到 3–5 类核心 evidence。
> 不是因为研究不够，而是因为市场结构只提供极少的独立信息源。

---

### 原因 1：市场真正的"独立信息源"非常少

Token 市场的 microstructure 本质只有三类驱动：

1. **流动性（Liquidity）**
2. **波动率（Volatility）**
3. **杠杆 / 清算（Leverage）**

几乎所有 evidence 特征都可归入这三类：

| 特征                    | 本质       |
| ----------------------- | ---------- |
| compression length      | volatility |
| ATR contraction         | volatility |
| range tightness         | volatility |
| volume spike            | liquidity  |
| price impact efficiency | liquidity  |
| OI change               | leverage   |
| funding                 | leverage   |

看起来特征很多，但只是同一信息的不同表达。几十个特征最终收敛为 volatility / liquidity / leverage 三类。

---

### 原因 2：特征之间高度共线

Crypto 数据中 feature correlation 很高。例如 compression duration、ATR contraction、range width 的相关性经常达到 0.6–0.8。
加入大量类似特征后 edge 不会增加，只是噪声增加，live trading Sharpe 下降。

---

### 原因 3：交易信号的"可解释维度"很少

Token 市场的交易结构本质上只有三种模式：breakout / trend / liquidation。

Evidence 也只是在回答三个问题：

1. Breakout 是否可能成功？
2. Trend 是否健康？
3. Squeeze 是否可能发生？

特征过多只是在重复回答同一个问题。

---

### 原因 4：样本量限制

Token 策略特别严重的问题。假设 5 年数据、1000 个交易，如果 evidence 有 20 个特征，模型自由度过高 → overfit。
所以 desk 通常强制限制 3–5 个核心 evidence。

---

### 收敛后的 Evidence 三大类

大多数成熟 crypto desk 最终收敛到的 evidence 结构：

#### 类别 1：波动率状态（Volatility State）

代表市场是否在积累能量。

- 常见特征：compression duration、ATR contraction、range percentile
- 作用：判断 breakout tail potential

#### 类别 2：流动性 / 参与度（Liquidity / Participation）

代表是否有真实资金参与。

- 常见特征：volume expansion、price impact efficiency、orderflow imbalance
- 作用：判断 move 是否真实

#### 类别 3：杠杆 / 挤压（Leverage / Positioning）

Crypto 特有维度。

- 常见特征：funding rate、OI change、liquidation clusters
- 作用：判断 squeeze / cascade 可能性

---

### 成熟系统的 Evidence 两通道结构

很多 desk 最终采用的结构：

```text
confidence     = f(volatility_state) + f(liquidity)       → position sizing
tail_potential = f(volatility_state) + f(leverage)         → TP scaling
```

实际只需 4–5 个特征（如 compression + volume expansion + funding extreme + OI change）即可覆盖 volatility / liquidity / leverage 三大类信息。

---

### Evidence 过多的危害

| 问题       | 表现                         |
| ---------- | ---------------------------- |
| 过拟合     | backtest 好、live 崩         |
| 信号不稳定 | 同一 setup 得分波动大        |
| 权重漂移   | 特征重要性随 regime 大幅变化 |
| live 衰减  | Sharpe 从 2 → 0.7            |

经验法则：**Evidence ≤ 5**。超过必须证明它带来新的独立信息。

---

### 设计原则

在设计 evidence 时问自己：**这个特征是否属于新的信息类别？**

如果只是同一信息的不同表达（例如 compression duration / range tightness / ATR contraction 都是 volatility compression），则只需保留一个代表性指标。
