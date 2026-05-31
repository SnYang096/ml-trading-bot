# fast_scalp Phase 1 — IC 剪枝 + pooled 训练 + holdout τ

**实验目录：** `config/experiments/20260529_fast_scalp/`  
**rd_loop：** `rd_loop_fast_scalp_ic_plateau.yaml`  
**结果根：** `results/rd_loop/fast_scalp_ic_plateau/`  
**Holdout（近期段）：** 2025-10-01 → 2026-04-01（6 币 pooled）

## 假设

| 编号 | 假设 | 结论 |
|------|------|------|
| H1 | lag≤5、\|IC\|≥0.02 筛特征 → 浅树 holdout Pearson 转正 | ✅ Pearson +0.025（Phase1 node）；**+0.034（top-20 列 scoped）** |
| H2 | top-35 node + FeatureStore 命中 → 可重复训练 | ✅ ~11–15s train，store 全命中 |
| H3 | holdout τ plateau → 可 promote 6 币一体 live | ❌ majors 拖后腿；→ Phase 2 拆分 alts+majors |
| H4 | IC 列级写回（非 whole node）→ 控维 + τ 更稳 | ✅ 20 列 scoped；τ 网格宽于 Phase1 node 版 |

## 决策（当前）

| 项 | 结论 |
|----|------|
| **6 币一体 live** | **reject**（与 Phase1 一致；scoped 略好但未改 majors 分裂事实） |
| **R&D 管线** | **保留并升级**：`writeback_mode=columns` + `model_features.yaml` + train scoped |
| **Promote 候选 artifact** | `train_top20_cols_scoped_20260531`（20 IC 列，Pearson +0.034） |
| **Live / shadow 前门禁** | dual-period **τ 已跑**（见下）；`event_backtest` 树 score 接线 **待补** |
| **部署路径** | → [`20260530_fast_scalp_alts_majors/`](../20260530_fast_scalp_alts_majors/) |

---

## Phase 1 基线（top-35 node，2026-05-30）

| 指标 | 值 |
|------|-----|
| Holdout Pearson | +0.025 |
| Pooled Sharpe @ q=0.05 | 0.45 |
| SOL/ADA/XRP | 正 |
| BTC/ETH | 负 |

**流水线：** prepare → ic-prune → `train_final_20260530_141451_ic_top35` → `holdout_rr_top35/`

---

## 2026-05-31 迭代：IC 命令 + 列级写回 + scoped train

### 管线改动

| 组件 | 说明 |
|------|------|
| `mlbot research ic-prune` | `writeback_mode=columns`，`top_n_columns=20` |
| `features.yaml` | `atr_f` + **20 列 singleton**（非 whole node） |
| `archetypes/model_features.yaml` | 规则侧风格列清单 + IC 元数据 |
| `determine_feature_columns` | **scoped**：仅 `requested_features` 展开列进模型（修 parquet passthrough 泄漏） |

### 候选 artifact：`train_top20_cols_scoped_20260531`

| 指标 | top-35 node (~199d) | 21d（泄漏 cvd） | **20d scoped** |
|------|---------------------|----------------|--------------|
| Holdout Pearson | +0.019 ~ +0.025 | +0.026 | **+0.034** |
| CV metric | ~-0.007 | +0.019 | **+0.025** |
| Pooled Sharpe @ q=0.05 | 0.23 ~ 0.45 | 0.70 | **0.73** |
| τ 网格 | q>0.05 易负 | 仅 q=0.05 稳 | **q=0.05–0.20 均为正**；峰值 **q=0.15 → 0.76** |

**Per-coin @ q=0.05（scoped）：** SOL **2.18**，ADA 1.45，XRP 1.02；BTC 0.49；ETH/BNB 仍负。

**对比 21d vs 20d scoped：** 去掉 `cvd_change_5_normalized` passthrough 后 Pearson 升、τ plateau 变宽；per-coin 有互换（非全面优于 21d）。

### 产物路径

| 路径 | 内容 |
|------|------|
| `ic_prune_label_top20/` | 列级 IC 表 |
| `config/strategies/tree_strategies/fast_scalp/features.yaml` | 20 列 + `atr_f` |
| `config/strategies/tree_strategies/fast_scalp/archetypes/model_features.yaml` | 列 manifest |
| `train_top20_cols_scoped_20260531/` | **当前最佳 train artifact** |
| `holdout_rr_top20_cols_scoped/` | holdout-only τ（9733 rows，`filter_split=holdout`） |
| `dual_period_top20/H_recent/` | dual-period OOS τ（artifact 推 score） |
| `dual_period_top20/H_bull_2024/` | dual-period in-sample τ（辅助） |
| `results/fast_scalp/experiments/` | event_backtest grid（**0 trades**，待树 score 接线） |
| `train_top20_cols_20260531/` | 泄漏版对照（勿 promote） |

### 验收（`target=label`，rd_loop 重跑）

ic-prune 列级与 node 级在相同 parquet 上 **34 node 一致**；35 node → 199 维问题是 **node 展开**，不是 IC 表本身。列级写回解决控维问题。

---

## Dual-period 验证（2026-05-31，`train_top20_cols_scoped_20260531`）

### 1) `event_backtest --variant-grid`（B 通道同款）

已跑 `fast_scalp_direction_grid.yaml` → `results/fast_scalp/experiments/EXPERIMENT_INDEX.json`

| 段 | 窗口 | 交易数 | Pooled Sharpe | 漏斗 |
|----|------|--------|---------------|------|
| `fast_scalp_recent` | 2025-10-01 → 2026-04-01 | **0** | 0 | `reject_no_direction` 9611，`reject_regime` 4874 |
| `fast_scalp_bull_2024` | 2024-01-01 → 2024-06-30 | **0** | 0 | 同上 |

**原因：** 树通道 `direction.yaml` 依赖 **`score` 列**，当前 `event_backtest` **未加载 ModelArtifact 推理**（只算了 regime 等 archetype 列）。  
→ 对 fast_scalp **不能**用 event_backtest 结果 sign-off；需补树 score 注入，或沿用下述 τ 路径。

### 2) 树通道等价验证：`tree_holdout_tau_rr_scan` + artifact（推荐 interim）

同一 artifact，在两段日期窗上 **现推 score → quantile τ → vectorbt RR**：

| 段 | 窗口 | OOS? | rows | 推荐 τ | Pooled Sharpe | Return% | 6 币 Sharpe>0 |
|----|------|------|------|--------|---------------|---------|---------------|
| **H_recent** | 2025-10-01 → 2026-04-01 | ✅ **OOS**（train &lt; 2025-10-01） | 13176 | **q=0.05** | **0.52** | +10.7% | **5/6**（BNB **-0.53**） |
| **H_bull_2024** | 2024-01-01 → 2024-06-30 | ⚠️ **in-sample**（在训练窗内） | 13104 | q=0.20 | 4.45 | +257% | **6/6** |

**产物：** `results/rd_loop/fast_scalp_ic_plateau/dual_period_top20/{H_recent,H_bull_2024}/`

**H_recent per-coin @ q=0.05：** ADA **1.94**，BTC **0.68**，SOL 0.55，ETH/XRP ~0.2；BNB -0.53。

**H_bull_2024 @ q=0.20（in-sample，仅作 regime 回忆参考）：** 6 币 Sharpe **3.5–5.6** — 数字偏乐观，**不能**当作 OOS promote 依据。

### 3) Pareto 判读（scoped top-20）

| 门禁 | 结论 |
|------|------|
| OOS **H_recent** τ | ✅ 弱正（Sharpe ~0.5 @ q=0.05；q=0.10–0.15 仍正） |
| OOS majors | ⚠️ BNB 仍负；BTC/ETH 弱正（较 Phase1 node 版改善） |
| In-sample bull 2024 | ✅ 全币正，但 **非 OOS** |
| B 通道 event_backtest 双段 | ❌ 0 笔（基础设施未就绪） |

**Promote 结论（6 币一体）：仍 reject** — OOS 段未达「6 币一体 live」；但 **scoped 20 列 artifact + τ 管线**可进入 Phase2 **alts/majors 拆分** 与 shadow。  
**下一步：** ① 补 `event_backtest` 树 score 接线；② Phase2 分 slug 重训 + 各段 τ；③ 可选 walk-forward（bull 段用 **未见过** 的日期窗）。

---

## 验证范围：近期 holdout vs 牛熊 dual-period

### 已做了什么

| 验证 | 状态 | 说明 |
|------|------|------|
| **时间 holdout（OOS）** | ✅ | Train &lt; 2025-10-01；τ on 2025-10-01 → 2026-04-01 |
| **6 币 pooled τ** | ✅ | `tree_holdout_tau_rr_scan` + vectorbt RR |
| **Dual-period τ（artifact）** | ✅ | `dual_period_top20/H_recent` + `H_bull_2024` |
| **Dual-period event_backtest** | ⚠️ 已跑但 **0 trades** | 树 score 未注入；见上节 |
| **分 bull/bear 各训一棵树** | ❌ | 非默认；Phase2 slug 拆分 |

### Dual-period 命令记录

```bash
# B 通道 grid（树通道当前 0 trades — 仅登记 EXPERIMENT_INDEX）
PYTHONPATH=src:scripts python -m scripts.event_backtest --variant-grid \
  config/experiments/20260529_fast_scalp/fast_scalp_direction_grid.yaml

# 树通道有效判据（artifact + 两段日期窗）
# H_recent + H_bull_2024 → results/rd_loop/fast_scalp_ic_plateau/dual_period_top20/
```

### 要不要「一个模型牛熊都跑」？

**训练节奏：** 浅树 ~11s/轮 → **滚动 IC 剪枝 + 重训**即可，不必一根树用多年。

**Dual-period 含义（树 vs B）：**

| 概念 | 含义 |
|------|------|
| **Dual-period τ（当前已做）** | 同一 artifact 在两段历史上推 score + τ RR；**OOS 只看 H_recent** |
| **Dual-period event_backtest（待接线）** | 全 archetype 栈 + execution；与 B Pareto 对齐 |
| **Rolling retrain** | Live 运维节奏；不是 promote 时跳过的理由 |
| **Regime 层** | `regime.yaml` EMA1200 过滤；不是 bull/bear 双模型 |

**结论：** 快训 ≠ 跳过 dual-period；但树通道 **H_bull_2024 在现训练切分下是 in-sample**，真正 promote 门禁以 **H_recent OOS** 为准，bull 段仅作辅助诊断（且需 event_backtest 或 walk-forward 补强）。

---

## 历史记录（Phase1 / 命令验收）

<details>
<summary>Phase1 流水线、6 币 reject、ic-prune 命令验收（折叠）</summary>

### 流水线（已跑）

1. **prepare-only** → `results/train_final/fast_scalp/prepare_20260530_140243/`
2. **IC prune** → `ic_prune_h5/`
3. **train** top-35 → `train_final_20260530_141451_ic_top35/`
4. **τ scan** → `holdout_rr_top35/`（q=0.05 Sharpe 0.45，BTC/ETH 负）
5. **更紧 IC** top-20 node → `train_final_20260530_145723_ic_top20/`（Pearson +0.029，τ q=0.15）

### 验收（2026-05-31，`mlbot research ic-prune` + `label` target）

| 步骤 | 结果 |
|------|------|
| ic-prune vs Phase1 top-35 | 34/35 node 相同 |
| 199 维 node 重训 | Pearson +0.019；τ @ q=0.10 Sharpe 0.23 |
| 结论 | 命令可用；node 写回 **未优于** Phase1 τ |

</details>
