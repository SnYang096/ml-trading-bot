# Tree IC: label vs forward_rr — small curated pools

**实验目录：** `config/experiments/20260601_1322_tree_forward_rr_ic_small_pool/`  
**结果根：** `results/rd_loop/tree_forward_rr_ic_small_pool/`  
**分析脚本：** `scripts/research/analyze_feature_family_overfit.py`

---

## 关键时间窗与数据集

| 项 | 值 |
|----|-----|
| **Train** | 2024-01-01 → 2025-10-01（6 币 pooled，120T） |
| **OOS (`recent_6m_oos`)** | 2025-10-01 → 2026-03-31 |
| **特征池** | 各 slug canonical ~20 列 curated 列表（非 880-col wide） |
| **训练目标** | 四臂均用 floored `label` 回归；IC 选材 target 为 `label` 或 `forward_rr_hN` |

---

## 1. 核心对比表（同数据、同 OOS）

| Arm | IC target | n_feat | Pearson | CV | Sharpe@q0.05 | Return%@q0.05 | 推荐 q | Sharpe@推荐 q |
|-----|-----------|--------|---------|-----|--------------|---------------|--------|---------------|
| **A1** fast_scalp | label | 12 | **+0.032** | 0.027 | -0.72 | -17.0 | 0.10 | **+0.57** |
| **A2** fast_scalp | forward_rr_h3 | 12 | **+0.032** | 0.027 | -0.72 | -17.0 | 0.10 | **+0.57** |
| **B1** short_term_swing | label | 17 | -0.060 | 0.023 | -1.59 | -35.9 | 0.25 | -1.30 |
| **B2** short_term_swing | forward_rr_h20 | 17 | -0.060 | 0.023 | -1.59 | -35.9 | 0.25 | -1.30 |
| **wide ref** top-100 | label | 100 | -0.068 | 0.017 | -1.23 | -28.5 | 0.05 | -1.23 |

**结论 1 — IC target（label vs forward_rr）：** 修复 `forward_rr_h*` 泄漏后，**两策略上选材与下游指标完全一致**。rank IC 对单调变换（rr_floor 子集）在 holdout 上产生相同排序。

**结论 2 — 小集合 vs wide：** fast_scalp 小池 Pearson 弱正、τ@q=0.10 6 币 pooled 可正；swing 小池仍 reject；wide top-100 全面更差。

---

## 2. 回答用户的三个问题

### Q1: 小特征集合 + 相同 label/IC 工具，fast_scalp 和 short_term_swing 表现如何？

- **fast_scalp（小池 12 列）**：Holdout Pearson **+0.032**；6 币 pooled @ q=0.10 Sharpe **+0.57**（+8.4%），但 @ q=0.05 仍负（-0.72）。失败率 lift **<1**（选中 trades 略优于 baseline），优于 wide/swing。
- **short_term_swing（小池 17 列）**：Pearson **-0.060**；6 币 pooled τ 全面负（最佳 q=0.25 仍 Sharpe -1.30）。**reject**。

与 5 月 fast_scalp 历史（alt 子集 Sharpe 1.31 @ q=0.05）对比：本次是 **6 币 pooled + recent_6m_oos + 修复泄漏后的小池**；majors/BNB 仍拖后腿，未复现 alt 子集条件 promote。

### Q2: 相同 train/OOS 数据上 label vs forward_rr IC 有无差异？

**无实质差异**（选材相同、train/tau 相同）。  
`forward_rr_hN` 作为 IC target 在概念上更贴近执行层 raw return，但本实验小池上 **不改变 rank IC 排序**。

### Q3: 小集合更好时，是否因大集合 math 特征过拟合？能否分析？

见 `results/rd_loop/tree_forward_rr_ic_small_pool/feature_family_analysis.md`：

| Run | math% (IC选中) | struct% | imp math% | CV−Pearson |
|-----|----------------|---------|-----------|------------|
| fast_scalp small | 50% | 50% | 46% | -0.005 |
| swing small | 35% | 53% | 44% | **+0.082** |
| wide top-100 | 30% | 55% | **52%** | **+0.085** |

**根因分析：**

1. **Wide 池 IC 多重检验**：880+ 列 → 252 pass → top-100 富含 **dtw_*/spectrum_*** 高 |IC| 噪声特征；选中列虽 math% 不高，但 **模型 importance 52% 集中在 math 族**。
2. **Swing 小池仍含 dtw 族**（top-5 全为 dtw/box/cvd），但池子受限，无法引入更多 spurious dtw 变体。
3. **CV−Pearson gap** wide ≈ swing（~0.085），说明过拟合不只在 wide；但 wide **OOS Pearson/τ 更差**。
4. **与规则策略一致**：TPC/BPC/ME/SRB 均用手工小集合（语义特征 + gate/prefilter），不用 `features_all` 自动 IC。树通道应 **默认 curated hypothesis pool**，拒绝 wide auto-IC。

---

## 3. 基础设施改动（已落地）

1. **`forward_rr_hN` 列**：`forward_rr_signed_label.py` + `train_strategy_pipeline.py` prepare 导出。
2. **IC 泄漏修复**：`ic_prune.py` 跳过 `forward_rr_h*` 及 IC target 列（测试 `test_screen_features_skips_forward_rr_h_and_target`）。
3. **4 个 isolated slug**：`*_label_ic_small` / `*_forward_rr_ic_small`。
4. **分析脚本**：`scripts/research/analyze_feature_family_overfit.py`。

---

## 4. Go / No-Go

- [ ] **采用 forward_rr_hN 作为 canonical IC target** — 概念更清晰，但本实验 **无收益**；可文档化推荐，非必须切换。
- [x] **维持 curated small pool**；**reject wide auto-IC**（与 TPC/BPC/ME/SRB 纪律一致）。
- [x] **fast_scalp**：6 币 pooled **仍 reject live**；可继续 alt 子集拆分（Phase2 路径）。
- [x] **short_term_swing**：**reject**（小池 + forward_rr IC 均无效）。

---

## 5. 产物路径

| 内容 | 路径 |
|------|------|
| rd_loop | `config/experiments/20260601_1322_tree_forward_rr_ic_small_pool/rd_loop_tree_forward_rr_comparison.yaml` |
| 特征族分析 | `results/rd_loop/tree_forward_rr_ic_small_pool/feature_family_analysis.md` |
| fast_scalp τ | `.../fast_scalp/tau_label/`、`.../tau_forward_rr_h3/` |
| swing τ | `.../short_term_swing/tau_label/`、`.../tau_forward_rr_h20/` |
| Train | `results/train_final/tree_forward_rr_ic_small_pool/` |

**更新时间：** 2026-06-01（实验完成）
