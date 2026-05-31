# short_term_swing — IC prune (H≈20 bar)

**实验目录：** `config/experiments/20260531_short_term_swing/`  
**rd_loop：** `rd_loop_short_term_swing_ic_plateau.yaml`  
**IC 规则：** `config/strategies/tree_strategies/short_term_swing/ic_screen.yaml`  
**Holdout：** 2025-10-01 → 2026-04-01（6 币 pooled）  
**标签：** `label` = signed forward RR @ **H=20**

## 决策（当前）

| 项 | 结论 |
|----|------|
| **6 币一体 live** | **reject** — holdout Pearson / τ 均为负 |
| **Pass 2（canonical）** | `ic_screen.yaml`：**horizons 含 H=50** + peak∈{10,20} → **6 列**（vol/ema 在 H=50 更强 → 正确剔除） |
| **下一步** | 补 `invert_features`（`bb_width`/`macd_atr` 负 IC）；扩 slow 假设池；勿去掉 H=50 换更多列 |

---

## Pass 2 canonical（2026-05-31）— `ic_screen.yaml` + 6 列

**规则（设计文档 §5.1 完整版）：**

| 字段 | 值 |
|------|-----|
| `horizons` | 1,3,5,10,15,20,**50** |
| `allowed_best_lags` | {10, 20} |
| `reject_peak_at` | 50 |
| Prepare 池 | slow 假设 9 node + `atr_f`（非 fast_scalp evt/box） |

**被 H=50 扫描剔除的 4 列（峰值挪到 lag=50，不属于 swing slug）：**  
`vol_leverage_asymmetry`, `vol_persistence`, `vol_clustering_strength`, `ema_1200_slope_10`

**保留 6 列（peak∈{10,20}）：**  
`bb_width_normalized_pct`, `macd_atr`, `wpt_ignition_score`, `wpt_absorption_score`, `wpt_exhaustion_score`, `wpt_compression_score`

**流水线：**

1. **prepare-only** → `prepare_slow_h20_20260531/`
2. **ic-prune** → `ic_prune_ic_screen_h50_6cols/`
3. **train** → `train_ic_screen_6cols_20260531/`（**6** 模型列 + atr）
4. **τ scan** → `holdout_rr_ic_screen_6cols/`

**关键数字：**

| 指标 | Pass 2 canonical (6 col) | Pass 2  interim (10 col, 无 H=50) | Pass 1 宽池 |
|------|--------------------------|-------------------------------------|-------------|
| IC pass | **6** | 10 | 150 → top-20 |
| Holdout Pearson | **-0.061** | -0.086 | -0.114 |
| CV metric | -0.006 | +0.037 | +0.034 |
| Pooled Sharpe @ q=0.05 | **-2.25** | -0.60 | -0.95 |
| 推荐 τ | q=0.30（仍负） | q=0.15 | q=0.05 |

**判读：** 6 列版 Pearson 略好于 10 列 interim，但 τ 更差；**仍 reject**。canonical 口径以 **含 H=50 的 ic_screen** 为准，不以 interim 10 列为 promote 依据。

---

## Pass 1（已 supersede）— 宽池误用 fast_scalp 节点

prepare `prepare_20260531/` → ic-prune lag≤20 无白名单 → Pearson **-0.114**, τ **-0.95**。

---

## 产物

| 路径 | 内容 |
|------|------|
| `ic_screen.yaml` | **canonical IC 规则** |
| `ic_prune_ic_screen_h50_6cols/` | 6 列 IC 表 |
| `train_ic_screen_6cols_20260531/` | **当前 artifact（勿 promote）** |
| `holdout_rr_ic_screen_6cols/` | τ |
| `ic_prune_label_lag1020/` | interim 10 列（无 H=50，仅对照） |
| `train_slow_lag1020_top20_20260531/` | interim train |
| `features.yaml` | 6 列 singleton 写回 |
