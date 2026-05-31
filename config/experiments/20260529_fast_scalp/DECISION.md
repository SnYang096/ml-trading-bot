# fast_scalp Phase 1 — IC 剪枝 + pooled 训练 + holdout τ

**实验目录：** `config/experiments/20260529_fast_scalp/`  
**rd_loop：** `rd_loop_fast_scalp_ic_plateau.yaml`  
**结果根：** `results/rd_loop/fast_scalp_ic_plateau/`  
**Holdout：** 2025-10-01 → 2026-04-01（6 币）

## 假设

| 编号 | 假设 | 结论 |
|------|------|------|
| H1 | lag≤5、\|IC\|≥0.02 筛特征 → 浅树 holdout Pearson 转正 | ✅ Pearson +0.025（弱正） |
| H2 | top-35 node + FeatureStore 命中 → 可重复训练 | ✅ ~15s train，store 全命中 |
| H3 | holdout τ plateau → 可 promote 6 币一体 live | ❌ majors 拖后腿；进入 Phase 2 拆分 |

## 流水线（已跑）

1. **prepare-only** → `results/train_final/fast_scalp/prepare_20260530_140243/`
2. **IC prune**（`mlbot research ic-prune`；Phase 1 时 sidecar `fast_scalp_ic_prune.py`）→ `ic_prune_h5/`
3. **train** top-35 → `train_final_20260530_141451_ic_top35/`
4. **τ scan** → `holdout_rr_top35/`（q=0.05 mean Sharpe 0.45，但 BTC/ETH 负）
5. **更紧 IC** top-20 → `train_final_20260530_145723_ic_top20/`（Pearson +0.029，τ 更稳 q=0.15）

## Phase 1 关键数字（6 币 pooled @ q=0.05）

| 指标 | 值 |
|---|---|
| Holdout Pearson | +0.025 |
| mean Sharpe | 0.45 |
| mean Return | +12.8% |
| SOL/ADA/XRP | 正 |
| BTC/ETH | 负 |

## 决策

- **`fast_scalp` 6 币一体 live：reject**（majors 负贡献）
- **特征/训练管线：保留**（作为 pooled artifact 与 R&D 源）
- **后续：** → [`20260530_fast_scalp_alts_majors/`](../20260530_fast_scalp_alts_majors/) Phase 2 拆分部署

## 产物路径

| 路径 | 内容 |
|---|---|
| `results/rd_loop/fast_scalp_ic_plateau/ic_prune_h5/` | IC 表 |
| `results/train_final/fast_scalp/train_final_20260530_141451_ic_top35/` | **pooled promote artifact** |
| `results/rd_loop/fast_scalp_ic_plateau/holdout_rr_top35/` | 6 币 τ 扫描 |
| `config/strategies/tree_strategies/fast_scalp/features.yaml` | top-35 IC 特征 |

## 验收（2026-05-31，`mlbot research ic-prune` + forward_rr 内核）

**Parquet：** `prepare_20260530_140243/features_labeled.parquet`  
**注意：** fast_scalp labeled parquet 的 target 列是 **`label`**（非 `forward_rr`），IC 剪枝需 `--target label`。

| 步骤 | 结果 |
|------|------|
| ic-prune vs Phase1 top-35 | **34/35 相同**；仅移除 `wpt_volume_energy_f` |
| 重训 artifact | `results/train_final/fast_scalp/train_ic_prune_validate_20260531/` |
| Holdout Pearson | **+0.019**（Phase1 +0.025，略弱） |
| τ scan @ q=0.10 pooled Sharpe | **0.23**（Phase1 @ q=0.05 **0.45**） |
| per-coin | SOL 仍强（Sharpe 2.0）；ADA/XRP 弱于 Phase1；BTC/ETH 仍负 |

**结论：** 新命令 + `label` target 下特征池**高度稳定**；holdout τ **未优于** Phase1 ATR-forward 剪枝模型，**不推翻** Phase1「6 币一体 reject / 拆分 alts+majors」决策。产物见 `results/rd_loop/fast_scalp_ic_plateau/ic_prune_forward_rr/`、`holdout_rr_ic_prune_validate/`。
