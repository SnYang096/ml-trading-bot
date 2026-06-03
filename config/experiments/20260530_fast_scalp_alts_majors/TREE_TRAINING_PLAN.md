# fast_scalp 树模型训练计划（canonical，2026-06）

| 字段 | 值 |
|------|-----|
| 策略模板 | **`config/strategies/tree_strategies/fast_scalp`**（唯一，6 币 pooled） |
| 币种 cohort | **[`cohorts.yaml`](cohorts.yaml)**（实验层 `symbols`，非 strategy fork） |
| 实验计划 | **`config/experiments/20260530_fast_scalp_alts_majors/`** |
| 冻结快照 | **`config_experiments/fast_scalp_alpha_G*_...`** |
| 架构说明 | [`CONFIG_LAYOUT.md`](CONFIG_LAYOUT.md) |
| G 编号 | [`G_VARIANTS.md`](G_VARIANTS.md) |
| Bug 审计 | [`TREE_BUG_AUDIT.md`](TREE_BUG_AUDIT.md) |

---

## 当前验证重点（两轨）

| 轨道 | 假设 | Event 快照 | Promote 门禁 |
|------|------|------------|--------------|
| **A — 双 head** | `P(long_win)` + `P(short_win)` agreement 优于 signed H=3 | **G7** | recent_6m_oos 优于 G3 short-only |
| **B — exec-aligned + gate** | G5 execution-aligned label + IC-prune adverse gate | **G14**（无 gate）/ **G16**（+ gate） | G14 recent_6m 正 + gate 降 adverse 率 |

**已拒绝：** G10 execution-aligned（8R SL 压扁 label，pred 退化）— 不再作为候选。

---

## 公共常量

```bash
export ROOT=/home/yin/trading/ml_trading_bot
export PY="PYTHONPATH=src:scripts"
export CFG="$ROOT/config/strategies/tree_strategies/fast_scalp"
export EXP="$ROOT/config/experiments/20260530_fast_scalp_alts_majors"
export OVR="$EXP/overrides"
export SYMS=BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT   # pooled_6 — 训练默认
export ALTS=SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT                    # alts_4 — 仅 event/score 子集
export MAJORS=BTCUSDT,ETHUSDT                                  # majors_2 — 可选 ablation 行
export TF=120T
export FS_LAYER=features_tree_core_120T_c005db49f7
export HOLDOUT_START=2025-10-01
export HOLDOUT_END=2026-04-01
export TRAIN_START=2024-01-01
export TRAIN_END=2026-04-01
```

---

## 轨道 A — 双 head 模型（G7）

### A1. 基线特征 prepare（H=3 signed label，pooled 6 币）

若已有 `results/train_final/fast_scalp/train_final_*/fast_scalp/features_labeled.parquet` 可跳过。

```bash
cd $ROOT
$PY python scripts/train_strategy_pipeline.py \
  --config $CFG \
  --symbol $SYMS --timeframe $TF \
  --start-date $TRAIN_START --end-date $TRAIN_END \
  --holdout-start-date $HOLDOUT_START --holdout-end-date $HOLDOUT_END \
  --feature-store-layer $FS_LAYER \
  --output-root results/train_final/fast_scalp/prepare_baseline \
  --prepare-only
```

### A2. IC prune → writeback `model_features.yaml`

```bash
$PY python scripts/research/ic_prune_holdout.py \
  --config $CFG \
  --features-parquet results/train_final/fast_scalp/prepare_baseline/fast_scalp/features_labeled.parquet \
  --holdout-start $HOLDOUT_START --holdout-end $HOLDOUT_END \
  --target label --min-ic 0.02 --top-n 20 \
  --writeback-mode columns \
  --out results/rd_loop/fast_scalp_ic_plateau/track_dual_head/ic_prune
```

（或整段 rd_loop：`rd_loop_track_dual_head.yaml` 的 prepare + ic-prune 步。）

### A3. 训练 signed H=3 回归树（dual head 的 feature 来源）

```bash
$PY python scripts/train_strategy_pipeline.py \
  --config $CFG \
  --symbol $SYMS --timeframe $TF \
  --start-date $TRAIN_START --end-date $TRAIN_END \
  --holdout-start-date $HOLDOUT_START --holdout-end-date $HOLDOUT_END \
  --feature-store-layer $FS_LAYER \
  --output-root results/train_final/fast_scalp/train_baseline_h3
```

产物：`…/train_baseline_h3/fast_scalp/predictions.parquet`

### A4. 训练双 binary head

```bash
$PY python scripts/research/train_tree_dual_head.py \
  --config $CFG \
  --predictions results/train_final/fast_scalp/train_baseline_h3/fast_scalp/predictions.parquet \
  --symbols $ALTS \
  --output-dir results/rd_loop/fast_scalp_ic_plateau/track_dual_head/alts \
  --train-end-date 2026-01-01 \
  --horizon 3 --rr-floor 0.30
```

产物：
- `long_head.joblib` / `short_head.joblib`
- `dual_head_holdout.parquet`（含 `score_long`, `score_short`）

### A5. 导出 event score + 生成 G7 快照

```bash
# 按 direction.yaml dual_head agreement 块合并 score → event parquet
# （见 train_tree_dual_head.py 输出的 scored parquet 或专用 export 脚本）

$PY python scripts/research/prepare_fast_scalp_alpha_snapshots.py \
  --only fast_scalp_alpha_G7_dual_head_strategies
```

### A6. Event 四段验证

```bash
$PY python -m scripts.event_backtest \
  --variant-grid $EXP/fast_scalp_dual_head_validate.yaml
```

Event grid 使用 **`strategy: fast_scalp`** + **`symbols:` cohort**（见 `cohorts.yaml`），不用 `fast_scalp_alts` strategy 名。

**对照组：** G3（short + regime off）同窗口 — 见 `fast_scalp_alpha_phase0.yaml`。

**判定：** dual head recent_6m_oos 须 **显著优于** G3；否则维持 short-biased 部署，双 head 不 promote。

---

## 轨道 B — Execution-aligned label + Gate（G14 / G16）

### B1. Prepare execution-aligned label（G5 profile）

```bash
$PY python scripts/train_strategy_pipeline.py \
  --config $CFG \
  --labels $OVR/labels_execution_aligned_g5.yaml \
  --symbol $SYMS --timeframe $TF \
  --start-date $TRAIN_START --end-date $TRAIN_END \
  --holdout-start-date $HOLDOUT_START --holdout-end-date $HOLDOUT_END \
  --feature-store-layer $FS_LAYER \
  --output-root results/train_final/fast_scalp/prepare_exec_aligned_g5 \
  --prepare-only
```

### B2. IC prune（target = execution-aligned label）

```bash
$PY python scripts/research/ic_prune_holdout.py \
  --config $CFG \
  --features-parquet results/train_final/fast_scalp/prepare_exec_aligned_g5/fast_scalp/features_labeled.parquet \
  --holdout-start $HOLDOUT_START --holdout-end $HOLDOUT_END \
  --target label --min-ic 0.02 --top-n 20 \
  --writeback-mode columns \
  --out results/rd_loop/fast_scalp_ic_plateau/track_exec_aligned/ic_prune_g5
```

### B3. 训练 entry ranker（g5-label）

```bash
$PY python scripts/train_strategy_pipeline.py \
  --config $CFG \
  --labels $OVR/labels_execution_aligned_g5.yaml \
  --symbol $SYMS --timeframe $TF \
  --start-date $TRAIN_START --end-date $TRAIN_END \
  --holdout-start-date $HOLDOUT_START --holdout-end-date $HOLDOUT_END \
  --feature-store-layer $FS_LAYER \
  --output-root results/train_final/fast_scalp/train_exec_aligned_g5
```

### B4. Holdout τ-scan（必须用 g5-label 自己的分布）

```bash
$PY python scripts/research/tree_holdout_tau_rr_scan.py \
  --config $CFG \
  --predictions results/train_final/fast_scalp/train_exec_aligned_g5/fast_scalp/predictions.parquet \
  --out results/rd_loop/fast_scalp_ic_plateau/track_exec_aligned/tau_scan_g5 \
  --segment-label recent_6m_oos \
  --quantile-grid "0.05,0.08,0.10,0.12,0.15,0.20,0.25,0.30" \
  --filter-split holdout
```

τ 写入 G14/G16 snapshot：`prepare_fast_scalp_alpha_snapshots.py` 内 `G5LABEL_TAU`（来自 `tau_scan_g5` q=0.10）。

### B5. 导出 full-history score（含 gate 特征列）

```bash
$PY python scripts/research/export_tree_scores_from_artifact.py \
  --artifact-dir results/train_final/fast_scalp/train_exec_aligned_g5/fast_scalp \
  --config $CFG \
  --symbols $ALTS \
  --start-date 2022-01-01 --end-date $HOLDOUT_END \
  --validate-short-entry -0.39530742466452556 \
  --include-gate-features \
  --output results/rd_loop/fast_scalp_ic_plateau/track_exec_aligned/scores/alts_g5label_full_history.parquet
```

退化 score 会写 `.DEGENERATE` 并拒绝 export（勿用 `--no-validate` promote）。

---

### B6. Gate — 宽候选特征 prepare

```bash
$PY python scripts/train_strategy_pipeline.py \
  --config $CFG \
  --features $OVR/features_gate_candidates.yaml \
  --symbol $SYMS --timeframe $TF \
  --start-date $TRAIN_START --end-date $TRAIN_END \
  --output-root results/train_final/fast_scalp/gate_features_wide \
  --prepare-only
```

### B7. Gate — IC-prune + 训练（正规管线）

**Label：** entry 点 1min MAE ≥ 1.5R → adverse（非 pseudo_ret）。  
**特征：** 宽候选池 IC-prune + lift 筛选（`train_tree_adverse_gate.py`）。

```bash
$PY python scripts/research/train_tree_adverse_gate.py \
  --config $CFG \
  --predictions results/train_final/fast_scalp/train_exec_aligned_g5/fast_scalp/predictions.parquet \
  --gate-features results/train_final/fast_scalp/gate_features_wide/fast_scalp/features_labeled.parquet \
  --features-gate-yaml $OVR/features_gate_candidates.yaml \
  --symbols $ALTS \
  --start-date 2022-01-01 --end-date $HOLDOUT_END \
  --train-end-date $HOLDOUT_START \
  --min-abs-ic 0.03 --min-lift 0.05 --top-k 8 \
  --output-dir results/rd_loop/fast_scalp_ic_plateau/track_exec_aligned/gate/ic_prune_v2
```

检查 `train_summary.json`：
- `selected_features`（IC 选出，非写死 6 个）
- `metrics.adverse_avoided` > 0
- `metrics.false_reject_rate` 可接受

### B8. 生成 G14 / G16 快照 + event 四段

```bash
$PY python scripts/research/prepare_fast_scalp_alpha_snapshots.py \
  --only fast_scalp_alpha_G14_g5label_g5exec_strategies \
           fast_scalp_alpha_G16_g5label_g5exec_gate_strategies

$PY python -m scripts.event_backtest \
  --variant-grid $EXP/fast_scalp_g14_g16_revalidation.yaml
```

**对照：**
- G5 + H=3 score（Step 0 修正后 baseline）
- G14 = g5-label + g5 τ + G5 exec（无 gate）
- G16 = G14 + IC-prune gate

---

## rd_loop 一键编排

```bash
# 轨道 A
$PY python scripts/rd_loop.py \
  --hypothesis-yaml $EXP/rd_loop_track_dual_head.yaml

# 轨道 B
$PY python scripts/rd_loop.py \
  --hypothesis-yaml $EXP/rd_loop_track_exec_aligned_gate.yaml
```

---

## 产物目录约定

| run_id | 路径 | 用途 |
|--------|------|------|
| `prepare_baseline` | `results/train_final/fast_scalp/prepare_baseline/` | H=3 label 特征 |
| `train_baseline_h3` | `…/train_baseline_h3/` | dual head 输入 predictions |
| `prepare_exec_aligned_g5` | `…/prepare_exec_aligned_g5/` | g5-label prepare |
| `train_exec_aligned_g5` | `…/train_exec_aligned_g5/` | G14/G16 entry artifact |
| `gate_features_wide` | `…/gate_features_wide/` | gate IC 候选池 |
| `track_dual_head/` | `results/rd_loop/.../track_dual_head/` | 双 head joblib + event score |
| `track_exec_aligned/` | `results/rd_loop/.../track_exec_aligned/` | τ-scan、gate、score export |

**禁止**再创建 `config/strategies/tree_strategies/fast_scalp_*` 实验 slug。

---

## Promote 路径

1. Event segment_matrix 过门禁（`LAYER_PROMOTION_CRITERIA.md`）
2. 优胜 **G 快照** 的 `fast_scalp/archetypes/*` → 复制回 deploy `fast_scalp/archetypes/`
3. Artifact 指针 → `fast_scalp/meta.yaml` 或 deploy 文档（**不再** split alts/majors slug）
4. Gate overlay → `fast_scalp/archetypes/gate.yaml`

---

## 历史 rd_loop（deprecated）

| 文件 | 状态 |
|------|------|
| `rd_loop_execution_aligned_labels.yaml` | 引用废弃 slug `fast_scalp_realized_g5/g10` → 用 **`rd_loop_track_exec_aligned_gate.yaml`** |
| `fast_scalp_alpha_rebuild_rd_loop.yaml` | Phase 0–4 已完成；新工作走两轨 plan |
