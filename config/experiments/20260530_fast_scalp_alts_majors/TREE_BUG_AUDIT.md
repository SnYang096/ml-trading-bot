# fast_scalp 树实验 — 过程 Bug 审计

区分 **「树 alpha 不行」** vs **「流程 bug 导致结论失真」**。

> **20260602 两轨验证** 结论见 [`../20260602_fast_scalp_tree_validate/DECISION.md`](../20260602_fast_scalp_tree_validate/DECISION.md)。

---

## 历史（20260530 alpha rebuild）

| Bug | 影响 | 状态 |
|-----|------|------|
| **`take_profit.r` 未读** | Step 0 前 G5/G12/G13 等 TP 未生效 | ✅ 已修 + Step 0 重跑 |
| **Score export 用 deploy IC yaml** | score 常数 → 0 trades | ✅ 已改 `predict_tree_from_artifact` |
| **G14 用 H=3 deploy τ** | g5-label 入场失真 | ✅ `G5LABEL_TAU` @ q=0.30 |
| **G16 gate 特征未注入（旧 6 列写死）** | fail-closed → 0 trades | ✅ 见下节 20260602 gate 重写 |

---

## 20260602 两轨验证 — 新增 Bug（已修）

| Bug | 影响 | 修复 |
|-----|------|------|
| `export_tree_scores_from_artifact.py` 漏 `import pandas` | 全历史导出 NameError | `import pandas as pd` |
| τ-scan CLI `--filter-split` | 阶段1 τ-scan 失败 | 用 `--output-dir`；rd_loop 仍可用 `filter_split: holdout` |
| `train_tree_adverse_gate` 用 `_prepare_df` | 只留 holdout + `train_end_date` → **0 训练行** | `_prepare_entry_scores`（不过滤 holdout） |
| gate 用 H=3 阈值 (0.55/0.45) | g5-label 分布下 cross 无 entry | `--long-entry` / `--short-entry` = ranker τ |
| gate `atr` 取自宽 parquet（无列） | excursion 全 None → 0 行 | 从 score parquet 取 `atr` |
| gate `--predictions` 用 holdout-only parquet | 与 train_end 互斥 → 0 行 | 用 export `--save-predictions` 全历史表 |
| export 注入 parquet 仅默认 6 gate 列 | 缺 IC 选的 evt_* 等 → G16 **0 trades** | gate 训练后按 `train_summary.selected_features` 重建 inject |
| `train_tree_dual_head` tz-naive vs UTC | A5 崩溃 | `pd.to_datetime(..., utc=True)` + UTC `train_cut` |
| 文档 `ic_prune_holdout.py` | rd_loop/手工命令失败 | 实为 `ic_prune.py`，`PYTHONPATH=.:src` |
| event grid 无 `inject_scores`（G3/G5） | baseline 变体 0 trades | G3 已加 `h3_baseline_full_history.parquet` 注入 |

---

## Gate 方法学（20260602 起，替代旧「写死 6 列」）

见 `scripts/research/train_tree_adverse_gate.py`：

1. 宽候选池（`overrides/features_gate_candidates.yaml`）→ IC + **真实 MAE lift**（非 `pseudo_ret`）
2. Entry 点：ranker score + **与 ranker 一致的 τ**（`--long-entry` / `--short-entry`）
3. 标签：1min MAE ≥ `mae_bad_r` × ATR → adverse
4. 训练窗：`train_end_date`（如 2025-10-01）之前；**predictions 必须全历史**
5. Export：inject parquet 须含 **`selected_features` 全部列**（见 `rd_loop_track_b` 重建步骤）

本轮选中 8 列：`vol_accel, me_accel_5k, evt_scale, evt_scale_left, me_accel_5k_long, evt_var_99, evt_var_99_left, spectrum_cvd_low_freq_ratio`  
`adverse_avoided=0.145`（`track_b/gate/ic_prune_v2/train_summary.json`）。

---

## 非 Bug（真实信号，20260602）

| 观察 | 解读 |
|------|------|
| g5-label 四段全负 + vector τ-scan 全负 | **ranker 无 edge**（非流程问题） |
| G7 样本内强、OOS −3.75% | **双 head 过拟合** |
| G3 OOS +12.02% | 简单 H=3 short 仍是赢家 |
| G16 OOS +3.61% < G3 | gate 有效但救不了无 edge ranker |
| g10-label 退化 | 8R SL 压扁 label → 拒绝 g10 |

---

## Promote（20260602）

- **不 promote** g5-label ranker、双 head。
- **保留** H=3 short baseline（G3）；**下一步**在 G3 上叠 adverse gate 看 OOS 是否 > +12%。
