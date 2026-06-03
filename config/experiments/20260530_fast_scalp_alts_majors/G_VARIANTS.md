# fast_scalp Alpha Rebuild — G 变体编号说明

G 是 **alpha rebuild 实验快照编号**（Grid variant）。每个 G 在 deploy 基线上改 1–2 个 yaml 维度，冻结在 `config_experiments/fast_scalp_alpha_G*_.../`，event backtest 通过 `strategies_root` 引用。

生成脚本：`scripts/research/prepare_fast_scalp_alpha_snapshots.py`

```bash
# 全部快照
PYTHONPATH=src:scripts python scripts/research/prepare_fast_scalp_alpha_snapshots.py

# 指定若干
PYTHONPATH=src:scripts python scripts/research/prepare_fast_scalp_alpha_snapshots.py \
  --only fast_scalp_alpha_G5_short_regimeoff_tight_exec_strategies \
         fast_scalp_alpha_G12_short_regimeoff_gate_tight_exec_strategies
```

---

## 三个实验维度

| 维度 | 改什么 | 典型 patch |
|------|--------|------------|
| **Direction** | 只做空 / 双头 | `direction_filter: short`、dual_head block |
| **Regime** | EMA1200 死区规则 | 清空 `regime.yaml` → `rules: []` |
| **Execution** | SL / TP / H | timeout / tight / trail / wide+tight |
| **Gate** | 入场 adverse veto | OOS `adverse_gate_oos` overlay |
| **Score**（仅 G14–G16） | 树预测来源 | H=3 旧 score vs execution-aligned 新 artifact |

**默认 score 约定：** 除 G14–G16 外，所有 G 使用同一份 **H=3 holdout score**（`results/.../scores/alts_full_history_v2.parquet`），只变规则层（direction / regime / exec / gate）。

**Step 0 分水岭（2026-06-02）：** 含 tight TP 的快照（G5/G6/G9/G10/G12/G13 等）在 `take_profit.r → target_r` 修复前，event 结果中 **TP 实际未生效**；Step 0 重跑后的数字才有效。见 [`DECISION.md`](DECISION.md) §11–§12 脚注。

---

## Phase 0 — Alpha 分解（G0–G3）

| G | 快照名后缀 | 含义 | 相对 G0 改动 |
|---|-----------|------|-------------|
| **G0** | `G0_baseline` | Deploy 基线 | 无（signed 回归 + regime ON + deploy exec） |
| **G1** | `G1_short_only` | 只做空 | `direction_filter: short`（alts + majors） |
| **G2** | `G2_regime_off` | Regime OFF | `regime.yaml` → `rules: []` |
| **G3** | `G3_short_regime_off` | Short + Regime OFF | G1 + G2（多数后续实验的 alpha 底盘） |

Grid：`fast_scalp_alpha_phase0.yaml`

---

## Phase 4 — 执行层 grid（G4–G6，在 G3 上叠 exec）

| G | Exec profile | SL / TP / H |
|---|-------------|-------------|
| **G4** | Timeout | SL 50R，无 TP，H=6 |
| **G5** | **Tight** | SL 1.5R / **TP target_r 1.0** / H=6 ← promote 候选 chassis |
| **G6** | Trail | SL 2.5R + trailing，TP 1.5R，H=12（已否决） |

Grid：`fast_scalp_exec_grid.yaml`、`fast_scalp_alpha_phase4_validate.yaml`

---

## Phase 3 — 双头 / Gate（G7–G8）

| G | 快照名后缀 | 含义 |
|---|-----------|------|
| **G7** | `G7_dual_head` | Dual head direction（long/short 分开训；event 未优于 G5） |
| **G8** | `G8_short_regimeoff_gate` | G3 + **OOS adverse gate**（仍用 timeout/default exec） |

Grid：`fast_scalp_dual_head_validate.yaml`、`fast_scalp_gate_validate.yaml`

---

## Phase 5 — Trend-style exec（G9–G11）

宽 SL + 紧 TP（trend_scalp 风格）：SL 8R / TP 0.12R / H=24。

| G | 快照名后缀 | 组合 |
|---|-----------|------|
| **G9** | `G9_short_wide_tight_regimeon` | Short + wide+tight + **regime ON** |
| **G10** | `G10_short_wide_tight_regimeoff` | Short + wide+tight + **regime OFF** |
| **G11** | `G11_short_wide_tight_regimeon_gate` | G9 方向 + gate（regime ON） |

Grid：`fast_scalp_trend_style_exec_grid.yaml`

---

## Phase 5 — Gate × Exec 组合（G12–G13）

| G | 快照名后缀 | 组合 | Score |
|---|-----------|------|-------|
| **G12** | `G12_short_regimeoff_gate_tight_exec` | G3 gate + **G5 tight exec** | H=3 v2 |
| **G13** | `G13_short_regimeoff_gate_wide_tight_exec` | G3 gate + **G10 wide exec** | H=3 v2 |

Grid：`fast_scalp_gate_exec_combo.yaml`、`fast_scalp_g12_segment_full.yaml`、`fast_scalp_step0_tp_fix.yaml`

---

## Phase E — Execution-aligned label 重训（G14–G16）

换 **树 score 来源**（`fast_scalp_realized_g5` / `fast_scalp_realized_g10` artifact），测 label 与 exec 对齐效果。

| G | 快照名后缀 | Score 来源 | Exec | Gate |
|---|-----------|-----------|------|------|
| **G14** | `G14_g5label_g5exec` | g5-label artifact | G5 tight | off |
| **G15** | `G15_g10label_g10exec` | g10-label artifact | G10 wide | off |
| **G16** | `G16_g5label_g5exec_gate` | g5-label artifact | G5 tight | OOS gate |

Score 产物：

- `results/.../scores/alts_g5label_full_history.parquet`
- `results/.../scores/alts_g10label_full_history.parquet`

Grid：`fast_scalp_execution_aligned_segment.yaml`、`fast_scalp_g16_gate_exec.yaml`  
训练编排：`rd_loop_execution_aligned_labels.yaml`

---

## 继承关系

```
G0 baseline
 ├─ G1 short_only
 ├─ G2 regime_off
 └─ G3 = G1 + G2          ← 多数后续实验的 alpha 底盘
      ├─ G4 / G5 / G6     (+ exec 变体)
      ├─ G7               (+ dual head)
      ├─ G8               (+ gate, 默认 exec)
      ├─ G9 / G10         (+ wide+tight exec)
      ├─ G11              (G10 方向 + gate)
      ├─ G12              (gate + G5 exec)
      └─ G13              (gate + G10 exec)

G14 / G15 / G16           ← 换 score（新 label 树）；exec/gate 同 G5/G10/G16
```

---

## 读结果时的要点

1. **G 编号 ≠ 训练 slug** — G 是 event 快照；`fast_scalp_realized_g5/g10` 是训练用 slug，产物供 G14–G16 注入 score。
2. **同 exec、不同 score** — G5（H=3 score）vs G14（g5-label score）：exec 相同，对比的是 **label 对齐** 是否改善 event KPI。
3. **Promote 口径** — 四段 segment（`bear_2022` | `bull_2023_2024` | `recent_range_to_bear` | `recent_6m_oos`）+ `LAYER_PROMOTION_CRITERIA.md` §8；单窗 recent_6m 不足以下 promote 结论。
4. **TPC 流程** — 变体 = 冻结整树；grid 只改 `strategies_root`；promote 时把优胜 yaml **复制**到 `config/strategies/tree_strategies/`（见 [`PREPARE_STRATEGY_TREE.md`](PREPARE_STRATEGY_TREE.md)）。

---

## 相关文档

| 文件 | 内容 |
|------|------|
| [`PREPARE_STRATEGY_TREE.md`](PREPARE_STRATEGY_TREE.md) | 快照生成与 grid 引用 |
| [`DECISION.md`](DECISION.md) | 各 phase 数值结论与 promote 判决 |
| [`TREE_ENTRY_SCORE_AND_EXECUTION.md`](TREE_ENTRY_SCORE_AND_EXECUTION.md) | 树打分 × 执行底盘方法论 |
| [`fast_scalp_tree_alpha_rebuild_PLAN.md`](fast_scalp_tree_alpha_rebuild_PLAN.md) | 实验矩阵与推进顺序 |
