# 准备 fast_scalp alpha rebuild 策略树快照（TPC 流程）

与 TPC gate 系列一致：**变体 = `config_experiments/` 下整棵冻结树**，grid 里只改 `strategies_root`，不用 deploy 上的 `_experiment/` overlay。

**训练配置不在此目录。** 模板见 `config/strategies/tree_strategies/fast_scalp`；实验 override 见 [`overrides/`](overrides/)；命令见 [`TREE_TRAINING_PLAN.md`](TREE_TRAINING_PLAN.md)。

## 一键生成快照（G0–G16）

完整编号含义见 **[`G_VARIANTS.md`](G_VARIANTS.md)**。

```bash
PYTHONPATH=src:scripts python scripts/research/prepare_fast_scalp_alpha_snapshots.py
```

| 快照 | diff vs deploy |
|------|----------------|
| `fast_scalp_alpha_G0_baseline_strategies` | deploy 镜像（无 `_experiment/`） |
| `fast_scalp_alpha_G1_short_only_strategies` | `direction_filter: short`（alts + majors） |
| `fast_scalp_alpha_G2_regime_off_strategies` | `fast_scalp/archetypes/regime.yaml` → `rules: []` |
| `fast_scalp_alpha_G3_short_regime_off_strategies` | G1 + G2 |
| `fast_scalp_alpha_G4_exec_timeout_strategies` | `initial_r: 50` timeout-only execution |
| `fast_scalp_alpha_G5_short_regimeoff_tight_exec_strategies` | G3 + tight SL/TP（SL 1.5R / TP 1.0R / H=6） |
| `fast_scalp_alpha_G6_short_regimeoff_trail_exec_strategies` | G3 + trail TP |
| `fast_scalp_alpha_G7_dual_head_strategies` | dual_head direction block |
| `fast_scalp_alpha_G8_short_regimeoff_gate_strategies` | G3 + OOS adverse gate |
| `fast_scalp_alpha_G9_short_wide_tight_regimeon_strategies` | short + wide+tight exec + regime ON |
| `fast_scalp_alpha_G10_short_wide_tight_regimeoff_strategies` | short + wide+tight exec + regime OFF |
| `fast_scalp_alpha_G11_short_wide_tight_regimeon_gate_strategies` | G9 + gate |
| `fast_scalp_alpha_G12_short_regimeoff_gate_tight_exec_strategies` | gate + G5 tight exec |
| `fast_scalp_alpha_G13_short_regimeoff_gate_wide_tight_exec_strategies` | gate + G10 wide exec |
| `fast_scalp_alpha_G14_g5label_g5exec_strategies` | G5 exec（g5-label score 注入） |
| `fast_scalp_alpha_G15_g10label_g10exec_strategies` | G10 exec（g10-label score 注入） |
| `fast_scalp_alpha_G16_g5label_g5exec_gate_strategies` | g5-label score + gate + G5 exec |

每棵树 **仅含** `fast_scalp/`（单包）。币种子集在 event grid 的 `symbols:` 指定，见 [`cohorts.yaml`](cohorts.yaml)。

```bash
diff -ru config/experiments/20260530_fast_scalp_alts_majors/variants/fast_scalp_alpha_G0_baseline_strategies/fast_scalp/archetypes/direction.yaml \
         config/experiments/20260530_fast_scalp_alts_majors/variants/fast_scalp_alpha_G1_short_only_strategies/fast_scalp/archetypes/direction.yaml
```

## Grid 引用

| Phase | Grid yaml | G 范围 |
|-------|-----------|--------|
| 0 | `fast_scalp_alpha_phase0.yaml` | G0–G3 |
| 4 exec | `fast_scalp_exec_grid.yaml` | G4–G6 |
| 3 dual head | `fast_scalp_dual_head_validate.yaml` | G7 |
| 3 gate | `fast_scalp_gate_validate.yaml` | G8 |
| 5 trend exec | `fast_scalp_trend_style_exec_grid.yaml` | G9–G11 |
| 5 gate×exec | `fast_scalp_gate_exec_combo.yaml` | G12–G13 |
| Step 0 复验 | `fast_scalp_step0_tp_fix.yaml` | G5/G12/G13 |
| E aligned | `fast_scalp_execution_aligned_segment.yaml` | G14–G15 vs G5 baseline |
| E gate | `fast_scalp_g16_gate_exec.yaml` | G14/G16 |

每 run 指定 `strategies_root: config_experiments/fast_scalp_alpha_G*_...`。

Promote 前：把优胜快照里的 yaml **复制**到 `config/strategies/tree_strategies/`（见 `LAYER_PROMOTION_CRITERIA.md`），不要保留 overlay 路径。
