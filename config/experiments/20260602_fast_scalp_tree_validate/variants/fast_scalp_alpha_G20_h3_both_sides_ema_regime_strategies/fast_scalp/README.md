# fast_scalp — 唯一树策略模板（pooled 6 币）

| 层 | 路径 |
|----|------|
| 模板（本目录） | `config/strategies/tree_strategies/fast_scalp/` |
| 实验 override / 两轨验证 | [`config/experiments/20260602_fast_scalp_tree_validate/`](../../../experiments/20260602_fast_scalp_tree_validate/) |
| 冻结 G 快照 | `config_experiments/fast_scalp_alpha_G*_.../fast_scalp/` |

**不再 fork 策略 slug**（`fast_scalp_alts`、`fast_scalp_majors`、`fast_scalp_realized_g5` 等已删除）。
币种子集、label override、gate 池均在实验层用 `symbols` / `overrides/` / `cohorts.yaml` 表达。

训练与 event 命令：[`TRAINING.md`](../../../experiments/20260602_fast_scalp_tree_validate/TRAINING.md)  
结论：[`DECISION.md`](../../../experiments/20260602_fast_scalp_tree_validate/DECISION.md)
