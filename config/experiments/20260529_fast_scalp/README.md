# fast_scalp IC plateau（Phase 1）

| 字段 | 值 |
|------|-----|
| 目录 | `20260529_fast_scalp/` |
| 日期 | 2026-05-29（Phase 1 跑数 2026-05-30） |
| 策略 | `fast_scalp`（tree_strategies） |
| 决策 | [`DECISION.md`](DECISION.md) — **6 币一体 reject live**；管线保留 |
| 后续 | [`20260530_fast_scalp_alts_majors/`](../20260530_fast_scalp_alts_majors/) Phase 2 拆分 |

## 假设

IC@H≤5 剪枝 → top-35 浅树 → holdout τ plateau；验证树通道相对 legacy sr_breakout。

## 物料

- `rd_loop_fast_scalp_ic_plateau.yaml` — 扫描编排 + 产物路径锚点
- `fast_scalp_direction_grid.yaml` — dual-period event_backtest（可选）
- `fast_scalp_vs_baseline_grid.yaml` — vs TPC baseline（可选）

## 跑法

```bash
# 一条命令：prepare → IC prune → train → holdout τ scan
PYTHONPATH=src:scripts python scripts/rd_loop.py \
  --hypothesis-yaml config/experiments/20260529_fast_scalp/rd_loop_fast_scalp_ic_plateau.yaml
```

`tree_steps` 模式说明见 [`config/experiments/README.md`](../README.md)。

## 结果产物

- `results/rd_loop/fast_scalp_ic_plateau/`
- `results/train_final/fast_scalp/train_final_20260530_141451_ic_top35/`（pooled artifact）

## 说明

本目录在 2026-05-29 已建 rd_loop **骨架**；Phase 1 实际跑数在 2026-05-30 通过 `scripts/research/*` 完成，**DECISION 与路径锚点于 2026-05-30 回填**，便于实验管理升级。
