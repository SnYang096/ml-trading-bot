# fast_scalp alt/majors 拆分部署（Phase 2）

| 字段 | 值 |
|------|-----|
| 目录 | `20260530_fast_scalp_alts_majors/` |
| 日期 | 2026-05-30 |
| 策略 | `fast_scalp_alts` + `fast_scalp_majors` |
| 前置 | [`20260529_fast_scalp/`](../20260529_fast_scalp/) Phase 1 pooled 训练 |
| 决策 | [`DECISION.md`](DECISION.md) — 两 slug **条件 promote** |

## 假设

| 编号 | 假设 |
|------|------|
| H1 | 4 alt 单独重训 OOS 优于 pooled 6 币 → alt 子集 |
| H2 | BTC/ETH 单独重训 + 单独 τ 优于 pooled 6 币 → majors 子集 |
| H3 | 两 slug 并行 PCM 优于 monolithic `fast_scalp` live |

## 物料

- `rd_loop_fast_scalp_alts_majors.yaml` — 训练 / τ 扫描路径锚点
- `fast_scalp_deploy_slugs.yaml` — 部署 slug 与 artifact 对照（机器可读）
- 策略配置：`config/strategies/tree_strategies/fast_scalp_alts/`、`fast_scalp_majors/`

## 跑法

```bash
# 一条命令：filter → τ scan → train → compare
PYTHONPATH=src:scripts python scripts/rd_loop.py \
  --hypothesis-yaml config/experiments/20260530_fast_scalp_alts_majors/rd_loop_fast_scalp_alts_majors.yaml
```

运行前把 `rd_loop_fast_scalp_alts_majors.yaml` 里 `pooled_predictions` 指到 Phase 1 artifact。

## 结果产物

- `results/rd_loop/fast_scalp_ic_plateau/alts_holdout_rr_from_6coin/`
- `results/rd_loop/fast_scalp_ic_plateau/majors_holdout_rr/`
- `results/train_final/fast_scalp_alts/train_final_latest/`
- `results/train_final/fast_scalp_majors/train_final_latest/`

## 结论摘要

- **`fast_scalp_alts`**：用 **pooled 6 币 artifact** + q=0.05（非 4 币重训）
- **`fast_scalp_majors`**：用 **BTC/ETH dedicated artifact** + q=0.08
- 详见 [`DECISION.md`](DECISION.md)
