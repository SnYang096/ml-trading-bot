# fast_scalp IC plateau

| 字段 | 值 |
|------|-----|
| 目录 | `20260529_fast_scalp/` |
| 日期 | 2026-05-29 |
| 策略 | fast_scalp |

## 假设

fast_scalp 独立策略：IC decay + entry plateau → direction / vs TPC baseline grid。

## 物料

- `rd_loop_fast_scalp_ic_plateau.yaml`
- `fast_scalp_direction_grid.yaml`
- `fast_scalp_vs_baseline_grid.yaml`

## 跑法

```bash
PYTHONPATH=src:scripts python scripts/rd_loop.py \
  --hypothesis-yaml config/experiments/20260529_fast_scalp/rd_loop_fast_scalp_ic_plateau.yaml

PYTHONPATH=src:scripts python -m scripts.event_backtest \
  --variant-grid config/experiments/20260529_fast_scalp/fast_scalp_direction_grid.yaml --quiet-signal-logs
```

## 结果产物

- `results/rd_loop/fast_scalp_ic_plateau`

## 结论

TODO（跑完后在此填写 promote / reject 与要点）。

## 关联

- 策略实验树（变体 yaml）：仓库根 `config_experiments/`（本目录不含整棵树）
- 历史决策文档（如有）：`docs/decisions/`（不强制迁入本 README）
