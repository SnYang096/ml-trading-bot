# ME direction 优化

| 字段 | 值 |
|------|-----|
| 目录 | `20260528_me_direction/` |
| 日期 | 2026-05-28 |
| 策略 | me |

## 假设

ME direction 动量级联 + EMA 带通：scan → baseline/smoke/holdout grid。

## 物料

- `rd_loop_me_direction.yaml`
- `me_direction_grid.yaml`
- `me_direction_smoke_grid.yaml`
- `me_direction_v6_holdout_grid.yaml`

## 跑法

```bash
PYTHONPATH=src:scripts python scripts/rd_loop.py \
  --hypothesis-yaml config/experiments/20260528_me_direction/rd_loop_me_direction.yaml

PYTHONPATH=src:scripts python -m scripts.event_backtest \
  --variant-grid config/experiments/20260528_me_direction/me_direction_grid.yaml --quiet-signal-logs
```

## 结果产物

- `results/rd_loop/me_direction`

## 结论

TODO（跑完后在此填写 promote / reject 与要点）。

## 关联

- 策略实验树（变体 yaml）：仓库根 `config_experiments/`（本目录不含整棵树）
- 决策文档：（暂无，跑完后写 `DECISION.md`）
