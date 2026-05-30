# ME prod holdout

| 字段 | 值 |
|------|-----|
| 目录 | `20260528_me_prod_holdout/` |
| 日期 | 2026-05-28 |
| 策略 | me |

## 假设

ME prod 配置 recent holdout 双段 event_backtest。

## 物料

- `me_prod_holdout_grid.yaml`

## 跑法

```bash
PYTHONPATH=src:scripts python -m scripts.event_backtest \
  --variant-grid config/experiments/20260528_me_prod_holdout/me_prod_holdout_grid.yaml --quiet-signal-logs
```

## 结果产物

- `results/me/experiments/`

## 结论

TODO（跑完后在此填写 promote / reject 与要点）。

## 关联

- 策略实验树（变体 yaml）：仓库根 `config_experiments/`（本目录不含整棵树）
- 决策文档：（暂无，跑完后写 `DECISION.md`）
