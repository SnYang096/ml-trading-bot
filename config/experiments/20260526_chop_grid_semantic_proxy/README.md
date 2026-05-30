# chop_grid 语义代理

| 字段 | 值 |
|------|-----|
| 目录 | `20260526_chop_grid_semantic_proxy/` |
| 日期 | 2026-05-26 |
| 策略 | chop_grid |

## 假设

C 层 chop_grid 用语义代理特征做 baseline / recent 双段 event_backtest。

## 物料

- `chop_grid_semantic_proxy_grid.yaml`
- `chop_grid_semantic_proxy_smoke.yaml`

## 跑法

```bash
PYTHONPATH=src:scripts python -m scripts.event_backtest \
  --variant-grid config/experiments/20260526_chop_grid_semantic_proxy/chop_grid_semantic_proxy_grid.yaml --quiet-signal-logs
```

## 结果产物

- `results/chop_grid/experiments/`

## 结论

TODO（跑完后在此填写 promote / reject 与要点）。

## 关联

- 策略实验树（变体 yaml）：仓库根 `config_experiments/`（本目录不含整棵树）
- 历史决策文档（如有）：`docs/decisions/`（不强制迁入本 README）
