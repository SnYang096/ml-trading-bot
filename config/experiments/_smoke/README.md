# CI / 工具 smoke

| 字段 | 值 |
|------|-----|
| 目录 | `_smoke/` |
| 日期 | — |
| 策略 | tpc |

## 假设

rd_loop + variant_grid 管线 smoke（非正式实验）。

## 物料

- `rd_loop_validation_smoke.yaml`
- `rd_loop_tpc_smoke.yaml`
- `tpc_variant_grid_smoke.yaml`

## 跑法

```bash
PYTHONPATH=src:scripts python scripts/rd_loop.py \
  --hypothesis-yaml config/experiments/_smoke/rd_loop_tpc_smoke.yaml

PYTHONPATH=src:scripts python -m scripts.event_backtest \
  --variant-grid config/experiments/_smoke/tpc_variant_grid_smoke.yaml --quiet-signal-logs
```

## 结果产物

- `results/rd_loop/tpc_smoke；results/validation_smoke/rd_loop`

## 结论

TODO（跑完后在此填写 promote / reject 与要点）。

## 关联

- 策略实验树（变体 yaml）：仓库根 `config_experiments/`（本目录不含整棵树）
