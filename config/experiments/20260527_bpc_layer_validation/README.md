# BPC 分层验证

| 字段 | 值 |
|------|-----|
| 目录 | `20260527_bpc_layer_validation/` |
| 日期 | 2026-05-27 |
| 策略 | bpc |

## 假设

BPC regime/prefilter/direction/gate/entry 分层扫描 + ABH gate 变体因果验证。

## 物料

- `rd_loop_bpc.yaml`
- `bpc_abh_variant_grid.yaml`

## 跑法

```bash
PYTHONPATH=src:scripts python scripts/rd_loop.py \
  --hypothesis-yaml config/experiments/20260527_bpc_layer_validation/rd_loop_bpc.yaml

PYTHONPATH=src:scripts python -m scripts.event_backtest \
  --variant-grid config/experiments/20260527_bpc_layer_validation/bpc_abh_variant_grid.yaml --quiet-signal-logs
```

## 结果产物

- `results/rd_loop/bpc`

## 结论

TODO（跑完后在此填写 promote / reject 与要点）。

## 关联

- 策略实验树（变体 yaml）：仓库根 `config_experiments/`（本目录不含整棵树）
- 历史决策文档（如有）：`docs/decisions/`（不强制迁入本 README）
