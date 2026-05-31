# BPC entry v2

| 字段 | 值 |
|------|-----|
| 目录 | `20260527_bpc_entry_v2/` |
| 日期 | 2026-05-27 |
| 策略 | bpc |

## 假设

BPC entry 层 v2 阈值与触发器 offline + event_backtest 验证。

## 物料

- `rd_loop_bpc_entry.yaml`
- `bpc_entry_v2_grid.yaml`

## 跑法

```bash
PYTHONPATH=src:scripts python scripts/rd_loop.py \
  --hypothesis-yaml config/experiments/20260527_bpc_entry_v2/rd_loop_bpc_entry.yaml

PYTHONPATH=src:scripts python -m scripts.event_backtest \
  --variant-grid config/experiments/20260527_bpc_entry_v2/bpc_entry_v2_grid.yaml --quiet-signal-logs
```

## 结果产物

- `results/rd_loop/bpc_entry`

## 结论

TODO（跑完后在此填写 promote / reject 与要点）。

## 关联

- 策略实验树（变体 yaml）：仓库根 `config_experiments/`（本目录不含整棵树）
