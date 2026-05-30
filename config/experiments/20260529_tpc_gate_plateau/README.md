# TPC gate plateau

| 字段 | 值 |
|------|-----|
| 目录 | `20260529_tpc_gate_plateau/` |
| 日期 | 2026-05-29 |
| 策略 | tpc |

## 假设

TPC gate 语义 chop 特征 plateau + lift；链 tpc_gate_refinement_grid（待建）。

## 物料

- `rd_loop_tpc_gate_plateau.yaml`

## 跑法

```bash
PYTHONPATH=src:scripts python scripts/rd_loop.py \
  --hypothesis-yaml config/experiments/20260529_tpc_gate_plateau/rd_loop_tpc_gate_plateau.yaml

PYTHONPATH=src:scripts python -m scripts.event_backtest \
  --variant-grid config/experiments/20260529_tpc_gate_plateau/tpc_gate_refinement_grid.yaml --quiet-signal-logs
```

## 结果产物

- `results/rd_loop/tpc_gate_plateau`

## 结论

TODO（跑完后在此填写 promote / reject 与要点）。

## 关联

- 策略实验树（变体 yaml）：仓库根 `config_experiments/`（本目录不含整棵树）
- 历史决策文档（如有）：`docs/decisions/`（不强制迁入本 README）
