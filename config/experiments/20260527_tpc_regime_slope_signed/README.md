# TPC regime slope signed

| 字段 | 值 |
|------|-----|
| 目录 | `20260527_tpc_regime_slope_signed/` |
| 日期 | 2026-05-27 |
| 策略 | tpc |

## 假设

TPC regime EMA slope 分符号变体：offline IC + 双段 grid（grid 待 F' 变体完成后链 variant_grid）。

## 物料

- `rd_loop_tpc_regime_slope_signed.yaml`
- `tpc_regime_slope_signed_grid.yaml`

## 跑法

```bash
PYTHONPATH=src:scripts python scripts/rd_loop.py \
  --hypothesis-yaml config/experiments/20260527_tpc_regime_slope_signed/rd_loop_tpc_regime_slope_signed.yaml

PYTHONPATH=src:scripts python -m scripts.event_backtest \
  --variant-grid config/experiments/20260527_tpc_regime_slope_signed/tpc_regime_slope_signed_grid.yaml --quiet-signal-logs
```

## 结果产物

- `results/rd_loop/tpc_slope_signed`

## 结论

TODO（跑完后在此填写 promote / reject 与要点）。

## 关联

- 策略实验树（变体 yaml）：仓库根 `config_experiments/`（本目录不含整棵树）
- 历史决策文档（如有）：`docs/decisions/`（不强制迁入本 README）
