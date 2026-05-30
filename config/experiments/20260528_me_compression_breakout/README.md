# ME 压缩突破分层

| 字段 | 值 |
|------|-----|
| 目录 | `20260528_me_compression_breakout/` |
| 日期 | 2026-05-28 |
| 策略 | me |

## 假设

CompressionBreakout：regime 去 box + 分层 rd_loop + no_box 变体 grid。

## 物料

- `rd_loop_me_compression_breakout.yaml`
- `me_regime_no_box_grid.yaml`

## 跑法

```bash
PYTHONPATH=src:scripts python scripts/rd_loop.py \
  --hypothesis-yaml config/experiments/20260528_me_compression_breakout/rd_loop_me_compression_breakout.yaml

PYTHONPATH=src:scripts python -m scripts.event_backtest \
  --variant-grid config/experiments/20260528_me_compression_breakout/me_regime_no_box_grid.yaml --quiet-signal-logs
```

## 结果产物

- `results/rd_loop/me_compression_breakout`

## 结论

TODO（跑完后在此填写 promote / reject 与要点）。

## 关联

- 策略实验树（变体 yaml）：仓库根 `config_experiments/`（本目录不含整棵树）
- 历史决策文档（如有）：`docs/decisions/`（不强制迁入本 README）
