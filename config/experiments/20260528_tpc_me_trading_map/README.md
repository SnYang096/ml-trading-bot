# TPC × ME 交易地图

| 字段 | 值 |
|------|-----|
| 目录 | `20260528_tpc_me_trading_map/` |
| 日期 | 2026-05-28 |
| 策略 | tpc, me |

## 假设

TPC 与 ME 在 bull/bear 双段上的交易地图对照（cross-strategy grid）。

## 物料

- `tpc_me_trading_map_bull_bear.yaml`

## 跑法

```bash
PYTHONPATH=src:scripts python -m scripts.event_backtest \
  --variant-grid config/experiments/20260528_tpc_me_trading_map/tpc_me_trading_map_bull_bear.yaml --quiet-signal-logs
```

## 结果产物

- `results/tpc/experiments/（见 grid output_dir）`

## 结论

TODO（跑完后在此填写 promote / reject 与要点）。

## 关联

- 策略实验树（变体 yaml）：仓库根 `config_experiments/`（本目录不含整棵树）
- 决策文档：（暂无，跑完后写 `DECISION.md`）
