# ME prefilter v4

| 字段 | 值 |
|------|-----|
| 目录 | `20260528_me_prefilter_v4/` |
| 日期 | 2026-05-28 |
| 策略 | me |

## 假设

ME prefilter v4 漏斗修复：fix / funnel / smoke 三档 grid。

## 物料

- `me_prefilter_fix_grid.yaml`
- `me_prefilter_v4_funnel_grid.yaml`
- `me_prefilter_v4_smoke_grid.yaml`

## 跑法

```bash
PYTHONPATH=src:scripts python -m scripts.event_backtest \
  --variant-grid config/experiments/20260528_me_prefilter_v4/me_prefilter_v4_funnel_grid.yaml --quiet-signal-logs
```

## 结果产物

- `results/me/experiments/`

## 结论

TODO（跑完后在此填写 promote / reject 与要点）。

## 关联

- 策略实验树（变体 yaml）：仓库根 `config_experiments/`（本目录不含整棵树）
- 历史决策文档（如有）：`docs/decisions/`（不强制迁入本 README）
