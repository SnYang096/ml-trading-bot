# ME entry filter

| 字段 | 值 |
|------|-----|
| 目录 | `20260528_me_entry_filter/` |
| 日期 | 2026-05-28 |
| 策略 | me |

## 假设

ME entry_filter 订单流 trigger 候选扫描（含 orderflow 子 loop）。

## 物料

- `rd_loop_me_entry_filter.yaml`
- `rd_loop_me_entry_filter_orderflow.yaml`

## 跑法

```bash
PYTHONPATH=src:scripts python scripts/rd_loop.py \
  --hypothesis-yaml config/experiments/20260528_me_entry_filter/rd_loop_me_entry_filter.yaml
```

## 结果产物

- `results/rd_loop/me_entry_filter`

## 结论

TODO（跑完后在此填写 promote / reject 与要点）。

## 关联

- 策略实验树（变体 yaml）：仓库根 `config_experiments/`（本目录不含整棵树）
- 历史决策文档（如有）：`docs/decisions/`（不强制迁入本 README）
