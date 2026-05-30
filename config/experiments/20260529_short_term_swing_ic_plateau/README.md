# short_term_swing IC plateau

| 字段 | 值 |
|------|-----|
| 目录 | `20260529_short_term_swing_ic_plateau/` |
| 日期 | 2026-05-29 |
| 策略 | short_term_swing |

## 假设

短期树独立策略 IC decay + entry tau plateau 扫描。

## 物料

- `rd_loop_short_term_swing_ic_plateau.yaml`

## 跑法

```bash
PYTHONPATH=src:scripts python scripts/rd_loop.py \
  --hypothesis-yaml config/experiments/20260529_short_term_swing_ic_plateau/rd_loop_short_term_swing_ic_plateau.yaml
```

## 结果产物

- `results/rd_loop/short_term_swing_ic_plateau`

## 结论

TODO（跑完后在此填写 promote / reject 与要点）。

## 关联

- 策略实验树（变体 yaml）：仓库根 `config_experiments/`（本目录不含整棵树）
- 历史决策文档（如有）：`docs/decisions/`（不强制迁入本 README）
