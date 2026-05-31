# SRB entry plateau

| 字段 | 值 |
|------|-----|
| 目录 | `20260527_srb_entry_plateau/` |
| 日期 | 2026-05-27 |
| 策略 | srb |

## 假设

SRB entry 特征 plateau / IC 扫描（rd_loop step 1）。

## 物料

- `rd_loop_srb_entry_plateau.yaml`

## 跑法

```bash
PYTHONPATH=src:scripts python scripts/rd_loop.py \
  --hypothesis-yaml config/experiments/20260527_srb_entry_plateau/rd_loop_srb_entry_plateau.yaml
```

## 结果产物

- `results/rd_loop/srb_entry_plateau`

## 结论

TODO（跑完后在此填写 promote / reject 与要点）。

## 关联

- 策略实验树（变体 yaml）：仓库根 `config_experiments/`（本目录不含整棵树）
- 决策文档：（暂无，跑完后写 `DECISION.md`）
