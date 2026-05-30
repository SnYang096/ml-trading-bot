# ME gate anti

| 字段 | 值 |
|------|-----|
| 目录 | `20260528_me_gate_anti/` |
| 日期 | 2026-05-28 |
| 策略 | me |

## 假设

ME gate 层 anti 特征扫描与 lift 验证。

## 物料

- `rd_loop_me_gate_anti.yaml`

## 跑法

```bash
PYTHONPATH=src:scripts python scripts/rd_loop.py \
  --hypothesis-yaml config/experiments/20260528_me_gate_anti/rd_loop_me_gate_anti.yaml
```

## 结果产物

- `results/rd_loop/me_gate_anti`

## 结论

TODO（跑完后在此填写 promote / reject 与要点）。

## 关联

- 策略实验树（变体 yaml）：仓库根 `config_experiments/`（本目录不含整棵树）
- 决策文档：（暂无，跑完后写 `DECISION.md`）
