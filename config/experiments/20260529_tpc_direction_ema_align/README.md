# TPC direction EMA1200 对齐

| 字段 | 值 |
|------|-----|
| 目录 | `20260529_tpc_direction_ema_align/` |
| 日期 | 2026-05-29 |
| 策略 | tpc |

## 假设

修复 direction 与 structural_exit_ema1200 冲突：signal_match_position_band；segment 重测 + trail 变体 + 6coin promote。

## 物料

- `tpc_direction_align_smoke.yaml`
- `tpc_direction_align_promote_6coin.yaml`
- `tpc_trail_rerun_smoke.yaml`
- `tpc_segment_retest_smoke.yaml`

## 跑法

```bash
PYTHONPATH=src:scripts python -m scripts.event_backtest \
  --variant-grid config/experiments/20260529_tpc_direction_ema_align/tpc_direction_align_smoke.yaml --quiet-signal-logs
```

## 结果产物

- `results/tpc/experiments/direction_align_retest/；results/tpc/experiments/direction_align_promote/`

## 结论

**6 币 promote（prod direction align，`config/strategies`）已完成：**

| 窗口 | trades | CAGR | total R | maxDD | structural_exit_ema1200 |
|------|--------|------|---------|-------|-------------------------|
| bull_2023_2024 | 124 | 5.77% | 13.30 | -7.93% | 1 (0.8%) |
| recent | 34 | 15.79% | 15.93 | -7.73% | 0 (0%) |

秒平 bug 已消除；可进入深回撤实验（`../20260530_tpc_deep_pullback/`）。（跑完后在此填写 promote / reject 与要点）。

## 关联

- 策略实验树（变体 yaml）：仓库根 `config_experiments/`（本目录不含整棵树）
- 历史决策文档（如有）：`docs/decisions/`（不强制迁入本 README）
