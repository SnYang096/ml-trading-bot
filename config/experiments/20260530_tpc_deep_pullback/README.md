# TPC 深回撤 + 吸收

| 字段 | 值 |
|------|-----|
| 目录 | `20260530_tpc_deep_pullback/` |
| 日期 | 2026-05-30 |
| 策略 | tpc |

## 假设（摘要）

牛市深回踩 + 订单流吸收确认后入场；deny 高 path_efficiency 延续区（留给 BPC）。  
**完整假设表 / promote checklist → [`DECISION.md`](DECISION.md)**

## 物料

- `rd_loop_tpc_deep_pullback.yaml` — Pass 1：plateau + IC（**不定** deep_absorb 组合阈值）
- `rd_loop_tpc_deep_pullback_entry_compare.yaml` — Pass 2：读完 plateau 后填 τ，再跑 H3 vs prod
- `tpc_deep_pullback_ablation_grid.yaml` — Phase 3 E0–E5 ablation（smoke 先 E0）

## 跑法

```bash
# Phase 0 parquet（见 decision doc）
# Phase 1
PYTHONPATH=src:scripts python scripts/rd_loop.py \
  --hypothesis-yaml config/experiments/20260530_tpc_deep_pullback/rd_loop_tpc_deep_pullback.yaml

# Phase 3 smoke（E0 已启用；E4/E5 待变体树）
PYTHONPATH=src:scripts python -m scripts.event_backtest \
  --variant-grid config/experiments/20260530_tpc_deep_pullback/tpc_deep_pullback_ablation_grid.yaml \
  --quiet-signal-logs
```

## 结果产物

- `results/rd_loop/tpc_deep_pullback/`
- `results/tpc/experiments/deep_pullback_ablation/`
- Readout：`scripts/analysis/tpc_holding_readout.py` → `DIAGNOSIS.md`（待写）

## 结论

见 [`DECISION.md`](DECISION.md) 文末（跑完 Pass 1/2/3 后填写）。

## 决策文档

- [`DECISION.md`](DECISION.md) — 假设 H1–H4、Pass 1/2 口径、promote checklist

## 关联

- 6 币 direction promote：`../20260529_tpc_direction_ema_align/` → `results/tpc/experiments/direction_align_promote/`
- 变体树：`config_experiments/tpc_*_strategies/`（Phase 2）
