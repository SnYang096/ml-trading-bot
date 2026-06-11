# 20260612_tpc_e22_monitor_baseline

TPC E22 labeled regime monitoring baseline (regime_shares + PSI ref).

## Status

- **标定窗**: `results/monitoring/tier0/tpc_20260612/features_labeled_recent_6m_oos.parquet`（4270 rows，来自 `train_final_20260610_adx` · `recent_6m_oos` 2025-10~2026-03）
- **Draft + Promote**（2026-06-11）：smoke `drift_exit=0`, `watchdog_exit=0`
- **regime_shares**: bull **0%**, bear **82.1%**, neutral **17.9%**（E22 规则在该窗的真实分布）

## Commands

```bash
# Phase 1 (after updating features_parquet path)
PYTHONPATH=src:scripts python scripts/rd_loop.py \
  --hypothesis-yaml config/experiments/20260612_tpc_e22_monitor_baseline/rd_loop_20260612_tpc_e22_monitor_baseline_phase1.yaml

# Phase 5
mlbot research promote-baseline \
  --experiment-dir config/experiments/20260612_tpc_e22_monitor_baseline \
  --enable-drift-ready
```

See [`docs/strategy/研发与监控打通_CN.md`](../../docs/strategy/研发与监控打通_CN.md).
