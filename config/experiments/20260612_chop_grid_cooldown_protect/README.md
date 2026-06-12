# chop_grid_cooldown_protect

## Phase checklist

| Phase | 命令 | 产物 |
|-------|------|------|
| 0 | `mlbot train final --prepare-only` 或 Tier-0 parquet | `features_labeled.parquet` |
| 1 | `rd_loop` phase1 yaml | `quick_scan/` + `monitor_bundle/` draft |
| 2 | 人读 scan → `DECISION.md` | τ / lookback |
| 3 | `event_backtest --variant-grid` | segment R |
| 4 | trading maps | 语义核对 |
| 5 | **`mlbot research promote-baseline`** | git baseline |

## Phase 1

```bash
PYTHONPATH=src:scripts python scripts/rd_loop.py \
  --hypothesis-yaml config/experiments/chop_grid_cooldown_protect/rd_loop_chop_grid_cooldown_protect_phase1.yaml
```

## Phase 5

```bash
mlbot research promote-baseline \
  --experiment-dir config/experiments/chop_grid_cooldown_protect \
  --enable-drift-ready
```

标定 segment: **bear_2022,bull_2023_2024,recent_range_to_bear,recent_6m_oos**

See [`LAYER_PROMOTION_CRITERIA.md`](../LAYER_PROMOTION_CRITERIA.md) §4.3 and
[`docs/strategy/研发与监控打通_CN.md`](../../docs/strategy/研发与监控打通_CN.md).
