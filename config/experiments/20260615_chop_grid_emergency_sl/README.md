# chop_grid Emergency SL A/B Test

## Phase A — spacing × mult (`per_leg_sl_spacing_mult`)

See `chop_grid_emergency_sl_grid.yaml` and `results/.../emergency_sl_20260615/`.

## Phase B — entry-% emergency STOP (`emergency_stop_loss.trigger_pct`)

Per-leg entry price; simulates exchange STOP_MARKET. Independent of grid spacing SL.

| Variant | `trigger_pct` | SL distance |
|---------|---------------|-------------|
| **baseline** | off | — |
| **em_12** | 0.12 | -12% from leg entry |
| **em_15** | 0.15 | -15% from leg entry |
| **em_20** | 0.20 | -20% from leg entry |

```bash
PYTHONPATH=src:scripts python -m scripts.event_backtest \
  --variant-grid config/experiments/20260615_chop_grid_emergency_sl/chop_grid_emergency_sl_entry_pct_grid.yaml
```

Judgement (from design doc): promote if `emergency_sl` rate < 5% and Sharpe drop < 10% vs baseline.

## Phase C — extreme-window stress

Windows in `market_segment_stress.yaml`: `bear_2022`, `covid_crash_2020`, `luna_crash_2022`, `ftx_crash_2022`.

```bash
PYTHONPATH=src:scripts python -m scripts.event_backtest \
  --variant-grid config/experiments/20260615_chop_grid_emergency_sl/chop_grid_emergency_sl_stress_grid.yaml
```

Results: `results/chop_grid/experiments/emergency_sl_stress_20260615/QUICK_SUMMARY.md`
