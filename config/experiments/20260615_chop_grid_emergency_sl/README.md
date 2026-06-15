# chop_grid Emergency SL A/B Test

## Hypothesis

Adding a per-leg structural stop-loss (``per_leg_stop_loss``) at a wide spacing
(``per_leg_sl_spacing_mult``) will act as a catastrophic tail-risk hedge without
materially degrading normal-market Sharpe.

## Design

- **Baseline** (``no_sl``):  ``per_leg_stop_loss=false``
- **Weak SL** (``sl_4x``):   ``per_leg_stop_loss=true``, ``per_leg_sl_spacing_mult=4.0``
- **Medium SL** (``sl_6x``): ``per_leg_stop_loss=true``, ``per_leg_sl_spacing_mult=6.0``
- **Strong SL** (``sl_8x``): ``per_leg_stop_loss=true``, ``per_leg_sl_spacing_mult=8.0``

## Metrics

| Metric | Baseline | Weak | Medium | Strong | Judgement |
|--------|----------|------|--------|--------|-----------|
| Sharpe | ? | ? | ? | ? | Higher is better |
| Max Drawdown | ? | ? | ? | ? | Lower is better |
| SL Trigger Rate | 0% | ? | ? | ? | Track false positives |
| Segments with SL exit | 0 | ? | ? | ? | Tail-risk protection count |

## Running

```bash
PYTHONPATH=src:scripts python -m scripts.event_backtest \
  --variant-grid config/experiments/20260615_chop_grid_emergency_sl/chop_grid_emergency_sl_grid.yaml
```
