# chop_grid OOS tuning experiments (2026-06-03)

## Scope

Live-aligned execution (1min subbar), spacing/regime/prefilter/box_prefilter sweeps on `recent_6m_oos`.

## Artifacts (gitignored under `results/`)

| Run | Path |
|-----|------|
| Exec align | `results/chop_grid/experiments/exec_align_20260603/` |
| Layer sweep | `results/chop_grid/experiments/oos_layer_sweep_20260603/` |
| Phase-2 joint | `results/chop_grid/experiments/oos_phase2_20260603/` |
| box_prefilter | `results/chop_grid/experiments/box_prefilter_sweep_20260603/` |

## Headline

- Prod baseline OOS **-0.67%** → tuned stack **~+1.16%** (55 trades; thin sample).
- Spacing **1.18 / 1.1%** + box_pos **0.40–0.60** main levers; regime 0.52/0.33 marginal vs 0.50/0.32.
- `box_prefilter` thresholds inert under tight box_pos (spread ~0.06 pp).
- Backtest `block_stable_box` aligned with live (`exclude_box_prefilter=false`).

## Reproduce

```bash
python scripts/sweep_chop_oos_layers.py
python scripts/sweep_chop_oos_phase2.py
python scripts/sweep_chop_box_prefilter.py --compare-no-block
```

## Archetype promote (research)

See `config/strategies/chop_grid/archetypes/{regime,prefilter,execution}.yaml`.
