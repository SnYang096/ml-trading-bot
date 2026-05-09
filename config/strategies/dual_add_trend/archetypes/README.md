# dual_add_trend archetypes

| File | Role |
|------|------|
| `prefilter.yaml` | Multi-leg prefilter layer (trend/chop/box regime gates). |
| `execution.yaml` | Multi-leg execution/risk layer (inventory/add/tp/fees/risk caps). |
| `regime_thresholds.yaml` | Trend vs chop regime thresholds for bounded multi-leg inventory (≈ first research ring). |

See `docs/z实验_011_chopgrid_dualadd/README.md` and ADR §12.3 for column and multi-leg calibration responsibilities.
