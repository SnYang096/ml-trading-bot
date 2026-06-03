# chop_grid archetypes

| File | Role |
|------|------|
| `regime.yaml` | Chop hysteresis (`entry_chop_min` / `exit_chop_below`) + stable-box gate thresholds. |
| `prefilter.yaml` | Feature rules (`rules:`) — e.g. `box_pos_60` band. |
| `execution.yaml` | Grid spacing, inventory caps, segment risk. |

Same layout as TPC (`regime.yaml` + `prefilter.yaml` + …); multileg engine merges all layers via `load_multileg_effective_config`.
