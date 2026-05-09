# Dual add trend research pipeline entrypoints

This directory mirrors the BPC profile naming:

- `turbo.yaml`: fixed features/regime thresholds, monthly profile calibration.
- `slow.yaml`: slower structural health check plus monthly profile calibration.
- `non_rolling.yaml`: full-window static holdout validation before gate/review.

The profile YAMLs use the same pipeline language as BPC:

- `threshold_calibration` controls which layers calibrate.
- `strategies.dual_add_trend.kpi_gates` holds KPI constraints.
- `dual_add_backtest` holds execution/report parameters.
- Strategy-owned calibration candidates are selected by the multi-leg dispatcher in code.

Rolling exports adoptable bundles under `results/dual_add_trend/<history>/dual_add_trend/<timestamp>/strategies/dual_add_trend/` (see `auto_research_pipeline` multileg export).
