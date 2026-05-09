# Dual add trend research pipeline entrypoints

This directory mirrors the BPC profile naming:

- `turbo.yaml`: fixed features/regime thresholds, monthly profile calibration.
- `slow.yaml`: slower structural health check plus monthly profile calibration.
- `non_rolling.yaml`: full-window static holdout validation before gate/review.

Strategy research metadata and profile candidates now live in:

- `turbo.yaml`: `study`, `threshold_search`, `calibration_profiles`.
- `slow.yaml` / `non_rolling.yaml`: stage-specific overrides via `extends`.

Rolling exports adoptable bundles under `results/dual_add_trend/<history>/dual_add_trend/<timestamp>/strategies/dual_add_trend/` (see `auto_research_pipeline` multileg export).
