# Exported tree strategy: `trend_following`

- Tag: `20260110_tf_fast_blacklist`
- Stage: `C`

## Files

- `features.yaml`: **lite** (default) — heavy blocks removed to make iteration fast
- `features_lite.yaml`: same as above
- `features_full.yaml`: exact suggested config from the selected stage

## What was trimmed in lite?

- `trade_cluster_scene_semantic_scores_f`
- `wpt_cvd_fluctuation_f`

## Notes

- Lite removes Tier2/3 heavy nodes (DTW/Spectrum/WPT/Hilbert + ticks/orderflow).
- Re-introduce blocks gradually as you refactor and benchmark.

