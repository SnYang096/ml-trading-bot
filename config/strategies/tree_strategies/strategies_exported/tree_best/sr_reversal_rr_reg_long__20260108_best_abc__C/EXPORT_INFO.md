# Exported tree strategy: `sr_reversal_rr_reg_long`

- Tag: `20260108_best_abc`
- Stage: `C`

## Files

- `features.yaml`: **lite** (default) — heavy blocks removed to make iteration fast
- `features_lite.yaml`: same as above
- `features_full.yaml`: exact suggested config from the selected stage

## What was trimmed in lite?

- (nothing)

## Notes

- Lite removes Tier2/3 heavy nodes (DTW/Spectrum/WPT/Hilbert + ticks/orderflow).
- Re-introduce blocks gradually as you refactor and benchmark.

