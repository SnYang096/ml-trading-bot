# Baseline/Derived “Narrow IO” Migration Audit

This repo is migrating feature functions toward a **pure-feature DAG contract**:

- **Explicit inputs** via `required_columns`
- **Explicit outputs** via `output_columns`
- **Narrow inputs** (`pass_full_df: false` + `column_mappings`) whenever possible
- Avoid mutating wide DataFrames inside feature functions (prevents OOM / “wide table” explosions)

This doc is generated from `config/feature_dependencies.yaml` by auditing entries with `pass_full_df: true`.

## Summary

- **pass_full_df: true count**: 115
- **Already migrated in this refactor** (examples):
  - `bb_width` → `compute_bb_width_features_from_series`
  - `roc_5` → `compute_roc_5_from_series`
  - `atr` (Series inputs) → `BaselineFeatureEngineer.compute_atr`
  - Derived rolling/ratio features now narrow:
    - `atr_ratio` → `compute_atr_ratio_from_series`
    - `bb_width_ratio` → `compute_bb_width_ratio_from_series`
    - `compression_score` → `compute_compression_score_from_series`
    - `tbr_ma_5` → `compute_tbr_ma_from_series`
    - `tbr_spike` → `compute_tbr_spike_from_series`

## Highest-risk `pass_full_df: true` features (prioritized)

Heuristic: many required columns and/or many output columns means higher chance of wide-table writes/copies.

- **`vpin_features`** (`extract_order_flow_features`): req=5, out=74  
  - Likely heavy + wide. Candidate for “layered feature store” (`orderflow_v1`) and incremental cache.
- **`extended_volatility_features`** (`extract_extended_volatility_features`): req=2, out=42  
  - Outputs many columns; ensure it returns only declared `output_columns` and avoid intermediate wide df mutation.
- **`dl_sequence_features`** (`compute_dl_sequence_features`): req=5, out=64  
  - Heavy model features; prefer feature-store layer (`heavy_v1`) and strict output trimming.
- **`sr_strength_max` / `sqs_hal_high` / `sqs_hal_low`**: req≈8  
  - Baseline SR structure functions; need careful “pure” refactor since they currently use multiple OHLCV + derived inputs.

## Low-hanging fruit (cheap to migrate next)

These are `pass_full_df:true` but logically **Series → Series**:

- **`range_ratio_5bar`**: req=2, out=1  
  - Candidate: `compute_range_ratio_5bar_from_series(high, low) -> Series`
- **`price_range_symmetry`**: req=3, out=1  
  - Candidate: `compute_price_range_symmetry_from_series(high, low, close) -> Series`
- **`wick_ratios`**: req=4, out=2  
  - Candidate: `compute_wick_ratios_from_series(open, high, low, close) -> DataFrame/tuple(series,series)`
- **Most `talib` single-indicator entries** (req=2–4, out=1–2)  
  - If `compute_talib_indicator` already supports `pass_full_df:false` with `column_mappings`, migrate systematically.

## Notes on `tbr_ma_*`

Currently only **`tbr_ma_5`** is defined in YAML. If you add `tbr_ma_10`, `tbr_ma_20`, etc., they should reuse:

- `compute_tbr_ma_from_series(taker_buy_ratio, window=N)`

with:

```yaml
pass_full_df: false
column_mappings:
  taker_buy_ratio: taker_buy_ratio
```

## Next execution steps

1. Migrate the **low-hanging fruit** baseline series features above.
2. Migrate or isolate heavy/wide groups (`vpin_features`, `extended_volatility_features`) behind feature-store layers, and keep their runtime path slim.
3. (Later) Replace `FEATURE_FUNCTION_MAP` with decorator registration and a console-script entrypoint.


