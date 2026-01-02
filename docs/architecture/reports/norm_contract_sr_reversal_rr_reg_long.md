## Normalization Contract Report (config/strategies/sr_reversal_rr_reg_long/features.yaml)

- **total_output_columns**: 52
- **missing_method**: 0
- **raw_columns**: 0

### Sample of normalized/unitless columns (first 20)
- `atr`: **unitless**  (feature=atr_f)
- `rsi`: **bounded_0_100** (0.0, 100.0) (feature=rsi_f)
- `macd`: **unitless**  (feature=macd_f)
- `macd_signal`: **unitless**  (feature=macd_f)
- `macd_histogram`: **unitless**  (feature=macd_f)
- `acceleration_3`: **unitless**  (feature=acceleration_3_f)
- `bb_width_normalized`: **unitless**  (feature=bb_width_f)
- `bb_position`: **unitless**  (feature=bb_width_f)
- `price_range_symmetry`: **zscore_rolling**  (feature=price_range_symmetry_f)
- `wick_upper_ratio`: **bounded_0_1**  (feature=wick_ratios_f)
- `wick_lower_ratio`: **bounded_0_1**  (feature=wick_ratios_f)
- `volume_anomaly`: **zscore_rolling**  (feature=volume_anomaly_f)
- `roc_5`: **zscore_rolling**  (feature=roc_5_f)
- `trend_r2_20`: **bounded_0_1**  (feature=trend_r2_20_f)
- `poc`: **unitless**  (feature=poc_hal_features_close_f)
- `hal_high`: **unitless**  (feature=poc_hal_features_close_f)
- `hal_low`: **unitless**  (feature=poc_hal_features_close_f)
- `hal_mid`: **unitless**  (feature=poc_hal_features_close_f)
- `sqs_hal_high`: **unitless**  (feature=sqs_hal_high_f)
- `sqs_hal_low`: **unitless**  (feature=sqs_hal_low_f)
