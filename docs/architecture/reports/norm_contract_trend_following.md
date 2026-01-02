## Normalization Contract Report (config/strategies/trend_following/features.yaml)

- **total_output_columns**: 369
- **missing_method**: 0
- **raw_columns**: 0

### Sample of normalized/unitless columns (first 20)
- `atr`: **unitless**  (feature=atr_f)
- `rsi`: **bounded_0_100** (0.0, 100.0) (feature=rsi_f)
- `macd`: **unitless**  (feature=macd_f)
- `macd_signal`: **unitless**  (feature=macd_f)
- `macd_histogram`: **unitless**  (feature=macd_f)
- `trend_r2_20`: **bounded_0_1**  (feature=trend_r2_20_f)
- `wpt_price_trend`: **unitless**  (feature=wpt_price_reconstructed_f)
- `wpt_price_fluctuation`: **unitless**  (feature=wpt_price_reconstructed_f)
- `wpt_price_reconstructed`: **unitless**  (feature=wpt_price_reconstructed_f)
- `wpt_price_fluctuation`: **unitless**  (feature=wpt_price_fluctuation_f)
- `wpt_price_trend`: **unitless**  (feature=wpt_price_fluctuation_f)
- `wpt_price_energy_low_ratio`: **bounded_0_1** (0.0, 1.0) (feature=wpt_price_fluctuation_f)
- `wpt_price_energy_mid_ratio`: **bounded_0_1** (0.0, 1.0) (feature=wpt_price_fluctuation_f)
- `wpt_price_energy_high_ratio`: **bounded_0_1** (0.0, 1.0) (feature=wpt_price_fluctuation_f)
- `wpt_cvd_fluctuation`: **unitless**  (feature=wpt_cvd_fluctuation_f)
- `fp_poc`: **atr_distance**  (feature=footprint_basic_f)
- `fp_hvn`: **atr_distance**  (feature=footprint_basic_f)
- `fp_lvn`: **atr_distance**  (feature=footprint_basic_f)
- `fp_vah`: **atr_distance**  (feature=footprint_basic_f)
- `fp_val`: **atr_distance**  (feature=footprint_basic_f)
