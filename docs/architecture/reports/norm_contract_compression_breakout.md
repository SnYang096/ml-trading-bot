## Normalization Contract Report (config/strategies/compression_breakout/features.yaml)

- **total_output_columns**: 453
- **missing_method**: 0
- **raw_columns**: 0

### Sample of normalized/unitless columns (first 20)
- `atr`: **unitless**  (feature=atr_f)
- `rsi`: **bounded_0_100** (0.0, 100.0) (feature=rsi_f)
- `macd`: **unitless**  (feature=macd_f)
- `macd_signal`: **unitless**  (feature=macd_f)
- `macd_histogram`: **unitless**  (feature=macd_f)
- `bb_width_normalized`: **unitless**  (feature=bb_width_f)
- `bb_position`: **unitless**  (feature=bb_width_f)
- `trend_r2_20`: **bounded_0_1**  (feature=trend_r2_20_f)
- `compression_energy`: **unitless**  (feature=compression_energy_f)
- `wpt_price_trend`: **unitless**  (feature=wpt_price_reconstructed_f)
- `wpt_price_fluctuation`: **unitless**  (feature=wpt_price_reconstructed_f)
- `wpt_price_reconstructed`: **unitless**  (feature=wpt_price_reconstructed_f)
- `wpt_price_fluctuation`: **unitless**  (feature=wpt_price_fluctuation_f)
- `wpt_price_trend`: **unitless**  (feature=wpt_price_fluctuation_f)
- `wpt_price_energy_low_ratio`: **bounded_0_1** (0.0, 1.0) (feature=wpt_price_fluctuation_f)
- `wpt_price_energy_mid_ratio`: **bounded_0_1** (0.0, 1.0) (feature=wpt_price_fluctuation_f)
- `wpt_price_energy_high_ratio`: **bounded_0_1** (0.0, 1.0) (feature=wpt_price_fluctuation_f)
- `wpt_cvd_fluctuation`: **unitless**  (feature=wpt_cvd_fluctuation_f)
- `liquidity_void_detected`: **bounded_0_1** (0.0, 1.0) (feature=liquidity_void_f)
- `liquidity_void_speed`: **unitless**  (feature=liquidity_void_f)
