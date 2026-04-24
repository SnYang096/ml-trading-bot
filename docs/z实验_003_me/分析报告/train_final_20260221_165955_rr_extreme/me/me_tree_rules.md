# me 树模型规则导出

以下为从 LightGBM 模型中提取的**高频分裂条件**（按出现次数排序）。

| 特征 | 条件 | 出现次数 |
|------|------|----------|
| `cvd_change_5_normalized` | `cvd_change_5_normalized <= -0.008775` | 2 |
| `me_cvd_alignment` | `me_cvd_alignment <= 0` | 1 |
| `funding_rate_zscore_50` | `funding_rate_zscore_50 <= 4.195` | 1 |
| `evt_scale_left` | `evt_scale_left <= 0.7242` | 1 |
| `sma_200_position` | `sma_200_position <= 0.07786` | 1 |
| `macd_histogram_atr` | `macd_histogram_atr <= 0.07809` | 1 |
| `sma_200_position` | `sma_200_position <= 0.1335` | 1 |
| `spectrum_volume_entropy` | `spectrum_volume_entropy <= 0.8877` | 1 |
| `location_amplifier` | `location_amplifier <= 0.8163` | 1 |
| `wpt_price_energy_low_ratio` | `wpt_price_energy_low_ratio <= 1` | 1 |
| `me_multi_tf_alignment` | `me_multi_tf_alignment <= 0.6667` | 1 |
| `roc_20` | `roc_20 <= 1.518` | 1 |
| `evt_tail_shape_right` | `evt_tail_shape_right <= 0.6607` | 1 |
| `shd_pct` | `shd_pct <= 0.5724` | 1 |
| `cvd_exhaustion_score` | `cvd_exhaustion_score <= 0.01736` | 1 |
| `cvd_divergence_score` | `cvd_divergence_score <= -0.1531` | 1 |
| `jump_risk_pct` | `jump_risk_pct <= 0.474` | 1 |
| `oi_exhaustion_score` | `oi_exhaustion_score <= 0.007019` | 1 |
| `evt_scale_left` | `evt_scale_left <= 0.4405` | 1 |
| `sma_200_position` | `sma_200_position <= 0.07744` | 1 |
| `spectrum_volume_low_freq_ratio` | `spectrum_volume_low_freq_ratio <= 0.116` | 1 |
| `vp_entropy` | `vp_entropy <= 0.9053` | 1 |
| `me_vol_regime` | `me_vol_regime <= 0.635` | 1 |
| `wpt_vper_high` | `wpt_vper_high <= 4.834e+20` | 1 |
| `trend_div_tension` | `trend_div_tension <= 0.5549` | 1 |
| `vol_slope_20` | `vol_slope_20 <= 0.000515` | 1 |
| `funding_rate_zscore_50` | `funding_rate_zscore_50 <= 1.485` | 1 |
| `me_reflex_risk` | `me_reflex_risk <= 0.9564` | 1 |
| `evt_es_99_right` | `evt_es_99_right <= 0.4663` | 1 |
| `vpin_signed_imbalance_max` | `vpin_signed_imbalance_max <= 0.0513` | 1 |

**模型来源**：`results/train_final_20260221_165955_rr_extreme/me`
