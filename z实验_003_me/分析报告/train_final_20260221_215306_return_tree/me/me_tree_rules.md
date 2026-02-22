# me 树模型规则导出

以下为从 LightGBM 模型中提取的**高频分裂条件**（按出现次数排序）。

| 特征 | 条件 | 出现次数 |
|------|------|----------|
| `wpt_price_energy_low_ratio` | `wpt_price_energy_low_ratio <= 1` | 4 |
| `garch_leverage_gamma` | `garch_leverage_gamma <= 0.01` | 3 |
| `wpt_price_energy_high_ratio` | `wpt_price_energy_high_ratio <= 1e-06` | 3 |
| `fp_poc` | `fp_poc <= 1357` | 3 |
| `spectrum_volume_centroid` | `spectrum_volume_centroid <= 0.252` | 2 |
| `sma_200_position` | `sma_200_position <= 0.06423` | 2 |
| `evt_es_99_right` | `evt_es_99_right <= 0.998` | 2 |
| `garch_volatility` | `garch_volatility <= 0.007927` | 2 |
| `evt_tail_shape` | `evt_tail_shape <= 0.5813` | 2 |
| `wpt_price_energy_mid_ratio` | `wpt_price_energy_mid_ratio <= 5e-06` | 2 |
| `sma_200_position` | `sma_200_position <= 0.07473` | 1 |
| `wpt_cvd_fluctuation` | `wpt_cvd_fluctuation <= -3780` | 1 |
| `oi_zscore` | `oi_zscore <= 0` | 1 |
| `oi_compression_score` | `oi_compression_score <= 0.1387` | 1 |
| `oi_zscore` | `oi_zscore <= -1.067` | 1 |
| `wpt_price_energy_low_ratio` | `wpt_price_energy_low_ratio <= 1` | 1 |
| `sr_strength_max` | `sr_strength_max <= 3.298` | 1 |
| `oi_compression_score` | `oi_compression_score <= 0.1118` | 1 |
| `me_delta_net_flow` | `me_delta_net_flow <= 0.02096` | 1 |
| `evt_es_99_right` | `evt_es_99_right <= 0.998` | 1 |
| `wpt_vper_mid` | `wpt_vper_mid <= 9.488e+15` | 1 |
| `evt_scale` | `evt_scale <= 0.8591` | 1 |
| `spectrum_price_low_freq_ratio` | `spectrum_price_low_freq_ratio <= 0.2761` | 1 |
| `macd_histogram_atr` | `macd_histogram_atr <= -0.07007` | 1 |
| `fp_exhaustion_price` | `fp_exhaustion_price <= 25.09` | 1 |
| `oi_compression_score` | `oi_compression_score <= 0.1009` | 1 |
| `evt_scale_right` | `evt_scale_right <= 0.9544` | 1 |
| `oi_zscore` | `oi_zscore <= -1.012` | 1 |
| `wpt_price_energy_high_ratio` | `wpt_price_energy_high_ratio <= 3e-06` | 1 |
| `evt_scale_right` | `evt_scale_right <= 0.9266` | 1 |

**模型来源**：`results/train_final_20260221_215306_return_tree/me`
