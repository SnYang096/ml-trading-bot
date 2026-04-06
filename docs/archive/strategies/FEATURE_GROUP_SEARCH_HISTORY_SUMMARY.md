## Feature Group Search History (Auto Summary)

This document is **auto-generated** from `results/feature_group_search/**/feature_group_search_result.json` and writeback YAMLs under `config/strategies/*/features_suggested*.yaml`.

Regenerate:

```bash
python3 scripts/summarize_feature_group_search_history.py
```

Notes:
- **selected_groups** are *feature-group nodes* (semantic groups / Pool-B candidates).
- **final_features** are *model columns* (after expanding node outputs).
- **invert_features** are *column-level inversions* applied by `feature_pipeline.invert_features`.
- **status**: `completed` means result JSON exists; `pending` means directory exists but no final result yet; `cancelled` means the run was intentionally stopped/superseded.
- **latest_poolb_semantic** marks the latest run (per strategy) whose name contains `poolb_semantic`.

## Latest Pool-B + Semantic run per strategy

| strategy | run_dir | status | score | selected_groups | final_features | invert_features |
|---|---|---|---:|---|---|---|
| `compression_breakout` | `compression_breakout_pipeline_poolb_semantic_20260105_pipeline_wide_retry1` | pending |  | - | - | - |
| `sr_breakout` | `sr_breakout_pipeline_poolb_semantic_20260104_pipeline_wide_retry1` | completed | 0.961039 | kline_core__volume_ratio_f | atr_f, poc_hal_features_close_f, volume_ratio_f | hal_mid, trade_cluster_compression_score, trade_cluster_exhaustion_scene_score |
| `sr_reversal_rr_reg_long` | `sr_reversal_rr_reg_long_pipeline_poolb_semantic_20260105_pipeline_wide_retry1` | completed | 0.933971 | poolb__volume_profile_volatility_features_f | poc_hal_features_close_f, atr_f, volume_profile_volatility_features_f | hal_low, vpin_ignition_score |
| `trend_following` | `trend_following_pipeline_poolb_semantic_20260105_pipeline_wide_retry1_tf_solo3_tzfix` | pending |  | - | - | - |
| `unknown` | `sr_reversal_poolb_semantic` | completed | 0.224860 | poolb__dtw_features_breakout_f, poolb__evt_features_f | macd_f, rsi_f, sma_200_f, atr_f, trend_r2_20_f, bb_width_f, wick_ratios_f, roc_5_f, mom_10_f, bbands_f, trend_r2_50_f, acceleration_3_f, price_range_symmetry_f, vpin_scene_semantic_scores_f, poc_hal_features_close_f, sqs_hal_high_f, sqs_hal_low_f, sr_strength_max_close_f, dtw_features_breakout_f, evt_features_f | - |

## Strategy: `compression_breakout`

| run_dir | status | latest_poolb_semantic | search_algo | objective | score | selected_groups | final_features | invert_features |
|---|---|---|---|---|---:|---|---|---|
| `compression_breakout` | pending |  |  |  |  | - | - | - |
| `compression_breakout_best_combo_multisymbol_v1` | completed |  |  |  | -0.736350 | volume_profile_scene, vpin_scene | atr_f, wpt_price_reconstructed_f, wpt_price_fluctuation_f, wpt_volume_energy_f, wpt_cvd_fluctuation_f, spectrum_features_compression_breakout_f, liquidity_void_f, hilbert_phase_f, hurst_price_f, hurst_cvd_f, vpin_features_f, footprint_basic_f, liquidity_void_x_wpt_risk_f, compression_energy_x_ofi_short_f, vpin_x_compression_f, vpin_zscore_x_trade_cluster_max_buy_run_f, vpin_x_trade_cluster_entropy_f, dtw_features_compression_f, volume_profile_scene_semantic_scores_f, vpin_scene_semantic_scores_f | - |
| `compression_breakout_best_combo_quick` | pending |  |  |  |  | - | - | - |
| `compression_breakout_best_combo_quick3` | pending |  |  |  |  | - | - | - |
| `compression_breakout_best_combo_v4` | pending |  |  |  |  | - | - | - |
| `compression_breakout_best_combo_v5` | completed |  |  |  | -1.406253 | vpin_scene, wpt_scene | atr_f, wpt_price_reconstructed_f, wpt_price_fluctuation_f, wpt_volume_energy_f, wpt_cvd_fluctuation_f, spectrum_features_compression_breakout_f, liquidity_void_f, hilbert_phase_f, hurst_price_f, hurst_cvd_f, vpin_features_f, footprint_basic_f, liquidity_void_x_wpt_risk_f, compression_energy_x_ofi_short_f, vpin_x_compression_f, vpin_zscore_x_trade_cluster_max_buy_run_f, vpin_x_trade_cluster_entropy_f, dtw_features_compression_f, vpin_scene_semantic_scores_f, wpt_scene_semantic_scores_f | - |
| `compression_breakout_expanded` | pending |  |  |  |  | - | - | - |
| `compression_breakout_features_suggested` | completed |  |  | Sharpe_mean |  | vpin_scene, wpt_scene | atr_f, wpt_price_reconstructed_f, wpt_price_fluctuation_f, wpt_volume_energy_f, wpt_cvd_fluctuation_f, spectrum_features_compression_breakout_f, liquidity_void_f, hilbert_phase_f, hurst_price_f, hurst_cvd_f, vpin_features_f, footprint_basic_f, liquidity_void_x_wpt_risk_f, compression_energy_x_ofi_short_f, vpin_x_compression_f, vpin_zscore_x_trade_cluster_max_buy_run_f, vpin_x_trade_cluster_entropy_f, dtw_features_compression_f, vpin_scene_semantic_scores_f, wpt_scene_semantic_scores_f | - |
| `compression_breakout_greedy_20260102` | pending |  |  |  |  | - | - | - |
| `compression_breakout_greedy_20260102_rerun` | completed |  |  | Sharpe_mean | -1.017497 | market_cap_norm, vpin_scene | compression_duration_f, atr_f, market_cap_normalized_orderflow_f, vpin_scene_semantic_scores_f | - |
| `compression_breakout_greedy_poolb_semantic_20260103_norm_full` | completed |  |  | Sharpe_mean | -0.709909 | market_cap_norm, vpin_scene | compression_duration_f, atr_f, market_cap_normalized_orderflow_f, vpin_scene_semantic_scores_f | dtw_bull_flag_dist_w20, dtw_bull_flag_dist_w50, dtw_decline_consolidation_inverse_dist_w20, dtw_decline_consolidation_inverse_dist_w30, hurst_cvd_rolling, hurst_price_rolling, macd_signal, trade_cluster_imbalance_zscore_50, trade_cluster_max_buy_run_ma10, trade_cluster_max_buy_run_ma5, trade_cluster_net_runs, trade_cluster_net_runs_zscore_20, trade_cluster_total_run_length, wpt_price_energy_high_ratio, wpt_price_energy_mid_ratio, wpt_price_reconstructed, wpt_vper_high, wpt_vper_mid |
| `compression_breakout_multisymbol` | completed |  |  | Sharpe_mean |  | volume_profile_scene, vpin_scene | atr_f, wpt_price_reconstructed_f, wpt_price_fluctuation_f, wpt_volume_energy_f, wpt_cvd_fluctuation_f, spectrum_features_compression_breakout_f, liquidity_void_f, hilbert_phase_f, hurst_price_f, hurst_cvd_f, vpin_features_f, footprint_basic_f, liquidity_void_x_wpt_risk_f, compression_energy_x_ofi_short_f, vpin_x_compression_f, vpin_zscore_x_trade_cluster_max_buy_run_f, vpin_x_trade_cluster_entropy_f, dtw_features_compression_f, volume_profile_scene_semantic_scores_f, vpin_scene_semantic_scores_f | - |
| `compression_breakout_pipeline_poolb_semantic_20260104_pipeline` | completed |  | pipeline_sh_beam_sffs | Sharpe_mean | -999.000000 | compression_core__compression_energy_f | compression_duration_f, atr_f, compression_energy_f | dtw_bull_flag_dist_w20, dtw_bull_flag_dist_w50, dtw_decline_consolidation_inverse_dist_w20, dtw_decline_consolidation_inverse_dist_w30, hurst_cvd_rolling, hurst_price_rolling, macd_signal, trade_cluster_imbalance_zscore_50, trade_cluster_max_buy_run_ma10, trade_cluster_max_buy_run_ma5, trade_cluster_net_runs, trade_cluster_net_runs_zscore_20, trade_cluster_total_run_length, wpt_price_energy_high_ratio, wpt_price_energy_low_ratio, wpt_price_energy_mid_ratio, wpt_price_trend, wpt_vper_high, wpt_vper_mid |
| `compression_breakout_pipeline_poolb_semantic_20260105_pipeline_wide_retry1` | pending | ⏳ |  |  |  | - | - | - |
| `compression_breakout_quick` | completed |  |  | Sharpe_mean | -1.064692 | - | compression_duration_f, atr_f | - |
| `compression_breakout_quick2` | completed |  |  | Sharpe_mean | -0.086682 | market_cap_norm | compression_duration_f, atr_f | - |
| `compression_breakout_quick3` | completed |  |  | Sharpe_mean | 0.188308 | - | compression_duration_f, atr_f | - |

### Artifacts

- **`compression_breakout`**
  - **result_json**: *(pending)* (expected `/workspaces/ml_trading_bot/results/feature_group_search/compression_breakout/feature_group_search_result.json`)
- **`compression_breakout_best_combo_multisymbol_v1`**
  - **result_json**: `/workspaces/ml_trading_bot/results/feature_group_search/compression_breakout_best_combo_multisymbol_v1/feature_group_search_result.json`
  - **writeback_yaml**: *(pending)* (expected `/workspaces/ml_trading_bot/config/strategies/compression_breakout/features_suggested_best_combo_multisymbol.yaml`)
  - **stop_reason**: `no_improvement`
- **`compression_breakout_best_combo_quick`**
  - **result_json**: *(pending)* (expected `/workspaces/ml_trading_bot/results/feature_group_search/compression_breakout_best_combo_quick/feature_group_search_result.json`)
  - **writeback_yaml**: *(pending)* (expected `/workspaces/ml_trading_bot/config/strategies/compression_breakout/features_suggested_best_combo_quick.yaml`)
- **`compression_breakout_best_combo_quick3`**
  - **result_json**: *(pending)* (expected `/workspaces/ml_trading_bot/results/feature_group_search/compression_breakout_best_combo_quick3/feature_group_search_result.json`)
  - **writeback_yaml**: *(pending)* (expected `/workspaces/ml_trading_bot/config/strategies/compression_breakout/features_suggested_best_combo_quick3.yaml`)
- **`compression_breakout_best_combo_v4`**
  - **result_json**: *(pending)* (expected `/workspaces/ml_trading_bot/results/feature_group_search/compression_breakout_best_combo_v4/feature_group_search_result.json`)
  - **writeback_yaml**: *(pending)* (expected `/workspaces/ml_trading_bot/config/strategies/compression_breakout/features_suggested_best_combo_v4.yaml`)
- **`compression_breakout_best_combo_v5`**
  - **result_json**: `/workspaces/ml_trading_bot/results/feature_group_search/compression_breakout_best_combo_v5/feature_group_search_result.json`
  - **writeback_yaml**: *(pending)* (expected `/workspaces/ml_trading_bot/config/strategies/compression_breakout/features_suggested_best_combo_v5.yaml`)
  - **stop_reason**: `no_improvement`
- **`compression_breakout_expanded`**
  - **result_json**: *(pending)* (expected `/workspaces/ml_trading_bot/results/feature_group_search/compression_breakout_expanded/feature_group_search_result.json`)
  - **writeback_yaml**: *(pending)* (expected `/workspaces/ml_trading_bot/config/strategies/compression_breakout/features_suggested_expanded.yaml`)
- **`compression_breakout_features_suggested`**
  - **writeback_yaml**: `/workspaces/ml_trading_bot/config/strategies/compression_breakout/features_suggested.yaml`
  - **groups_source**: `groups_yaml:auto:config/feature_groups_compression_breakout_semantic.yaml`
  - **stop_reason**: `no_improvement`
- **`compression_breakout_greedy_20260102`**
  - **result_json**: *(pending)* (expected `/workspaces/ml_trading_bot/results/feature_group_search/compression_breakout_greedy_20260102/feature_group_search_result.json`)
  - **writeback_yaml**: *(pending)* (expected `/workspaces/ml_trading_bot/config/strategies/compression_breakout/features_suggested_greedy_20260102.yaml`)
- **`compression_breakout_greedy_20260102_rerun`**
  - **result_json**: `/workspaces/ml_trading_bot/results/feature_group_search/compression_breakout_greedy_20260102_rerun/feature_group_search_result.json`
  - **writeback_yaml**: `/workspaces/ml_trading_bot/config/strategies/compression_breakout/features_suggested_greedy_20260102_rerun.yaml`
  - **pool_b_yaml**: `/dev/null`
  - **groups_source**: `groups_yaml:auto:config/feature_groups_compression_breakout_semantic.yaml`
  - **stop_reason**: `no_improvement`
- **`compression_breakout_greedy_poolb_semantic_20260103_norm_full`**
  - **result_json**: `/workspaces/ml_trading_bot/results/feature_group_search/compression_breakout_greedy_poolb_semantic_20260103_norm_full/feature_group_search_result.json`
  - **writeback_yaml**: `/workspaces/ml_trading_bot/config/strategies/compression_breakout/features_suggested_greedy_poolb_semantic_20260103_norm_full.yaml`
  - **pool_b_yaml**: `/workspaces/ml_trading_bot/results/pools/compression_breakout/pool_b/20260103_norm_full/features_pool_b.yaml`
  - **groups_source**: `groups_yaml:auto:config/feature_groups_compression_breakout_semantic.yaml`
  - **stop_reason**: `no_improvement`
- **`compression_breakout_multisymbol`**
  - **writeback_yaml**: `/workspaces/ml_trading_bot/config/strategies/compression_breakout/features_suggested_multisymbol.yaml`
  - **groups_source**: `groups_yaml:auto:config/feature_groups_compression_breakout_semantic.yaml`
  - **stop_reason**: `no_improvement`
- **`compression_breakout_pipeline_poolb_semantic_20260104_pipeline`**
  - **result_json**: `/workspaces/ml_trading_bot/results/feature_group_search/compression_breakout_pipeline_poolb_semantic_20260104_pipeline/feature_group_search_result.json`
  - **writeback_yaml**: `/workspaces/ml_trading_bot/config/strategies/compression_breakout/features_suggested_pipeline_poolb_semantic_20260104_pipeline.yaml`
  - **pool_b_yaml**: `/workspaces/ml_trading_bot/results/pools/compression_breakout/pool_b/20260104_pipeline/features_pool_b.yaml`
  - **groups_source**: `groups_yaml:auto:config/feature_groups_compression_breakout_semantic.yaml`
  - **stop_reason**: `completed`
- **`compression_breakout_pipeline_poolb_semantic_20260105_pipeline_wide_retry1`**
  - **result_json**: *(pending)* (expected `/workspaces/ml_trading_bot/results/feature_group_search/compression_breakout_pipeline_poolb_semantic_20260105_pipeline_wide_retry1/feature_group_search_result.json`)
  - **writeback_yaml**: *(pending)* (expected `/workspaces/ml_trading_bot/config/strategies/compression_breakout/features_suggested_pipeline_poolb_semantic_20260105_pipeline_wide_retry1.yaml`)
- **`compression_breakout_quick`**
  - **result_json**: `/workspaces/ml_trading_bot/results/feature_group_search/compression_breakout_quick/feature_group_search_result.json`
  - **writeback_yaml**: `/workspaces/ml_trading_bot/config/strategies/compression_breakout/features_suggested_quick.yaml`
  - **pool_b_yaml**: `/dev/null`
  - **groups_source**: `groups_yaml:auto:config/feature_groups_compression_breakout_semantic.yaml`
  - **stop_reason**: `no_valid_candidates`
- **`compression_breakout_quick2`**
  - **result_json**: `/workspaces/ml_trading_bot/results/feature_group_search/compression_breakout_quick2/feature_group_search_result.json`
  - **writeback_yaml**: `/workspaces/ml_trading_bot/config/strategies/compression_breakout/features_suggested_quick2.yaml`
  - **pool_b_yaml**: `/dev/null`
  - **groups_source**: `groups_yaml:auto:config/feature_groups_compression_breakout_semantic.yaml`
  - **stop_reason**: `no_improvement`
- **`compression_breakout_quick3`**
  - **result_json**: `/workspaces/ml_trading_bot/results/feature_group_search/compression_breakout_quick3/feature_group_search_result.json`
  - **writeback_yaml**: `/workspaces/ml_trading_bot/config/strategies/compression_breakout/features_suggested_quick3.yaml`
  - **pool_b_yaml**: `/dev/null`
  - **groups_source**: `groups_yaml:auto:config/feature_groups_compression_breakout_semantic.yaml`
  - **stop_reason**: `no_improvement`

## Strategy: `sr_breakout`

| run_dir | status | latest_poolb_semantic | search_algo | objective | score | selected_groups | final_features | invert_features |
|---|---|---|---|---|---:|---|---|---|
| `sr_breakout` | pending |  |  |  |  | - | - | - |
| `sr_breakout_best_combo_multisymbol_v1` | pending |  |  |  |  | - | - | - |
| `sr_breakout_best_combo_multisymbol_v2` | completed |  |  |  | nan | - | wpt_price_reconstructed_f, wpt_price_fluctuation_f, wpt_volume_energy_f, wpt_cvd_fluctuation_f, hurst_price_f, hurst_cvd_f, liquidity_void_f, volume_profile_vpvr_f, spectrum_features_sr_breakout_f, liquidity_void_x_wpt_risk_f, hurst_x_trend_r2_f, hilbert_phase_f, hilbert_price_env, hilbert_cvd_env, hilbert_cvd_price_env_ratio, hilbert_volume_env, hilbert_env_price_vol_ratio, hilbert_triple_divergence, vpin_features_f, footprint_basic_f, vpin_zscore_x_trade_cluster_max_buy_run_f, vpin_signed_imbalance_x_trade_cluster_imbalance_f, dtw_features_breakout_f | - |
| `sr_breakout_best_combo_multisymbol_v3` | completed |  |  |  | -0.696498 | wpt_scene, market_cap_norm | wpt_price_reconstructed_f, wpt_price_fluctuation_f, wpt_volume_energy_f, wpt_cvd_fluctuation_f, hurst_price_f, hurst_cvd_f, liquidity_void_f, volume_profile_vpvr_f, spectrum_features_sr_breakout_f, liquidity_void_x_wpt_risk_f, hurst_x_trend_r2_f, hilbert_phase_f, hilbert_price_env, hilbert_cvd_env, hilbert_cvd_price_env_ratio, hilbert_volume_env, hilbert_env_price_vol_ratio, hilbert_triple_divergence, vpin_features_f, footprint_basic_f, vpin_zscore_x_trade_cluster_max_buy_run_f, vpin_signed_imbalance_x_trade_cluster_imbalance_f, dtw_features_breakout_f, wpt_scene_semantic_scores_f, market_cap_normalized_orderflow_f | - |
| `sr_breakout_best_combo_multisymbol_v3_quick` | completed |  |  |  | -0.214294 | volume_profile_scene | wpt_price_reconstructed_f, wpt_price_fluctuation_f, wpt_volume_energy_f, wpt_cvd_fluctuation_f, hurst_price_f, hurst_cvd_f, liquidity_void_f, volume_profile_vpvr_f, spectrum_features_sr_breakout_f, liquidity_void_x_wpt_risk_f, hurst_x_trend_r2_f, hilbert_phase_f, hilbert_price_env, hilbert_cvd_env, hilbert_cvd_price_env_ratio, hilbert_volume_env, hilbert_env_price_vol_ratio, hilbert_triple_divergence, vpin_features_f, footprint_basic_f, vpin_zscore_x_trade_cluster_max_buy_run_f, vpin_signed_imbalance_x_trade_cluster_imbalance_f, dtw_features_breakout_f, volume_profile_scene_semantic_scores_f | - |
| `sr_breakout_best_combo_quick` | pending |  |  |  |  | - | - | - |
| `sr_breakout_best_combo_quick3` | pending |  |  |  |  | - | - | - |
| `sr_breakout_best_combo_v4` | completed |  |  |  | 0.832229 | trade_cluster_scene, wick_scene | wpt_price_reconstructed_f, wpt_price_fluctuation_f, wpt_volume_energy_f, wpt_cvd_fluctuation_f, hurst_price_f, hurst_cvd_f, liquidity_void_f, volume_profile_vpvr_f, spectrum_features_sr_breakout_f, liquidity_void_x_wpt_risk_f, hurst_x_trend_r2_f, hilbert_phase_f, hilbert_price_env, hilbert_cvd_env, hilbert_cvd_price_env_ratio, hilbert_volume_env, hilbert_env_price_vol_ratio, hilbert_triple_divergence, vpin_features_f, footprint_basic_f, vpin_zscore_x_trade_cluster_max_buy_run_f, vpin_signed_imbalance_x_trade_cluster_imbalance_f, dtw_features_breakout_f, trade_cluster_scene_semantic_scores_f, wick_scene_semantic_scores_f | - |
| `sr_breakout_expanded` | pending |  |  |  |  | - | - | - |
| `sr_breakout_features_suggested` | completed |  |  | Sharpe_mean |  | trade_cluster_scene, wick_scene | wpt_price_reconstructed_f, wpt_price_fluctuation_f, wpt_volume_energy_f, wpt_cvd_fluctuation_f, hurst_price_f, hurst_cvd_f, liquidity_void_f, volume_profile_vpvr_f, spectrum_features_sr_breakout_f, liquidity_void_x_wpt_risk_f, hurst_x_trend_r2_f, hilbert_phase_f, hilbert_price_env, hilbert_cvd_env, hilbert_cvd_price_env_ratio, hilbert_volume_env, hilbert_env_price_vol_ratio, hilbert_triple_divergence, vpin_features_f, footprint_basic_f, vpin_zscore_x_trade_cluster_max_buy_run_f, vpin_signed_imbalance_x_trade_cluster_imbalance_f, dtw_features_breakout_f, trade_cluster_scene_semantic_scores_f, wick_scene_semantic_scores_f | - |
| `sr_breakout_greedy_20260102` | pending |  |  |  |  | - | - | - |
| `sr_breakout_greedy_20260102_rerun` | completed |  |  | Sharpe_mean | -0.858103 | kline_core | atr_f, poc_hal_features_close_f, macd_f, rsi_f, trend_r2_20_f, bb_width_f, wick_ratios_f, volume_ratio_f | - |
| `sr_breakout_greedy_poolb_semantic_20260103_norm_full` | completed |  |  | Sharpe_mean | -0.086236 | kline_core, trade_cluster_scene, poolb__sma_200_f | atr_f, poc_hal_features_close_f, macd_f, rsi_f, trend_r2_20_f, bb_width_f, wick_ratios_f, volume_ratio_f, trade_cluster_scene_semantic_scores_f, sma_200_f | dtw_double_top_dist_w40, dtw_double_top_dist_w50, dtw_double_top_dist_w60, dtw_head_shoulder_top_dist_w30, dtw_head_shoulder_top_dist_w40, dtw_head_shoulder_top_dist_w50, dtw_min_dist_w40, dtw_min_dist_w50, dtw_min_dist_w60, dtw_random_15_dist_w40, dtw_random_15_dist_w60, dtw_random_20_dist_w30, dtw_random_20_dist_w50, liquidity_void_price_impact, liquidity_void_retracement, liquidity_void_speed, spectrum_price_entropy, spectrum_price_flatness, trade_cluster_max_buy_run, trade_cluster_max_buy_run_zscore_50, trade_cluster_max_sell_run_zscore_20, trade_cluster_net_runs_ma10, wpt_cvd_fluctuation, wpt_price_energy_high_ratio, wpt_price_fluctuation |
| `sr_breakout_greedy_poolb_semantic_20260103_norm_full_singletons` | completed |  |  | Sharpe_mean | 0.457637 | poolb__sma_200_f, trade_cluster_scene__absorption | atr_f, poc_hal_features_close_f, sma_200_f, trade_cluster_absorption_scene_score | dtw_double_top_dist_w30, dtw_double_top_dist_w40, dtw_double_top_dist_w50, dtw_double_top_dist_w60, dtw_head_shoulder_bottom_inverse_dist_w60, dtw_head_shoulder_top_dist_w30, dtw_head_shoulder_top_dist_w40, dtw_min_dist_w40, dtw_min_dist_w50, dtw_min_dist_w60, dtw_random_15_dist_w60, dtw_random_20_dist_w50, liquidity_void_price_impact, liquidity_void_retracement, liquidity_void_speed, trade_cluster_max_buy_run, trade_cluster_max_buy_run_zscore_50, trade_cluster_max_run_ratio, trade_cluster_max_sell_run, trade_cluster_max_sell_run_zscore_20, trade_cluster_net_runs_ma10, trade_cluster_total_run_length, wpt_cvd_fluctuation, wpt_price_fluctuation |
| `sr_breakout_greedy_poolb_semantic_20260103_norm_test` | completed |  |  | Sharpe_mean | 0.785420 | kline_core | atr_f, poc_hal_features_close_f, macd_f, rsi_f, trend_r2_20_f, bb_width_f, wick_ratios_f, volume_ratio_f | dtw_double_top_dist_w40, dtw_double_top_dist_w50, dtw_double_top_dist_w60, dtw_head_shoulder_top_dist_w30, dtw_head_shoulder_top_dist_w40, dtw_head_shoulder_top_dist_w50, dtw_min_dist_w40, dtw_min_dist_w50, dtw_min_dist_w60, dtw_random_15_dist_w40, dtw_random_15_dist_w60, dtw_random_20_dist_w30, dtw_random_20_dist_w50, liquidity_void_price_impact, liquidity_void_retracement, liquidity_void_speed, spectrum_price_entropy, spectrum_price_flatness, trade_cluster_max_buy_run, trade_cluster_max_buy_run_zscore_50, trade_cluster_max_sell_run_zscore_20, trade_cluster_net_runs_ma10, wpt_cvd_fluctuation, wpt_price_energy_high_ratio, wpt_price_fluctuation |
| `sr_breakout_multisymbol` | completed |  |  | Sharpe_mean |  | wpt_scene, market_cap_norm | wpt_price_reconstructed_f, wpt_price_fluctuation_f, wpt_volume_energy_f, wpt_cvd_fluctuation_f, hurst_price_f, hurst_cvd_f, liquidity_void_f, volume_profile_vpvr_f, spectrum_features_sr_breakout_f, liquidity_void_x_wpt_risk_f, hurst_x_trend_r2_f, hilbert_phase_f, hilbert_price_env, hilbert_cvd_env, hilbert_cvd_price_env_ratio, hilbert_volume_env, hilbert_env_price_vol_ratio, hilbert_triple_divergence, vpin_features_f, footprint_basic_f, vpin_zscore_x_trade_cluster_max_buy_run_f, vpin_signed_imbalance_x_trade_cluster_imbalance_f, dtw_features_breakout_f, wpt_scene_semantic_scores_f, market_cap_normalized_orderflow_f | - |
| `sr_breakout_pipeline_poolb_semantic_20260104_pipeline` | completed |  | pipeline_sh_beam_sffs | Sharpe_mean | 0.878696 | kline_core__macd, liquidity_void_scene__absorption | atr_f, poc_hal_features_close_f, macd, liquidity_void_absorption_score | hal_mid, trade_cluster_compression_score, trade_cluster_exhaustion_scene_score |
| `sr_breakout_pipeline_poolb_semantic_20260104_pipeline_wide` | completed |  | pipeline_sh_beam_sffs | Sharpe_mean | -999.000000 | - | atr_f, poc_hal_features_close_f | hal_mid, trade_cluster_compression_score, trade_cluster_exhaustion_scene_score |
| `sr_breakout_pipeline_poolb_semantic_20260104_pipeline_wide_retry1` | completed | ✅ | pipeline_sh_beam_sffs | Sharpe_mean | 0.961039 | kline_core__volume_ratio_f | atr_f, poc_hal_features_close_f, volume_ratio_f | hal_mid, trade_cluster_compression_score, trade_cluster_exhaustion_scene_score |
| `sr_breakout_quick` | completed |  |  | Sharpe_mean | -0.018505 | - | atr_f, poc_hal_features_close_f | - |
| `sr_breakout_quick2` | completed |  |  | Sharpe_mean | -0.018505 | - | atr_f, poc_hal_features_close_f | - |
| `sr_breakout_quick3` | completed |  |  | Sharpe_mean | -0.018505 | - | atr_f, poc_hal_features_close_f | - |
| `sr_breakout_v3` | pending |  |  |  |  | - | - | - |
| `sr_breakout_v4` | pending |  |  |  |  | - | - | - |

### Artifacts

- **`sr_breakout`**
  - **result_json**: *(pending)* (expected `/workspaces/ml_trading_bot/results/feature_group_search/sr_breakout/feature_group_search_result.json`)
- **`sr_breakout_best_combo_multisymbol_v1`**
  - **result_json**: *(pending)* (expected `/workspaces/ml_trading_bot/results/feature_group_search/sr_breakout_best_combo_multisymbol_v1/feature_group_search_result.json`)
  - **writeback_yaml**: *(pending)* (expected `/workspaces/ml_trading_bot/config/strategies/sr_breakout/features_suggested_best_combo_multisymbol.yaml`)
- **`sr_breakout_best_combo_multisymbol_v2`**
  - **result_json**: `/workspaces/ml_trading_bot/results/feature_group_search/sr_breakout_best_combo_multisymbol_v2/feature_group_search_result.json`
  - **writeback_yaml**: *(pending)* (expected `/workspaces/ml_trading_bot/config/strategies/sr_breakout/features_suggested_best_combo_multisymbol_v2.yaml`)
  - **stop_reason**: `no_valid_candidates`
- **`sr_breakout_best_combo_multisymbol_v3`**
  - **result_json**: `/workspaces/ml_trading_bot/results/feature_group_search/sr_breakout_best_combo_multisymbol_v3/feature_group_search_result.json`
  - **writeback_yaml**: *(pending)* (expected `/workspaces/ml_trading_bot/config/strategies/sr_breakout/features_suggested_best_combo_multisymbol_v3.yaml`)
  - **stop_reason**: `no_improvement`
- **`sr_breakout_best_combo_multisymbol_v3_quick`**
  - **result_json**: `/workspaces/ml_trading_bot/results/feature_group_search/sr_breakout_best_combo_multisymbol_v3_quick/feature_group_search_result.json`
  - **writeback_yaml**: *(pending)* (expected `/workspaces/ml_trading_bot/config/strategies/sr_breakout/features_suggested_best_combo_multisymbol_v3_quick.yaml`)
  - **stop_reason**: `max_steps_reached`
- **`sr_breakout_best_combo_quick`**
  - **result_json**: *(pending)* (expected `/workspaces/ml_trading_bot/results/feature_group_search/sr_breakout_best_combo_quick/feature_group_search_result.json`)
  - **writeback_yaml**: *(pending)* (expected `/workspaces/ml_trading_bot/config/strategies/sr_breakout/features_suggested_best_combo_quick.yaml`)
- **`sr_breakout_best_combo_quick3`**
  - **result_json**: *(pending)* (expected `/workspaces/ml_trading_bot/results/feature_group_search/sr_breakout_best_combo_quick3/feature_group_search_result.json`)
  - **writeback_yaml**: *(pending)* (expected `/workspaces/ml_trading_bot/config/strategies/sr_breakout/features_suggested_best_combo_quick3.yaml`)
- **`sr_breakout_best_combo_v4`**
  - **result_json**: `/workspaces/ml_trading_bot/results/feature_group_search/sr_breakout_best_combo_v4/feature_group_search_result.json`
  - **writeback_yaml**: *(pending)* (expected `/workspaces/ml_trading_bot/config/strategies/sr_breakout/features_suggested_best_combo_v4.yaml`)
  - **stop_reason**: `no_improvement`
- **`sr_breakout_expanded`**
  - **result_json**: *(pending)* (expected `/workspaces/ml_trading_bot/results/feature_group_search/sr_breakout_expanded/feature_group_search_result.json`)
  - **writeback_yaml**: *(pending)* (expected `/workspaces/ml_trading_bot/config/strategies/sr_breakout/features_suggested_expanded.yaml`)
- **`sr_breakout_features_suggested`**
  - **writeback_yaml**: `/workspaces/ml_trading_bot/config/strategies/sr_breakout/features_suggested.yaml`
  - **groups_source**: `groups_yaml:auto:config/feature_groups_sr_breakout_semantic.yaml`
  - **stop_reason**: `no_improvement`
- **`sr_breakout_greedy_20260102`**
  - **result_json**: *(pending)* (expected `/workspaces/ml_trading_bot/results/feature_group_search/sr_breakout_greedy_20260102/feature_group_search_result.json`)
  - **writeback_yaml**: *(pending)* (expected `/workspaces/ml_trading_bot/config/strategies/sr_breakout/features_suggested_greedy_20260102.yaml`)
- **`sr_breakout_greedy_20260102_rerun`**
  - **result_json**: `/workspaces/ml_trading_bot/results/feature_group_search/sr_breakout_greedy_20260102_rerun/feature_group_search_result.json`
  - **writeback_yaml**: `/workspaces/ml_trading_bot/config/strategies/sr_breakout/features_suggested_greedy_20260102_rerun.yaml`
  - **pool_b_yaml**: `/dev/null`
  - **groups_source**: `groups_yaml:auto:config/feature_groups_sr_breakout_semantic.yaml`
  - **stop_reason**: `no_improvement`
- **`sr_breakout_greedy_poolb_semantic_20260103_norm_full`**
  - **result_json**: `/workspaces/ml_trading_bot/results/feature_group_search/sr_breakout_greedy_poolb_semantic_20260103_norm_full/feature_group_search_result.json`
  - **writeback_yaml**: `/workspaces/ml_trading_bot/config/strategies/sr_breakout/features_suggested_greedy_poolb_semantic_20260103_norm_full.yaml`
  - **pool_b_yaml**: `/workspaces/ml_trading_bot/results/pools/sr_breakout/pool_b/20260103_norm_full/features_pool_b.yaml`
  - **groups_source**: `groups_yaml:auto:config/feature_groups_sr_breakout_semantic.yaml`
  - **stop_reason**: `no_improvement`
- **`sr_breakout_greedy_poolb_semantic_20260103_norm_full_singletons`**
  - **result_json**: `/workspaces/ml_trading_bot/results/feature_group_search/sr_breakout_greedy_poolb_semantic_20260103_norm_full_singletons/feature_group_search_result.json`
  - **writeback_yaml**: `/workspaces/ml_trading_bot/config/strategies/sr_breakout/features_suggested_greedy_poolb_semantic_20260103_norm_full_singletons.yaml`
  - **pool_b_yaml**: `/workspaces/ml_trading_bot/results/pools/sr_breakout/pool_b/20260103_norm_full_singletons/features_pool_b.yaml`
  - **groups_source**: `groups_yaml:auto:config/feature_groups_sr_breakout_semantic.yaml`
  - **stop_reason**: `no_improvement`
- **`sr_breakout_greedy_poolb_semantic_20260103_norm_test`**
  - **result_json**: `/workspaces/ml_trading_bot/results/feature_group_search/sr_breakout_greedy_poolb_semantic_20260103_norm_test/feature_group_search_result.json`
  - **writeback_yaml**: `/workspaces/ml_trading_bot/config/strategies/sr_breakout/features_suggested_greedy_poolb_semantic_20260103_norm_test.yaml`
  - **pool_b_yaml**: `/workspaces/ml_trading_bot/results/pools/sr_breakout/pool_b/20260103_norm_test/features_pool_b.yaml`
  - **groups_source**: `groups_yaml:auto:config/feature_groups_sr_breakout_semantic.yaml`
  - **stop_reason**: `max_steps_reached`
- **`sr_breakout_multisymbol`**
  - **writeback_yaml**: `/workspaces/ml_trading_bot/config/strategies/sr_breakout/features_suggested_multisymbol.yaml`
  - **groups_source**: `groups_yaml:auto:config/feature_groups_sr_breakout_semantic.yaml`
  - **stop_reason**: `no_improvement`
- **`sr_breakout_pipeline_poolb_semantic_20260104_pipeline`**
  - **result_json**: `/workspaces/ml_trading_bot/results/feature_group_search/sr_breakout_pipeline_poolb_semantic_20260104_pipeline/feature_group_search_result.json`
  - **writeback_yaml**: `/workspaces/ml_trading_bot/config/strategies/sr_breakout/features_suggested_pipeline_poolb_semantic_20260104_pipeline.yaml`
  - **pool_b_yaml**: `/workspaces/ml_trading_bot/results/pools/sr_breakout/pool_b/20260104_pipeline/features_pool_b.yaml`
  - **groups_source**: `groups_yaml:auto:config/feature_groups_sr_breakout_semantic.yaml`
  - **stop_reason**: `completed`
- **`sr_breakout_pipeline_poolb_semantic_20260104_pipeline_wide`**
  - **result_json**: `/workspaces/ml_trading_bot/results/feature_group_search/sr_breakout_pipeline_poolb_semantic_20260104_pipeline_wide/feature_group_search_result.json`
  - **writeback_yaml**: `/workspaces/ml_trading_bot/config/strategies/sr_breakout/features_suggested_pipeline_poolb_semantic_20260104_pipeline_wide.yaml`
  - **pool_b_yaml**: `results/pools/sr_breakout/pool_b/20260104_pipeline/features_pool_b.yaml`
  - **groups_source**: `groups_yaml:auto:config/feature_groups_sr_breakout_semantic.yaml`
  - **stop_reason**: `completed`
- **`sr_breakout_pipeline_poolb_semantic_20260104_pipeline_wide_retry1`**
  - **result_json**: `/workspaces/ml_trading_bot/results/feature_group_search/sr_breakout_pipeline_poolb_semantic_20260104_pipeline_wide_retry1/feature_group_search_result.json`
  - **writeback_yaml**: `/workspaces/ml_trading_bot/config/strategies/sr_breakout/features_suggested_pipeline_poolb_semantic_20260104_pipeline_wide_retry1.yaml`
  - **pool_b_yaml**: `results/pools/sr_breakout/pool_b/20260104_pipeline/features_pool_b.yaml`
  - **groups_source**: `groups_yaml:auto:config/feature_groups_sr_breakout_semantic.yaml`
  - **stop_reason**: `completed`
- **`sr_breakout_quick`**
  - **result_json**: `/workspaces/ml_trading_bot/results/feature_group_search/sr_breakout_quick/feature_group_search_result.json`
  - **writeback_yaml**: `/workspaces/ml_trading_bot/config/strategies/sr_breakout/features_suggested_quick.yaml`
  - **pool_b_yaml**: `/dev/null`
  - **groups_source**: `groups_yaml:auto:config/feature_groups_sr_breakout_semantic.yaml`
  - **stop_reason**: `no_valid_candidates`
- **`sr_breakout_quick2`**
  - **result_json**: `/workspaces/ml_trading_bot/results/feature_group_search/sr_breakout_quick2/feature_group_search_result.json`
  - **writeback_yaml**: `/workspaces/ml_trading_bot/config/strategies/sr_breakout/features_suggested_quick2.yaml`
  - **pool_b_yaml**: `/dev/null`
  - **groups_source**: `groups_yaml:auto:config/feature_groups_sr_breakout_semantic.yaml`
  - **stop_reason**: `no_improvement`
- **`sr_breakout_quick3`**
  - **result_json**: `/workspaces/ml_trading_bot/results/feature_group_search/sr_breakout_quick3/feature_group_search_result.json`
  - **writeback_yaml**: `/workspaces/ml_trading_bot/config/strategies/sr_breakout/features_suggested_quick3.yaml`
  - **pool_b_yaml**: `/dev/null`
  - **groups_source**: `groups_yaml:auto:config/feature_groups_sr_breakout_semantic.yaml`
  - **stop_reason**: `no_improvement`
- **`sr_breakout_v3`**
  - **result_json**: *(pending)* (expected `/workspaces/ml_trading_bot/results/feature_group_search/sr_breakout_v3/feature_group_search_result.json`)
  - **writeback_yaml**: *(pending)* (expected `/workspaces/ml_trading_bot/config/strategies/sr_breakout/features_suggested_v3.yaml`)
- **`sr_breakout_v4`**
  - **result_json**: *(pending)* (expected `/workspaces/ml_trading_bot/results/feature_group_search/sr_breakout_v4/feature_group_search_result.json`)
  - **writeback_yaml**: *(pending)* (expected `/workspaces/ml_trading_bot/config/strategies/sr_breakout/features_suggested_v4.yaml`)

## Strategy: `sr_reversal_rr_reg_long`

| run_dir | status | latest_poolb_semantic | search_algo | objective | score | selected_groups | final_features | invert_features |
|---|---|---|---|---|---:|---|---|---|
| `sr_reversal_rr_reg_long` | pending |  |  |  |  | - | - | - |
| `sr_reversal_rr_reg_long_demo` | completed |  |  |  |  | vpin_scene, kline_core | vpin_scene_semantic_scores_f, macd_f, rsi_f, sma_200_f, atr_f, trend_r2_20_f, bb_width_f, wick_ratios_f | - |
| `sr_reversal_rr_reg_long_fast` | pending |  |  |  |  | - | - | - |
| `sr_reversal_rr_reg_long_features_suggested` | completed |  |  | Sharpe_mean |  | poolb__dtw_features_breakout_f, poolb__evt_features_f | macd_f, rsi_f, sma_200_f, atr_f, trend_r2_20_f, bb_width_f, wick_ratios_f, roc_5_f, mom_10_f, bbands_f, trend_r2_50_f, acceleration_3_f, price_range_symmetry_f, vpin_scene_semantic_scores_f, poc_hal_features_close_f, sqs_hal_high_f, sqs_hal_low_f, sr_strength_max_close_f, dtw_features_breakout_f, evt_features_f | direction_to_nearest_sr, dl_seq_f19, dl_seq_f31, dl_seq_f35, dl_seq_f40, dtw_bear_flag_dist_w45, dtw_decline_consolidation_dist_w40, dtw_decline_consolidation_dist_w50, dtw_double_bottom_dist_w50, dtw_hammer_dist_w30, dtw_head_shoulder_top_dist_w60, dtw_random_20_dist_w60, evt_scale_right, garch_alpha, garch_beta, garch_leverage_gamma, hurst_volume_rolling, sqs_hal_high, sqs_hal_low, sr_strength_max, trade_cluster_imbalance_zscore_50, trade_cluster_net_runs_ratio, trade_cluster_net_runs_zscore_20, trend_r2_50, vol_range_10, vp_poc_deviation, wpt_multi_scale_consistency, wpt_price_energy_low_ratio |
| `sr_reversal_rr_reg_long_greedy_20260102_rerun` | completed |  |  | Sharpe_mean | 0.815995 | volume_profile_scene, volume_profile | poc_hal_features_close_f, atr_f, volume_profile_scene_semantic_scores_f, volume_profile_volatility_features_f | - |
| `sr_reversal_rr_reg_long_greedy_poolb_semantic_20260103` | completed |  |  | Sharpe_mean | 1.504081 | poolb__dist_to_zz_high_f, poolb__garch_features_f | poc_hal_features_close_f, atr_f, dist_to_zz_high_f, garch_features_f | dl_seq_f1, dl_seq_f35, dl_seq_f45, dl_seq_f57, dtw_bear_flag_dist_w40, dtw_bear_flag_dist_w45, dtw_random_20_dist_w40, dtw_random_30_dist_w40, dtw_triangle_dist_w30, dtw_triangle_dist_w40, evt_es_99_right, evt_scale, evt_scale_right, evt_var_99, garch_alpha, garch_beta, garch_leverage_gamma, garch_persistence, slope_consistency_score, sqs_hal_high, trade_cluster_directional_entropy_change, trade_cluster_imbalance_zscore_50, trade_cluster_max_buy_run_ma10, trade_cluster_max_buy_run_ma20, trade_cluster_max_run_ratio, trade_cluster_net_runs_ma10, trade_cluster_net_runs_ma5, trade_cluster_net_runs_ratio, trade_cluster_net_runs_zscore_20, trend_r2_50, … (+8) |
| `sr_reversal_rr_reg_long_greedy_poolb_semantic_20260103_norm` | completed |  |  | Sharpe_mean | 1.504081 | - | poc_hal_features_close_f, atr_f | direction_to_nearest_sr, hal_high, sr_strength_max, vp_poc_deviation |
| `sr_reversal_rr_reg_long_greedy_poolb_semantic_20260103_norm_full` | completed |  |  | Sharpe_mean | 1.504081 | - | poc_hal_features_close_f, atr_f | direction_to_nearest_sr, hal_high, sr_strength_max, vp_poc_deviation |
| `sr_reversal_rr_reg_long_greedy_poolb_semantic_20260103_norm_full_singletons` | completed |  |  | Sharpe_mean | 0.936193 | compression__compression_energy_f | poc_hal_features_close_f, atr_f, compression_energy_f | hal_low, vpin_ignition_score |
| `sr_reversal_rr_reg_long_normalized` | pending |  |  |  |  | - | - | - |
| `sr_reversal_rr_reg_long_pipeline_poolb_semantic_20260104_pipeline` | completed |  | pipeline_sh_beam_sffs | Sharpe_mean | 1.054697 | sr_structure_min__sqs_hal_high_f | poc_hal_features_close_f, atr_f, sqs_hal_high_f | hal_low, vpin_ignition_score |
| `sr_reversal_rr_reg_long_pipeline_poolb_semantic_20260105_pipeline_wide_retry1` | completed | ✅ | pipeline_sh_beam_sffs | Sharpe_mean | 0.933971 | poolb__volume_profile_volatility_features_f | poc_hal_features_close_f, atr_f, volume_profile_volatility_features_f | hal_low, vpin_ignition_score |
| `sr_reversal_rr_reg_long_quick` | completed |  |  | Sharpe_mean | 2.132804 | fp_scene | poc_hal_features_close_f, atr_f, fp_imbalance_scene_semantic_scores_f | - |

### Artifacts

- **`sr_reversal_rr_reg_long`**
  - **result_json**: *(pending)* (expected `/workspaces/ml_trading_bot/results/feature_group_search/sr_reversal_rr_reg_long/feature_group_search_result.json`)
- **`sr_reversal_rr_reg_long_demo`**
  - **result_json**: `/workspaces/ml_trading_bot/results/feature_group_search/sr_reversal_rr_reg_long_demo/feature_group_search_result.json`
  - **writeback_yaml**: *(pending)* (expected `/workspaces/ml_trading_bot/config/strategies/sr_reversal_rr_reg_long/features_suggested_demo.yaml`)
- **`sr_reversal_rr_reg_long_fast`**
  - **result_json**: *(pending)* (expected `/workspaces/ml_trading_bot/results/feature_group_search/sr_reversal_rr_reg_long_fast/feature_group_search_result.json`)
  - **writeback_yaml**: *(pending)* (expected `/workspaces/ml_trading_bot/config/strategies/sr_reversal_rr_reg_long/features_suggested_fast.yaml`)
- **`sr_reversal_rr_reg_long_features_suggested`**
  - **writeback_yaml**: `/workspaces/ml_trading_bot/config/strategies/sr_reversal_rr_reg_long/features_suggested.yaml`
  - **pool_b_yaml**: `results/pools/sr_reversal_rr_reg_long/pool_b/features_pool_b.yaml`
  - **groups_source**: `groups_yaml:config/feature_groups_sr_reversal_semantic.yaml`
  - **stop_reason**: `no_improvement`
- **`sr_reversal_rr_reg_long_greedy_20260102_rerun`**
  - **result_json**: `/workspaces/ml_trading_bot/results/feature_group_search/sr_reversal_rr_reg_long_greedy_20260102_rerun/feature_group_search_result.json`
  - **writeback_yaml**: `/workspaces/ml_trading_bot/config/strategies/sr_reversal_rr_reg_long/features_suggested_greedy_20260102_rerun.yaml`
  - **pool_b_yaml**: `/dev/null`
  - **groups_source**: `groups_yaml:auto:config/feature_groups.yaml`
  - **stop_reason**: `no_improvement`
- **`sr_reversal_rr_reg_long_greedy_poolb_semantic_20260103`**
  - **result_json**: `/workspaces/ml_trading_bot/results/feature_group_search/sr_reversal_rr_reg_long_greedy_poolb_semantic_20260103/feature_group_search_result.json`
  - **writeback_yaml**: `/workspaces/ml_trading_bot/config/strategies/sr_reversal_rr_reg_long/features_suggested_greedy_poolb_semantic_20260103.yaml`
  - **pool_b_yaml**: `/workspaces/ml_trading_bot/results/pools/sr_reversal_rr_reg_long/pool_b/features_pool_b.yaml`
  - **groups_source**: `groups_yaml:auto:config/feature_groups.yaml`
  - **stop_reason**: `no_improvement`
- **`sr_reversal_rr_reg_long_greedy_poolb_semantic_20260103_norm`**
  - **result_json**: `/workspaces/ml_trading_bot/results/feature_group_search/sr_reversal_rr_reg_long_greedy_poolb_semantic_20260103_norm/feature_group_search_result.json`
  - **writeback_yaml**: `/workspaces/ml_trading_bot/config/strategies/sr_reversal_rr_reg_long/features_suggested_greedy_poolb_semantic_20260103_norm.yaml`
  - **pool_b_yaml**: `/workspaces/ml_trading_bot/results/pools/sr_reversal_rr_reg_long/pool_b/features_pool_b.yaml`
  - **groups_source**: `groups_yaml:auto:config/feature_groups.yaml`
  - **stop_reason**: `no_improvement`
- **`sr_reversal_rr_reg_long_greedy_poolb_semantic_20260103_norm_full`**
  - **result_json**: `/workspaces/ml_trading_bot/results/feature_group_search/sr_reversal_rr_reg_long_greedy_poolb_semantic_20260103_norm_full/feature_group_search_result.json`
  - **writeback_yaml**: `/workspaces/ml_trading_bot/config/strategies/sr_reversal_rr_reg_long/features_suggested_greedy_poolb_semantic_20260103_norm_full.yaml`
  - **pool_b_yaml**: `/workspaces/ml_trading_bot/results/pools/sr_reversal_rr_reg_long/pool_b/20260103_norm_full/features_pool_b.yaml`
  - **groups_source**: `groups_yaml:auto:config/feature_groups.yaml`
  - **stop_reason**: `no_improvement`
- **`sr_reversal_rr_reg_long_greedy_poolb_semantic_20260103_norm_full_singletons`**
  - **result_json**: `/workspaces/ml_trading_bot/results/feature_group_search/sr_reversal_rr_reg_long_greedy_poolb_semantic_20260103_norm_full_singletons/feature_group_search_result.json`
  - **writeback_yaml**: `/workspaces/ml_trading_bot/config/strategies/sr_reversal_rr_reg_long/features_suggested_greedy_poolb_semantic_20260103_norm_full_singletons.yaml`
  - **pool_b_yaml**: `/workspaces/ml_trading_bot/results/pools/sr_reversal_rr_reg_long/pool_b/20260103_norm_full_singletons/features_pool_b.yaml`
  - **groups_source**: `groups_yaml:auto:config/feature_groups.yaml`
  - **stop_reason**: `no_improvement`
- **`sr_reversal_rr_reg_long_normalized`**
  - **result_json**: *(pending)* (expected `/workspaces/ml_trading_bot/results/feature_group_search/sr_reversal_rr_reg_long_normalized/feature_group_search_result.json`)
  - **writeback_yaml**: *(pending)* (expected `/workspaces/ml_trading_bot/config/strategies/sr_reversal_rr_reg_long/features_suggested_normalized.yaml`)
- **`sr_reversal_rr_reg_long_pipeline_poolb_semantic_20260104_pipeline`**
  - **result_json**: `/workspaces/ml_trading_bot/results/feature_group_search/sr_reversal_rr_reg_long_pipeline_poolb_semantic_20260104_pipeline/feature_group_search_result.json`
  - **writeback_yaml**: `/workspaces/ml_trading_bot/config/strategies/sr_reversal_rr_reg_long/features_suggested_pipeline_poolb_semantic_20260104_pipeline.yaml`
  - **pool_b_yaml**: `/workspaces/ml_trading_bot/results/pools/sr_reversal_rr_reg_long/pool_b/20260104_pipeline/features_pool_b.yaml`
  - **groups_source**: `groups_yaml:auto:config/feature_groups.yaml`
  - **stop_reason**: `completed`
- **`sr_reversal_rr_reg_long_pipeline_poolb_semantic_20260105_pipeline_wide_retry1`**
  - **result_json**: `/workspaces/ml_trading_bot/results/feature_group_search/sr_reversal_rr_reg_long_pipeline_poolb_semantic_20260105_pipeline_wide_retry1/feature_group_search_result.json`
  - **writeback_yaml**: `/workspaces/ml_trading_bot/config/strategies/sr_reversal_rr_reg_long/features_suggested_pipeline_poolb_semantic_20260105_pipeline_wide_retry1.yaml`
  - **pool_b_yaml**: `/workspaces/ml_trading_bot/results/pools/sr_reversal_rr_reg_long/pool_b/20260105_pipeline_wide_retry1/features_pool_b.yaml`
  - **groups_source**: `groups_yaml:auto:config/feature_groups.yaml`
  - **stop_reason**: `completed`
- **`sr_reversal_rr_reg_long_quick`**
  - **result_json**: `/workspaces/ml_trading_bot/results/feature_group_search/sr_reversal_rr_reg_long_quick/feature_group_search_result.json`
  - **writeback_yaml**: `/workspaces/ml_trading_bot/config/strategies/sr_reversal_rr_reg_long/features_suggested_quick.yaml`
  - **pool_b_yaml**: `/dev/null`
  - **groups_source**: `groups_yaml:auto:config/feature_groups.yaml`
  - **stop_reason**: `no_improvement`

## Strategy: `trend_following`

| run_dir | status | latest_poolb_semantic | search_algo | objective | score | selected_groups | final_features | invert_features |
|---|---|---|---|---|---:|---|---|---|
| `trend_following` | pending |  |  |  |  | - | - | - |
| `trend_following_best_combo_quick` | pending |  |  |  |  | - | - | - |
| `trend_following_best_combo_quick3` | completed |  |  |  | 1.170728 | liquidity_void_scene | wpt_price_reconstructed_f, wpt_price_fluctuation_f, wpt_cvd_fluctuation_f, hurst_price_f, hurst_cvd_f, hurst_volume_f, hilbert_phase_f, hilbert_price_env, hilbert_cvd_env, hilbert_cvd_price_env_ratio, hilbert_price_env_slope, hilbert_cvd_env_slope, hilbert_price_env_qnorm, hilbert_cvd_env_qnorm, hilbert_volume_env, hilbert_env_price_vol_ratio, spectrum_features_trend_following_f, atr_f, rsi_f, vpin_features_f, footprint_basic_f, hurst_x_trend_r2_f, evt_x_trend_r2_f, sma_slope_x_price_pos_f, vpin_signed_imbalance_x_trade_cluster_imbalance_f, vpin_x_trade_cluster_entropy_f, dtw_features_trend_f, liquidity_void_scene_semantic_scores_f | - |
| `trend_following_best_combo_v5` | completed |  |  |  | 1.372796 | - | wpt_price_reconstructed_f, wpt_price_fluctuation_f, wpt_cvd_fluctuation_f, hurst_price_f, hurst_cvd_f, hurst_volume_f, hilbert_phase_f, hilbert_price_env, hilbert_cvd_env, hilbert_cvd_price_env_ratio, hilbert_price_env_slope, hilbert_cvd_env_slope, hilbert_price_env_qnorm, hilbert_cvd_env_qnorm, hilbert_volume_env, hilbert_env_price_vol_ratio, spectrum_features_trend_following_f, atr_f, rsi_f, vpin_features_f, footprint_basic_f, hurst_x_trend_r2_f, evt_x_trend_r2_f, sma_slope_x_price_pos_f, vpin_signed_imbalance_x_trade_cluster_imbalance_f, vpin_x_trade_cluster_entropy_f, dtw_features_trend_f | - |
| `trend_following_expanded` | pending |  |  |  |  | - | - | - |
| `trend_following_features_suggested` | completed |  |  | Sharpe_mean |  | - | wpt_price_reconstructed_f, wpt_price_fluctuation_f, wpt_cvd_fluctuation_f, hurst_price_f, hurst_cvd_f, hurst_volume_f, hilbert_phase_f, hilbert_price_env, hilbert_cvd_env, hilbert_cvd_price_env_ratio, hilbert_price_env_slope, hilbert_cvd_env_slope, hilbert_price_env_qnorm, hilbert_cvd_env_qnorm, hilbert_volume_env, hilbert_env_price_vol_ratio, spectrum_features_trend_following_f, atr_f, rsi_f, vpin_features_f, footprint_basic_f, hurst_x_trend_r2_f, evt_x_trend_r2_f, sma_slope_x_price_pos_f, vpin_signed_imbalance_x_trade_cluster_imbalance_f, vpin_x_trade_cluster_entropy_f, dtw_features_trend_f | - |
| `trend_following_greedy_20260102` | pending |  |  |  |  | - | - | - |
| `trend_following_greedy_20260102_rerun` | completed |  |  | Sharpe_mean | -1.661683 | kline_core, trend_core | atr_f, macd_f, rsi_f, trend_r2_20_f, bb_width_f, wick_ratios_f, volume_ratio_f, trend_r2_50_f, slope_consistency_score_f, trend_volatility_alignment_f | - |
| `trend_following_pipeline_poolb_semantic_20260105_pipeline_wide_retry1_tf_solo2` | cancelled |  |  |  |  | - | - | - |
| `trend_following_pipeline_poolb_semantic_20260105_pipeline_wide_retry1_tf_solo3_tzfix` | pending | ⏳ |  |  |  | - | - | - |
| `trend_following_quick` | completed |  |  | Sharpe_mean | -1.837755 | - | atr_f | - |
| `trend_following_quick2` | completed |  |  | Sharpe_mean | -1.837755 | - | atr_f | - |
| `trend_following_v3` | pending |  |  |  |  | - | - | - |
| `trend_following_v4` | pending |  |  |  |  | - | - | - |

### Artifacts

- **`trend_following`**
  - **result_json**: *(pending)* (expected `/workspaces/ml_trading_bot/results/feature_group_search/trend_following/feature_group_search_result.json`)
- **`trend_following_best_combo_quick`**
  - **result_json**: *(pending)* (expected `/workspaces/ml_trading_bot/results/feature_group_search/trend_following_best_combo_quick/feature_group_search_result.json`)
  - **writeback_yaml**: *(pending)* (expected `/workspaces/ml_trading_bot/config/strategies/trend_following/features_suggested_best_combo_quick.yaml`)
- **`trend_following_best_combo_quick3`**
  - **result_json**: `/workspaces/ml_trading_bot/results/feature_group_search/trend_following_best_combo_quick3/feature_group_search_result.json`
  - **writeback_yaml**: *(pending)* (expected `/workspaces/ml_trading_bot/config/strategies/trend_following/features_suggested_best_combo_quick3.yaml`)
  - **stop_reason**: `no_improvement`
- **`trend_following_best_combo_v5`**
  - **result_json**: `/workspaces/ml_trading_bot/results/feature_group_search/trend_following_best_combo_v5/feature_group_search_result.json`
  - **writeback_yaml**: *(pending)* (expected `/workspaces/ml_trading_bot/config/strategies/trend_following/features_suggested_best_combo_v5.yaml`)
  - **stop_reason**: `no_improvement`
- **`trend_following_expanded`**
  - **result_json**: *(pending)* (expected `/workspaces/ml_trading_bot/results/feature_group_search/trend_following_expanded/feature_group_search_result.json`)
  - **writeback_yaml**: *(pending)* (expected `/workspaces/ml_trading_bot/config/strategies/trend_following/features_suggested_expanded.yaml`)
- **`trend_following_features_suggested`**
  - **writeback_yaml**: `/workspaces/ml_trading_bot/config/strategies/trend_following/features_suggested.yaml`
  - **groups_source**: `groups_yaml:auto:config/feature_groups_trend_following_semantic.yaml`
  - **stop_reason**: `no_improvement`
- **`trend_following_greedy_20260102`**
  - **result_json**: *(pending)* (expected `/workspaces/ml_trading_bot/results/feature_group_search/trend_following_greedy_20260102/feature_group_search_result.json`)
  - **writeback_yaml**: *(pending)* (expected `/workspaces/ml_trading_bot/config/strategies/trend_following/features_suggested_greedy_20260102.yaml`)
- **`trend_following_greedy_20260102_rerun`**
  - **result_json**: `/workspaces/ml_trading_bot/results/feature_group_search/trend_following_greedy_20260102_rerun/feature_group_search_result.json`
  - **writeback_yaml**: `/workspaces/ml_trading_bot/config/strategies/trend_following/features_suggested_greedy_20260102_rerun.yaml`
  - **pool_b_yaml**: `/dev/null`
  - **groups_source**: `groups_yaml:auto:config/feature_groups_trend_following_semantic.yaml`
  - **stop_reason**: `no_improvement`
- **`trend_following_pipeline_poolb_semantic_20260105_pipeline_wide_retry1_tf_solo2`**
  - **result_json**: *(pending)* (expected `/workspaces/ml_trading_bot/results/feature_group_search/trend_following_pipeline_poolb_semantic_20260105_pipeline_wide_retry1_tf_solo2/feature_group_search_result.json`)
  - **writeback_yaml**: *(pending)* (expected `/workspaces/ml_trading_bot/config/strategies/trend_following/features_suggested_pipeline_poolb_semantic_20260105_pipeline_wide_retry1_tf_solo2.yaml`)
- **`trend_following_pipeline_poolb_semantic_20260105_pipeline_wide_retry1_tf_solo3_tzfix`**
  - **result_json**: *(pending)* (expected `/workspaces/ml_trading_bot/results/feature_group_search/trend_following_pipeline_poolb_semantic_20260105_pipeline_wide_retry1_tf_solo3_tzfix/feature_group_search_result.json`)
  - **writeback_yaml**: *(pending)* (expected `/workspaces/ml_trading_bot/config/strategies/trend_following/features_suggested_pipeline_poolb_semantic_20260105_pipeline_wide_retry1_tf_solo3_tzfix.yaml`)
- **`trend_following_quick`**
  - **result_json**: `/workspaces/ml_trading_bot/results/feature_group_search/trend_following_quick/feature_group_search_result.json`
  - **writeback_yaml**: `/workspaces/ml_trading_bot/config/strategies/trend_following/features_suggested_quick.yaml`
  - **pool_b_yaml**: `/dev/null`
  - **groups_source**: `groups_yaml:auto:config/feature_groups_trend_following_semantic.yaml`
  - **stop_reason**: `no_improvement`
- **`trend_following_quick2`**
  - **result_json**: `/workspaces/ml_trading_bot/results/feature_group_search/trend_following_quick2/feature_group_search_result.json`
  - **writeback_yaml**: `/workspaces/ml_trading_bot/config/strategies/trend_following/features_suggested_quick2.yaml`
  - **pool_b_yaml**: `/dev/null`
  - **groups_source**: `groups_yaml:auto:config/feature_groups_trend_following_semantic.yaml`
  - **stop_reason**: `no_improvement`
- **`trend_following_v3`**
  - **result_json**: *(pending)* (expected `/workspaces/ml_trading_bot/results/feature_group_search/trend_following_v3/feature_group_search_result.json`)
  - **writeback_yaml**: *(pending)* (expected `/workspaces/ml_trading_bot/config/strategies/trend_following/features_suggested_v3.yaml`)
- **`trend_following_v4`**
  - **result_json**: *(pending)* (expected `/workspaces/ml_trading_bot/results/feature_group_search/trend_following_v4/feature_group_search_result.json`)
  - **writeback_yaml**: *(pending)* (expected `/workspaces/ml_trading_bot/config/strategies/trend_following/features_suggested_v4.yaml`)

## Strategy: `unknown`

| run_dir | status | latest_poolb_semantic | search_algo | objective | score | selected_groups | final_features | invert_features |
|---|---|---|---|---|---:|---|---|---|
| `_smoke_feature_group_search` | completed |  |  |  | 2.197677 | - | macd_f, rsi_f, sma_200_f, atr_f, roc_5_f, mom_10_f, bbands_f, bb_width_f, trend_r2_20_f, trend_r2_50_f, acceleration_3_f, wick_ratios_f, price_range_symmetry_f, poc_hal_features_close_f, sqs_hal_high_f, sqs_hal_low_f, sr_strength_max_close_f | - |
| `_smoke_feature_group_search2` | completed |  |  |  | 2.137735 | volume_profile_scene | macd_f, rsi_f, sma_200_f, atr_f, roc_5_f, mom_10_f, bbands_f, bb_width_f, trend_r2_20_f, trend_r2_50_f, acceleration_3_f, wick_ratios_f, price_range_symmetry_f, poc_hal_features_close_f, sqs_hal_high_f, sqs_hal_low_f, sr_strength_max_close_f, volume_profile_scene_semantic_scores_f | - |
| `sr_reversal_best_combo` | pending |  |  |  |  | - | - | - |
| `sr_reversal_best_combo_v3` | pending |  |  |  |  | - | - | - |
| `sr_reversal_expanded` | completed |  |  |  | 1.528694 | poolb__sqs_hal_high_f | sqs_hal_high_f | - |
| `sr_reversal_poolb_semantic` | completed | ✅ |  |  | 0.224860 | poolb__dtw_features_breakout_f, poolb__evt_features_f | macd_f, rsi_f, sma_200_f, atr_f, trend_r2_20_f, bb_width_f, wick_ratios_f, roc_5_f, mom_10_f, bbands_f, trend_r2_50_f, acceleration_3_f, price_range_symmetry_f, vpin_scene_semantic_scores_f, poc_hal_features_close_f, sqs_hal_high_f, sqs_hal_low_f, sr_strength_max_close_f, dtw_features_breakout_f, evt_features_f | - |
| `strategy_bases` | pending |  |  |  |  | - | - | - |

### Artifacts

- **`_smoke_feature_group_search`**
  - **result_json**: `/workspaces/ml_trading_bot/results/feature_group_search/_smoke_feature_group_search/feature_group_search_result.json`
  - **stop_reason**: `no_improvement`
- **`_smoke_feature_group_search2`**
  - **result_json**: `/workspaces/ml_trading_bot/results/feature_group_search/_smoke_feature_group_search2/feature_group_search_result.json`
  - **stop_reason**: `max_steps_reached`
- **`sr_reversal_best_combo`**
  - **result_json**: *(pending)* (expected `/workspaces/ml_trading_bot/results/feature_group_search/sr_reversal_best_combo/feature_group_search_result.json`)
- **`sr_reversal_best_combo_v3`**
  - **result_json**: *(pending)* (expected `/workspaces/ml_trading_bot/results/feature_group_search/sr_reversal_best_combo_v3/feature_group_search_result.json`)
- **`sr_reversal_expanded`**
  - **result_json**: `/workspaces/ml_trading_bot/results/feature_group_search/sr_reversal_expanded/feature_group_search_result.json`
  - **stop_reason**: `no_improvement`
- **`sr_reversal_poolb_semantic`**
  - **result_json**: `/workspaces/ml_trading_bot/results/feature_group_search/sr_reversal_poolb_semantic/feature_group_search_result.json`
  - **stop_reason**: `no_improvement`
- **`strategy_bases`**
  - **result_json**: *(pending)* (expected `/workspaces/ml_trading_bot/results/feature_group_search/strategy_bases/feature_group_search_result.json`)
