# 树模型策略收尾：各策略最有效特征（自动生成）

此文档由脚本生成：`scripts/tree_model_finalize.py`

## 总览

| strategy | tag | stage | objective | seeds | selected_groups | n_final_features | n_invert | suggested_yaml | export_dir |
|---|---|---|---|---:|---:|---:|---:|---|---|
| sr_reversal_rr_reg_long | 20260108_best_abc | C | Sharpe_mean | 5 | 1 | 3 | 0 | /workspaces/ml_trading_bot/config/strategies/sr_reversal_rr_reg_long/features_suggested_pipeline_poolb_semantic_20260108_best_abc_C.yaml | /workspaces/ml_trading_bot/config/strategies_exported/tree_best/sr_reversal_rr_reg_long__20260108_best_abc__C |
| sr_breakout | 20260108_best_abc | C | Sharpe_mean | 5 | 1 | 3 | 0 | /workspaces/ml_trading_bot/config/strategies/sr_breakout/features_suggested_pipeline_poolb_semantic_20260108_best_abc_C.yaml | /workspaces/ml_trading_bot/config/strategies_exported/tree_best/sr_breakout__20260108_best_abc__C |
| compression_breakout | 20260108_best_abc | B | CV_mean | 3 | 3 | 5 | 0 | /workspaces/ml_trading_bot/config/strategies/compression_breakout/features_suggested_pipeline_poolb_semantic_20260108_best_abc_B.yaml | /workspaces/ml_trading_bot/config/strategies_exported/tree_best/compression_breakout__20260108_best_abc__B |
| trend_following | 20260110_tf_fast_blacklist | C | Sharpe_mean | 5 | 5 | 10 | 0 | /workspaces/ml_trading_bot/config/strategies/trend_following/features_suggested_pipeline_poolb_semantic_20260110_tf_fast_blacklist_C.yaml | /workspaces/ml_trading_bot/config/strategies_exported/tree_best/trend_following__20260110_tf_fast_blacklist__C |

## 逐策略详情

### sr_reversal_rr_reg_long

- **tag/stage**: `20260108_best_abc` / `C`
- **objective**: `Sharpe_mean`
- **search_algo**: `pipeline_sh_beam_sffs`
- **result_json**: `/workspaces/ml_trading_bot/results/feature_group_search/sr_reversal_rr_reg_long_pipeline_poolb_semantic_20260108_best_abc_C/feature_group_search_result.json`
- **suggested_yaml（可直接用）**: `/workspaces/ml_trading_bot/config/strategies/sr_reversal_rr_reg_long/features_suggested_pipeline_poolb_semantic_20260108_best_abc_C.yaml`
- **export_dir（默认 lite，已剪掉重特征）**: `/workspaces/ml_trading_bot/config/strategies_exported/tree_best/sr_reversal_rr_reg_long__20260108_best_abc__C`

**selected_groups**:
- `poolb__volume_profile_volatility_features_f`

**final_features（建议传给模型的 feature nodes）**:
- `poc_hal_features_close_f`
- `atr_f`
- `volume_profile_volatility_features_f`

**final_invert_features（最终确认需要取反的输出列）**:
- (none)

### sr_breakout

- **tag/stage**: `20260108_best_abc` / `C`
- **objective**: `Sharpe_mean`
- **search_algo**: `pipeline_sh_beam_sffs`
- **result_json**: `/workspaces/ml_trading_bot/results/feature_group_search/sr_breakout_pipeline_poolb_semantic_20260108_best_abc_C/feature_group_search_result.json`
- **suggested_yaml（可直接用）**: `/workspaces/ml_trading_bot/config/strategies/sr_breakout/features_suggested_pipeline_poolb_semantic_20260108_best_abc_C.yaml`
- **export_dir（默认 lite，已剪掉重特征）**: `/workspaces/ml_trading_bot/config/strategies_exported/tree_best/sr_breakout__20260108_best_abc__C`

**selected_groups**:
- `poolb__poc_hal_features_f__hal_low`

**final_features（建议传给模型的 feature nodes）**:
- `atr_f`
- `poc_hal_features_close_f`
- `hal_low`

**final_invert_features（最终确认需要取反的输出列）**:
- (none)

### compression_breakout

- **tag/stage**: `20260108_best_abc` / `B`
- **objective**: `CV_mean`
- **search_algo**: `pipeline_sh_beam_sffs`
- **result_json**: `/workspaces/ml_trading_bot/results/feature_group_search/compression_breakout_pipeline_poolb_semantic_20260108_best_abc_B/feature_group_search_result.json`
- **suggested_yaml（可直接用）**: `/workspaces/ml_trading_bot/config/strategies/compression_breakout/features_suggested_pipeline_poolb_semantic_20260108_best_abc_B.yaml`
- **export_dir（默认 lite，已剪掉重特征）**: `/workspaces/ml_trading_bot/config/strategies_exported/tree_best/compression_breakout__20260108_best_abc__B`

**selected_groups**:
- `kline_core__volume_ratio_f__volume_ratio_f`
- `poolb__liquidity_void_f`
- `poolb__trend_r2_20_f`

**final_features（建议传给模型的 feature nodes）**:
- `compression_duration_f`
- `atr_f`
- `volume_ratio_f`
- `liquidity_void_f`
- `trend_r2_20_f`

**final_invert_features（最终确认需要取反的输出列）**:
- (none)

### trend_following

- **tag/stage**: `20260110_tf_fast_blacklist` / `C`
- **objective**: `Sharpe_mean`
- **search_algo**: `pipeline_sh_beam_sffs`
- **result_json**: `/workspaces/ml_trading_bot/results/feature_group_search/trend_following_pipeline_poolb_semantic_20260110_tf_fast_blacklist_C/feature_group_search_result.json`
- **suggested_yaml（可直接用）**: `/workspaces/ml_trading_bot/config/strategies/trend_following/features_suggested_pipeline_poolb_semantic_20260110_tf_fast_blacklist_C.yaml`
- **export_dir（默认 lite，已剪掉重特征）**: `/workspaces/ml_trading_bot/config/strategies_exported/tree_best/trend_following__20260110_tf_fast_blacklist__C`

**selected_groups**:
- `kline_core`
- `trade_cluster_scene`
- `poolb__wpt_cvd_fluctuation_f`
- `funding_scene`
- `poolb__atr_f`

**final_features（建议传给模型的 feature nodes）**:
- `atr_f`
- `macd_f`
- `rsi_f`
- `trend_r2_20_f`
- `bb_width_f`
- `wick_ratios_f`
- `volume_ratio_f`
- `trade_cluster_scene_semantic_scores_f`
- `wpt_cvd_fluctuation_f`
- `funding_scene_semantic_scores_f`

**final_invert_features（最终确认需要取反的输出列）**:
- (none)

