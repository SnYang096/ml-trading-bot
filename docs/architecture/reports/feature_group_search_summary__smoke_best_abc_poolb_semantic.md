# Feature Group Search Summary (Pool-B + semantic, staged A→B→C) — _smoke_best_abc

This report summarizes runs that include **semantic groups + Pool-B singletons**.

## Runs included

- **sr_breakout**:
  - pool_b: `results/pools/sr_breakout/pool_b/_smoke_best_abc/features_pool_b.yaml`
  - stage_A_result: `results/feature_group_search/sr_breakout_pipeline_poolb_semantic__smoke_best_abc_A/feature_group_search_result.json`
  - stage_B_result: `results/feature_group_search/sr_breakout_pipeline_poolb_semantic__smoke_best_abc_B/feature_group_search_result.json`
  - stage_C_result: `results/feature_group_search/sr_breakout_pipeline_poolb_semantic__smoke_best_abc_C/feature_group_search_result.json`
  - final_writeback: `config/strategies/sr_breakout/features_suggested_pipeline_poolb_semantic__smoke_best_abc_C.yaml`

Common params:
- symbol: `BTCUSDT`
- timeframe: `240T`
- date range: `2024-01-01 .. 2024-06-30`
- stages: `A (CV_mean, 2 seeds) -> B (CV_mean, 3 seeds) -> C (Sharpe_mean, 5 seeds)`

## Results

### sr_breakout

- **Stage A**: `results/feature_group_search/sr_breakout_pipeline_poolb_semantic__smoke_best_abc_A/feature_group_search_result.json`
  - objective: `CV_mean`
  - baseline_score: **-999.0000**
  - last_score: **N/A** (step=1)
  - selected_groups: `kline_core__trend_r2_20_f`
  - final_requested_features: `3` nodes
  - stop_reason: `completed`
- **Stage B**: `results/feature_group_search/sr_breakout_pipeline_poolb_semantic__smoke_best_abc_B/feature_group_search_result.json`
  - objective: `CV_mean`
  - baseline_score: **-999.0000**
  - selected_groups: *(none)*
  - final_requested_features: `2` nodes
  - stop_reason: `completed`
  - seed1: `n_train=653`, `n_test=110`, `n_features=11`, `sharpe=4.5188`, `trades=8`
- **Stage C**: `results/feature_group_search/sr_breakout_pipeline_poolb_semantic__smoke_best_abc_C/feature_group_search_result.json`
  - objective: `Sharpe_mean`
  - baseline_score: **-999.0000**
  - selected_groups: *(none)*
  - final_requested_features: `2` nodes
  - stop_reason: `completed`
  - final_features: `atr_f, poc_hal_features_close_f`
  - seed1: `n_train=653`, `n_test=110`, `n_features=11`, `sharpe=4.5188`, `trades=8`
