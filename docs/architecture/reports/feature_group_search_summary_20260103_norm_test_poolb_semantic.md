# Feature Group Search Summary (Pool-B + semantic, greedy multi-seed) — 20260103_norm_test

This report summarizes runs that include **semantic groups + Pool-B singletons**.

## Runs included

- **sr_breakout**: `results/feature_group_search/sr_breakout_greedy_poolb_semantic_20260103_norm_test/feature_group_search_result.json`
  - pool_b: `results/pools/sr_breakout/pool_b/20260103_norm_test/features_pool_b.yaml`
  - writeback: `config/strategies/sr_breakout/features_suggested_greedy_poolb_semantic_20260103_norm_test.yaml`

Common params:
- symbol: `BTCUSDT`
- timeframe: `240T`
- date range: `2023-01-01 .. 2025-12-31`
- seeds: `1,2`
- objective: `Sharpe_mean`

## Results

### sr_breakout

- **Selected groups**: `kline_core`
- **Final requested_features (8 nodes)**:
  - `atr_f, poc_hal_features_close_f, macd_f, rsi_f, trend_r2_20_f, bb_width_f, wick_ratios_f, volume_ratio_f`
- **Sharpe_mean (multi-seed)**:
  - baseline: **0.7854**
  - +`kline_core`: **1.5969**
- **stop_reason**: `max_steps_reached`
- **Seed-1 training stats** (`results/feature_group_search/sr_breakout_greedy_poolb_semantic_20260103_norm_test/runs/step1_add_kline_core/seed_1/sr_breakout__step1_add_kline_core/results.json`)
  - `n_train_samples=4026`, `n_test_samples=1622`
  - `n_features=21`
  - backtest: `sharpe=1.5013`, `total_trades=151`
