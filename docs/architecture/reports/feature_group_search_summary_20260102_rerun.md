# Feature Group Search Summary (Greedy, multi-seed) — 2026-01-02 rerun

This report summarizes the rerun results after a machine reboot interrupted the previous runs.

## Runs included

- **sr_breakout**: `results/feature_group_search/sr_breakout_greedy_20260102_rerun/feature_group_search_result.json`
  - writeback: `config/strategies/sr_breakout/features_suggested_greedy_20260102_rerun.yaml`
- **compression_breakout**: `results/feature_group_search/compression_breakout_greedy_20260102_rerun/feature_group_search_result.json`
  - writeback: `config/strategies/compression_breakout/features_suggested_greedy_20260102_rerun.yaml`
- **trend_following**: `results/feature_group_search/trend_following_greedy_20260102_rerun/feature_group_search_result.json`
  - writeback: `config/strategies/trend_following/features_suggested_greedy_20260102_rerun.yaml`

Common params:
- symbol: `BTCUSDT`
- timeframe: `240T`
- date range: `2023-01-01 .. 2025-12-31`
- seeds: `1,2,3,4,5`
- objective: `Sharpe_mean`

## Key note: “requested_features 很少”是正常的

In this repo, `feature_pipeline.requested_features` are **feature compute nodes** (functions ending with `_f`), **not raw model columns**.

One feature node may produce **many columns** (e.g. scene semantic score blocks, VPVR blocks, etc.). So it’s normal to see “only a few requested_features” while the model actually trains on ~20+ columns.

### Node-level vs column-level selection (why we keep it node-level for now)

- Current `feature-group-search` operates at **feature node** granularity (`*_f`), **not per-output-column** granularity.
- For tree models (LightGBM), this is usually fine: trees are relatively robust to some redundant columns inside a well-designed block.
- We only consider “column-level masking” later when:
  - a single node outputs a very wide block (e.g. embeddings), or
  - we see instability/overfit and need to prune within a block for cost or robustness.

Concrete examples from this rerun:
- `sr_breakout`: **8 feature nodes** → model trained with **21 columns** (seed=1)
- `compression_breakout`: **4 feature nodes** → model trained with **18 columns** (seed=1)
- `trend_following`: **10 feature nodes** → model trained with **20 columns** (seed=1)

## Rerun results (recommended YAML writeback)

### sr_breakout

- **Selected groups**: `kline_core`
- **Final requested_features (8 nodes)**:
  - `atr_f`, `poc_hal_features_close_f`, `macd_f`, `rsi_f`, `trend_r2_20_f`, `bb_width_f`, `wick_ratios_f`, `volume_ratio_f`
- **Sharpe_mean (5 seeds)**:
  - baseline: **-0.8581**
  - after `kline_core`: **1.6618**
- **Seed-1 training stats** (`runs/step1_add_kline_core/seed_1/.../results.json`)
  - `n_train_samples=4026`, `n_test_samples=1720`
  - `n_features=21`
  - backtest: `sharpe=2.1003`, `total_trades=148`

### compression_breakout

- **Selected groups**: `market_cap_norm`, `vpin_scene`
- **Final requested_features (4 nodes)**:
  - `compression_duration_f`, `atr_f`, `market_cap_normalized_orderflow_f`, `vpin_scene_semantic_scores_f`
- **Sharpe_mean (5 seeds)**:
  - baseline: **-1.0175**
  - +`market_cap_norm`: **0.7940**
  - +`vpin_scene`: **2.3153**
- **Seed-1 training stats** (`runs/step2_add_vpin_scene/seed_1/.../results.json`)
  - `n_train_samples=3793`, `n_test_samples=1572`
  - `n_features=18`
  - backtest: `sharpe=2.4813`, `total_trades=80`

### trend_following

- **Selected groups**: `kline_core`, `trend_core`
- **Final requested_features (10 nodes)**:
  - `atr_f`, `macd_f`, `rsi_f`, `trend_r2_20_f`, `bb_width_f`, `wick_ratios_f`, `volume_ratio_f`,
    `trend_r2_50_f`, `slope_consistency_score_f`, `trend_volatility_alignment_f`
- **Sharpe_mean (5 seeds)**:
  - baseline: **-1.6617**
  - +`kline_core`: **-0.0326**
  - +`trend_core`: **0.6501**
- **Seed-1 training stats** (`runs/step2_add_trend_core/seed_1/.../results.json`)
  - `n_train_samples=4368`, `n_test_samples=1816`
  - `n_features=20`

### sr_reversal_rr_reg_long

- **Selected groups**: `volume_profile_scene`, `volume_profile`
- **Final requested_features (4 nodes)**:
  - `poc_hal_features_close_f`, `atr_f`, `volume_profile_scene_semantic_scores_f`, `volume_profile_volatility_features_f`
- **Sharpe_mean (5 seeds)**:
  - baseline: **0.8160**
  - +`volume_profile_scene`: **1.2530**
  - +`volume_profile`: **2.0260**
- **Seed-1 training stats** (`runs/step2_add_volume_profile/seed_1/.../results.json`)
  - `n_train_samples=4403`, `n_test_samples=1739`
  - `n_features=22`
  - backtest: `sharpe=1.9480`, `total_trades=27`

## Comparison to the “previous semantic run” (historical best_combo runs)

These are **not apples-to-apples** vs the rerun above because the historical runs used **very different base feature pools** (and their `train_mean` indicates different effective datasets / label filters / ranges).

Still, this explains why you saw “semantic run chose many more features”:

| Strategy | Historical run | Final feature nodes | Selected groups | train_mean (from summary) |
|---|---|---:|---|---:|
| sr_breakout | `sr_breakout_best_combo_v4` | 25 | `trade_cluster_scene`, `wick_scene` | 4185 |
| compression_breakout | `compression_breakout_best_combo_v5` | 20 | `vpin_scene`, `wpt_scene` | 1754 |
| trend_following | `trend_following_best_combo_v5` | 27 | *(none; baseline already best)* | 1825 |

Meanwhile, the rerun above intentionally started from **minimal base_features + semantic groups**, so the greedy search can stop early when additional semantic blocks don’t improve `Sharpe_mean`.

## If you want a strict “two-round” apples-to-apples comparison

We should rerun the historical configuration with the **same**:
- base strategy config (same `features_base.yaml` / base pool),
- date range + timeframe,
- seeds + objective,
- and the same groups-yaml / singleton-expansion settings.

Then we can compare:
- requested feature nodes count
- model column count (`n_features`)
- train/test sample counts
- Sharpe distribution across seeds

## Doc-to-doc comparison (INTERIM vs RERUN)

This section compares:
- **INTERIM**: `docs/strategies/FEATURE_GROUP_SEARCH_INTERIM_RESULTS.md` (2026-01-01, partial / in-progress)
- **RERUN**: this report (2026-01-02 rerun, completed for 3 strategies)

### sr_breakout

- **INTERIM**: `TBD` (not recorded in the interim doc; table shows `sr_breakout_v4: TBD`)
- **RERUN**:
  - selected group(s): `kline_core`
  - Sharpe_mean: baseline **-0.8581** → after `kline_core` **1.6618**
  - final requested_features: `atr_f`, `poc_hal_features_close_f`, `macd_f`, `rsi_f`, `trend_r2_20_f`, `bb_width_f`, `wick_ratios_f`, `volume_ratio_f`

### compression_breakout

- **INTERIM**: `TBD` (table shows `compression_breakout_expanded: TBD`)
- **RERUN**:
  - selected group(s): `market_cap_norm`, `vpin_scene`
  - Sharpe_mean: baseline **-1.0175** → +`market_cap_norm` **0.7940** → +`vpin_scene` **2.3153**
  - final requested_features: `compression_duration_f`, `atr_f`, `market_cap_normalized_orderflow_f`, `vpin_scene_semantic_scores_f`

### trend_following

- **INTERIM**: `TBD` (table shows `trend_following_v4: TBD`)
- **RERUN**:
  - selected group(s): `kline_core`, `trend_core`
  - Sharpe_mean: baseline **-1.6617** → +`kline_core` **-0.0326** → +`trend_core` **0.6501**
  - final requested_features: `atr_f`, `macd_f`, `rsi_f`, `trend_r2_20_f`, `bb_width_f`, `wick_ratios_f`, `volume_ratio_f`, `trend_r2_50_f`, `slope_consistency_score_f`, `trend_volatility_alignment_f`

### sr_reversal_rr_reg_long

- **INTERIM** (sr_reversal_expanded):
  - baseline Sharpe_mean: **1.529** (3 seeds)
  - best Step 1 group: `sqs_hal_high_f` → **2.088**
  - negative groups called out (suggest invert): `trend_r2_50_f`, `dtw_features_reversal_f`, `order_flow_all_features_f`, `dtw_features_trend_f`
- **RERUN**:
  - selected group(s): `volume_profile_scene`, `volume_profile`
  - Sharpe_mean: baseline **0.8160** → +`volume_profile_scene` **1.2530** → +`volume_profile` **2.0260**
  - final requested_features: `poc_hal_features_close_f`, `atr_f`, `volume_profile_scene_semantic_scores_f`, `volume_profile_volatility_features_f`


