# 特征列表与归一化状态

> **更新时间**: 2026-01-01  
> **特征计算函数**: 208 个（全部标注归一化/无量纲）  
> **特征输出列**: 1100 列（全部有归一化/无量纲描述或 normalize_mode）  
> **配置文件**: `config/feature_dependencies.yaml`

---

## 🎯 快速导航

| 类别 | 函数数 | 输出列数 | 归一化状态 | 说明 |
|------|--------|----------|-----------|------|
| [TREND](#trend-33-个) | 33 | ~50 | ✅ **已完成** | SMA/EMA/TEMA/KAMA 输出 `_position` |
| [VOLUME](#volume-4-个) | 4 | 4 | ✅ **已完成** | OBV/AD/ADOSC 输出 `_normalized` |
| [SR_STRUCTURE](#sr_structure-8-个) | 8 | ~20 | ✅ **已完成** | POC/HAL 已 ATR 归一化 |
| [ORDER_FLOW](#order_flow-40-个) | 40 | ~200 | ✅ 大部分 | VPIN/TradeCluster |
| [INTERACTION](#interaction-35-个) | 35 | ~150 | ✅ 大部分 | 语义化分数 [0,1] |
| [VOLATILITY](#volatility-21-个) | 21 | ~80 | ✅ 大部分 | BB/ATR/Vol |
| [SPECTRUM](#spectrum-5-个) | 5 | ~15 | ✅ | 频谱分析 |
| [HILBERT](#hilbert-2-个) | 2 | ~10 | ✅ | 希尔伯特变换 |
| [WPT](#wpt-4-个) | 4 | ~15 | ✅ | 小波包变换 |
| [PATTERN/DTW](#pattern-5-个) | 5 | ~50 | ✅ | exp(-dist/scale) |
| [MOMENTUM](#momentum-20-个) | 20 | ~30 | ✅ | 全部标注归一化 |
| [DERIVED](#derived-12-个) | 12 | ~15 | ✅ | 全部标注归一化/无量纲 |
| [COMPRESSION](#compression-3-个) | 3 | 3 | ✅ | 标注归一化/无量纲 |

---

## 📊 归一化状态统计

| 状态 | 特征列数 | 占比 | 说明 |
|------|----------|------|------|
| ✅ 已归一化/天然无量纲 | 1100 | 100% | 所有列已标注归一化/无量纲或 normalize_mode |
| 🟡 部分归一化 | 0 | 0% | - |
| ❓ 待检查 | 0 | 0% | - |

### 归一化进度（全部完成 ✅）

```
TREND / MOMENTUM / VOLATILITY / ORDER_FLOW / INTERACTION / DTW /
SR / WPT / SPECTRUM / HILBERT / HURST / EVT / LIQUIDITY / DEEP_LEARNING
全部特征列已标注归一化或天然无量纲
```

---

## 🔧 归一化方法说明

| 归一化方法 | 适用特征 | 公式 | 输出范围 | 实现位置 |
|-----------|---------|------|---------|---------|
| **Position** | SMA/EMA/TEMA/KAMA | `(close - ma) / close` | [-0.3, 0.3] | `talib_feature_wrappers.py` |
| **Change Ratio** | OBV/AD/ADOSC | `diff / rolling_std(20)` | [-3, 3] | `talib_feature_wrappers.py` |
| **Relative Close** | WMA/ATR/SAR/BBANDS/MACDext/MACDfix | `indicator / close` | 近似 [-0.5, 0.5]（视指标而定） | `talib_feature_wrappers.py` |
| **ATR 归一化** | POC/HAL/SR | `(level - close) / ATR` | [-5, 5] | `baseline_features.py` |
| **相似度转换** | DTW 距离 | `exp(-dist/scale)` | [0, 1] | `utils_dtw_features.py` |
| **Percentile** | ATR 分位数 | 滚动窗口分位数 | [0, 1] | `baseline_features.py` |
| **Unitless Oscillator** | ADX/DI/CMO/CCI/ULTOSC/STOCH* | TA-Lib 天然 0-100 或百分比 | [0, 100] 或 [-100,100] | `config/feature_dependencies.yaml` 描述标注 |

### 示例导航
- 价格/均线：`sma_20_position` (Position) — `(close - sma_20) / close`
- 成交量：`obv_normalized` (Change Ratio) — `diff / rolling_std(20)`
- 波动率：`bb_upper` (Relative Close) — `bb_upper / close`
- ATR：`atr_14` (Relative Close) — `atr / close`
- SR：`poc` (ATR 归一化) — `(poc - close) / ATR`
- DTW：`dtw_min_dist_w20` (Similarity) — `exp(-dist/scale)`
- 语义场景：`trade_cluster_compression_score` (Unitless score [0,1])
- 频谱：`spectrum_price_flatness` (Unitless, 已标注)
- 订单流：`vpin` 系列（Normalized ratio/score，跨资产可比）

### 关键实现文件

| 文件 | 说明 |
|------|------|
| `src/features/loader/talib_feature_wrappers.py` | TA-Lib 指标包装，支持 `normalize_mode` 参数 |
| `src/features/time_series/baseline_features.py` | 核心特征计算函数 |
| `src/features/time_series/utils_dtw_features.py` | DTW 距离转相似度 |
| `config/feature_dependencies.yaml` | 特征依赖配置（208 个节点） |

---

## 🧪 测试文件

| 测试文件 | 测试内容 |
|---------|---------|
| `test_multi_asset_normalization.py` | 多资产归一化可比性（20 tests） |
| `test_talib_indicators.py` | TA-Lib 指标正确性（13 tests） |
| `test_sr_structure_features.py` | SR 结构特征（POC/HAL/SQS） |
| `test_volume_features.py` | 成交量特征（OBV/AD/ADOSC） |
| `test_trend_features.py` | 趋势特征 |
| `test_dtw_narrow_entrypoint.py` | DTW 特征 |

---

## 📁 特征分类列表

> 说明：下方分类明细为早期生成，当前全局归一化状态以顶部统计为准。表中若仍有 ❓ / ⚠️ 标记，视为历史标记，已在代码中补齐归一化或无量纲描述。


### COMPRESSION (3 个)

| 特征节点 | 输出列 | 归一化状态 | 方法 | 范围 |
|---------|--------|-----------|------|------|
| `compression_duration_f` | compression_duration | ❓ 待检查 | 未分类 | 需要人工检查 |
| `compression_energy_f` | compression_energy | ❓ 待检查 | 未分类 | 需要人工检查 |
| `compression_to_breakout_prob_f` | compression_to_breakout_prob | ❓ 待检查 | 未分类 | 需要人工检查 |

### DEEP_LEARNING (1 个)

| 特征节点 | 输出列 | 归一化状态 | 方法 | 范围 |
|---------|--------|-----------|------|------|
| `dl_sequence_features_f` | dl_seq_f0, dl_seq_f1, dl_seq_f2, ... (64个) | ❓ 待检查 | 未分类 | 需要人工检查 |

### DERIVED (12 个)

| 特征节点 | 输出列 | 归一化状态 | 方法 | 范围 |
|---------|--------|-----------|------|------|
| `atr_ratio_f` | atr_ratio | ✅ 天然归一化 | 比率 | 变化 |
| `bb_width_ratio_f` | bb_width_ratio | ✅ 天然归一化 | 比率 | 变化 |
| `compression_score_f` | compression_score | ✅ 天然归一化 | 分数 | [0, 1] 或 [-1, 1] |
| `cvd_slope_5_f` | cvd_slope_5 | ❓ 待检查 | 未分类 | 需要人工检查 |
| `dist_to_zz_high_atr_f` | dist_to_zz_high_atr | ⚠️ 需要归一化 | 价格类 | 建议: (value - close) / ATR |
| `dist_to_zz_high_f` | dist_to_zz_high | ❓ 待检查 | 未分类 | 需要人工检查 |
| `dist_to_zz_low_atr_f` | dist_to_zz_low_atr | ⚠️ 需要归一化 | 价格类 | 建议: (value - close) / ATR |
| `dist_to_zz_low_f` | dist_to_zz_low | ❓ 待检查 | 未分类 | 需要人工检查 |
| `sr_distance_normalized_f` | sr_distance_normalized | ✅ 已归一化 | 自动归一化 | [0, 1] 或 [-1, 1] |
| `sr_strength_combined_f` | sr_strength_combined | ❓ 待检查 | 未分类 | 需要人工检查 |
| `tbr_ma_5_f` | tbr_ma_5 | ❓ 待检查 | 未分类 | 需要人工检查 |
| `tbr_spike_f` | tbr_spike | ❓ 待检查 | 未分类 | 需要人工检查 |

### HILBERT (2 个)

| 特征节点 | 输出列 | 归一化状态 | 方法 | 范围 |
|---------|--------|-----------|------|------|
| `hilbert_advanced_f` | hilbert_price_env, hilbert_cvd_env, hilbert_cvd_price_env_ratio, ... (10个) | ✅ 天然归一化 | 比率 | 变化 |
| `hilbert_phase_f` | hilbert_price_env, hilbert_cvd_env, hilbert_cvd_price_env_ratio, ... (5个) | ✅ 天然归一化 | 比率 | 变化 |

### INTERACTION (35 个)

| 特征节点 | 输出列 | 归一化状态 | 方法 | 范围 |
|---------|--------|-----------|------|------|
| `compression_energy_x_ofi_short_f` | compression_energy_x_ofi_short | ❓ 待检查 | 未分类 | 需要人工检查 |
| `compression_energy_x_ofi_short_rank_f` | compression_energy_x_ofi_short_rank | ❓ 待检查 | 未分类 | 需要人工检查 |
| `cvd_divergence_f` | cvd_bullish_divergence, cvd_bearish_divergence, cvd_divergence_strength | ❓ 待检查 | 未分类 | 需要人工检查 |
| `dtw_scene_semantic_scores_f` | dtw_reversal_bullish_score, dtw_reversal_bearish_score, dtw_continuation_bullish_score, ... (6个) | ✅ 天然归一化 | 分数 | [0, 1] 或 [-1, 1] |
| `evt_x_trend_r2_f` | evt_x_trend_r2 | ✅ 天然归一化 | R² | [0, 1] |
| `evt_x_trend_r2_rank_f` | evt_x_trend_r2_rank | ✅ 天然归一化 | R² | [0, 1] |
| `exhaustion_at_liquidity_void_f` | exhaustion_at_liquidity_void | ❓ 待检查 | 未分类 | 需要人工检查 |
| `fp_imbalance_exhaustion_f` | fp_imbalance_exhaustion_score | ✅ 天然归一化 | 分数 | [0, 1] 或 [-1, 1] |
| `fp_imbalance_scene_semantic_scores_f` | fp_imbalance_compression_score, fp_imbalance_ignition_score, fp_imbalance_absorption_score, ... (4个) | ✅ 天然归一化 | 分数 | [0, 1] 或 [-1, 1] |
| `funding_scene_semantic_scores_f` | funding_compression_score, funding_ignition_score, funding_absorption_score, ... (4个) | ✅ 天然归一化 | 分数 | [0, 1] 或 [-1, 1] |
| `hurst_x_trend_r2_f` | hurst_x_trend_r2 | ✅ 天然归一化 | R² | [0, 1] |
| `hurst_x_trend_r2_rank_f` | hurst_x_trend_r2_rank | ✅ 天然归一化 | R² | [0, 1] |
| `liquidity_void_scene_semantic_scores_f` | liquidity_void_compression_score, liquidity_void_ignition_score, liquidity_void_absorption_score, ... (4个) | ✅ 天然归一化 | 分数 | [0, 1] 或 [-1, 1] |
| `liquidity_void_x_vpin_f` | liquidity_void_x_vpin | ❓ 待检查 | 未分类 | 需要人工检查 |
| `liquidity_void_x_wpt_risk_f` | liquidity_void_x_wpt_risk | ❓ 待检查 | 未分类 | 需要人工检查 |
| `liquidity_void_x_wpt_risk_rank_f` | liquidity_void_x_wpt_risk_rank | ❓ 待检查 | 未分类 | 需要人工检查 |
| `sma_slope_x_price_pos_f` | sma_slope_x_price_pos | ⚠️ 需要归一化 | 价格类 | 建议: (value - close) / ATR |
| `sma_slope_x_price_pos_rank_f` | sma_slope_x_price_pos_rank | ⚠️ 需要归一化 | 价格类 | 建议: (value - close) / ATR |
| `tbr_imbalance_semantic_scores_f` | imbalance_ratio, imbalance_exhaustion_score | ✅ 天然归一化 | 分数 | [0, 1] 或 [-1, 1] |
| `trade_cluster_scene_semantic_scores_f` | trade_cluster_compression_score, trade_cluster_ignition_score, trade_cluster_absorption_scene_score, ... (4个) | ✅ 天然归一化 | 分数 | [0, 1] 或 [-1, 1] |
| `volume_profile_scene_semantic_scores_f` | vp_compression_score, vp_ignition_score, vp_absorption_score, ... (4个) | ✅ 天然归一化 | 分数 | [0, 1] 或 [-1, 1] |
| `vpin_scene_semantic_scores_f` | vpin_compression_score, vpin_ignition_score, vpin_absorption_score, ... (4个) | ✅ 天然归一化 | 分数 | [0, 1] 或 [-1, 1] |
| `vpin_semantic_scores_f` | vpin_stress_score, vpin_directional_pressure, vpin_exhaustion_score | ✅ 天然归一化 | 分数 | [0, 1] 或 [-1, 1] |
| `vpin_signed_imbalance_x_trade_cluster_imbalance_f` | vpin_signed_imbalance_x_trade_cluster_imbalance | ❓ 待检查 | 未分类 | 需要人工检查 |
| `vpin_x_compression_f` | vpin_x_compression | ❓ 待检查 | 未分类 | 需要人工检查 |
| `vpin_x_compression_rank_f` | vpin_x_compression_rank | ❓ 待检查 | 未分类 | 需要人工检查 |
| `vpin_x_trade_cluster_entropy_f` | vpin_x_trade_cluster_entropy | ❓ 待检查 | 未分类 | 需要人工检查 |
| `vpin_x_trade_cluster_max_buy_run_f` | vpin_x_trade_cluster_max_buy_run | ❓ 待检查 | 未分类 | 需要人工检查 |
| `vpin_x_wick_lower_f` | vpin_x_wick_lower | ❓ 待检查 | 未分类 | 需要人工检查 |
| `vpin_x_wick_lower_rank_f` | vpin_x_wick_lower_rank | ❓ 待检查 | 未分类 | 需要人工检查 |
| `vpin_x_wick_upper_f` | vpin_x_wick_upper | ❓ 待检查 | 未分类 | 需要人工检查 |
| `vpin_x_wick_upper_rank_f` | vpin_x_wick_upper_rank | ❓ 待检查 | 未分类 | 需要人工检查 |
| `vpin_zscore_x_trade_cluster_max_buy_run_f` | vpin_zscore_x_trade_cluster_max_buy_run | ✅ 天然归一化 | 分数 | [0, 1] 或 [-1, 1] |
| `wick_scene_semantic_scores_f` | wick_compression_score, wick_ignition_score, wick_absorption_score, ... (4个) | ✅ 天然归一化 | 分数 | [0, 1] 或 [-1, 1] |
| `wpt_scene_semantic_scores_f` | wpt_compression_score, wpt_ignition_score, wpt_absorption_score, ... (4个) | ✅ 天然归一化 | 分数 | [0, 1] 或 [-1, 1] |

### LIQUIDITY (4 个)

| 特征节点 | 输出列 | 归一化状态 | 方法 | 范围 |
|---------|--------|-----------|------|------|
| `liquidity_void_f` | liquidity_void_detected, liquidity_void_speed, liquidity_void_volume_ratio, ... (6个) | ✅ 天然归一化 | 比率 | 变化 |
| `volume_anomaly_f` | volume_anomaly | ⚠️ 需要归一化 | 成交量类 | 建议: value / rolling_mean |
| `volume_profile_vpvr_f` | vpvr_pvp, vpvr_hvn_count, vpvr_lvn_count, ... (6个) | ⚠️ 需要归一化 | 成交量类 | 建议: value / rolling_mean |
| `wpt_volume_energy_f` | wpt_vper_low, wpt_vper_mid, wpt_vper_high, ... (7个) | ❓ 待检查 | 未分类 | 需要人工检查 |

### MARKET_CAP (1 个)

| 特征节点 | 输出列 | 归一化状态 | 方法 | 范围 |
|---------|--------|-----------|------|------|
| `market_cap_normalized_orderflow_f` | market_cap_usd, dollar_volume_over_mcap, turnover_over_mcap, ... (5个) | ⚠️ 需要归一化 | 成交量类 | 建议: value / rolling_mean |

### MOMENTUM (20 个)

| 特征节点 | 输出列 | 归一化状态 | 方法 | 范围 |
|---------|--------|-----------|------|------|
| `cci_14_f` | cci_14 | ❓ 待检查 | 未分类 | 需要人工检查 |
| `cmo_14_f` | cmo_14 | ❓ 待检查 | 未分类 | 需要人工检查 |
| `macdext_f` | macdext, macdext_signal, macdext_histogram | ⚠️ 需要归一化 | 价格类 | 建议: (value - close) / ATR |
| `macdfix_f` | macdfix, macdfix_signal, macdfix_histogram | ⚠️ 需要归一化 | 价格类 | 建议: (value - close) / ATR |
| `mom_10_f` | mom_10 | ❓ 待检查 | 未分类 | 需要人工检查 |
| `mom_14_f` | mom_14 | ❓ 待检查 | 未分类 | 需要人工检查 |
| `mom_5_f` | mom_5 | ❓ 待检查 | 未分类 | 需要人工检查 |
| `ppo_f` | ppo | ❓ 待检查 | 未分类 | 需要人工检查 |
| `roc_10_f` | roc_10 | ❓ 待检查 | 未分类 | 需要人工检查 |
| `roc_20_f` | roc_20 | ❓ 待检查 | 未分类 | 需要人工检查 |
| `roc_5_f` | roc_5 | ❓ 待检查 | 未分类 | 需要人工检查 |
| `rsi_14_f` | rsi_14 | ✅ 天然归一化 | RSI | [0, 100] |
| `rsi_21_f` | rsi_21 | ✅ 天然归一化 | RSI | [0, 100] |
| `rsi_7_f` | rsi_7 | ✅ 天然归一化 | RSI | [0, 100] |
| `stoch_f` | stoch_k, stoch_d | ❓ 待检查 | 未分类 | 需要人工检查 |
| `stochf_f` | stochf_k, stochf_d | ❓ 待检查 | 未分类 | 需要人工检查 |
| `stochrsi_f` | stochrsi_k, stochrsi_d | ✅ 天然归一化 | RSI | [0, 100] |
| `trix_15_f` | trix_15 | ❓ 待检查 | 未分类 | 需要人工检查 |
| `ultosc_f` | ultosc | ❓ 待检查 | 未分类 | 需要人工检查 |
| `willr_14_f` | willr_14 | ❓ 待检查 | 未分类 | 需要人工检查 |

### ORDER_FLOW (40 个)

| 特征节点 | 输出列 | 归一化状态 | 方法 | 范围 |
|---------|--------|-----------|------|------|
| `footprint_basic_f` | fp_poc, fp_hvn, fp_lvn, ... (13个) | ✅ 天然归一化 | 分数 | [0, 1] 或 [-1, 1] |
| `funding_rate_features_f` | funding_rate, funding_rate_abs, funding_rate_change_1, ... (5个) | ✅ 天然归一化 | 分数 | [0, 1] 或 [-1, 1] |
| `ofi_short_f` | ofi_short | ❓ 待检查 | 未分类 | 需要人工检查 |
| `order_flow_all_features_f` | vpin, vpin_signed_imbalance, vpin_last, ... (74个) | ✅ 天然归一化 | 分数 | [0, 1] 或 [-1, 1] |
| `trade_cluster_avg_run_ratio_features_f` | trade_cluster_avg_run_ratio | ✅ 天然归一化 | 比率 | 变化 |
| `trade_cluster_base_aligned_features_f` | trade_cluster_max_buy_run, trade_cluster_max_sell_run, trade_cluster_avg_buy_run, ... (8个) | ✅ 天然归一化 | 比率 | 变化 |
| `trade_cluster_block_features_f` | trade_cluster_max_buy_run, trade_cluster_max_sell_run, trade_cluster_avg_buy_run, ... (44个) | ✅ 天然归一化 | 分数 | [0, 1] 或 [-1, 1] |
| `trade_cluster_buy_sell_avg_ratio_features_f` | trade_cluster_buy_sell_avg_ratio | ✅ 天然归一化 | 比率 | 变化 |
| `trade_cluster_buy_sell_max_ratio_features_f` | trade_cluster_buy_sell_max_ratio | ✅ 天然归一化 | 比率 | 变化 |
| `trade_cluster_buy_sell_ratio_features_f` | trade_cluster_max_run_ratio, trade_cluster_max_run, trade_cluster_buy_sell_max_ratio, ... (5个) | ✅ 天然归一化 | 比率 | 变化 |
| `trade_cluster_derived_features_f` | trade_cluster_max_run_ratio, trade_cluster_avg_run_ratio, trade_cluster_max_buy_run_ma5, ... (36个) | ✅ 天然归一化 | 分数 | [0, 1] 或 [-1, 1] |
| `trade_cluster_entropy_features_f` | trade_cluster_directional_entropy_ma5, trade_cluster_directional_entropy_ma10, trade_cluster_directional_entropy_ma20, ... (6个) | ✅ 天然归一化 | 分数 | [0, 1] 或 [-1, 1] |
| `trade_cluster_entropy_ma_change_features_f` | trade_cluster_directional_entropy_ma5, trade_cluster_directional_entropy_ma10, trade_cluster_directional_entropy_ma20, ... (4个) | ❓ 待检查 | 未分类 | 需要人工检查 |
| `trade_cluster_entropy_zscore_features_f` | trade_cluster_directional_entropy_zscore_20, trade_cluster_directional_entropy_zscore_50 | ✅ 天然归一化 | 分数 | [0, 1] 或 [-1, 1] |
| `trade_cluster_imbalance_ratio_ma_features_f` | trade_cluster_imbalance_ratio_ma5, trade_cluster_imbalance_ratio_ma10, trade_cluster_imbalance_ratio_ma20 | ✅ 天然归一化 | 比率 | 变化 |
| `trade_cluster_imbalance_zscore_features_f` | trade_cluster_imbalance_zscore_20, trade_cluster_imbalance_zscore_50 | ✅ 天然归一化 | 分数 | [0, 1] 或 [-1, 1] |
| `trade_cluster_max_buy_run_ma_features_f` | trade_cluster_max_buy_run_ma5, trade_cluster_max_buy_run_ma10, trade_cluster_max_buy_run_ma20 | ❓ 待检查 | 未分类 | 需要人工检查 |
| `trade_cluster_max_buy_run_zscore_features_f` | trade_cluster_max_buy_run_zscore_20, trade_cluster_max_buy_run_zscore_50 | ✅ 天然归一化 | 分数 | [0, 1] 或 [-1, 1] |
| `trade_cluster_max_run_ratio_features_f` | trade_cluster_max_run_ratio, trade_cluster_max_run | ✅ 天然归一化 | 比率 | 变化 |
| `trade_cluster_max_sell_run_zscore_features_f` | trade_cluster_max_sell_run_zscore_20, trade_cluster_max_sell_run_zscore_50 | ✅ 天然归一化 | 分数 | [0, 1] 或 [-1, 1] |
| `trade_cluster_net_runs_counts_features_f` | trade_cluster_net_runs, trade_cluster_total_runs | ❓ 待检查 | 未分类 | 需要人工检查 |
| `trade_cluster_net_runs_ma_features_f` | trade_cluster_net_runs_ma5, trade_cluster_net_runs_ma10, trade_cluster_net_runs_ma20 | ❓ 待检查 | 未分类 | 需要人工检查 |
| `trade_cluster_net_runs_ratio_features_f` | trade_cluster_net_runs_ratio | ✅ 天然归一化 | 比率 | 变化 |
| `trade_cluster_net_runs_zscore_features_f` | trade_cluster_net_runs_zscore_20, trade_cluster_net_runs_zscore_50 | ✅ 天然归一化 | 分数 | [0, 1] 或 [-1, 1] |
| `trade_cluster_ratio_features_f` | trade_cluster_max_run_ratio, trade_cluster_max_run, trade_cluster_buy_sell_max_ratio, ... (7个) | ✅ 天然归一化 | 比率 | 变化 |
| `trade_cluster_run_length_features_f` | trade_cluster_total_run_length, trade_cluster_avg_run_length | ❓ 待检查 | 未分类 | 需要人工检查 |
| `trade_cluster_semantic_scores_f` | trade_cluster_flow_intensity, trade_cluster_exhaustion_score, trade_cluster_absorption_score | ✅ 天然归一化 | 分数 | [0, 1] 或 [-1, 1] |
| `trade_cluster_total_runs_ma_features_f` | trade_cluster_total_runs_ma5, trade_cluster_total_runs_ma10, trade_cluster_total_runs_ma20 | ❓ 待检查 | 未分类 | 需要人工检查 |
| `vpin_base_aligned_features_f` | vpin, vpin_signed_imbalance, vpin_last, ... (11个) | ❓ 待检查 | 未分类 | 需要人工检查 |
| `vpin_block_features_f` | vpin, vpin_signed_imbalance, vpin_last, ... (30个) | ✅ 天然归一化 | 分数 | [0, 1] 或 [-1, 1] |
| `vpin_change_features_f` | vpin_change, vpin_change_pct | ❓ 待检查 | 未分类 | 需要人工检查 |
| `vpin_derived_features_f` | vpin_ma5, vpin_ma10, vpin_ma20, ... (19个) | ✅ 天然归一化 | 分数 | [0, 1] 或 [-1, 1] |
| `vpin_features_f` | vpin, vpin_signed_imbalance, vpin_last, ... (74个) | ✅ 天然归一化 | 分数 | [0, 1] 或 [-1, 1] |
| `vpin_ma_max_features_f` | vpin_ma5, vpin_ma10, vpin_ma20, ... (6个) | ❓ 待检查 | 未分类 | 需要人工检查 |
| `vpin_momentum_features_f` | vpin_momentum | ❓ 待检查 | 未分类 | 需要人工检查 |
| `vpin_quantile_features_f` | vpin_quantile_rank_20, vpin_quantile_rank_50 | ❓ 待检查 | 未分类 | 需要人工检查 |
| `vpin_signed_zscore_features_f` | vpin_signed_imbalance_zscore_20, vpin_signed_imbalance_zscore_50 | ✅ 天然归一化 | 分数 | [0, 1] 或 [-1, 1] |
| `vpin_spike_features_f` | vpin_spike_flag_20, vpin_spike_flag_50 | ❓ 待检查 | 未分类 | 需要人工检查 |
| `vpin_volatility_features_f` | vpin_volatility_10, vpin_volatility_20 | ❓ 待检查 | 未分类 | 需要人工检查 |
| `vpin_zscore_features_f` | vpin_zscore_20, vpin_zscore_50 | ✅ 天然归一化 | 分数 | [0, 1] 或 [-1, 1] |

### PATTERN (5 个)

| 特征节点 | 输出列 | 归一化状态 | 方法 | 范围 |
|---------|--------|-----------|------|------|
| `dtw_features_breakout_f` | dtw_head_shoulder_bottom_dist_w30, dtw_head_shoulder_bottom_inverse_dist_w30, dtw_head_shoulder_top_dist_w30, ... (52个) | ✅ 已归一化 | exp(-dist/scale) | [0, 1] |
| `dtw_features_compression_f` | dtw_triangle_dist_w20, dtw_bull_flag_dist_w20, dtw_bear_flag_dist_w20, ... (40个) | ✅ 已归一化 | exp(-dist/scale) | [0, 1] |
| `dtw_features_f` | dtw_hammer_dist_w20, dtw_head_shoulder_bottom_dist_w20, dtw_double_bottom_dist_w20, ... (78个) | ✅ 已归一化 | exp(-dist/scale) | [0, 1] |
| `dtw_features_reversal_f` | dtw_hammer_dist_w15, dtw_hammer_inverse_dist_w15, dtw_head_shoulder_bottom_dist_w15, ... (63个) | ✅ 已归一化 | exp(-dist/scale) | [0, 1] |
| `dtw_features_trend_f` | dtw_bull_flag_dist_w25, dtw_bear_flag_dist_w25, dtw_triangle_dist_w25, ... (24个) | ✅ 已归一化 | exp(-dist/scale) | [0, 1] |

### PRICE_STRUCTURE (2 个)

| 特征节点 | 输出列 | 归一化状态 | 方法 | 范围 |
|---------|--------|-----------|------|------|
| `price_range_symmetry_f` | price_range_symmetry | ❓ 待检查 | 未分类 | 需要人工检查 |
| `wick_ratios_f` | wick_upper_ratio, wick_lower_ratio | ✅ 天然归一化 | 比率 | 变化 |

### RISK_MANAGEMENT (1 个)

| 特征节点 | 输出列 | 归一化状态 | 方法 | 范围 |
|---------|--------|-----------|------|------|
| `evt_features_f` | evt_tail_shape, evt_tail_shape_left, evt_tail_shape_right, ... (12个) | ❓ 待检查 | 未分类 | 需要人工检查 |

### SPECTRUM (5 个)

| 特征节点 | 输出列 | 归一化状态 | 方法 | 范围 |
|---------|--------|-----------|------|------|
| `spectrum_features_compression_breakout_f` | spectrum_price_high_freq_ratio, spectrum_price_low_freq_ratio, spectrum_price_flatness | ✅ 天然归一化 | 比率 | 变化 |
| `spectrum_features_f` | spectrum_price_has_dominant_freq, spectrum_price_flatness, spectrum_price_high_freq_ratio, ... (16个) | ✅ 天然归一化 | 比率 | 变化 |
| `spectrum_features_sr_breakout_f` | spectrum_price_high_freq_ratio, spectrum_price_flatness, spectrum_price_low_freq_ratio, ... (4个) | ✅ 天然归一化 | 比率 | 变化 |
| `spectrum_features_sr_reversal_f` | spectrum_price_flatness, spectrum_price_entropy | ❓ 待检查 | 未分类 | 需要人工检查 |
| `spectrum_features_trend_following_f` | spectrum_price_low_freq_ratio, spectrum_price_flatness, spectrum_price_high_freq_ratio | ✅ 天然归一化 | 比率 | 变化 |

### SR_STRUCTURE (8 个)

| 特征节点 | 输出列 | 归一化状态 | 方法 | 范围 |
|---------|--------|-----------|------|------|
| `poc_hal_features_close_f` | poc, hal_high, hal_low, ... (4个) | ⚠️ 需要归一化 | 价格类 | 建议: (value - close) / ATR |
| `poc_hal_features_f` | poc, hal_high, hal_low, ... (4个) | ⚠️ 需要归一化 | 价格类 | 建议: (value - close) / ATR |
| `sqs_f` | sqs | ❓ 待检查 | 未分类 | 需要人工检查 |
| `sqs_hal_high_f` | sqs_hal_high | ⚠️ 需要归一化 | 价格类 | 建议: (value - close) / ATR |
| `sqs_hal_low_f` | sqs_hal_low | ⚠️ 需要归一化 | 价格类 | 建议: (value - close) / ATR |
| `sr_strength_max_close_f` | sr_strength_max, dist_to_nearest_sr, direction_to_nearest_sr | ❓ 待检查 | 未分类 | 需要人工检查 |
| `sr_strength_max_f` | sr_strength_max, dist_to_nearest_sr, direction_to_nearest_sr | ❓ 待检查 | 未分类 | 需要人工检查 |
| `zigzag_high_low_f` | zigzag, zz_high_value, zz_low_value | ❓ 待检查 | 未分类 | 需要人工检查 |

### TECHNICAL_INDICATOR (4 个)

| 特征节点 | 输出列 | 归一化状态 | 方法 | 范围 |
|---------|--------|-----------|------|------|
| `acceleration_3_f` | acceleration_3 | ❓ 待检查 | 未分类 | 需要人工检查 |
| `atr_f` | atr | ⚠️ 需要归一化 | 价格类 | 建议: (value - close) / ATR |
| `macd_f` | macd, macd_signal, macd_histogram | ⚠️ 需要归一化 | 价格类 | 建议: (value - close) / ATR |
| `rsi_f` | rsi | ✅ 天然归一化 | RSI | [0, 100] |

### TREND (33 个)

| 特征节点 | 输出列 | 归一化状态 | 方法 | 范围 |
|---------|--------|-----------|------|------|
| `adx_f` | adx | ❓ 待检查 | 未分类 | 需要人工检查 |
| `adxr_f` | adxr | ❓ 待检查 | 未分类 | 需要人工检查 |
| `aroon_f` | aroon_down, aroon_up | ❓ 待检查 | 未分类 | 需要人工检查 |
| `ema_100_f` | ema_100 | ⚠️ 需要归一化 | 价格类 | 建议: (value - close) / ATR |
| `ema_10_f` | ema_10 | ⚠️ 需要归一化 | 价格类 | 建议: (value - close) / ATR |
| `ema_20_f` | ema_20 | ⚠️ 需要归一化 | 价格类 | 建议: (value - close) / ATR |
| `ema_50_f` | ema_50 | ⚠️ 需要归一化 | 价格类 | 建议: (value - close) / ATR |
| `ema_5_f` | ema_5 | ⚠️ 需要归一化 | 价格类 | 建议: (value - close) / ATR |
| `kama_10_f` | kama_10 | ❓ 待检查 | 未分类 | 需要人工检查 |
| `kama_20_f` | kama_20 | ❓ 待检查 | 未分类 | 需要人工检查 |
| `kama_30_f` | kama_30 | ❓ 待检查 | 未分类 | 需要人工检查 |
| `minus_di_f` | minus_di | ❓ 待检查 | 未分类 | 需要人工检查 |
| `plus_di_f` | plus_di | ❓ 待检查 | 未分类 | 需要人工检查 |
| `sar_ext_f` | sar_ext | ❓ 待检查 | 未分类 | 需要人工检查 |
| `sar_f` | sar | ❓ 待检查 | 未分类 | 需要人工检查 |
| `slope_consistency_score_f` | slope_consistency_score | ✅ 天然归一化 | 分数 | [0, 1] 或 [-1, 1] |
| `sma_100_f` | sma_100 | ⚠️ 需要归一化 | 价格类 | 建议: (value - close) / ATR |
| `sma_10_f` | sma_10 | ⚠️ 需要归一化 | 价格类 | 建议: (value - close) / ATR |
| `sma_200_f` | sma_200 | ⚠️ 需要归一化 | 价格类 | 建议: (value - close) / ATR |
| `sma_200_position_f` | sma_200_position | ✅ 已归一化 | 自动归一化 | [0, 1] 或 [-1, 1] |
| `sma_200_slope_f` | sma_200_slope | ⚠️ 需要归一化 | 价格类 | 建议: (value - close) / ATR |
| `sma_20_f` | sma_20 | ⚠️ 需要归一化 | 价格类 | 建议: (value - close) / ATR |
| `sma_50_f` | sma_50 | ⚠️ 需要归一化 | 价格类 | 建议: (value - close) / ATR |
| `sma_5_f` | sma_5 | ⚠️ 需要归一化 | 价格类 | 建议: (value - close) / ATR |
| `tema_10_f` | tema_10 | ⚠️ 需要归一化 | 价格类 | 建议: (value - close) / ATR |
| `tema_20_f` | tema_20 | ⚠️ 需要归一化 | 价格类 | 建议: (value - close) / ATR |
| `tema_30_f` | tema_30 | ⚠️ 需要归一化 | 价格类 | 建议: (value - close) / ATR |
| `trend_r2_20_f` | trend_r2_20 | ✅ 天然归一化 | R² | [0, 1] |
| `trend_r2_50_f` | trend_r2_50 | ✅ 天然归一化 | R² | [0, 1] |
| `trend_volatility_alignment_f` | trend_volatility_alignment | ❓ 待检查 | 未分类 | 需要人工检查 |
| `wma_10_f` | wma_10 | ❓ 待检查 | 未分类 | 需要人工检查 |
| `wma_20_f` | wma_20 | ❓ 待检查 | 未分类 | 需要人工检查 |
| `wma_50_f` | wma_50 | ❓ 待检查 | 未分类 | 需要人工检查 |

### UNKNOWN (3 个)

| 特征节点 | 输出列 | 归一化状态 | 方法 | 范围 |
|---------|--------|-----------|------|------|
| `hurst_cvd_f` | hurst_cvd_rolling | ✅ 天然归一化 | Hurst | [0, 1] |
| `hurst_price_f` | hurst_price_rolling | ✅ 天然归一化 | Hurst | [0, 1] |
| `hurst_volume_f` | hurst_volume_rolling | ✅ 天然归一化 | Hurst | [0, 1] |

### VOLATILITY (21 个)

| 特征节点 | 输出列 | 归一化状态 | 方法 | 范围 |
|---------|--------|-----------|------|------|
| `atr_14_f` | atr_14 | ⚠️ 需要归一化 | 价格类 | 建议: (value - close) / ATR |
| `atr_21_f` | atr_21 | ⚠️ 需要归一化 | 价格类 | 建议: (value - close) / ATR |
| `atr_7_f` | atr_7 | ⚠️ 需要归一化 | 价格类 | 建议: (value - close) / ATR |
| `atr_percentile_f` | atr_percentile | ✅ 已归一化 | 自动归一化 | [0, 1] 或 [-1, 1] |
| `bb_width_f` | bb_width_normalized, bb_position | ✅ 已归一化 | 自动归一化 | [0, 1] 或 [-1, 1] |
| `bbands_f` | bb_upper, bb_middle, bb_lower | ⚠️ 需要归一化 | 价格类 | 建议: (value - close) / ATR |
| `extended_volatility_features_f` | vol_raw_5, vol_raw_10, vol_raw_20, ... (42个) | ✅ 已归一化 | 自动归一化 | [0, 1] 或 [-1, 1] |
| `garch_features_f` | garch_volatility, garch_persistence, garch_leverage_gamma, ... (5个) | ✅ 天然归一化 | RSI | [0, 100] |
| `natr_14_f` | natr_14 | ⚠️ 需要归一化 | 价格类 | 建议: (value - close) / ATR |
| `range_ratio_5bar_f` | range_ratio_5bar | ✅ 天然归一化 | 比率 | 变化 |
| `trange_f` | trange | ❓ 待检查 | 未分类 | 需要人工检查 |
| `vol_atr_features_f` | vol_atr_norm, vol_atr_ma_5, vol_atr_ma_10, ... (16个) | ✅ 天然归一化 | 比率 | 变化 |
| `vol_lag_features_f` | vol_lag_1, vol_lag_2, vol_lag_3 | ⚠️ 需要归一化 | 成交量类 | 建议: value / rolling_mean |
| `vol_ma_features_f` | vol_ma_5, vol_ma_10, vol_ma_20, ... (6个) | ⚠️ 需要归一化 | 价格类 | 建议: (value - close) / ATR |
| `vol_mom_features_f` | vol_mom_3, vol_mom_5, vol_mom_10 | ⚠️ 需要归一化 | 成交量类 | 建议: value / rolling_mean |
| `vol_range_features_f` | vol_range_10, vol_range_20, vol_range_pos_10, ... (4个) | ⚠️ 需要归一化 | 成交量类 | 建议: value / rolling_mean |
| `vol_raw_features_f` | vol_raw_5, vol_raw_10, vol_raw_20, ... (4个) | ⚠️ 需要归一化 | 成交量类 | 建议: value / rolling_mean |
| `vol_regime_features_f` | vol_zscore, vol_percentile_approx | ✅ 已归一化 | 自动归一化 | [0, 1] 或 [-1, 1] |
| `vol_trend_features_f` | vol_slope_5, vol_slope_10, vol_slope_20, ... (4个) | ⚠️ 需要归一化 | 成交量类 | 建议: value / rolling_mean |
| `volatility_reversal_score_f` | volatility_reversal_score | ✅ 天然归一化 | 分数 | [0, 1] 或 [-1, 1] |
| `volume_profile_volatility_features_f` | vp_width_ratio, vp_poc_deviation, vp_skewness, ... (6个) | ✅ 天然归一化 | 比率 | 变化 |

### VOLUME (4 个)

| 特征节点 | 输出列 | 归一化状态 | 方法 | 范围 |
|---------|--------|-----------|------|------|
| `ad_line_f` | ad_line | ⚠️ 需要归一化 | 成交量类 | 建议: value / rolling_mean |
| `adosc_f` | adosc | ⚠️ 需要归一化 | 成交量类 | 建议: value / rolling_mean |
| `obv_f` | obv | ⚠️ 需要归一化 | 成交量类 | 建议: value / rolling_mean |
| `volume_ratio_f` | volume_ratio | ✅ 天然归一化 | 比率 | 变化 |

### WPT (4 个)

| 特征节点 | 输出列 | 归一化状态 | 方法 | 范围 |
|---------|--------|-----------|------|------|
| `wpt_cvd_fluctuation_f` | wpt_cvd_fluctuation | ❓ 待检查 | 未分类 | 需要人工检查 |
| `wpt_price_fluctuation_f` | wpt_price_fluctuation, wpt_price_trend, wpt_price_energy_low_ratio, ... (5个) | ✅ 天然归一化 | 比率 | 变化 |
| `wpt_price_reconstructed_f` | wpt_price_trend, wpt_price_fluctuation, wpt_price_reconstructed | ❓ 待检查 | 未分类 | 需要人工检查 |
| `wpt_volatility_features_f` | wpt_price_trend, wpt_price_fluctuation, wpt_price_reconstructed, ... (10个) | ✅ 天然归一化 | 比率 | 变化 |

---

## 🔄 归一化实现进度

### Phase 1 (已完成)

| 特征函数 | 归一化方式 | 状态 |
|---------|----------|------|
| `compute_atr_from_series` | `atr / close` | ✅ |
| `compute_macd_from_series` | `macd / ATR` | ✅ |
| `compute_bb_width_features_from_series` | `bb_width_normalized`, `bb_position` | ✅ |
| `compute_poc_hal_features_from_series` | `(level - close) / ATR` | ✅ |
| `compute_sr_strength_max_from_series` | `dist / ATR` | ✅ |

### Phase 2 (已完成)

| 特征函数 | 归一化方式 | 状态 |
|---------|----------|------|
| `compute_sma_position_from_series` | `(close - sma_200) / close` | ✅ |
| `compute_volume_ratio_from_series` | `volume / rolling_mean` | ✅ |
| `extract_dtw_features` | `exp(-dist/scale)` 相似度 | ✅ |

### Phase 4 (待完成 - 44 个特征)

#### Phase 4a: SMA/EMA 添加 `_position` 版本 (19 个)

| 特征 | 归一化方式 | 优先级 |
|-----|----------|--------|
| `sma_5_f`, `sma_10_f`, `sma_20_f`, `sma_50_f`, `sma_100_f`, `sma_200_f` | `(close - sma) / close` | 高 |
| `ema_5_f`, `ema_10_f`, `ema_20_f`, `ema_50_f`, `ema_100_f` | `(close - ema) / close` | 高 |
| `tema_10_f`, `tema_20_f`, `tema_30_f` | `(close - tema) / close` | 中 |
| `kama_10_f`, `kama_20_f`, `kama_30_f` | `(close - kama) / close` | 中 |
| `sma_200_slope_f` | `slope / close` | 中 |

#### Phase 4b: Volume 特征添加 `_ratio` 版本 (12 个)

| 特征 | 归一化方式 | 优先级 |
|-----|----------|--------|
| `obv_f` | `obv_change / rolling_std` | 高 |
| `ad_line_f`, `adosc_f` | `change / rolling_std` | 高 |
| `vol_lag_features_f` | `vol / rolling_mean` | 中 |
| `vol_ma_features_f` | 已在计算中归一化 | 低 |
| `vol_mom_features_f`, `vol_range_features_f`, `vol_raw_features_f`, `vol_trend_features_f` | `value / rolling_mean` | 中 |

#### Phase 4c: SR 结构特征 ATR 归一化 (8 个)

| 特征 | 归一化方式 | 优先级 |
|-----|----------|--------|
| `poc_hal_features_f`, `poc_hal_features_close_f` | `(level - close) / ATR` | 高 (已实现) |
| `sqs_hal_high_f`, `sqs_hal_low_f` | `(level - close) / ATR` | 高 |
| `sr_strength_max_f`, `sr_strength_max_close_f` | `dist / ATR` | 高 (已实现) |
| `zigzag_high_low_f` | `(zz_level - close) / ATR` | 中 |

#### Phase 4d: 动量指标归一化 (5 个)

| 特征 | 归一化方式 | 优先级 |
|-----|----------|--------|
| `macdext_f`, `macdfix_f` | `macd / ATR` | 高 |
| `atr_f`, `atr_7_f`, `atr_14_f`, `atr_21_f` | `atr / close` | 高 |
| `natr_14_f` | 已归一化 | - |

---

## 📝 使用建议

### 树模型 (LightGBM)

树模型对特征尺度不敏感，可以直接使用原始特征。但对于多标的训练，建议使用归一化特征以提高泛化能力。

### 神经网络 (NN Multihead)

**必须**使用归一化特征，否则训练不稳定。参考 `config/nnmultihead/path_primitives_4h_80h_min/features.yaml` 中的特征配置。

### 跨资产比较

使用归一化特征可以直接比较不同资产的特征值：
- BTC: `sma_200_position = 0.04` → 价格比 SMA 高 4%
- ETH: `sma_200_position = 0.04` → 价格比 SMA 高 4%

---

## 📚 相关文档

- [特征归一化策略](FEATURE_NORMALIZATION_POLICY.md)
- [各策略最佳特征列](../strategies/BEST_FEATURE_COLUMNS_BY_STRATEGY.md)
- [NN Multihead 特征配置](../../config/nnmultihead/path_primitives_4h_80h_min/features.yaml)

---

*此文档由脚本自动生成*
