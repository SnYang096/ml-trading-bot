# 特征测试覆盖分析报告

**生成时间**: 2025-01-01  
**分析范围**: `config/feature_dependencies.yaml` 中的所有特征节点

---

## 📊 总体情况

- **总特征节点数**: 205
- **有测试覆盖的节点数**: 74
- **无测试覆盖的节点数**: 131
- **总体测试覆盖率**: 36.1%

---

## 📋 按类别详细分析

### ✅ 完整覆盖（覆盖率 ≥ 80%）

#### 1. hilbert (2/2, 100.0%)
- **测试文件**: `test_hilbert_features_improved.py`
- **覆盖节点**: 
  - `hilbert_price_f`
  - `hilbert_cvd_f`

#### 2. market_cap (1/1, 100.0%)
- **测试文件**: `test_market_cap_features.py`
- **覆盖节点**: 
  - `market_cap_normalized_orderflow_f`

#### 3. pattern (5/5, 100.0%)
- **测试文件**: `test_advanced_features.py`, `test_dtw_narrow_entrypoint.py`
- **覆盖节点**: 
  - `dtw_features_f`
  - `dtw_features_reversal_f`
  - `dtw_features_breakout_f`
  - `dtw_features_compression_f`
  - `dtw_features_trend_f`

#### 4. risk_management (1/1, 100.0%)
- **测试文件**: `test_advanced_features.py`, `test_garch_evt_features.py`
- **覆盖节点**: 
  - `evt_features_f`

#### 5. spectrum (5/5, 100.0%)
- **测试文件**: `test_spectrum_features.py`
- **覆盖节点**: 
  - `spectrum_price_f`
  - `spectrum_volume_f`
  - `spectrum_cvd_f`
  - `spectrum_price_volume_f`
  - `spectrum_all_f`

#### 6. unknown (3/3, 100.0%)
- **测试文件**: `test_hurst_features_improved.py`
- **覆盖节点**: 
  - `hurst_price_f`
  - `hurst_cvd_f`
  - `hurst_volume_f`

#### 7. wpt (4/4, 100.0%)
- **测试文件**: `test_wpt_volatility_features.py`, `test_wpt_future_leak_and_multi_asset.py`
- **覆盖节点**: 
  - `wpt_price_f`
  - `wpt_volume_f`
  - `wpt_cvd_f`
  - `wpt_all_f`

---

### ⚠️ 部分覆盖（覆盖率 50-80%）

#### 8. compression (2/3, 66.7%)
- **测试文件**: 部分在 `test_baseline_remaining_narrow.py`
- **覆盖节点**: 
  - `compression_energy_f` ✅
  - `compression_to_breakout_prob_f` ✅
- **缺失节点**: 
  - `compression_duration_f` ❌

#### 9. derived (7/12, 58.3%)
- **测试文件**: 部分在 `test_baseline_remaining_narrow.py`, `test_volume_profile_volatility_features.py`
- **覆盖节点**: 
  - `atr_ratio_f` ✅
  - `bb_width_ratio_f` ✅
  - `compression_score_f` ✅
  - `tbr_ma_5_f` ✅
  - `tbr_spike_f` ✅
  - 其他 2 个 ✅
- **缺失节点**: 
  - `sr_strength_combined_f` ❌
  - `sr_distance_normalized_f` ❌
  - `dist_to_zz_high_f` ❌
  - `dist_to_zz_low_f` ❌
  - `cvd_slope_5_f` ❌
  - 其他 4 个 ❌

#### 10. liquidity (3/4, 75.0%)
- **测试文件**: `test_liquidity_features.py`
- **覆盖节点**: 
  - `liquidity_void_f` ✅
  - `wpt_volume_energy_f` ✅
  - `wpt_scene_semantic_scores_f` ✅
- **缺失节点**: 
  - `volume_profile_vpvr_f` ❌

#### 11. technical_indicator (3/4, 75.0%)
- **测试文件**: `test_baseline_remaining_narrow.py`, `test_momentum_features.py`
- **覆盖节点**: 
  - `atr_f` ✅
  - `rsi_f` ⚠️ (通过 `add_common_derived_features` 间接测试)
  - `acceleration_3_f` ✅
- **缺失节点**: 
  - `macd_f` ❌

---

### ❌ 低覆盖率（覆盖率 < 50%）

#### 12. interaction (11/34, 32.4%)
- **测试文件**: `test_interaction_features.py`
- **覆盖节点**: 11 个（主要是基础交互特征）
- **缺失节点**: 23 个
  - `funding_scene_semantic_scores_f` ❌
  - `fp_imbalance_exhaustion_f` ❌
  - `exhaustion_at_liquidity_void_f` ❌
  - `vpin_semantic_scores_f` ❌
  - `tbr_imbalance_semantic_scores_f` ❌
  - 其他 18 个 ❌

#### 13. momentum (1/20, 5.0%)
- **测试文件**: `test_momentum_features.py` (只测试了 `add_common_derived_features`)
- **覆盖节点**: 
  - `momentum_5_f`, `momentum_10_f`, `momentum_20_f` (通过 `add_common_derived_features`) ⚠️
- **缺失节点**: 19 个
  - `rsi_7_f`, `rsi_14_f`, `rsi_21_f` (使用 `compute_talib_indicator_from_series`) ❌
  - `mom_5_f`, `mom_10_f`, `mom_14_f` (使用 `compute_talib_indicator_from_series`) ❌
  - `roc_5_f` ✅ (在 `test_baseline_remaining_narrow.py` 中)
  - `cci_14_f`, `stochf_f`, `willr_14_f` 等 ❌

#### 14. order_flow (17/40, 42.5%)
- **测试文件**: `test_vpin_features.py`, `test_vpin_future_leak_and_multi_asset.py`, `test_interaction_features.py`
- **覆盖节点**: 17 个（主要是 VPIN 和 TradeCluster 相关）
- **缺失节点**: 23 个
  - `funding_rate_features_f` ❌ (使用 `compute_funding_rate_features_from_df`)
  - `order_flow_all_features_f` ❌ (使用 `select_columns_from_series`)
  - `vpin_derived_features_f` ❌ (使用 `select_columns_from_series`)
  - `vpin_block_features_f` ❌ (使用 `select_columns_from_series`)
  - `trade_cluster_block_features_f` ❌ (使用 `select_columns_from_series`)
  - 其他 18 个 ❌

#### 15. trend (4/32, 12.5%)
- **测试文件**: `test_trend_features.py`
- **覆盖节点**: 
  - `trend_r2_20_f` ✅
  - `trend_r2_50_f` ✅
  - `slope_consistency_score_f` ✅
  - `trend_volatility_alignment_f` ✅
- **缺失节点**: 28 个
  - `sma_5_f`, `sma_10_f`, `sma_20_f`, `sma_50_f`, `sma_100_f` (使用 `compute_talib_indicator_from_series`) ❌
  - `ema_5_f`, `ema_10_f`, `ema_20_f`, `ema_50_f` (使用 `compute_talib_indicator_from_series`) ❌
  - `wma_20_f`, `tema_20_f`, `dema_20_f` 等 ❌
  - 其他趋势相关特征 ❌

#### 16. volatility (4/21, 19.0%)
- **测试文件**: `test_volume_profile_volatility_features.py`, `test_volume_profile_volatility_future_leak_and_multi_asset.py`
- **覆盖节点**: 
  - `volume_profile_volatility_f` ✅
  - `atr_percentile_f` ✅
  - `volatility_reversal_score_f` ✅
  - 其他 1 个 ✅
- **缺失节点**: 17 个
  - `bb_width_f` ❌ (使用 `compute_bb_width_features_from_series`)
  - `range_ratio_5bar_f` ❌ (使用 `compute_range_ratio_5bar_from_series`)
  - `extended_volatility_features_f` ❌ (使用 `select_columns_from_series`)
  - `vol_raw_features_f` ❌ (使用 `compute_vol_raw_features_from_series`)
  - `vol_atr_features_f` ❌ (使用 `compute_vol_atr_features_from_series`)
  - 其他 12 个 ❌

#### 17. price_structure (0/2, 0.0%)
- **缺失节点**: 
  - `price_range_symmetry_f` ❌ (使用 `compute_price_range_symmetry_from_series`)
  - `wick_ratios_f` ❌ (使用 `compute_wick_ratios_from_series`)

#### 18. sr_structure (0/8, 0.0%)
- **缺失节点**: 
  - `poc_hal_features_f` ❌ (使用 `compute_poc_hal_features_from_series`)
  - `poc_hal_features_close_f` ❌ (使用 `compute_poc_hal_features_from_series`)
  - `sqs_hal_high_f` ❌ (使用 `compute_sqs_hal_high_from_series`)
  - `sqs_hal_low_f` ❌ (使用 `compute_sqs_hal_low_from_series`)
  - `sqs_f` ❌ (使用 `compute_sqs_combined_from_series`)
  - `sr_strength_max_f` ❌ (使用 `compute_sr_strength_max_from_series`)
  - `sr_strength_max_close_f` ❌ (使用 `compute_sr_strength_max_from_series`)
  - `zigzag_high_low_f` ❌ (使用 `compute_zigzag_high_low_from_series`)

#### 19. volume (0/3, 0.0%)
- **缺失节点**: 
  - `obv_f` ❌ (使用 `compute_talib_indicator_from_series`)
  - `ad_line_f` ❌ (使用 `compute_talib_indicator_from_series`)
  - `adosc_f` ❌ (使用 `compute_talib_indicator_from_series`)

#### 20. deep_learning (0/1, 0.0%)
- **缺失节点**: 
  - `dl_sequence_features_f` ❌

---

## 🔍 关键发现

### 1. 使用 `compute_talib_indicator_from_series` 的特征

**数量**: 约 50+ 个节点  
**类别**: 主要是 `trend`, `momentum`, `volume`, `technical_indicator`  
**问题**: 这些特征都使用同一个通用函数 `compute_talib_indicator_from_series`，但该函数本身可能没有专门的测试。

**建议**: 
- 为 `compute_talib_indicator_from_series` 创建通用测试
- 或者为常用的 TA-Lib 指标（如 SMA, EMA, RSI, MACD）创建专门的测试

### 2. 使用 `select_columns_from_series` 的特征

**数量**: 约 20+ 个节点  
**类别**: 主要是 `order_flow`, `volatility`  
**问题**: 这些是"选择器"特征，只是从其他特征中选择列，本身不进行计算。

**建议**: 
- 这些特征可能不需要单独测试（因为它们只是选择列）
- 但需要确保被选择的源特征有测试覆盖

### 3. 缺失测试的主要类别

1. **trend** (28 个节点无测试): 主要是 SMA, EMA, WMA 等移动平均线
2. **interaction** (23 个节点无测试): 主要是各种 semantic scores
3. **order_flow** (23 个节点无测试): 主要是 funding_rate, select_columns 特征
4. **momentum** (19 个节点无测试): 主要是各种 TA-Lib 动量指标
5. **volatility** (17 个节点无测试): 主要是 BB, range_ratio, vol_raw 等
6. **sr_structure** (8 个节点无测试): POC, HAL, SQS, SR strength 等

---

## 📝 测试建议

### 高优先级（需要立即补充）

1. **TA-Lib 指标通用测试**
   - 创建 `test_talib_indicators.py`
   - 测试 `compute_talib_indicator_from_series` 函数
   - 覆盖常用的指标：SMA, EMA, RSI, MACD, OBV, AD 等

2. **SR Structure 特征测试**
   - 创建 `test_sr_structure_features.py`
   - 测试 POC, HAL, SQS, SR strength 等特征

3. **Price Structure 特征测试**
   - 创建 `test_price_structure_features.py`
   - 测试 `price_range_symmetry`, `wick_ratios` 等特征

4. **Volume 特征测试**
   - 创建 `test_volume_features.py`
   - 测试 OBV, AD Line, ADOSC 等特征

### 中优先级（可以逐步补充）

1. **Interaction Semantic Scores 测试**
   - 扩展 `test_interaction_features.py`
   - 测试各种 semantic scores（funding, vpin, tbr 等）

2. **Volatility 扩展测试**
   - 扩展 `test_volatility_features.py`
   - 测试 BB, range_ratio, vol_raw, vol_atr 等特征

3. **Order Flow 扩展测试**
   - 扩展 `test_order_flow_features.py`
   - 测试 funding_rate, select_columns 特征

### 低优先级（可选）

1. **Deep Learning 特征测试**
   - 创建 `test_dl_sequence_features.py`
   - 测试 `dl_sequence_features_f`

2. **Compression 扩展测试**
   - 扩展现有测试
   - 测试 `compression_duration_f`

---

## 📊 测试覆盖统计表

| 类别 | 总节点数 | 有测试 | 无测试 | 覆盖率 | 状态 |
|------|---------|--------|--------|--------|------|
| hilbert | 2 | 2 | 0 | 100.0% | ✅ |
| market_cap | 1 | 1 | 0 | 100.0% | ✅ |
| pattern | 5 | 5 | 0 | 100.0% | ✅ |
| risk_management | 1 | 1 | 0 | 100.0% | ✅ |
| spectrum | 5 | 5 | 0 | 100.0% | ✅ |
| unknown | 3 | 3 | 0 | 100.0% | ✅ |
| wpt | 4 | 4 | 0 | 100.0% | ✅ |
| liquidity | 4 | 3 | 1 | 75.0% | ⚠️ |
| technical_indicator | 4 | 3 | 1 | 75.0% | ⚠️ |
| compression | 3 | 2 | 1 | 66.7% | ⚠️ |
| derived | 12 | 7 | 5 | 58.3% | ⚠️ |
| interaction | 34 | 11 | 23 | 32.4% | ❌ |
| order_flow | 40 | 17 | 23 | 42.5% | ❌ |
| volatility | 21 | 4 | 17 | 19.0% | ❌ |
| trend | 32 | 4 | 28 | 12.5% | ❌ |
| momentum | 20 | 1 | 19 | 5.0% | ❌ |
| price_structure | 2 | 0 | 2 | 0.0% | ❌ |
| sr_structure | 8 | 0 | 8 | 0.0% | ❌ |
| volume | 3 | 0 | 3 | 0.0% | ❌ |
| deep_learning | 1 | 0 | 1 | 0.0% | ❌ |

---

## 🎯 下一步行动

1. **立即补充**（高优先级）:
   - 创建 `test_talib_indicators.py` 测试 TA-Lib 指标
   - 创建 `test_sr_structure_features.py` 测试 SR 结构特征
   - 创建 `test_price_structure_features.py` 测试价格结构特征
   - 创建 `test_volume_features.py` 测试成交量特征

2. **逐步补充**（中优先级）:
   - 扩展 `test_interaction_features.py` 测试更多 semantic scores
   - 扩展 `test_volatility_features.py` 测试更多波动率特征
   - 扩展 `test_order_flow_features.py` 测试 funding_rate 等

3. **可选补充**（低优先级）:
   - 创建 `test_dl_sequence_features.py`
   - 补充 compression_duration 测试

---

## 📌 注意事项

1. **`compute_talib_indicator_from_series`**: 这是一个通用函数，用于计算各种 TA-Lib 指标。建议创建一个通用测试文件，测试常用的指标（SMA, EMA, RSI, MACD 等）。

2. **`select_columns_from_series`**: 这是一个"选择器"函数，只是从其他特征中选择列。这些特征可能不需要单独测试，但需要确保被选择的源特征有测试覆盖。

3. **Semantic Scores**: 很多 interaction 特征都是 semantic scores，它们可能共享相似的计算逻辑。可以考虑创建通用的 semantic scores 测试。

4. **测试粒度**: 当前测试主要关注核心计算函数（如 `extract_*`, `compute_*`），而不是每个特征节点。这是合理的，因为很多节点共享同一个 compute_func。

---

**最后更新**: 2025-01-01

