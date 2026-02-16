# Compression Breakout 特征配置历史

## 📋 文件对比

### 1. `features_suggested_20260128.yaml` ⭐ **最新版本（推荐使用）**

**日期**: 2026-01-28  
**状态**: ✅ 当前使用版本

**特征列表** (5个):
- `compression_duration_f` (基础)
- `atr_f` (基础)
- `volume_ratio_f`
- `liquidity_void_f`
- `trend_r2_20_f`

**关键信息**:
- **Objective**: `CV_mean` (交叉验证均值)
- **Selected Groups**: 
  - `kline_core__volume_ratio_f`
  - `poolb__liquidity_void_f`
  - `poolb__trend_r2_20_f`
- **来源**: `20260108_best_abc_A` (Stage A 最佳结果)
- **Pool B**: `20260108_best_abc`

**特点**:
- ✅ 基于 feature-group-search 最佳结果
- ✅ 特征精简（5个），避免过拟合
- ✅ 使用 CV_mean 目标，更稳健
- ✅ 包含完整的 feature_group_search 元数据

---

### 2. `features_suggested_pipeline_poolb_semantic_20260108_best_abc_A.yaml.bak.20260111_111728`

**日期**: 2026-01-11 (备份)  
**状态**: ⚠️ 历史版本

**特征列表** (5个):
- `compression_duration_f` (基础)
- `atr_f` (基础)
- `liquidity_void_f`
- `volume_ratio_f`
- `rsi_f` ⚠️ **不同点**

**关键信息**:
- **Objective**: `CV_mean`
- **Selected Groups**: 
  - `poolb__liquidity_void_f`
  - `kline_core__volume_ratio_f`
  - `kline_core__rsi_f` ⚠️ **包含 RSI**
- **来源**: `20260108_best_abc_A`

**与最新版本的区别**:
- ❌ 包含 `rsi_f`，最新版本已移除
- ✅ 最新版本用 `trend_r2_20_f` 替代了 `rsi_f`

**为什么被替换**:
- `trend_r2_20_f` 在 feature-group-search 中表现更好
- `rsi_f` 可能与其他特征有冗余

---

### 3. `features_suggested_greedy_poolb_semantic_20260103_norm_full.yaml.bak.20260103_203211`

**日期**: 2026-01-03 (备份)  
**状态**: ⚠️ 早期实验版本

**特征列表** (3个):
- `compression_duration_f` (基础)
- `atr_f` (基础)
- `market_cap_normalized_orderflow_f` ⚠️ **不同点**

**关键信息**:
- **Objective**: `Sharpe_mean` (不是 CV_mean)
- **Selected Groups**: 
  - `market_cap_norm` ⚠️ **只有1个组**
- **来源**: `20260103_norm_full`
- **Stop Reason**: `no_improvement` (没有改进)

**特点**:
- ⚠️ 特征最少（只有3个）
- ⚠️ 使用 `market_cap_normalized_orderflow_f`，后续版本已移除
- ⚠️ 搜索提前停止（no_improvement）
- ⚠️ 使用 Sharpe_mean，可能过拟合

**为什么被替换**:
- 特征太少，可能欠拟合
- `market_cap_normalized_orderflow_f` 在后续搜索中表现不佳
- 后续版本改用 CV_mean 更稳健

---

### 4. `features_suggested.yaml.bak.20251229_095547`

**日期**: 2025-12-29 (备份)  
**状态**: ❌ 最早期版本（已废弃）

**特征列表** (17个):
- `atr_f`
- `wpt_price_reconstructed_f`
- `wpt_price_fluctuation_f`
- `wpt_volume_energy_f`
- `wpt_cvd_fluctuation_f`
- `spectrum_features_compression_breakout_f`
- `liquidity_void_f`
- `hilbert_phase_f`
- `hurst_price_f`
- `hurst_cvd_f`
- `vpin_features_f`
- `footprint_basic_f`
- `liquidity_void_x_wpt_risk_f`
- `compression_energy_x_ofi_short_f`
- `vpin_x_compression_f`
- `vpin_zscore_x_trade_cluster_max_buy_run_f`
- `vpin_x_trade_cluster_entropy_f`
- `dtw_features_compression_f`

**关键信息**:
- **Objective**: `Sharpe_mean`
- **Selected Groups**: `[]` (空！)
- **Stop Reason**: `no_valid_candidates` (没有有效候选)

**特点**:
- ❌ 特征太多（17个），容易过拟合
- ❌ Feature Group Search 没有找到有效组合
- ❌ 可能是手动配置，没有经过系统搜索
- ❌ 包含很多交互特征（`_x_`），可能冗余

**为什么被替换**:
- 特征过多，过拟合风险高
- 没有经过 feature-group-search 验证
- 后续版本通过系统搜索找到了更优的5特征组合

---

## 📊 演进路径

```
2025-12-29: 17个特征 (手动配置，未搜索)
    ↓
2026-01-03: 3个特征 (Greedy搜索，Sharpe_mean，提前停止)
    ↓
2026-01-11: 5个特征 (Pool B搜索，CV_mean，包含rsi_f)
    ↓
2026-01-28: 5个特征 (Pool B搜索，CV_mean，rsi_f → trend_r2_20_f) ⭐ 当前版本
```

## 🎯 结论

### ✅ **只看最新的就可以了**

**推荐使用**: `features_suggested_20260128.yaml`

**理由**:
1. ✅ **最新版本** - 基于最新的 feature-group-search 结果
2. ✅ **最优特征组合** - 从 17 个特征精简到 5 个，避免过拟合
3. ✅ **更稳健的目标** - 使用 `CV_mean` 而非 `Sharpe_mean`
4. ✅ **完整元数据** - 包含完整的搜索历史，可追溯
5. ✅ **已验证** - 基于 `20260108_best_abc_A` Stage A 最佳结果

### 📝 其他文件的作用

- **`.bak` 文件**: 历史备份，用于追溯演进过程
- **可以删除**: 如果不需要历史记录，可以删除所有 `.bak` 文件
- **保留建议**: 建议保留最新一个备份（20260111），以防需要回退

### 🔍 关键改进点

1. **特征数量**: 17 → 5 (减少 70%)
2. **目标函数**: Sharpe_mean → CV_mean (更稳健)
3. **特征替换**: rsi_f → trend_r2_20_f (更好的表现)
4. **搜索质量**: no_valid_candidates → completed (成功完成搜索)

---

**最后更新**: 2026-01-28
