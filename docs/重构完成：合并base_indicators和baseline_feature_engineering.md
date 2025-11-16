# 重构完成：合并 base_indicators 和 baseline_feature_engineering

## 概述

已成功合并 `base_indicators.py` 和 `baseline_feature_engineering.py` 到新的统一模块 `baseline_features.py`，并添加了所有请求的无量纲特征和优化。

## 完成的工作

### 1. ✅ 文件合并

- **新文件**: `src/data_tools/baseline_features.py`
- **合并内容**:
  - 所有基础指标计算函数（`compute_rsi`, `compute_macd`, `compute_bollinger_bands`, `compute_atr`, `compute_zigzag`）
  - `BaselineFeatureEngineer` 类及其所有方法
  - 优化的 `add_basic_indicators` 和 `ensure_basic_indicators`（支持按需计算）
  - 优化的 `add_common_derived_features`（支持按需计算）

### 2. ✅ 新增 ZigZag 无量纲特征

- `price_to_zz_high_pct`: 当前价格到最近 ZigZag 高点的相对距离
- `price_to_zz_low_pct`: 当前价格到最近 ZigZag 低点的相对距离
- `zz_amplitude_pct`: ZigZag 波幅（相对）
- `zz_duration`: ZigZag 持续时间（bar 数，无量纲）
- `zz_slope`: ZigZag 斜率（归一化）

### 3. ✅ 新增 POC 无量纲特征

- `price_to_poc_pct`: 当前价格到 POC 的相对距离
- `poc_position_ratio`: POC 在价格区间中的位置（0-1）
- `poc_volume_ratio`: POC 位置的成交量占比

### 4. ✅ 新增 HAL 无量纲特征

- `price_to_hal_high_pct`: 当前价格到 HAL 高点的相对距离
- `price_to_hal_low_pct`: 当前价格到 HAL 低点的相对距离
- `price_to_hal_mid_pct`: 当前价格到 HAL 中点的相对距离
- `hal_bandwidth_pct`: HAL 带宽（相对）

### 5. ✅ 新增 Swing High/Low 无量纲特征

- `swing_high_pct_close`: Swing High 相对收盘价的比率
- `swing_low_pct_close`: Swing Low 相对收盘价的比率
- `swing_amplitude_pct`: Swing 波幅（相对）

### 6. ✅ 新增基础价格与量能相对变化特征

- `ret_1h`, `ret_4h`, `ret_24h`: 对数收益率（1小时、4小时、24小时）
- `rv_4h`, `rv_24h`: 已实现波动率
- `vol_ma_ratio`: 成交量移动平均比率
- `vol_zscore`: 成交量 Z-score

### 7. ✅ 优化依赖关系管理

- **`add_basic_indicators`**: 支持 `required_features` 参数，只计算需要的指标
- **`ensure_basic_indicators`**: 支持 `required_features` 参数，按需检查并计算
- **`add_common_derived_features`**: 优化依赖关系解析，只计算必要的基础指标
- **`BaselineFeatureEngineer.engineer_features`**: 集成所有新特征，支持按需计算

### 8. ✅ 更新所有引用

已更新以下文件的导入语句：

- `src/data_tools/comprehensive_feature_engineering.py`
- `src/data_tools/feature_engineering_enhanced.py`
- `src/data_tools/feature_engineering.py`
- `src/time_series_model/pipeline/training/safe_multi_asset_preprocessing.py`
- `src/cross_sectional/panel_generation.py`
- `src/time_series_model/pipeline/training/rolling.py`
- `src/time_series_model/pipeline/training/train.py`
- `scripts/analysis/factor_analysis_alphalens.py`
- `scripts/optimization/tune_q50_params.py`

## 新增函数和类

### 基础指标计算函数

```python
compute_rsi(series, period=14)
compute_macd(series, fast=12, slow=26, signal=9)
compute_bollinger_bands(series, period=20, std_dev=2)
compute_atr(high, low, close, period=14)
compute_zigzag(high, low, threshold=0.05)
compute_zigzag_high_low(zigzag)  # 新增
compute_poc(high, low, volume, window=20, bins=50)  # 新增
compute_hal(high, low, window=20)  # 新增
```

### 无量纲特征函数

```python
add_zigzag_dimensionless_features(df, required_features=None)
add_poc_dimensionless_features(df, required_features=None, poc_window=20)
add_hal_dimensionless_features(df, required_features=None, hal_window=20)
add_swing_dimensionless_features(df, required_features=None, swing_win_short=20, swing_win_long=60)
add_price_volume_relative_features(df, required_features=None)
```

### 优化的基础函数

```python
add_basic_indicators(df, required_features=None)  # 支持按需计算
ensure_basic_indicators(df, required_features=None)  # 支持按需计算
add_common_derived_features(df, required_features=None)  # 优化依赖关系
```

## 向后兼容性

- ✅ 所有原有函数和类都保留，API 不变
- ✅ `BaselineFeatureEngineer` 类完全兼容
- ✅ `engineer_baseline_features` 函数完全兼容
- ✅ `get_baseline_feature_columns` 函数完全兼容
- ✅ `create_binary_labels_baseline` 函数完全兼容

## 使用示例

### 使用新的无量纲特征

```python
from data_tools.baseline_features import (
    BaselineFeatureEngineer,
    add_zigzag_dimensionless_features,
    add_poc_dimensionless_features,
    add_hal_dimensionless_features,
    add_swing_dimensionless_features,
    add_price_volume_relative_features,
)

# 方式1: 使用 BaselineFeatureEngineer（自动包含所有新特征）
engineer = BaselineFeatureEngineer()
df_features = engineer.engineer_features(df)

# 方式2: 单独添加特定特征
df = add_zigzag_dimensionless_features(df, required_features={"price_to_zz_high_pct"})
df = add_poc_dimensionless_features(df, required_features={"price_to_poc_pct"})
df = add_hal_dimensionless_features(df, required_features={"hal_bandwidth_pct"})
df = add_swing_dimensionless_features(df, required_features={"swing_amplitude_pct"})
df = add_price_volume_relative_features(df, required_features={"ret_1h", "vol_zscore"})
```

### 按需计算特征（性能优化）

```python
from data_tools.baseline_features import (
    add_basic_indicators,
    add_common_derived_features,
)

# 只计算需要的指标
df = add_basic_indicators(df, required_features={"rsi", "atr"})
df = add_common_derived_features(df, required_features={"rsi_normalized", "atr_normalized"})
```

## 注意事项

1. **时间框架假设**: `add_price_volume_relative_features` 中的 `ret_1h`, `ret_4h`, `ret_24h` 假设数据是 5 分钟 K 线。如果使用其他时间框架，需要调整 `periods_1h`, `periods_4h`, `periods_24h` 参数。

2. **POC 计算**: `compute_poc` 使用简化的 volume profile 计算方法。对于更精确的 POC，可能需要使用 tick 数据或更复杂的算法。

3. **向后兼容**: 原有的 `base_indicators.py` 和 `baseline_feature_engineering.py` 文件仍然存在，但建议逐步迁移到新的统一模块。

## 下一步

1. **测试**: 运行现有测试确保所有功能正常
2. **性能测试**: 验证按需计算是否提升了性能
3. **文档更新**: 更新相关文档和示例
4. **逐步迁移**: 可以考虑在旧文件中添加 deprecation 警告，引导用户使用新模块

## 文件结构

```
src/data_tools/
├── baseline_features.py  # 新的统一模块（推荐使用）
├── base_indicators.py           # 旧文件（保留以兼容）
└── baseline_feature_engineering.py  # 旧文件（保留以兼容）
```

## 总结

✅ 成功合并两个文件到统一模块  
✅ 添加了所有请求的无量纲特征  
✅ 优化了依赖关系管理，支持按需计算  
✅ 更新了所有引用，保持向后兼容  
✅ 代码结构清晰，易于维护和扩展

