# 重构计划：合并 base_indicators 和 baseline_feature_engineering

## 目标

1. 合并 `base_indicators.py` 和 `baseline_feature_engineering.py` 到一个统一的模块
2. 增加 ZigZag 相关的无量纲指标
3. POC、HAL、Swing High/Low 转为无量纲比率
4. 添加基础价格与量能特征，转为相对变化

## 新增无量纲特征

### 1. ZigZag 无量纲特征

```python
# ZigZag 高点和低点（需要先计算 zigzag）
zz_high = zigzag 序列中的高点
zz_low = zigzag 序列中的低点

# 当前价格距离最近 ZigZag 高/低点的相对距离
price_to_zz_high_pct = (zz_high - close) / close
price_to_zz_low_pct = (close - zz_low) / close

# ZigZag 波幅（相对）
zz_amplitude_pct = (zz_high - zz_low) / zz_low

# ZigZag 持续时间（bar 数，无量纲）
zz_duration = 从上一个转折点至今的 bar 数

# ZigZag 斜率（归一化）
zz_slope = (zigzag.diff(5) / 5) / atr
```

### 2. POC 无量纲特征

```python
# POC (Point of Control) - 需要先计算 volume profile
poc = 最大成交量对应的价格

# 当前价格到 POC 的相对距离
price_to_poc_pct = (poc - close) / close

# POC 相对位置（在价格区间中的位置）
poc_position_ratio = (poc - low) / (high - low)

# POC 成交量占比
poc_volume_ratio = poc_volume / total_volume
```

### 3. HAL 无量纲特征

```python
# HAL (High Average Low) - 需要先计算
hal_high = high 的移动平均
hal_low = low 的移动平均
hal_mid = (hal_high + hal_low) / 2

# 当前价格到 HAL 的相对距离
price_to_hal_high_pct = (hal_high - close) / close
price_to_hal_low_pct = (close - hal_low) / close
price_to_hal_mid_pct = (hal_mid - close) / close

# HAL 带宽（相对）
hal_bandwidth_pct = (hal_high - hal_low) / hal_mid
```

### 4. Swing High/Low 无量纲特征

```python
# Swing High/Low（已有 roll_high_s, roll_low_s 等）
# 转为相对比率
swing_high_pct_close = (roll_high_s - close) / close
swing_low_pct_close = (close - roll_low_s) / close

# Swing 波幅（相对）
swing_amplitude_pct = (roll_high_s - roll_low_s) / roll_low_s
```

### 5. 基础价格与量能特征（相对变化）

```python
# 对数收益率（常用）
ret_1h = log(close / close.shift(1))
ret_4h = log(close / close.shift(4))
ret_24h = log(close / close.shift(24))

# 波动率（已实现波动）
rv_4h = ret_1h.rolling(4).std()
rv_24h = ret_1h.rolling(24).std()

# 成交量异常度
vol_ma_ratio = volume / volume.rolling(24).mean()
vol_zscore = (volume - volume.rolling(24).mean()) / volume.rolling(24).std()
```

## 实施步骤

1. **创建新文件** `baseline_features.py`
2. **合并现有功能**：
   - 从 `base_indicators.py` 导入所有基础指标计算函数
   - 从 `baseline_feature_engineering.py` 导入所有 baseline 特征
3. **添加新特征**：
   - ZigZag 无量纲特征
   - POC/HAL/Swing 无量纲特征
   - 基础价格与量能相对变化特征
4. **更新引用**：
   - 更新 `comprehensive_feature_engineering.py` 中的引用
   - 更新其他模块的引用

## 注意事项

1. **保持向后兼容**：确保现有代码仍能工作
2. **依赖关系**：新特征需要先计算基础指标（如 zigzag, atr 等）
3. **性能优化**：使用 `required_features` 参数只计算需要的特征
4. **测试**：确保所有特征计算正确

