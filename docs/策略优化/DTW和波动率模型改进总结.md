# DTW和波动率模型改进总结

## 一、DTW特征改进

### 1.1 修改内容
- **仅在SR附近计算DTW**：设置`compute_only_near_sr=True`，仅在`dist_to_nearest_sr < 1.5 * ATR`时计算
- **提高效率**：大幅减少DTW计算量，只在关键区域（SR附近）计算
- **作为辅助特征**：DTW特征仅用于SR Reversal策略，不用于波动率模型

### 1.2 配置变更
```yaml
dtw_features:
  compute_params:
    window: 20
    compute_only_near_sr: true
    sr_dist_col: "dist_to_nearest_sr"
    sr_threshold: 1.5  # 仅在距离SR < 1.5 * ATR时计算
```

### 1.3 实现细节
- DTW函数已更新，支持使用ATR归一化SR距离
- 如果`dist_to_nearest_sr`列不存在，会fallback到全量计算

## 二、GARCH特征改进

### 2.1 修改内容
- **增加窗口大小**：从60增加到120，减少过拟合
- **保持其他参数不变**：GJR-GARCH仍然启用，用于捕捉杠杆效应

### 2.2 配置变更
```yaml
garch_features:
  compute_params:
    window: 120  # 从60增加到120
    garch_p: 1
    garch_q: 1
    use_gjr: true
    use_figarch: false
```

### 2.3 预期效果
- 更大的窗口提供更多历史信息，减少过拟合
- 训练集和测试集的相关性差距应该缩小

## 三、扩展波动率特征

### 3.1 新增特征类别

#### 3.1.1 历史波动率特征
- `vol_historical_5`, `vol_historical_10`, `vol_historical_20`, `vol_historical_60`
- 不同窗口的滚动标准差（returns的std）

#### 3.1.2 ATR相关特征
- `vol_atr_price_ratio`: ATR/Price比率
- `vol_atr_ma_*`, `vol_atr_std_*`, `vol_atr_max_*`, `vol_atr_min_*`: ATR的统计特征
- `vol_atr_ratio_ma20`: 当前ATR相对于20期均值的比率
- `vol_atr_change`, `vol_atr_change_abs`: ATR的变化率

#### 3.1.3 滞后特征
- `vol_lag_1`, `vol_lag_2`, `vol_lag_3`, `vol_lag_5`, `vol_lag_10`
- 波动率的滞后值，捕捉时间序列依赖

#### 3.1.4 趋势特征
- `vol_trend_slope_5`, `vol_trend_slope_10`, `vol_trend_slope_20`: 波动率的线性趋势斜率
- `vol_acceleration`: 波动率的加速度（二阶导数）

#### 3.1.5 移动平均特征
- `vol_ma_5`, `vol_ma_10`, `vol_ma_20`: 波动率的简单移动平均
- `vol_ema_5`, `vol_ema_10`, `vol_ema_20`: 波动率的指数移动平均

#### 3.1.6 Regime特征
- `vol_zscore`: 波动率的Z-score（标准化）
- `vol_percentile_rank`: 波动率的百分位排名

#### 3.1.7 范围特征
- `vol_range_10`, `vol_range_20`: 波动率的范围
- `vol_range_ratio_10`, `vol_range_ratio_20`: 波动率的范围比率

#### 3.1.8 动量特征
- `vol_momentum_3`, `vol_momentum_5`, `vol_momentum_10`: 波动率的变化率

### 3.2 配置
```yaml
extended_volatility_features:
  module: enhanced
  compute_func: extract_extended_volatility_features
  dependencies: ["atr"]
  required_columns: ["close", "atr"]
  compute_params:
    window: 20
    lag_periods: [1, 2, 3, 5, 10]
```

### 3.3 预期效果
- 提供更多波动率相关的信息，提高预测准确性
- 通过滞后特征捕捉时间序列依赖
- 通过趋势特征捕捉波动率的动态变化
- 通过regime特征识别波动率状态

## 四、波动率模型训练更新

### 4.1 特征选择更新
波动率模型现在使用：
- GARCH特征（5个）
- EVT特征（6个）
- **扩展波动率特征（~50个）** ← 新增
- ATR相关特征（1个）
- 其他波动率相关特征

### 4.2 代码更新
- `sr_reversal_model_comparison.py`: 更新`train_volatility_model`函数
- `analyze_ml_volatility_model.py`: 更新特征选择逻辑

## 五、预期改进效果

### 5.1 DTW特征
- ✅ 计算效率大幅提升（仅在SR附近计算）
- ✅ 作为辅助特征，不过度依赖
- ⚠️ 需要验证效果是否改善

### 5.2 GARCH特征
- ✅ 窗口增大，减少过拟合
- ⚠️ 需要验证测试集相关性是否提升

### 5.3 扩展波动率特征
- ✅ 提供更多波动率相关信息
- ✅ 通过滞后和趋势特征捕捉动态变化
- ⚠️ 需要验证是否提高预测准确性

## 六、下一步验证

1. **运行模型对比**：`make ts-sr-reversal-model-comparison`
2. **分析波动率预测**：`make ts-analyze-dtw-volatility`
3. **检查特征重要性**：查看扩展波动率特征的重要性排名
4. **验证测试集相关性**：检查是否从0.1810提升

## 七、注意事项

1. **DTW特征**：如果`dist_to_nearest_sr`列不存在，会fallback到全量计算
2. **GARCH窗口**：窗口增大到120，计算时间会增加
3. **扩展波动率特征**：新增~50个特征，可能增加过拟合风险，需要监控
4. **特征选择**：建议使用特征重要性或相关性筛选最重要的特征

