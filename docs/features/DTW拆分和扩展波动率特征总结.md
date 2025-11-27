# DTW拆分和扩展波动率特征总结

## 一、已完成的工作

### 1.1 修复命名冲突 ✅
- **问题**：`extended_volatility_features`中的`vol_zscore`与`baseline_features`中的`vol_zscore`（成交量Z-score）冲突
- **修复**：将`vol_zscore`重命名为`vol_volatility_zscore`
- **文件**：
  - `src/features/time_series/utils_volatility_features.py`
  - `config/feature_dependencies.yaml`

### 1.2 创建独立DTW特征提取器 ✅
- **文件**：`src/features/time_series/utils_dtw_individual.py`
- **功能**：每个DTW模板一个独立的函数
- **支持**：12个独立的DTW特征提取函数

## 二、扩展波动率特征分析

### 2.1 特征数量
- **总计**：45个特征
- **分类**：
  - 历史波动率：4个
  - ATR相关：17个
  - 滞后特征：5个（由`lag_periods`参数控制）
  - 趋势特征：4个
  - 移动平均：6个
  - Regime特征：2个
  - 范围特征：4个
  - 动量特征：3个

### 2.2 重复特征检查结果

| 特征名 | extended_volatility_features | baseline_features | 状态 |
|--------|------------------------------|-------------------|------|
| `vol_zscore` | 波动率Z-score | 成交量Z-score | ✅ 已修复（重命名为`vol_volatility_zscore`） |
| `vol_atr_ratio_ma20` | ATR比率 | `atr_ratio` | ⚠️ 部分重叠，但更详细 |
| `vol_historical_*` | 历史波动率 | 无 | ✅ 新特征 |
| `vol_lag_*` | 滞后特征 | 无 | ✅ 新特征 |

### 2.3 聚合分类的目的

#### ✅ 方便模型使用
- **统一命名**：所有特征以`vol_`开头，便于筛选
- **功能分组**：相关特征聚合，便于理解
- **批量加载**：一次加载所有波动率特征

#### ✅ 提高效率
- **并行计算**：共享中间结果（如`vol_base`）
- **缓存优化**：整个特征组一起缓存

#### ✅ 便于维护
- **统一管理**：所有波动率特征在一个地方
- **版本控制**：特征组版本变更更容易追踪

## 三、DTW拆分方案

### 3.1 已创建的函数

#### 反转相关（8个）
- `extract_dtw_hammer` - 锤子线
- `extract_dtw_head_shoulder_bottom` - 头肩底
- `extract_dtw_double_bottom` - 双底
- `extract_dtw_bullish_engulfing` - 看涨吞没
- `extract_dtw_shooting_star` - 射击之星
- `extract_dtw_head_shoulder_top` - 头肩顶
- `extract_dtw_double_top` - 双顶
- `extract_dtw_bearish_engulfing` - 看跌吞没

#### 中继形态（4个）
- `extract_dtw_bull_flag` - 上升旗形
- `extract_dtw_bear_flag` - 下降旗形
- `extract_dtw_triangle` - 三角收敛
- `extract_dtw_decline_consolidation` - 下跌后横盘

### 3.2 配置示例（待添加到feature_dependencies.yaml）

```yaml
# 反转相关DTW特征（适合SR Reversal策略）
dtw_hammer:
  module: enhanced
  compute_func: extract_dtw_hammer
  dependencies: ["sr_strength_max"]
  required_columns: ["close", "dist_to_nearest_sr", "atr"]
  output_columns: ["dtw_hammer_dist"]
  category: pattern
  description: "DTW锤子线特征（看涨反转）"
  compute_params:
    window: 20
    compute_only_near_sr: true
    sr_dist_col: "dist_to_nearest_sr"
    sr_threshold: 1.5
  pass_full_df: true

dtw_head_shoulder_bottom:
  module: enhanced
  compute_func: extract_dtw_head_shoulder_bottom
  # ... 类似配置

# 中继形态DTW特征（适合趋势/压缩突破策略）
dtw_triangle:
  module: enhanced
  compute_func: extract_dtw_triangle
  # ... 类似配置
```

### 3.3 使用示例

#### SR Reversal策略（只需要反转相关）
```yaml
requested_features:
  - dtw_hammer
  - dtw_head_shoulder_bottom
  - dtw_double_bottom
  - dtw_bullish_engulfing
  - dtw_shooting_star
  - dtw_head_shoulder_top
  - dtw_double_top
  - dtw_bearish_engulfing
```

#### 压缩突破策略（只需要中继形态）
```yaml
requested_features:
  - dtw_triangle
  - dtw_decline_consolidation
  - dtw_bull_flag
  - dtw_bear_flag
```

### 3.4 优势
1. **按需加载**：只加载需要的DTW模板
2. **提高效率**：减少不必要的计算
3. **灵活配置**：不同策略选择不同的DTW模板
4. **易于维护**：每个模板独立配置

## 四、下一步行动

### 4.1 需要完成的工作
1. ✅ 修复`vol_zscore`命名冲突 - **已完成**
2. ✅ 创建独立DTW特征提取器 - **已完成**
3. ⏳ 添加DTW函数映射到`feature_function_mapping.py`
4. ⏳ 添加DTW独立配置到`feature_dependencies.yaml`
5. ⏳ 更新SR Reversal策略配置，使用独立的DTW特征

### 4.2 建议
- **保持`dtw_features`**：作为向后兼容的选项，仍然支持一次性加载所有DTW特征
- **新增独立DTW特征**：作为新的选项，支持按需加载
- **逐步迁移**：可以先在SR Reversal策略中测试独立DTW特征，确认无误后再推广

## 五、总结

### 5.1 扩展波动率特征
- ✅ **保持聚合**：45个特征聚合在一起是合理的
- ✅ **修复冲突**：已重命名`vol_zscore`为`vol_volatility_zscore`
- ✅ **功能明确**：所有特征都是波动率相关的，便于使用

### 5.2 DTW特征
- ✅ **支持拆分**：已创建独立的DTW特征提取器
- ⏳ **待配置**：需要在配置文件中添加独立DTW特征的定义
- ✅ **向后兼容**：保留原有的`dtw_features`配置

