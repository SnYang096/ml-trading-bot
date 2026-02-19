# 常数特征分析报告

## 问题总结

诊断发现 8 个常数或几乎常数的特征，这些特征对模型没有帮助。

## 根本原因

### 1. VPIN 相关特征（缺少 tick 数据）

**问题特征：**
- `vpin_x_wick_upper`: 值 = 0.0（常数）
- `vpin_x_wick_lower`: 值 = 0.0（常数）
- `vpin_x_trade_cluster_entropy`: 值 = 0.0（常数）
- `vpin_signed_imbalance_x_trade_cluster_imbalance`: 值 = 1.0（常数）

**根本原因：**
- VPIN 特征计算失败（缺少 tick 数据）
- 依赖列不存在：`vpin`, `vpin_signed_imbalance`, `trade_cluster_imbalance_ratio`, `trade_cluster_directional_entropy`
- 计算函数使用默认值：`df.get("vpin", pd.Series(0.0, index=df.index))`
- 结果：`0.0 * wick_upper_ratio = 0.0` 或 `0.0 * 0.0 = 0.0`

**代码位置：**
- `src/features/time_series/utils_interaction_features.py`:
  - `compute_vpin_x_wick_upper()`: 第 253-271 行
  - `compute_vpin_x_wick_lower()`: 第 274-292 行
  - `compute_vpin_x_trade_cluster_entropy()`: 第 201-223 行
  - `compute_vpin_signed_imbalance_x_trade_cluster_imbalance()`: 第 177-198 行

**解决方案：**
1. **短期**：从特征列表中移除这些 VPIN 交互特征（如果 tick 数据不可用）
2. **长期**：确保 tick 数据可用，或修改计算函数，在依赖列缺失时返回 NaN 而不是 0

### 2. EVT × Trend R² 特征（依赖列缺失）

**问题特征：**
- `evt_x_trend_r2`: 值 = 1.0（常数）

**根本原因：**
- 依赖列不存在：`evt_tail_shape`, `trend_r2_20`
- 计算函数使用默认值：`evt_tail_shape = 0.3`, `trend_r2_20 = 0.0`
- 但诊断显示值是 1.0，说明可能 `evt_tail_shape` 存在但值是 1.0，或者 `trend_r2_20` 存在但值是某个值

**代码位置：**
- `src/features/time_series/utils_interaction_features.py`:
  - `compute_evt_x_trend_r2()`: 第 87-105 行

**解决方案：**
1. 检查 `evt_features` 和 `trend_r2_20` 特征是否正确计算
2. 如果这些特征不可用，从特征列表中移除 `evt_x_trend_r2`

### 3. CVD Slope 特征（实际不是常数）

**问题特征：**
- `cvd_slope_5`: 诊断显示为常数，但实际检查显示有变化

**实际情况：**
- 范围：[-5197.26, 3067.14]
- 方差：4.24e+05（很大，不是常数）
- 非零值：1999/2000 (100%)

**结论：**
- 这个特征**不是常数**，诊断脚本可能误判
- 可能是方差阈值设置过低（1e-8）

### 4. ATR Ratio 特征（ATR 列缺失）

**问题特征：**
- `atr_ratio`: 值范围 [0.000000, 0.107176]，但方差很小（4.87e-10）

**根本原因：**
- `atr` 列不存在
- 计算函数：`df[atr_col] / df[price_col]`，当 `atr_col` 不存在时会报错
- 但特征值有变化，说明可能使用了其他方式计算或 `atr` 列存在但值很小

**代码位置：**
- `src/features/time_series/utils_interaction_features.py`:
  - `compute_atr_ratio()`: 第 546-573 行

**解决方案：**
1. 检查 `atr` 特征是否正确计算
2. 如果 `atr` 不可用，从特征列表中移除 `atr_ratio`

### 5. TBR MA 特征（实际不是常数）

**问题特征：**
- `tbr_ma_5`: 值范围 [0.406769, 0.569140]，方差很小（4.87e-10）

**实际情况：**
- 范围：[0.406769, 0.569140]
- 方差：5.13e-04（虽然小，但不是常数）
- 均值：0.499156（接近 0.5）

**结论：**
- 这个特征**不是常数**，但方差很小
- `taker_buy_ratio` 接近 0.5，所以移动平均也接近 0.5
- 这可能是因为数据中买卖比例接近平衡

## 建议

### 立即行动

1. **移除 VPIN 交互特征**（如果 tick 数据不可用）：
   - `vpin_x_wick_upper`
   - `vpin_x_wick_lower`
   - `vpin_x_trade_cluster_entropy`
   - `vpin_signed_imbalance_x_trade_cluster_imbalance`
   - `vpin_x_wick_upper_rank`
   - `vpin_x_wick_lower_rank`

2. **检查并修复依赖特征**：
   - 确保 `atr` 特征正确计算
   - 确保 `evt_features` 和 `trend_r2_20` 特征正确计算

3. **改进计算函数**：
   - 当依赖列缺失时，返回 NaN 而不是默认值
   - 添加警告日志，提示依赖列缺失

### 长期改进

1. **特征依赖检查**：
   - 在特征计算前检查所有依赖列是否存在
   - 如果依赖列缺失，跳过该特征或返回 NaN

2. **特征质量监控**：
   - 自动检测常数特征
   - 在训练前移除常数特征

3. **文档更新**：
   - 明确哪些特征需要 tick 数据
   - 在配置文件中标记必需的数据类型

## 代码修改建议

### 1. 修改交互特征计算函数，添加依赖检查

```python
def compute_vpin_x_wick_upper(
    df: pd.DataFrame,
    vpin_col: str = "vpin",
    wick_col: str = "wick_upper_ratio",
) -> pd.Series:
    """计算 VPIN × 上影线占比交互项"""
    if vpin_col not in df.columns:
        import warnings
        warnings.warn(f"Column '{vpin_col}' not found, returning NaN for vpin_x_wick_upper")
        return pd.Series(np.nan, index=df.index, name="vpin_x_wick_upper")
    
    if wick_col not in df.columns:
        import warnings
        warnings.warn(f"Column '{wick_col}' not found, returning NaN for vpin_x_wick_upper")
        return pd.Series(np.nan, index=df.index, name="vpin_x_wick_upper")
    
    state = df[vpin_col]
    momentum = df[wick_col]
    return (state.fillna(0) * momentum.fillna(0)).rename("vpin_x_wick_upper")
```

### 2. 在特征配置中标记必需的数据类型

```yaml
vpin_x_wick_upper:
  module: enhanced
  compute_func: compute_vpin_x_wick_upper
  dependencies: ["vpin_features", "wick_ratios"]
  required_columns: ["vpin", "wick_upper_ratio"]
  required_data_types: ["tick"]  # 新增：标记需要 tick 数据
  output_columns: ["vpin_x_wick_upper"]
```

