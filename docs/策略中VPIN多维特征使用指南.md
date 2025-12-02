# 策略中 VPIN 多维特征使用指南

## 概述

新的 VPIN 多维聚合特征已经**自动包含**在所有使用 `vpin_features` 的策略中。这些特征通过 `config/feature_dependencies.yaml` 中的 `output_columns` 定义，会在计算 `vpin_features` 时自动添加到数据集中。

## 四个策略的配置状态

### ✅ 所有策略已更新

所有四个策略的配置文件都已添加了注释说明新特征的用途和推荐组合：

1. **SR Reversal** (`config/strategies/sr_reversal/features.yaml`)
2. **SR Breakout** (`config/strategies/sr_breakout/features.yaml`)
3. **Compression Breakout** (`config/strategies/compression_breakout/features.yaml`)
4. **Trend Following** (`config/strategies/trend_following/features.yaml`)

## 新增的多维特征列表

| 特征名称 | 含义 | 关键程度 |
|---------|------|---------|
| `vpin` | 均值（原有，向后兼容） | ⭐⭐ |
| `vpin_max` | **峰值（关键！避免峰值被稀释）** | ⭐⭐⭐⭐⭐ |
| `vpin_last` | 最新值（反映最新情绪） | ⭐⭐⭐⭐ |
| `vpin_min` | 最小值 | ⭐⭐ |
| `vpin_std` | 标准差（衡量波动性） | ⭐⭐⭐ |
| `vpin_count` | 事件数（代理流动性） | ⭐⭐⭐ |
| `vpin_signed_imbalance_last` | Signed imbalance 最新值 | ⭐⭐⭐ |
| `vpin_signed_imbalance_max` | Signed imbalance 峰值 | ⭐⭐⭐ |

## 各策略的推荐特征组合

### 1. SR Reversal（反转策略）

**推荐组合：**
```python
[
    'vpin',              # 均值（整体不平衡度）
    'vpin_max',          # 峰值（捕捉极端知情交易）
    'vpin_last',         # 最新值（反映最新情绪）
    'vpin_count',        # 事件数（区分突发事件vs持续活跃）
    'vpin_signed_imbalance_max',  # 最大买卖压力方向
]
```

**使用场景：**
- `vpin_max > 0.6`：捕捉极端异常，触发反转信号
- `vpin_last < 0.3 & vpin_max > 0.6`：峰值高但最新值低 → 可能已经释放，准备反转
- `vpin_count` 低：突发事件，可能更可靠

### 2. SR Breakout（突破策略）

**推荐组合：**
```python
[
    'vpin_max',          # 峰值（突破确认，关键！）
    'vpin_count',        # 事件数（流动性确认）
    'vpin_zscore_20',    # Z-score（异常检测）
    'vpin_signed_imbalance_max',  # 最大买卖压力方向
]
```

**使用场景：**
- `vpin_max > 0.7`：突破时的峰值异常，确认真突破
- `vpin_count > 15`：突破时流动性增加，事件数增加
- `vpin_zscore_20 > 2`：异常高的订单流不平衡

### 3. Compression Breakout（压缩突破策略）

**推荐组合：**
```python
[
    'vpin_max',          # 峰值（突破确认，关键！）
    'vpin_last',         # 最新值（突破瞬间信号）
    'vpin_count',        # 事件数（流动性确认）
    'vpin_std',          # 标准差（突破时波动性上升）
]
```

**使用场景：**
- `vpin_max > 0.7 & vpin_last > 0.5`：峰值高且最新值高 → 突破信号强烈
- `vpin_std > 0.2`：突破时波动性上升
- `vpin_count` 增加：突破时流动性增加

### 4. Trend Following（趋势跟踪策略）

**推荐组合：**
```python
[
    'vpin',              # 均值（整体不平衡度）
    'vpin_signed_imbalance_last',  # 最新买卖压力方向（趋势方向）
    'vpin_std',          # 标准差（健康趋势应低波动）
    'vpin_count',        # 事件数（持续活跃）
]
```

**使用场景：**
- `vpin_signed_imbalance_last > 0`：最新买压，趋势向上
- `vpin_std < 0.15`：低波动性，健康趋势
- `vpin_count` 稳定：持续活跃，趋势延续

## 特征自动包含机制

### 工作原理

1. **配置层面**：`config/feature_dependencies.yaml` 中定义了 `vpin_features` 的所有 `output_columns`
2. **自动计算**：当策略请求 `vpin_features` 时，`extract_order_flow_features()` 函数会计算所有输出列
3. **自动添加**：所有在 `output_columns` 中定义的特征会自动添加到 DataFrame 中
4. **无需手动添加**：策略配置中只需包含 `vpin_features`，新特征会自动包含

### 验证方法

运行特征计算后，检查 DataFrame 的列名：

```python
# 检查是否包含新特征
new_vpin_features = [
    'vpin', 'vpin_max', 'vpin_last', 'vpin_min', 
    'vpin_std', 'vpin_count',
    'vpin_signed_imbalance_last', 'vpin_signed_imbalance_max'
]

for feat in new_vpin_features:
    assert feat in df.columns, f"Missing feature: {feat}"
```

## 使用示例

### 示例 1：在 SR Breakout 中使用峰值特征

```python
# 策略逻辑
def sr_breakout_signal(df):
    # 使用峰值特征捕捉极端异常
    high_vpin_peak = df['vpin_max'] > 0.7
    high_liquidity = df['vpin_count'] > 15
    abnormal_vpin = df['vpin_zscore_20'] > 2
    
    # 组合信号
    signal = high_vpin_peak & high_liquidity & abnormal_vpin
    return signal
```

### 示例 2：在 SR Reversal 中区分信号类型

```python
# 区分突发事件和持续活跃
def reversal_signal_classification(df):
    high_peak = df['vpin_max'] > 0.6
    high_count = df['vpin_count'] > 15
    low_count = df['vpin_count'] < 5
    
    # 持续活跃：峰值高且事件多
    sustained_activity = high_peak & high_count
    
    # 突发事件：峰值高但事件少
    sudden_event = high_peak & low_count
    
    # 反转信号：突发事件可能更可靠
    reversal_signal = sudden_event | (sustained_activity & (df['vpin_last'] < 0.3))
    return reversal_signal
```

### 示例 3：在 Trend Following 中使用方向特征

```python
# 趋势方向确认
def trend_direction(df):
    # 最新买卖压力方向
    buy_pressure = df['vpin_signed_imbalance_last'] > 0.3
    sell_pressure = df['vpin_signed_imbalance_last'] < -0.3
    
    # 低波动性（健康趋势）
    stable_trend = df['vpin_std'] < 0.15
    
    # 趋势信号
    long_signal = buy_pressure & stable_trend
    short_signal = sell_pressure & stable_trend
    
    return long_signal, short_signal
```

## 模型训练建议

### 特征重要性预期

1. **`vpin_max`**：预期会成为**最重要**的特征之一，因为它直接捕捉峰值信号
2. **`vpin_last`**：时间敏感的信号，可能在近期预测中更重要
3. **`vpin_count`**：流动性代理，可能有助于区分信号类型

### 特征选择

LightGBM 会自动选择有用特征，但可以：

1. **保留所有多维特征**：让模型自动选择
2. **重点关注峰值特征**：`vpin_max` 应该成为核心特征
3. **结合使用**：峰值 + 最新值 + 事件数的组合可能更有价值

### 阈值设置

- **峰值阈值**：可能需要更高（如 0.7），因为峰值本身就是异常值
- **最新值阈值**：可以较低（如 0.3-0.5），反映最新情绪
- **事件数阈值**：根据时间框架调整（1h：10-20，4h：20-40）

## 注意事项

1. **向后兼容**：原有的 `vpin`（均值）特征仍然保留，现有模型可以继续使用
2. **自动包含**：新特征会自动包含，无需修改策略配置
3. **特征数量**：VPIN 特征总数从 21 个增加到 28 个，注意特征维度
4. **计算性能**：多维统计不会显著增加计算时间（已经在一次计算中完成）

## 总结

✅ **新特征已自动包含**在所有策略中  
✅ **四个策略配置已更新**注释说明  
✅ **推荐使用峰值特征**（`vpin_max`）捕捉异常信号  
✅ **结合使用多维特征**可以更好地理解订单流状态  

所有新特征都会在特征计算时自动添加到数据集中，可以直接在模型训练和策略逻辑中使用。

