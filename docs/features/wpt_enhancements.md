# WPT 特征增强说明

## 概述

WPT (Wavelet Packet Transform) 量价能量特征新增了两个重要改进：
1. **Log Returns 预处理**：避免趋势干扰，更真实反映波动
2. **自适应窗口**：基于 ATR 动态调整窗口大小

## 1. Log Returns 预处理

### 问题

在强趋势市场中，原始价格序列做 WPT 分解时：
- 低频能量被趋势主导
- VPER (Volume-Price Energy Ratio) 计算失真
- 量价关系的真实情况被掩盖

### 解决方案

对 **log returns** 做 WPT 分解，而不是原始价格：

```
log_returns = diff(log(prices))
```

### 优势

1. **去除趋势**：log returns 更平稳，趋势不主导低频能量
2. **更真实的能量分布**：能量分布反映真实波动，而非趋势
3. **提高 VPER 准确性**：VPER 能更准确反映量价关系

### 使用

在 `compute_wpt_volume_energy_features` 中：

```python
result = compute_wpt_volume_energy_features(
    df,
    price_col="close",
    volume_col="volume",
    use_log_returns=True,  # 启用 log returns 预处理
)
```

或者在 narrow-IO 版本中：

```python
result = compute_wpt_volume_energy_features_from_series(
    close=close,
    volume=volume,
    use_log_returns=True,
)
```

### 示例

**强趋势市场**：
```
原始价格: [100, 105, 110, 115, 120, 125, 130]
Log Returns: [0.049, 0.047, 0.045, 0.043, 0.041, 0.039]
```

原始价格的低频能量会被趋势主导，而 log returns 的能量分布更真实。

## 2. 自适应窗口

### 问题

固定窗口大小在不同波动率市场可能不合适：
- 低波动率市场：窗口太小，信号不稳定
- 高波动率市场：窗口太大，响应滞后

### 解决方案

基于 ATR（Average True Range）动态调整窗口大小：

```
adaptive_window = max(min_window, int(14 * ATR_ma / price_ma))
```

- 波动率越高，窗口越大
- 波动率越低，窗口越小
- 限制在 `[min_window, max_window]` 范围内

### 使用

```python
result = compute_wpt_volume_energy_features(
    df,
    price_col="close",
    volume_col="volume",
    adaptive_window=True,  # 启用自适应窗口
    atr_col="atr",         # 提供 ATR 列
    lookback_window=20,    # 基础窗口大小
)
```

### 参数

- `adaptive_window` (bool): 是否启用自适应窗口
- `atr_col` (str): ATR 列名（如果为 None，会从价格估算）
- `lookback_window` (int): 基础窗口大小，自适应窗口会围绕此值调整

## 3. 多尺度一致性改进

### 频率中心分类

之前使用路径字符串匹配（如 `"aa"` 开头）来分类频段，现在使用**频率中心**：

```python
freq_center = a_count / level  # a_count 是路径中 'a' 的个数
```

分类标准：
- **Low freq** (< 0.25): 最高频噪声
- **Mid freq** (0.25-0.75): 中频信号（有用的信息）
- **High freq** (>= 0.75): 低频趋势

### 优势

1. **适用于任意 level**：不再依赖路径字符串模式
2. **更准确的分类**：基于频率中心，而非路径字符串
3. **更好的可扩展性**：易于调整分类阈值

## 配置建议

### 推荐配置

```python
# 对于趋势明显的市场
compute_wpt_volume_energy_features(
    df,
    use_log_returns=True,      # 推荐：去除趋势干扰
    adaptive_window=True,       # 推荐：适应波动率
    atr_col="atr",
    lookback_window=20,
)

# 对于震荡市场（趋势不明显）
compute_wpt_volume_energy_features(
    df,
    use_log_returns=False,     # 可选：如果趋势不明显
    adaptive_window=True,       # 仍然推荐：适应波动率
    atr_col="atr",
    lookback_window=20,
)
```

## 相关特征

- `wpt_vper_low` / `wpt_vper_mid` / `wpt_vper_high` - VPER 各频段
- `wpt_energy_cascade` - 能量下移指标
- `wpt_multi_scale_consistency` - 多尺度一致性
- `wpt_breakout_confidence` - 突破置信度
- `wpt_false_breakout_risk` - 假突破风险

## 注意事项

1. **Log Returns 需要足够数据**：至少需要 `level * 2` 个数据点
2. **自适应窗口需要 ATR**：如果没有提供 ATR，会从价格估算
3. **性能考虑**：自适应窗口会增加少量计算开销，但通常可以忽略

## 更新历史

- **2024-12-19**: 
  - 新增 `use_log_returns` 参数
  - 新增 `adaptive_window` 参数
  - 改进频率分类方法（基于频率中心）

