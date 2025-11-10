# CVD 滚动窗口和变化率改进

## 改进日期
2025-10-22

## 问题背景

### 原始CVD实现的问题

```python
# 原始实现：使用全局cumsum
cvd = (buy_qty - sell_qty).cumsum()
```

**潜在问题：**

1. **跨期不连续性：** 如果训练和测试数据分开计算，CVD值会不连续
2. **数值漂移：** 长期累计可能导致数值过大或过小
3. **趋势依赖：** CVD绝对值依赖于起始点，不同时期不可比
4. **数据泄露风险：** 虽然在单文件内计算是安全的，但跨文件合并时需要谨慎

## 改进方案

### 1. 多时间窗口CVD

使用不同时间窗口的滚动累计，而非全局cumsum：

```python
# Delta (buy - sell)
delta = per_interval['buy_qty'] - per_interval['sell_qty']

# 短期CVD (20个周期，约100分钟 for 5min bars)
per_interval['cvd_short'] = delta.rolling(window=20, min_periods=1).sum()

# 中期CVD (60个周期，约5小时)
per_interval['cvd_medium'] = delta.rolling(window=60, min_periods=1).sum()

# 长期CVD (288个周期，约24小时)
per_interval['cvd_long'] = delta.rolling(window=288, min_periods=1).sum()
```

**优势：**
- ✅ 每个时间窗口独立，不依赖起始点
- ✅ 数值范围可控
- ✅ 多时间框架捕捉不同周期的订单流
- ✅ 避免数值漂移

### 2. CVD变化率

预计算CVD的变化率，避免在特征工程中使用diff()：

```python
# 当前周期的delta
per_interval['cvd_change_1'] = delta

# 5周期累计变化
per_interval['cvd_change_5'] = delta.rolling(window=5).sum()

# 20周期累计变化
per_interval['cvd_change_20'] = delta.rolling(window=20).sum()
```

**优势：**
- ✅ 避免对cumsum结果做diff()（等价于直接用delta的rolling sum）
- ✅ 更清晰的语义：直接表达"最近N期的买卖压力"
- ✅ 计算更高效

### 3. CVD归一化

相对于成交量归一化CVD：

```python
total_volume = per_interval['buy_qty'] + per_interval['sell_qty']
per_interval['cvd_normalized'] = delta / total_volume.replace(0, np.nan)
per_interval['cvd_normalized'] = per_interval['cvd_normalized'].fillna(0)
```

**优势：**
- ✅ 不同市场阶段可比（高波动期vs低波动期）
- ✅ 消除成交量规模的影响
- ✅ 值域在[-1, 1]之间，易于理解

### 4. 向后兼容

保留原始CVD用于向后兼容：

```python
# 保留原始CVD
per_interval['cvd'] = delta.cumsum()
```

这样旧代码仍然可以运行，但建议使用新的滚动窗口版本。

## 新增特征列表

| 特征名 | 描述 | 时间窗口 | 用途 |
|--------|------|----------|------|
| `cvd_short` | 短期CVD | 20周期 (约100分钟) | 捕捉短期买卖压力 |
| `cvd_medium` | 中期CVD | 60周期 (约5小时) | 捕捉中期趋势 |
| `cvd_long` | 长期CVD | 288周期 (约24小时) | 捕捉长期趋势 |
| `cvd_change_1` | 当前周期delta | 1周期 | 即时买卖压力 |
| `cvd_change_5` | 5周期CVD变化 | 5周期 | 短期momentum |
| `cvd_change_20` | 20周期CVD变化 | 20周期 | 中期momentum |
| `cvd_normalized` | 归一化CVD | 1周期 | 标准化的买卖压力 |
| `cvd` | 原始CVD (向后兼容) | 全局cumsum | 传统CVD |

## Feature Engineering中的改进

### 增强的CVD衍生特征

在 `feature_engineering_enhanced.py::add_order_flow_features()` 中新增：

#### 1. 多时间框架CVD趋势

```python
# 短期CVD趋势
df['cvd_short_trend'] = df['cvd_short'].diff(5)
df['cvd_short_momentum'] = df['cvd_short_trend'].diff()

# 中期CVD趋势
df['cvd_medium_trend'] = df['cvd_medium'].diff(10)
df['cvd_medium_momentum'] = df['cvd_medium_trend'].diff()

# 长期CVD趋势
df['cvd_long_trend'] = df['cvd_long'].diff(20)
```

#### 2. CVD跨周期关系

```python
# 短/中期比率
df['cvd_short_medium_ratio'] = df['cvd_short'] / (df['cvd_medium'].abs() + 1e-10)

# 中/长期比率
df['cvd_medium_long_ratio'] = df['cvd_medium'] / (df['cvd_long'].abs() + 1e-10)
```

#### 3. CVD趋势一致性

```python
# 短中长期同向性
cvd_short_sign = np.sign(df['cvd_short'])
cvd_medium_sign = np.sign(df['cvd_medium'])
cvd_long_sign = np.sign(df['cvd_long'])

df['cvd_trend_alignment'] = (
    (cvd_short_sign == cvd_medium_sign) & 
    (cvd_medium_sign == cvd_long_sign)
).astype(int)
```

**解释：** 当短中长期CVD方向一致时，表示强烈的趋势信号

#### 4. CVD归一化特征

```python
df['cvd_norm_momentum'] = df['cvd_normalized'].rolling(5).mean()
df['cvd_norm_extreme'] = (df['cvd_normalized'].abs() > 0.6).astype(int)
```

### Order Flow Imbalance (OFI) 改进

同样的逻辑应用于OFI：

```python
# 改用滚动窗口OFI（避免全局cumsum）
df['ofi_short'] = df['order_flow_imbalance'].rolling(20, min_periods=1).sum()
df['ofi_medium'] = df['order_flow_imbalance'].rolling(60, min_periods=1).sum()
df['ofi_long'] = df['order_flow_imbalance'].rolling(288, min_periods=1).sum()

# 向后兼容：保留cumulative_ofi
df['cumulative_ofi'] = df['order_flow_imbalance'].cumsum()
```

## 使用示例

### 基础用法

```python
from data_utils import load_and_process_file, add_order_flow_features

# 加载数据
df = load_and_process_file('BTCUSDT-aggTrades-2024-10.zip')

# 添加订单流特征（包括新的CVD特征）
df = add_order_flow_features('BTCUSDT-aggTrades-2024-10.zip', df)

# 现在df包含：
# - cvd_short, cvd_medium, cvd_long (滚动窗口CVD)
# - cvd_change_1, cvd_change_5, cvd_change_20 (CVD变化率)
# - cvd_normalized (归一化CVD)
# - cvd (原始CVD，向后兼容)
```

### 特征工程

```python
from data_tools.feature_engineering_enhanced import EnhancedFeatureEngineer

# 创建特征工程器
fe = EnhancedFeatureEngineer()

# 工程化特征（会自动使用新的CVD特征）
multi_tf_data = {'5min': df}
engineered_data = fe.engineer_features(multi_tf_data, fit=True)

# 现在包含增强的CVD衍生特征：
# - cvd_short_trend, cvd_short_momentum
# - cvd_medium_trend, cvd_medium_momentum
# - cvd_long_trend
# - cvd_short_medium_ratio, cvd_medium_long_ratio
# - cvd_trend_alignment
# - cvd_norm_momentum, cvd_norm_extreme
```

## 特征数量对比

| 版本 | CVD相关特征数 | 说明 |
|------|--------------|------|
| 原始版本 | 5 | cvd, cvd_slope_3/10/30, cvd_acceleration |
| 改进版本 | 15+ | 增加多时间框架、趋势、一致性等特征 |

## 预期效果

### 优势

1. **更稳定的特征：** 滚动窗口CVD不受起始点影响
2. **更丰富的信息：** 多时间框架捕捉不同周期的订单流
3. **更清晰的语义：** cvd_short/medium/long直观表达时间尺度
4. **避免数值问题：** 不会出现累计值过大或过小的问题
5. **趋势识别：** cvd_trend_alignment可以识别强趋势信号

### 潜在应用场景

1. **趋势确认：** cvd_trend_alignment=1时，短中长期订单流同向
2. **背离检测：** cvd_short vs cvd_long方向相反时，可能是转折点
3. **强度判断：** cvd_normalized表示买卖压力的相对强度
4. **动量追踪：** cvd_change特征直接表达最近N期的净买入压力

## 数据泄露检查

### ✅ 滚动窗口是安全的

```python
# 滚动窗口只使用历史数据
cvd_short = delta.rolling(window=20).sum()
# 第i个时刻的cvd_short只依赖 delta[i-19:i+1]
# 这是安全的，因为我们使用的是K线数据，第i根K线完成后才有第i个值
```

### ✅ 每个文件独立计算

```python
# 每个月的数据文件独立加载和计算
# 不会跨文件累计，因此不会泄露
```

### ⚠️ 注意事项

如果未来要合并多个月份的数据并继续使用CVD的累计值，需要注意：

1. **方案A：** 按时间顺序连接数据，让cumsum自然延续（需要确保时间顺序）
2. **方案B（推荐）：** 只使用滚动窗口版本，避免跨期累计
3. **方案C：** 使用CVD的diff()或标准化版本，消除起始点影响

## 与原始CVD的对比

### 原始CVD

```python
import matplotlib.pyplot as plt

# 模拟数据
delta = np.random.randn(1000).cumsum()  # 模拟buy-sell delta

# 原始CVD
cvd_original = delta.cumsum()

plt.figure(figsize=(12, 4))
plt.plot(cvd_original, label='Original CVD (cumsum)')
plt.legend()
plt.title('原始CVD：无界累计')
```

**问题：** CVD值可能漂移到很大或很小的数值

### 改进CVD

```python
# 滚动窗口CVD
cvd_short = pd.Series(delta).rolling(20).sum()
cvd_medium = pd.Series(delta).rolling(60).sum()
cvd_long = pd.Series(delta).rolling(288).sum()

plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(cvd_short, label='CVD Short (20)')
plt.legend()
plt.subplot(3, 1, 2)
plt.plot(cvd_medium, label='CVD Medium (60)')
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(cvd_long, label='CVD Long (288)')
plt.legend()
plt.tight_layout()
```

**优势：** 每个窗口的CVD值在合理范围内

## 总结

### 主要改进

1. ✅ **多时间窗口CVD** - 短期(20)/中期(60)/长期(288)
2. ✅ **CVD变化率** - 直接使用delta的rolling sum
3. ✅ **CVD归一化** - 相对于成交量标准化
4. ✅ **增强衍生特征** - 趋势、momentum、一致性等
5. ✅ **向后兼容** - 保留原始CVD

### 使用建议

1. **优先使用：** cvd_short/medium/long 而非 cvd
2. **趋势判断：** cvd_trend_alignment
3. **强度判断：** cvd_normalized
4. **Momentum：** cvd_change_5/20

### 文件修改

- ✅ `ml_project/scripts/common/data_utils.py` - CVD计算逻辑
- ✅ `ml_project/src/data_tools/feature_engineering_enhanced.py` - 特征工程增强

---

**更新日期:** 2025-10-22  
**作者:** AI Assistant  
**相关文档:** [数据泄露检查和最佳实践](./数据泄露检查和最佳实践.md)

