# VPIN 多维聚合特征改进说明

## 问题背景

### 原有实现的问题

之前的 VPIN 对齐到 K 线时，**只使用均值聚合**，这会导致严重的**信息损失**：

1. **峰值信号被稀释**：如果在 4H 内某个时刻 VPIN 异常高（如 0.8），但被该 4H 内的其他低 VPIN 值平均后可能只有 0.3，**峰值信号完全丢失**。

2. **无法区分突发事件和持续活跃**：
   - 场景 A：4H 内有 20 个 VPIN bucket，均值 0.3 → 持续活跃
   - 场景 B：4H 内有 2 个 VPIN bucket，均值 0.3 → 可能是突发事件
   - 仅用均值无法区分这两种情况

3. **时间信息丢失**：无法知道异常 VPIN 发生在 K 线的哪个时间点（开始、中间、还是结束）

### VPIN 的本质特性

VPIN 是基于 **volume bucket** 的异步事件流：
- 每个 bucket 有固定的 volume（如 100 BTC），而非固定的时间
- 在高流动性时期，bucket 填满快，VPIN 事件密集
- 在低流动性时期，bucket 填满慢，VPIN 事件稀疏
- **VPIN 事件的时间戳不均匀，信息密度由成交量决定**

## 改进方案

### 核心思想

**不再使用单一均值，而是提取多维统计特征**，完整保留 K 线周期内的 VPIN 信息。

### 新增特征（7个）

对每个 K 线周期（如 4H），提取以下 VPIN 统计特征：

| 特征 | 含义 | 作用 |
|------|------|------|
| `vpin` | 均值（原有，保留向后兼容） | 整体不平衡度 |
| `vpin_last` | 最新值 | 反映最新情绪，时间最接近 K 线结束 |
| **`vpin_max`** | **峰值** | **捕捉极端知情交易，避免峰值被稀释（关键！）** |
| `vpin_min` | 最小值 | 捕捉最低不平衡度 |
| `vpin_std` | 标准差 | 衡量 VPIN 在 K 线内的波动性 |
| `vpin_count` | 事件数 | 代理流动性，区分突发事件 vs 持续活跃 |
| `vpin_signed_imbalance_last` | Signed imbalance 最新值 | 最新买卖压力方向 |
| `vpin_signed_imbalance_max` | Signed imbalance 峰值 | 最大买卖压力 |

### 特征使用建议

#### 1. 捕捉峰值信号（关键改进）

```python
# 之前：只能使用均值，峰值被稀释
signal = df['vpin'] > 0.6  # 可能漏掉峰值信号

# 现在：可以直接使用峰值特征
signal = df['vpin_max'] > 0.6  # 捕捉极端异常，不会漏掉
```

#### 2. 区分突发事件和持续活跃

```python
# 场景识别
high_vpin_peak = df['vpin_max'] > 0.6  # 峰值高
high_vpin_count = df['vpin_count'] > 15  # 事件多

if high_vpin_peak & high_vpin_count:
    # 持续活跃：峰值高且事件多 → 知情交易者持续入场
elif high_vpin_peak & ~high_vpin_count:
    # 突发事件：峰值高但事件少 → 大单突袭
```

#### 3. 结合最新值和峰值

```python
# 最新值和峰值都高 → 强信号
strong_signal = (df['vpin_last'] > 0.5) & (df['vpin_max'] > 0.6)

# 峰值高但最新值低 → 可能已经释放
peak_but_fading = (df['vpin_max'] > 0.6) & (df['vpin_last'] < 0.3)
```

#### 4. 波动性分析

```python
# 高波动性 → VPIN 在 K 线内变化大，可能是不稳定信号
unstable = df['vpin_std'] > 0.2

# 低波动性 + 高均值 → 稳定的不平衡
stable_imbalance = (df['vpin_std'] < 0.1) & (df['vpin'] > 0.5)
```

## 技术实现

### 对齐逻辑改进

**之前（单一均值）**：
```python
# 简单均值聚合
vpin_aggregated = vpin_series.groupby(kline_idx).mean()
```

**现在（多维统计）**：
```python
# 多维统计聚合
aligned_stats = temp_df.groupby('kline_idx').agg({
    'vpin': ['mean', 'max', 'min', 'std', 'last', 'count'],
    'signed': ['mean', 'max', 'last']
})
```

### 代码位置

- **实现文件**：`src/features/time_series/utils_order_flow_features.py`
- **对齐函数**：`extract_order_flow_features()` 中的对齐逻辑（第 488-640 行）
- **配置位置**：`config/feature_dependencies.yaml` 中的 `vpin_features`

## 实际效果对比

### 场景示例

假设某个 4H K 线内有以下 VPIN bucket：

| 时间 | VPIN 值 |
|------|---------|
| 00:00 | 0.3 |
| 00:30 | 0.2 |
| 01:15 | **0.85** ← 极端异常 |
| 02:00 | 0.4 |
| 02:45 | 0.3 |
| 03:30 | 0.35 |

**之前（均值聚合）**：
- `vpin = 0.4` （峰值 0.85 被稀释，信号丢失）

**现在（多维统计）**：
- `vpin = 0.4` （均值，保留向后兼容）
- `vpin_max = 0.85` （峰值，关键信号保留！）
- `vpin_last = 0.35` （最新值）
- `vpin_count = 6` （事件数）
- `vpin_std = 0.23` （波动性）

**结果**：峰值信号（0.85）完全保留，模型可以识别异常。

## 对模型的影响

### 特征重要性预期

1. **`vpin_max`**：预期会成为重要特征，因为它直接捕捉峰值信号
2. **`vpin_count`**：可能有助于区分信号类型（突发事件 vs 持续活跃）
3. **`vpin_last`**：时间敏感的信号（反映最新情绪）

### 向后兼容性

- ✅ **完全向后兼容**：原有的 `vpin`（均值）特征仍然保留
- ✅ 现有模型不需要修改即可使用新特征
- ✅ 新特征作为**额外列**添加，不会破坏现有特征结构

## 使用建议

### 1. 策略开发

优先使用多维特征：
```python
# 推荐组合
features = [
    'vpin',           # 均值（整体）
    'vpin_max',       # 峰值（异常）
    'vpin_last',      # 最新值（时效）
    'vpin_count',     # 事件数（流动性）
]
```

### 2. 特征选择

根据策略类型选择：
- **突破策略**：重点使用 `vpin_max`（捕捉极端信号）
- **反转策略**：可以使用 `vpin_last` + `vpin_max` 组合
- **流动性策略**：使用 `vpin_count` 作为流动性代理

### 3. 阈值设置

峰值特征可能需要更高的阈值：
```python
# 均值阈值（原有）
vpin_mean_threshold = 0.5

# 峰值阈值（新特征，可能需要更高）
vpin_max_threshold = 0.7  # 因为峰值本身就是异常值
```

## 总结

### 关键改进

1. ✅ **保留峰值信号**：`vpin_max` 不会因均值而被稀释
2. ✅ **区分信号类型**：`vpin_count` 帮助区分突发事件 vs 持续活跃
3. ✅ **时间信息**：`vpin_last` 反映最新情绪
4. ✅ **向后兼容**：原有 `vpin` 特征保留，不影响现有模型

### 核心价值

这个改进解决了 **VPIN 作为异步事件流与固定时间框架（K 线）对齐时的信息损失问题**，是微观结构特征与多尺度时间框架融合的标准做法（参考《Advances in Financial Machine Learning》）。

### 下一步

1. 回测验证新特征的有效性
2. 特征重要性分析（LightGBM feature importance）
3. 根据回测结果调整阈值和特征组合

