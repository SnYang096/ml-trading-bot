# LVN (Low Volume Node) 检测改进说明

## 概述

LVN（Low Volume Node，低成交量节点）检测逻辑已改进，从简单的全局最小值方法升级为基于局部极小值的检测方法。

## 改进内容

### 之前的方法（已移除）

**全局最小值方法**：
- 在整个价格区间内找到成交量最小（但 > 0）的那个 bin
- 问题：可能选择到价格边缘的噪声 bin，而不是真正有意义的低成交量"谷底"

### 现在的方法（推荐）

**局部极小值方法**：
- 使用 `scipy.signal.find_peaks` 检测局部极小值
- 选择被高成交量区域包围的低成交量"谷底"
- 更符合专业交易分析中的 LVN 概念

## 技术实现

### 函数

`_find_lvn_local_minimum()` - 在 `src/features/time_series/utils_footprint.py` 中

### 关键参数

1. **`lvn_min_prominence`** (默认: 0.15)
   - 最小 prominence，相对于平均成交量的比例
   - 控制局部极小值的显著性要求

2. **`lvn_min_distance`** (默认: 2)
   - 局部极小值之间的最小距离（bin 数）
   - 避免选择过于接近的局部极小值

### 检测逻辑

```python
1. 计算成交量分布的平均值
2. 设置 prominence 阈值 = mean_volume * lvn_min_prominence
3. 使用 find_peaks(-volumes) 找到局部极小值
4. 选择最深的 LVN（成交量最小的局部极小值）
5. 如果未找到，回退到 POC（Point of Control）
```

## 配置

在 `FootprintConfig` 中：

```python
from src.features.time_series.utils_footprint import FootprintConfig

cfg = FootprintConfig(
    price_bin_size=0.1,
    value_area_pct=0.7,
    lvn_min_prominence=0.15,  # 默认值，可根据需要调整
    lvn_min_distance=2,        # 默认值，可根据需要调整
)
```

## 使用示例

```python
from src.features.time_series.utils_footprint import (
    compute_kline_footprint_features,
    FootprintConfig
)

# 使用默认配置（局部极小值方法）
cfg = FootprintConfig()
result = compute_kline_footprint_features(ticks, klines, cfg=cfg)

# 访问 LVN 价格
lvn_price = result['fp_lvn']
```

## 优势

1. **更准确的 LVN 识别**：识别被高成交量包围的低成交量区域
2. **减少噪声影响**：避免选择价格边缘的随机噪声 bin
3. **更符合交易逻辑**：LVN 作为"快速穿越通道"的概念更准确

## 限制

1. **需要足够的数据**：至少需要 3 个 bin 才能进行检测
2. **参数敏感性**：`prominence` 和 `distance` 参数可能需要根据市场特征调整
3. **回退机制**：如果未找到局部极小值，会回退到 POC（这是合理的默认行为）

## 相关特征

- `fp_poc` - Point of Control（成交量最大的价格）
- `fp_hvn` - High Volume Node（高成交量节点）
- `fp_vah` / `fp_val` - Value Area High/Low（价值区域）

## 更新历史

- **2024-12-19**: 
  - 移除全局最小值方法
  - 实现局部极小值检测
  - 移除 `lvn_method` 配置选项（只保留局部极小值方法）

