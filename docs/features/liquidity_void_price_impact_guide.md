# Liquidity Void Price Impact 特征使用指南

## 概述

`liquidity_void_price_impact` 是 `liquidity_void_f` 特征的新输出列，用于衡量价格冲击（Price Impact），即单位成交量推动的价格距离。

## 计算公式

```
price_impact = (high - low) / volume
```

其中：
- `high`: K线最高价
- `low`: K线最低价  
- `volume`: K线成交量

## 含义

- **值越大**：说明流动性越差，少量成交量就能推动较大的价格变动
- **值越小**：说明流动性越好，需要更多成交量才能推动价格变动

## 使用场景

### 1. 流动性真空识别

价格冲击是识别流动性真空区的重要指标之一。当以下条件同时满足时，通常表示存在流动性真空：

- `liquidity_void_detected == 1.0`
- `liquidity_void_price_impact` 较大（相对历史水平）
- `liquidity_void_speed` 较高（价格快速穿越）
- `liquidity_void_volume_ratio` 较低（成交量低于历史均值）

### 2. 假突破风险判断

高价格冲击可能意味着：
- 市场深度不足，容易被大单推动
- 突破后可能快速回撤（假突破）
- 需要结合 `liquidity_void_false_breakout_risk` 综合判断

### 3. 订单执行策略

在实盘交易中，`price_impact` 可以帮助：
- 评估订单执行成本
- 选择合适的订单拆分策略
- 避免在流动性真空区下单

## 输出列

`liquidity_void_f` 特征包含以下输出列：

1. `liquidity_void_detected` - 是否检测到流动性真空 (0.0/1.0)
2. `liquidity_void_speed` - 价格速度（归一化）
3. `liquidity_void_volume_ratio` - 成交量比率
4. **`liquidity_void_price_impact`** - 价格冲击（新增）
5. `liquidity_void_retracement` - 回撤幅度
6. `liquidity_void_false_breakout_risk` - 假突破风险评分

## 注意事项

1. **仅在检测到流动性真空时有效**：`price_impact` 只在 `liquidity_void_detected == 1.0` 时有非零值，其他情况下为 0.0

2. **需要结合其他指标使用**：单独使用 `price_impact` 可能不够，建议结合 `volume_ratio`、`speed` 等指标综合判断

3. **相对值更重要**：关注 `price_impact` 相对于历史水平的异常，而不是绝对值

4. **跨资产比较需谨慎**：不同资产的价格和成交量尺度不同，直接比较 `price_impact` 可能没有意义

## 配置示例

在策略配置文件中使用：

```yaml
feature_pipeline:
  requested_features:
    - liquidity_void_f  # 包含 price_impact 输出列
```

## 相关特征

- `wpt_volume_energy_f` - WPT 量价能量特征
- `vpvr_*` - Volume Profile 相关特征
- `footprint_basic_f` - Footprint 特征（包含 POC/HVN/LVN）

## 更新历史

- **2024-12-19**: 新增 `liquidity_void_price_impact` 输出列

