# ZigZag 特征未被选中的原因分析

## 问题描述

在 `top_factors.json` 中，zigzag 特征没有被选中，而且选中的特征效果都特别差。

## 根本原因

### 1. **ZigZag 特征未归一化**

- `zigzag` 是一个**原始价格量纲**的特征（类似于 `atr`）
- 代码中有 `atr_normalized`，但**没有 `zigzag_normalized`**
- 在多资产场景下，原始价格量纲的特征会导致：
  - 不同资产间的特征值不可比
  - 特征选择时 IC 值被稀释
  - 模型训练时产生偏差

### 2. **特征选择机制排除原始量纲特征**

在 `get_feature_columns()` 中：
- `atr` 被明确排除（使用 `atr_normalized` 替代）
- 但 `zigzag` **没有被明确排除**，导致：
  - 如果 zigzag 被包含，会因为量纲问题影响模型
  - 如果 zigzag 被排除，就没有 zigzag 相关的特征可用

### 3. **ZigZag 作为结构特征的特性**

ZigZag 是一个**结构特征**，用于：
- 识别价格的高低点
- 识别趋势转折点
- 提供结构确认信号

**不是直接的预测特征**，因此：
- 与未来收益的 IC 值可能较低
- 在基于 IC 的特征选择中排名靠后
- 需要与其他特征组合使用才能发挥价值

### 4. **特征化导致的信息丢失**

ZigZag 的核心价值在于：
- **转折点识别**：标识重要的价格反转点
- **结构确认**：确认趋势的延续或反转
- **距离测量**：当前价格到结构点的距离

如果只是简单地将 zigzag 值作为特征，会丢失这些结构化信息。

## 解决方案

### ✅ 已实施的改进

1. **创建 ZigZag 归一化特征**
   - `zigzag_normalized`: 类似 `atr_normalized`，归一化到价格比例
   - 公式: `zigzag_normalized = (zigzag / close) - 1.0`

2. **创建 ZigZag 衍生特征**
   - `zigzag_distance`: 当前价格到 zigzag 点的距离（ATR 归一化）
   - `zigzag_turn`: 转折点标记（0/1）
   - `zigzag_slope`: ZigZag 段的斜率（归一化）

3. **更新特征选择逻辑**
   - 在 `get_feature_columns()` 中排除原始 `zigzag`
   - 保留所有归一化的 zigzag 衍生特征

### 📋 新增特征列表

| 特征名称 | 类型 | 说明 | 归一化方法 |
|---------|------|------|-----------|
| `zigzag_normalized` | 归一化 | ZigZag 值相对于收盘价的比例 | `(zigzag / close) - 1.0` |
| `zigzag_distance` | 距离 | 当前价格到 ZigZag 点的距离 | `(close - zigzag) / ATR` |
| `zigzag_turn` | 标记 | ZigZag 转折点标记 | 0/1 二值 |
| `zigzag_slope` | 斜率 | ZigZag 段的斜率 | `diff(zigzag, 5) / ATR` |

## 使用建议

### 1. **重新运行特征选择**

```bash
make ts-dim-compare SYMBOL=BTCUSDT,ETHUSDT \
  START_DATE=2020-01-01 END_DATE=2021-12-31 \
  DIM_COMPARE_FEATURE_TYPE=comprehensive
```

### 2. **检查新特征是否被选中**

查看新的 `top_factors.json`，应该包含：
- `zigzag_normalized`
- `zigzag_distance`
- `zigzag_turn`
- `zigzag_slope`

### 3. **特征组合使用**

ZigZag 特征应该与其他特征组合使用：
- **趋势确认**: `zigzag_slope` + `hurst_trend_signal`
- **结构突破**: `zigzag_distance` + `compression_to_breakout_prob`
- **反转信号**: `zigzag_turn` + `rsi_divergence`

### 4. **调整 ZigZag 参数**

如果新特征仍然效果不佳，可以尝试：
- 调整 `compute_zigzag()` 中的 `threshold` 参数（默认 0.05）
- 使用不同时间周期的 zigzag
- 结合多个 zigzag 特征（不同阈值）

## 技术细节

### 代码修改位置

1. **`src/data_tools/base_indicators.py`**
   - 在 `add_common_derived_features()` 中添加 zigzag 衍生特征

2. **`src/data_tools/comprehensive_feature_engineering.py`**
   - 在 `get_feature_columns()` 中排除原始 `zigzag`

### 特征计算逻辑

```python
# 1. 归一化
zigzag_normalized = (zigzag / close) - 1.0

# 2. 距离（ATR 归一化）
zigzag_distance = (close - zigzag) / ATR

# 3. 转折点
zigzag_turn = (zigzag.diff() * zigzag.diff().shift(1) < 0).astype(float)

# 4. 斜率（ATR 归一化）
zigzag_slope = (zigzag.diff(5) / 5) / ATR
```

## 预期效果

1. **特征选择**: zigzag 衍生特征应该出现在 top factors 中
2. **模型性能**: 结构特征与其他特征组合，提升模型预测能力
3. **可解释性**: zigzag 特征提供结构确认，增强模型可解释性

## 注意事项

1. **特征工程顺序**: zigzag 衍生特征依赖于 `zigzag` 和 `atr`，确保这些基础特征已计算
2. **数据质量**: zigzag 对数据质量敏感，确保 OHLC 数据准确
3. **计算成本**: zigzag 计算相对简单，但衍生特征会增加特征数量

## 后续优化方向

1. **多周期 ZigZag**: 计算不同时间周期的 zigzag，捕捉不同尺度的结构
2. **ZigZag 模式识别**: 识别常见的 zigzag 模式（如双顶、双底）
3. **ZigZag 与订单流结合**: 结合 CVD 和订单流数据，增强结构确认

