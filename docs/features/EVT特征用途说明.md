# EVT特征用途说明

## 核心定位

**EVT特征用于风险管理/仓位控制，不用于波动率预测**

## 为什么EVT不用于波动率预测？

### 1. 概念差异

| 特征类型 | 描述内容 | 预测目标 |
|---------|---------|---------|
| **波动率特征** | 描述价格波动的**平均水平** | 预测未来波动率水平（如ATR、GARCH） |
| **EVT特征** | 描述**极端事件**的概率和严重程度 | 预警黑天鹅风险（尾部风险） |

### 2. 理论依据

- **波动率预测**：关注的是"正常波动"的平均水平
  - GARCH模型：预测条件波动率（σ²）
  - ATR：历史波动率的移动平均
  - 扩展波动率特征：历史波动率、滞后特征、趋势特征

- **EVT特征**：关注的是"极端事件"的概率
  - `evt_tail_shape (ξ)`：尾部形状参数，描述极端事件的概率分布
  - `evt_var_99`：99% VaR，极端损失的下界
  - `evt_es_99`：99% Expected Shortfall，极端损失的期望值

### 3. 实际应用

**EVT特征的正确用途**：

1. **控制离场**：
   - 当 `evt_tail_shape > 0.5` 时，表示尾部风险突增
   - 提前离场，避免黑天鹅事件

2. **不加仓**：
   - 当尾部风险高时，避免在极端市场条件下加仓
   - 降低组合风险暴露

3. **仓位调整**：
   - 根据 `evt_tail_shape` 动态调整仓位大小
   - 高风险时降低仓位，低风险时正常仓位

## 波动率模型应该使用什么特征？

### 推荐特征

1. **GARCH特征**（关键）：
   - `garch_volatility`：预测的波动率
   - `garch_persistence`：波动持续性
   - `garch_leverage_gamma`：杠杆效应

2. **扩展波动率特征**（关键）：
   - `vol_historical_*`：历史波动率
   - `vol_lag_*`：波动率滞后特征
   - `vol_trend_*`：波动率趋势特征
   - `vol_ma_*`：波动率移动平均

3. **ATR相关特征**：
   - `atr`：平均真实波幅
   - `atr_ratio`：ATR比率
   - `atr_*`：ATR的统计特征

### 不推荐特征

- ❌ **EVT特征**：描述极端事件概率，而非波动率水平
- ❌ **DTW特征**：用于形态匹配，不用于波动率预测

## 代码实现

### 波动率模型特征选择

```python
# ✅ 正确：选择波动率相关特征
volatility_relevant_features = []

# GARCH特征（关键）
garch_features = [col for col in X.columns if col.startswith("garch_")]
volatility_relevant_features.extend(garch_features)

# 扩展波动率特征（关键）
extended_vol_features = [col for col in X.columns if col.startswith("vol_")]
volatility_relevant_features.extend(extended_vol_features)

# ATR相关特征
atr_features = [col for col in X.columns if "atr" in col.lower()]
volatility_relevant_features.extend(atr_features)

# ❌ 错误：不要包含EVT特征
# evt_features = [col for col in X.columns if col.startswith("evt_")]
# volatility_relevant_features.extend(evt_features)  # 不要这样做！
```

### 风险管理使用EVT特征

```python
# ✅ 正确：在风险管理逻辑中使用EVT特征
def should_exit_position(evt_tail_shape: float, threshold: float = 0.5) -> bool:
    """根据EVT特征决定是否离场"""
    if evt_tail_shape > threshold:
        return True  # 尾部风险高，提前离场
    return False

def should_add_position(evt_tail_shape: float, threshold: float = 0.4) -> bool:
    """根据EVT特征决定是否加仓"""
    if evt_tail_shape > threshold:
        return False  # 尾部风险高，不加仓
    return True
```

## 总结

| 特征类型 | 用途 | 模型 |
|---------|------|------|
| **GARCH特征** | 波动率预测 | 波动率模型 ✅ |
| **扩展波动率特征** | 波动率预测 | 波动率模型 ✅ |
| **ATR特征** | 波动率预测 | 波动率模型 ✅ |
| **EVT特征** | 风险管理/仓位控制 | 风险管理逻辑 ✅ |
| **DTW特征** | 形态匹配 | 策略模型 ✅ |

**关键原则**：
- 波动率模型预测"正常波动"的平均水平
- EVT特征预警"极端事件"的概率
- 两者目标不同，不应混用

