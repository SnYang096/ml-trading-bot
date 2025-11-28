# Hurst 特征 CVD 和 Volume 配置说明

## 📋 概述

Hurst 特征现在支持三个维度的信号：
1. **价格 Hurst** (`hurst_price_rolling`): 价格收益率的滚动 Hurst
2. **CVD Hurst** (`hurst_cvd_rolling`): CVD 单期变化的滚动 Hurst
3. **Volume Hurst** (`hurst_volume_rolling`): 成交量收益率的滚动 Hurst

## 🔧 特征配置

### 1. 特征定义

在 `config/feature_dependencies.yaml` 中定义了三个独立的特征：

#### `hurst_price`
- **输出列**: `hurst_price_rolling`
- **必需列**: `["close"]`
- **用途**: 捕捉价格收益率的持续性（趋势 vs 均值回复）

#### `hurst_cvd`
- **输出列**: `hurst_cvd_rolling`
- **必需列**: `["close", "cvd"]`
- **用途**: 捕捉资金流的持续性，识别价格与资金流的背离

#### `hurst_volume`
- **输出列**: `hurst_volume_rolling`
- **必需列**: `["close", "volume"]`
- **用途**: 捕捉成交量的持续性，识别量价背离

### 2. 策略配置

#### SR Breakout 策略
```yaml
requested_features:
  - hurst_price
  - hurst_cvd  # CVD 资金流的持续性（识别假突破）
```

**用途**: 
- 价格 Hurst 识别趋势持续性
- CVD Hurst 识别资金流持续性，与价格 Hurst 结合识别假突破

#### Trend Following 策略
```yaml
requested_features:
  - hurst_price
  - hurst_cvd  # CVD 资金流的持续性（确认趋势质量）
  - hurst_volume  # 成交量的持续性（确认趋势质量）
```

**用途**:
- 价格 Hurst 识别趋势持续性
- CVD Hurst 确认资金流支持趋势
- Volume Hurst 确认成交量支持趋势
- 三者结合，提高趋势信号质量

#### SR Reversal 策略
```yaml
requested_features:
  - hurst_price
  - hurst_cvd  # CVD 资金流的持续性（识别反转信号）
```

**用途**:
- 价格 Hurst 识别趋势持续性
- CVD Hurst 识别资金流反转，提前捕捉反转信号

## 📊 特征解释

### CVD Hurst 特征

**物理意义**:
- CVD (Cumulative Volume Delta) 是累积买卖量差，反映资金流向
- CVD Hurst 捕捉资金流的持续性

**解读**:
- **Hurst > 0.5**: 资金流持续流入/流出，趋势性强
- **Hurst < 0.5**: 资金流快速反转，均值回复
- **与价格 Hurst 结合**:
  - 价格 Hurst > 0.5 + CVD Hurst > 0.5 → 趋势确认（价格和资金流都持续）
  - 价格 Hurst > 0.5 + CVD Hurst < 0.5 → 背离信号（价格趋势但资金流反转）

### Volume Hurst 特征

**物理意义**:
- Volume Hurst 捕捉成交量的持续性

**解读**:
- **Hurst > 0.5**: 成交量持续放大/缩小，趋势性强
- **Hurst < 0.5**: 成交量快速波动，震荡市场
- **与价格 Hurst 结合**:
  - 价格 Hurst > 0.5 + Volume Hurst > 0.5 → 量价配合（趋势确认）
  - 价格 Hurst > 0.5 + Volume Hurst < 0.5 → 量价背离（无量上涨，可能假突破）

## 🎯 使用建议

### 场景 1: 趋势确认（Trend Following）

**组合使用**:
```python
# 三个 Hurst 特征都高 → 强趋势确认
if (hurst_price > 0.6 and 
    hurst_cvd > 0.6 and 
    hurst_volume > 0.6):
    # 强趋势信号，可以加仓
    signal = "strong_trend"
```

### 场景 2: 假突破识别（SR Breakout）

**组合使用**:
```python
# 价格 Hurst 高但 CVD Hurst 低 → 假突破
if (hurst_price > 0.6 and 
    hurst_cvd < 0.4):
    # 价格趋势但资金流反转，可能是假突破
    signal = "false_breakout"
```

### 场景 3: 反转信号识别（SR Reversal）

**组合使用**:
```python
# 价格 Hurst 低但 CVD Hurst 高 → 反转信号
if (hurst_price < 0.4 and 
    hurst_cvd > 0.6):
    # 价格震荡但资金流持续，可能反转
    signal = "reversal_signal"
```

## ⚙️ 技术细节

### 参数自动调整

所有 Hurst 特征都支持自动参数调整：

- **`update_freq: "auto"`**: 根据数据频率自动调整
  - 高频数据（< 15分钟）: `update_freq=5`
  - 中频数据（15分钟-1小时）: `update_freq=3`
  - 低频数据（>= 1小时）: `update_freq=1`

- **`clip_pct: "auto"`**: 根据品种波动特性自动调整
  - 高波动（>3%）: `clip_pct=0.8-1.0`
  - 中波动（1-3%）: `clip_pct=0.5-0.7`
  - 低波动（<1%）: `clip_pct=0.3`

### 计算效率

- 当同时请求多个 Hurst 特征时，函数会一次性计算所有特征
- 特征加载器会根据 `output_columns` 配置筛选需要的列
- 避免重复计算，提升效率

### 数据要求

- **CVD 特征**: 需要 `cvd` 列（Cumulative Volume Delta）
- **Volume 特征**: 需要 `volume` 列
- 如果数据中没有对应列，特征会返回 NaN

## 📚 参考

- [Hurst 特征实现文档](../../src/features/time_series/utils_hurst_features.py)
- [Hurst 特征自动参数调优说明](./Hurst特征自动参数调优说明.md)
- [特征依赖配置](../../config/feature_dependencies.yaml)

