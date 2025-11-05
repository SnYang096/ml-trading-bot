# 多时间窗口标签创建指南

## 概述

当前系统默认使用**单一时间窗口**的收益预测（通常为1根或3根K线），但实际交易中，不同的特征可能在**不同时间窗口**下表现不同。支持**多时间窗口**（1、5、10、15根K线）可以让系统发现：

1. **短期特征**: 在1根K线上表现好的特征（捕捉短期波动）
2. **中期特征**: 在5-10根K线上表现好的特征（捕捉回调再启动模式）
3. **长期特征**: 在15根K线上表现好的特征（捕捉趋势延续）

## 当前实现的问题

### 问题 1: 单一时间窗口

**当前实现**:
- `dimensionality_comparison.py`: 使用未来**1根**K线收益
  ```python
  y = df_features["close"].pct_change().shift(-1).dropna().values
  ```
- `rolling_data.py`: 使用未来**3根**K线收益（默认）
  ```python
  def create_labels(df, *, forward_bars: int = 3, threshold: float = 0.005):
      df["future_return"] = df["close"].shift(-forward_bars) / df["close"] - 1
  ```

**问题**:
- 只能发现单一时间窗口下的特征表现
- 可能错过在某些时间窗口下表现很好的特征（如回调再启动模式）

### 问题 2: 无法对比不同时间窗口

**场景**:
- 某个特征在1根K线上表现不好（R² = 0.01）
- 但在15根K线上表现很好（R² = 0.15）
- 这可能是"回调再启动"模式：短期回调，长期趋势向好

**当前问题**:
- 无法发现这种差异
- 可能丢弃了有用的长期特征

## 解决方案

### 方案 1: 多时间窗口训练（推荐）

**实现方式**:
1. 为每个时间窗口（1、5、10、15根K线）创建独立的标签
2. 为每个时间窗口训练独立的模型
3. 评估每个时间窗口的模型性能
4. 选择表现最好的时间窗口（或组合多个时间窗口）

**优点**:
- 可以发现不同时间窗口下的特征表现
- 可以选择最适合的时间窗口
- 可以组合多个时间窗口的预测

**缺点**:
- 训练时间增加（需要训练多个模型）
- 需要更多的计算资源

### 方案 2: 多时间窗口标签（当前建议）

**实现方式**:
1. 为每个时间窗口（1、5、10、15根K线）创建独立的标签列
2. 训练单个模型，但使用多个标签（多任务学习）
3. 评估每个时间窗口的预测性能

**优点**:
- 只需要训练一个模型
- 可以发现不同时间窗口下的特征表现

**缺点**:
- 需要修改模型架构（支持多任务学习）
- 可能需要更复杂的训练流程

### 方案 3: 多时间窗口特征对比（最简单）

**实现方式**:
1. 为每个时间窗口（1、5、10、15根K线）创建独立的标签
2. 分别训练每个时间窗口的模型
3. 在 `dim-compare` 中对比不同时间窗口的特征表现
4. 选择表现最好的时间窗口用于生产

**优点**:
- 实现简单，不需要修改模型架构
- 可以直接对比不同时间窗口的表现
- 可以选择最适合的时间窗口

**缺点**:
- 训练时间增加（需要训练多个模型）

## 推荐实现：方案 3（多时间窗口特征对比）

### 实现步骤

#### 步骤 1: 修改标签创建函数

修改 `create_labels` 函数，支持多个时间窗口：

```python
def create_labels_multi_horizon(
    df: pd.DataFrame,
    *,
    horizons: list[int] = [1, 5, 10, 15],
    threshold: float = 0.005
) -> pd.DataFrame:
    """Create future-return based labels for multiple horizons.
    
    Args:
        df: DataFrame with OHLCV data
        horizons: List of forward bars to look ahead (e.g., [1, 5, 10, 15])
        threshold: Threshold for signal classification
    
    Returns:
        DataFrame with multiple label columns (e.g., signal_1, signal_5, signal_10, signal_15)
    """
    df = df.copy()
    
    for horizon in horizons:
        # Create future return for this horizon
        future_return_col = f"future_return_{horizon}"
        df[future_return_col] = df["close"].shift(-horizon) / df["close"] - 1
        
        # Create signal for this horizon
        signal_col = f"signal_{horizon}"
        df[signal_col] = 0
        df.loc[df[future_return_col] > threshold, signal_col] = 1
        df.loc[df[future_return_col] < -threshold, signal_col] = -1
        
        # Create binary signal for this horizon
        binary_signal_col = f"binary_signal_{horizon}"
        df[binary_signal_col] = (df[signal_col] == 1).astype(int)
    
    return df
```

#### 步骤 2: 修改训练流程

修改 `dim-compare` 和 `auto-rolling-update`，支持多时间窗口训练：

```python
# 在 dim-compare 中，对每个时间窗口分别训练和评估
for horizon in [1, 5, 10, 15]:
    # 创建该时间窗口的标签
    df_horizon = create_labels_multi_horizon(df, horizons=[horizon])
    y_horizon = df_horizon[f"binary_signal_{horizon}"].values
    
    # 训练模型
    model_horizon = train_lightgbm_model(X_train, y_horizon)
    
    # 评估模型
    perf_horizon = evaluate_model_performance(model_horizon, X_test, y_horizon)
    
    # 记录结果
    results[f"horizon_{horizon}"] = perf_horizon
```

#### 步骤 3: 对比不同时间窗口的表现

在 `dim-compare` 的报告中，添加多时间窗口对比表：

```
时间窗口对比表:
┌─────────┬─────────┬─────────┬─────────┬─────────┐
│ 时间窗口│   R²     │  RMSE   │   MAE   │ Sharpe  │
├─────────┼─────────┼─────────┼─────────┼─────────┤
│ 1根K线  │  0.01    │  0.02   │  0.01   │  0.5    │
│ 5根K线  │  0.05    │  0.03   │  0.02   │  0.8    │
│ 10根K线 │  0.12    │  0.04   │  0.03   │  1.2    │
│ 15根K线 │  0.15    │  0.05   │  0.04   │  1.5    │ ← 最佳
└─────────┴─────────┴─────────┴─────────┴─────────┘
```

## 使用场景示例

### 场景：发现回调再启动模式

**场景描述**:
- 某个特征在1根K线上表现不好（R² = 0.01）
- 但在15根K线上表现很好（R² = 0.15）
- 这可能是"回调再启动"模式：短期回调，长期趋势向好

**实现方式**:
```bash
# 使用多时间窗口进行 dim-compare
make dim-compare \
  SYMBOL=BTCUSDT \
  START_DATE=2025-04-01 \
  END_DATE=2025-07-31 \
  ENCODING_GRID="16,32,64" \
  DIM_COMPARE_ARGS="--horizons 1,5,10,15"
```

**输出**:
- 每个时间窗口的特征选择结果
- 每个时间窗口的模型性能
- 多时间窗口对比报告

## 参数配置

### 新增参数

```bash
# 在 Makefile 中添加
HORIZONS ?= 1,5,10,15  # 默认使用4个时间窗口

# 在 dim-compare 中使用
make dim-compare \
  SYMBOL=BTCUSDT \
  START_DATE=2025-04-01 \
  END_DATE=2025-07-31 \
  HORIZONS=1,5,10,15
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `HORIZONS` | `1,5,10,15` | 时间窗口列表（逗号分隔） |
| `--horizons` | `1,5,10,15` | 命令行参数格式 |

## 实现优先级

### 优先级 1: 修改 `create_labels` 函数（必需）

**文件**: `src/ml_trading/data_tools/rolling_data.py`

**修改内容**:
- 添加 `create_labels_multi_horizon` 函数
- 支持多个时间窗口的标签创建

### 优先级 2: 修改 `dim-compare`（必需）

**文件**: `src/ml_trading/pipeline/dimensionality/dimensionality_comparison.py`

**修改内容**:
- 添加 `--horizons` 参数
- 为每个时间窗口分别训练和评估
- 在报告中添加多时间窗口对比

### 优先级 3: 修改 `auto-rolling-update`（可选）

**文件**: `scripts/rolling/auto_rolling_update.py`

**修改内容**:
- 支持多时间窗口训练
- 可以选择最适合的时间窗口用于生产

## 预期效果

### 效果 1: 发现不同时间窗口的特征表现

**示例**:
- 特征A在1根K线上：R² = 0.01（表现不好）
- 特征A在15根K线上：R² = 0.15（表现很好）
- **结论**: 特征A适合长期预测，不适合短期预测

### 效果 2: 选择最适合的时间窗口

**示例**:
- 时间窗口1根：整体性能不好（R² = 0.05）
- 时间窗口15根：整体性能最好（R² = 0.15）
- **结论**: 选择15根K线作为生产模型的时间窗口

### 效果 3: 发现回调再启动模式

**示例**:
- 某些特征在短期（1根）表现不好
- 但在中长期（10、15根）表现很好
- **结论**: 可能是回调再启动模式，适合中长期策略

## 下一步行动

1. **修改 `create_labels` 函数**: 支持多个时间窗口
2. **修改 `dim-compare`**: 添加多时间窗口对比
3. **修改 `auto-rolling-update`**: 支持多时间窗口训练
4. **更新文档**: 添加多时间窗口使用说明

