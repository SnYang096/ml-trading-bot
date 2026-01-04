# Rank IC 训练模块使用指南

## 概述

新的 `rank_ic_trainer.py` 模块实现了文档中推荐的最佳实践：
- ✅ 波动率标准化目标（Sharpe-like target）
- ✅ 历史分位数标签（用于评估和信号生成）
- ✅ 可交易掩码（过滤低质量样本）
- ✅ 趋势强度作为样本权重
- ✅ Rank IC（Spearman correlation）作为核心评估指标
- ✅ 时间序列交叉验证（防止时间泄漏）
- ✅ 置信度过滤信号生成

## 快速开始

### 1. 准备数据

```python
import pandas as pd
from time_series_model.pipeline.training.rank_ic_trainer import prepare_rank_ic_labels

# 假设你有包含价格数据的 DataFrame
df = your_data.copy()  # 包含 'close', 'date', 可选 'symbol'

# 准备 Rank IC 优化的标签
df = prepare_rank_ic_labels(
    df,
    price_col="close",
    asset_col="symbol",  # 多资产时使用
    date_col="date",
    hold_period=5,  # 持有期
    lookback_window=60,  # 历史窗口
    vol_mult=0.5,  # 波动率阈值倍数
)

# 现在 df 包含：
# - future_return: 原始未来收益
# - rolling_vol: 滚动波动率
# - volatility_normalized_target: 波动率标准化目标（用于训练）
# - return_quantile: 历史分位数标签（用于评估）
# - tradable: 可交易掩码（过滤低质量样本）
# - trend_strength: 趋势强度（用作样本权重）
# - momentum: 动量特征（自动计算）
```

### 2. 训练模型

```python
from time_series_model.pipeline.training.rank_ic_trainer import train_rank_ic_model

# 定义特征列
feature_cols = ["feature1", "feature2", "momentum", "rolling_vol", ...]

# 训练 Rank IC 优化模型
models, avg_rank_ic, results_df = train_rank_ic_model(
    df,
    feature_cols=feature_cols,
    target_col="volatility_normalized_target",
    tradable_col="tradable",
    weight_col="trend_strength",  # 使用趋势强度作为样本权重
    date_col="date",
    n_splits=5,  # 5折时间序列交叉验证
    filter_high_confidence=True,  # 只训练高置信度样本
    min_trend_strength=1.0,  # 最小趋势强度阈值
    smooth_target=False,  # 可选：平滑目标变量
)

print(f"Average Rank IC: {avg_rank_ic:.4f}")
# Rank IC > 0.03: 优秀
# Rank IC > 0.02: 良好
# Rank IC < 0.01: 需要改进特征
```

### 3. 生成交易信号

```python
from time_series_model.pipeline.training.rank_ic_trainer import generate_ensemble_signals

# 使用集成模型生成信号
df_signals = generate_ensemble_signals(
    df,
    models=models,
    feature_cols=feature_cols,
    confidence_threshold=0.85,  # 只交易高置信度信号
    long_threshold=0.9,  # Long 信号阈值
    short_threshold=0.1,  # Short 信号阈值
    asset_col="symbol",  # 多资产时使用
)

# df_signals 现在包含：
# - pred: 集成预测值
# - pred_quantile: 预测分位数
# - confidence_score: 置信度评分
# - signal: 交易信号 (1=Long, -1=Short, 0=Hold)
```

## 核心功能说明

### 1. 波动率标准化目标

```python
target = future_return / rolling_vol
```

**优势**：
- 自动适应高/低波动环境
- 目标分布更接近正态分布
- 模型更容易学习

### 2. 历史分位数标签

```python
return_quantile = (historical_returns < current_return).mean()
```

**用途**：
- 自适应评估（什么是"强信号"取决于历史上下文）
- 信号生成（只在极端分位数交易）

### 3. 可交易掩码

```python
tradable = (|future_return| > vol_mult * rolling_vol) & 
           (return_quantile in [0.1, 0.9])
```

**过滤**：
- 噪声交易（弱信号）
- 极端异常值（可能是数据错误）

### 4. 趋势强度权重

```python
trend_strength = |momentum| / rolling_vol
```

**效果**：
- 给强趋势样本更高权重
- 模型专注学习可预测模式，而非噪声

### 5. Rank IC 评估

```python
rank_ic = spearman_correlation(predictions, true_returns)
```

**优势**：
- 关注排序能力，而非绝对值
- 即使预测值不准，只要排序对，策略也能盈利

## 与现有代码的集成

### 选项1：在新 pipeline 中使用

创建新的训练脚本，使用 `rank_ic_trainer` 模块。

### 选项2：在现有 pipeline 中集成

在 `dimensionality_comparison.py` 或 `train.py` 中：
1. 使用 `prepare_rank_ic_labels` 准备标签
2. 使用 `train_rank_ic_model` 训练模型
3. 使用 `generate_ensemble_signals` 生成信号

### 选项3：渐进式迁移

保持现有分类模型训练流程不变，逐步引入 Rank IC 评估：
- 在现有训练函数中添加 Rank IC 计算
- 使用新的标签生成函数作为可选功能
- 逐步迁移到 Rank IC 优化训练

## 最佳实践

1. **评估指标**：关注 Rank IC，而非 MSE/R²
2. **信号生成**：只依赖预测排序，不依赖绝对值
3. **样本过滤**：使用可交易掩码和趋势强度过滤
4. **多模型集成**：使用多个 CV fold 模型的平均预测
5. **置信度过滤**：只交易高置信度信号（confidence_score >= 0.85）

## 预期效果

根据文档，使用这些改进后：
- **交易次数**：降低 30-60%（只做高置信信号）
- **胜率**：从 ~52% 提升到 ~58-62%
- **最大回撤**：明显缩小
- **夏普比率**：提升 15-30%

## 4. 模型评估

```python
from time_series_model.pipeline.training.rank_ic_trainer import evaluate_model_performance

# 评估模型性能（包含分位数分布分析和置信度统计）
evaluation_results = evaluate_model_performance(
    df_signals,
    signals=df_signals["signal"],
    return_quantile_col="return_quantile",
    pred_quantile_col="pred_quantile",
    confidence_col="confidence_score",
    true_return_col="future_return",
    confidence_threshold=0.85,
)

# 输出包含：
# - quantile_distribution: 分位数分布分析
#   - return_quantile 和 pred_quantile 的统计信息
#   - 分布均匀性评分
#   - 相关性分析
# - confidence_statistics: 置信度统计
#   - 高置信度信号比例
#   - 胜率、夏普比率、最大回撤
#   - Long/Short 分别的统计
```

### 分位数分布分析

评估标签质量和预测分布：
- **均匀性评分**：接近 1.0 表示分布均匀（理想）
- **偏度和峰度**：检测分布偏差
- **相关性**：return_quantile 和 pred_quantile 的相关性

### 置信度统计

评估高置信度信号的交易表现：
- **胜率**：高置信度信号的盈利比例
- **夏普比率**：风险调整后的收益
- **最大回撤**：最大峰值到谷值的跌幅
- **Long/Short 分别统计**：多空分别的表现

## 5. 自动确保特征包含 volatility

```python
# prepare_rank_ic_labels 会自动检查并计算 volatility 特征
df = prepare_rank_ic_labels(
    df,
    price_col="close",
    ensure_volatility=True,  # 默认 True，自动确保 volatility 特征存在
)

# 如果 volatility 特征不存在，会自动从价格数据计算
# 使用 rolling_rms_volatility 函数
```

## 下一步

1. ✅ 在现有 pipeline 中集成这些功能
2. ✅ 添加分位数分布分析和置信度统计
3. ✅ 支持多资产面板数据（已支持）
4. ✅ 确保向后兼容现有分类模型训练流程
5. ✅ 自动确保特征包含 volatility

