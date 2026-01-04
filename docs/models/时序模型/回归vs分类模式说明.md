# 回归 vs 分类模式说明

## 当前实现

**所有策略（reversal、breakout、compression_breakout）都使用 `train_rank_ic_model`，且都是 regression 模式。**

```python
# rank_ic_trainer.py 第 814 行和 836 行
default_params = {
    "objective": "regression",  # ✅ 使用回归模式
    "metric": "rmse",
    "boosting_type": "gbdt",
    ...
}
```

## 为什么使用 Regression 而不是 Binary Classification？

### 1. Rank IC 评估需要连续值

**Rank IC（Spearman 相关系数）需要连续的预测值来排序：**

```python
# Rank IC 计算
def compute_rank_ic(predictions, true_returns):
    # 需要将 predictions 和 true_returns 都转换为排名
    pred_ranks = predictions.rank()
    true_ranks = true_returns.rank()
    # 计算 Spearman 相关系数
    ic = spearmanr(pred_ranks, true_ranks)
```

**如果使用 binary classification：**
- 输出是概率值（0-1）
- 虽然也是连续值，但范围受限
- 对于 Rank IC 来说，只要值可以排序即可

**使用 regression 的优势：**
- 输出是连续值，范围不受限
- 可以更好地表达"强度"或"置信度"
- 更适合 Rank IC 评估

### 2. 标签类型支持

当前系统支持两种标签类型：

#### A. 传统标签（连续值）

```python
# future_return: 连续值（如 0.05, -0.02, 0.10）
# volatility_normalized_target: 连续值（future_return / rolling_vol）
volatility_normalized_target = future_return / rolling_vol
```

**这种标签天然适合 regression。**

#### B. R/R 标签（二元值，但作为连续值处理）

```python
# rr_achieved: 二元值（0 或 1）
# 但在 regression 模式下，模型会学习预测 0-1 之间的连续值
# 这个连续值可以理解为"成功概率"或"R/R 强度"
rr_achieved = compute_rr_label(...)  # 返回 0.0 或 1.0
volatility_normalized_target = rr_achieved  # 作为连续值使用
```

**即使标签是二元的（0/1），使用 regression 也可以：**
- 模型输出 0-1 之间的连续值（可以理解为概率）
- 这个连续值可以用于 Rank IC 评估
- 也可以用于信号生成（通过阈值）

### 3. 信号生成的灵活性

**使用 regression 输出可以更灵活地生成信号：**

```python
# Regression 输出：连续值（如 0.3, 0.7, 0.9）
# 可以通过分位数或阈值生成信号
pred_quantile = prediction_quantile(predictions, window=30)
confidence = confidence_score(predictions, window=30)

# 生成信号
long_mask = (pred_quantile > 0.8) & (confidence > 0.85)
short_mask = (pred_quantile < 0.2) & (confidence > 0.85)
```

**如果使用 binary classification：**
- 输出是概率值（0-1）
- 需要设置阈值（如 0.5）来生成信号
- 灵活性较低

## Regression vs Binary Classification 对比

| 维度 | Regression（当前） | Binary Classification |
|------|-------------------|----------------------|
| **Objective** | `"regression"` | `"binary"` |
| **输出范围** | 无限制（理论上） | 0-1（概率） |
| **标签类型** | 连续值或二元值（作为连续值处理） | 必须是二元值（0/1） |
| **Rank IC 评估** | ✅ 直接使用 | ✅ 也可以使用（概率值可排序） |
| **信号生成** | ✅ 灵活（分位数、阈值） | ⚠️ 需要阈值 |
| **可解释性** | ⚠️ 输出值含义不明确 | ✅ 输出是概率，含义明确 |
| **模型复杂度** | 中等 | 中等 |

## 当前实现的问题与改进建议

### 问题 1：R/R 标签使用 Regression 可能不是最优

**当前实现：**
```python
# R/R 标签是二元的（0/1）
rr_achieved = compute_rr_label(...)  # 返回 0.0 或 1.0

# 但使用 regression 模式训练
model = train_rank_ic_model(
    df,
    target_col="rr_achieved",  # 二元标签
    # objective="regression"  # 使用回归模式
)
```

**潜在问题：**
- Regression 假设输出是连续值，但标签是二元的
- 模型可能学习到 0-1 之间的值，但标签只有 0 和 1
- 对于二元标签，binary classification 可能更合适

### 改进建议 1：根据标签类型选择模型模式

```python
def train_rank_ic_model(
    df: pd.DataFrame,
    target_col: str = "volatility_normalized_target",
    auto_detect_objective: bool = True,  # 新增：自动检测目标类型
    ...
):
    # 自动检测目标变量类型
    if auto_detect_objective:
        target_values = df[target_col].dropna()
        unique_values = target_values.unique()
        
        # 如果只有 0 和 1，使用 binary classification
        if len(unique_values) <= 2 and set(unique_values).issubset({0.0, 1.0}):
            objective = "binary"
            metric = "binary_logloss"
        else:
            objective = "regression"
            metric = "rmse"
    else:
        objective = "regression"
        metric = "rmse"
    
    default_params = {
        "objective": objective,
        "metric": metric,
        ...
    }
```

### 改进建议 2：为 R/R 标签使用 Binary Classification

```python
# 在 train_sr_reversal_model.py 中
# 如果使用 R/R 标签，使用 binary classification
if use_risk_reward_label:
    # 使用 binary classification
    models = train_rank_ic_model(
        df_train_reversal,
        feature_cols=reversal_features,
        target_col="rr_reversal_achieved",  # 二元标签
        lgbm_params={
            "objective": "binary",  # 使用二分类
            "metric": "binary_logloss",
        }
    )
else:
    # 使用 regression（传统标签）
    models = train_rank_ic_model(
        df_train_reversal,
        feature_cols=reversal_features,
        target_col="volatility_normalized_target",  # 连续标签
        # objective="regression"  # 默认
    )
```

### 改进建议 3：保持 Regression，但优化评估

**如果继续使用 regression：**

```python
# 模型输出连续值（0-1 之间，可以理解为概率）
predictions = model.predict(X_test)  # 输出如 [0.3, 0.7, 0.9, 0.2]

# 对于 Rank IC，直接使用（因为可以排序）
rank_ic = compute_rank_ic(predictions, rr_achieved)

# 对于信号生成，使用阈值
long_mask = predictions > 0.6  # 阈值可调
short_mask = predictions < 0.4
```

**优点：**
- 保持 Rank IC 评估的一致性
- 输出值可以表达"强度"或"置信度"
- 信号生成更灵活

**缺点：**
- 输出值含义不如概率明确
- 对于二元标签，可能不是最优选择

## 实际影响分析

### 当前实现的影响

1. **R/R 标签使用 Regression：**
   - ✅ Rank IC 评估正常（连续值可以排序）
   - ✅ 信号生成正常（通过阈值）
   - ⚠️ 模型可能学习到 0-1 之间的值，但标签只有 0 和 1
   - ⚠️ 对于二元标签，binary classification 可能更合适

2. **传统标签使用 Regression：**
   - ✅ 完全合适（标签是连续值）
   - ✅ Rank IC 评估正常
   - ✅ 信号生成灵活

### 性能对比（理论）

| 场景 | Regression | Binary Classification |
|------|-----------|---------------------|
| **连续标签** | ✅ 最优 | ⚠️ 可以但非最优 |
| **二元标签（R/R）** | ⚠️ 可以但非最优 | ✅ 最优 |
| **Rank IC 评估** | ✅ 正常 | ✅ 正常（概率值可排序） |
| **信号生成** | ✅ 灵活 | ⚠️ 需要阈值 |

## 总结

### 当前状态

**所有策略都使用 `train_rank_ic_model`，且都是 regression 模式：**

- ✅ **优点**：统一、灵活、适合 Rank IC 评估
- ⚠️ **潜在问题**：对于 R/R 标签（二元），binary classification 可能更合适

### 建议

1. **短期（保持现状）**：
   - 继续使用 regression 模式
   - 通过阈值生成信号
   - 监控模型输出分布（是否集中在 0-1 之间）

2. **中期（优化）**：
   - 根据标签类型自动选择模型模式
   - R/R 标签使用 binary classification
   - 传统标签使用 regression

3. **长期（实验）**：
   - 对比 regression 和 binary classification 的表现
   - 选择最优方案

### 关键点

**即使使用 regression 模式，对于 R/R 标签（二元）也是可行的：**
- 模型输出可以理解为"成功概率"或"R/R 强度"
- Rank IC 评估正常（连续值可以排序）
- 信号生成正常（通过阈值）

**但 binary classification 可能更合适：**
- 输出是概率，含义更明确
- 对于二元标签，理论上更匹配
- 可以使用概率阈值生成信号

