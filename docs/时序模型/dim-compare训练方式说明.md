# dim-compare 训练方式说明

## 当前状态

### 训练方式：**分类训练**（不是回归）

`dim-compare` 目前使用的是**多分类训练**：
- **标签类型**：3类分类（0=Hold, 1=Long, 2=Short）
- **训练目标**：`objective='multiclass'`, `num_class=3`
- **评估指标**：Accuracy, F1-score, R², RMSE（分类任务的 R²/RMSE）

### 我刚才添加的功能：**Rank IC 评估**（不是训练方式）

我只是添加了 Rank IC **评估**功能：
- 在分类训练完成后，计算预测值与真实收益的 Rank IC
- 用于评估分类模型的预测能力
- **不改变训练方式**（仍然是分类训练）

## 什么是"向后兼容"？

"向后兼容"的意思是：
- ✅ 添加 Rank IC 评估不会破坏现有的分类训练流程
- ✅ 如果 `future_return` 数据不可用，会跳过 Rank IC 评估，但训练照常进行
- ✅ 现有的分类训练代码完全不受影响

**但是**：
- ❌ 训练方式本身**还是分类训练**，不是回归训练
- ❌ **没有使用**新的 Rank IC 训练模块（回归 + 波动率标准化）

---

## 两种训练方式的区别

### 方式 1：当前 dim-compare 使用的（分类训练）

```python
# 标签：0=Hold, 1=Long, 2=Short（分类标签）
y_train = [0, 1, 2, 0, 1, ...]

# 训练：多分类
model = train_production_lightgbm(
    X_train, y_train,  # 分类标签
    objective='multiclass',
    num_class=3,
)

# 预测：类别概率
pred_proba = model.predict(X_test)  # shape: (n_samples, 3)
pred_class = np.argmax(pred_proba, axis=1)  # 0, 1, 2

# Rank IC 评估（新增）：用预测概率计算 Rank IC
rank_ic = compute_rank_ic(pred_proba[:, 1], true_returns)  # 使用 Long 类概率
```

**特点**：
- 预测的是类别（Hold/Long/Short）
- 使用分类损失函数（multi_logloss）
- Rank IC 是**事后评估**，用于衡量分类模型的预测能力

---

### 方式 2：新的 Rank IC 训练模块（回归训练）

```python
# 标签：波动率标准化目标（回归标签）
y_train = future_return / rolling_vol  # 连续值

# 训练：回归
models, avg_ic, _ = train_rank_ic_model(
    df,
    feature_cols,
    target_col="volatility_normalized_target",  # 回归目标
    objective='regression',
)

# 预测：连续值
pred = model.predict(X_test)  # 连续值

# Rank IC：直接计算预测值与真实收益的 Rank IC
rank_ic = compute_rank_ic(pred, true_returns)
```

**特点**：
- 预测的是连续值（波动率标准化收益）
- 使用回归损失函数（rmse）
- Rank IC 是**训练目标**，模型直接优化排序能力

---

## 对比总结

| 特性 | 当前 dim-compare（分类） | 新 Rank IC 训练（回归） |
|------|------------------------|----------------------|
| **标签类型** | 分类（0/1/2） | 回归（连续值） |
| **训练目标** | multi_logloss | rmse |
| **预测输出** | 类别概率 | 连续值 |
| **Rank IC** | 事后评估 | 训练目标 |
| **波动率标准化** | ❌ | ✅ |
| **历史分位数标签** | ❌ | ✅ |
| **可交易掩码** | ❌ | ✅ |
| **趋势强度权重** | ❌ | ✅ |
| **时间序列 CV** | ❌ | ✅ |

---

## 如何让 dim-compare 使用新的 Rank IC 训练？

### 方案 1：添加可选参数（推荐）

添加 `--use-rank-ic-training` 参数，让用户选择：

```python
if args.use_rank_ic_training:
    # 使用新的 Rank IC 训练模块（回归）
    from time_series_model.pipeline.training.rank_ic_trainer import (
        prepare_rank_ic_labels,
        train_rank_ic_model,
    )
    
    # 准备 Rank IC 标签
    df_with_labels = prepare_rank_ic_labels(
        df_features_full,
        price_col="close",
        asset_col="_symbol" if "_symbol" in df_features_full.columns else None,
    )
    
    # 训练 Rank IC 模型
    models, avg_ic, _ = train_rank_ic_model(
        df_with_labels,
        feature_cols=feature_names,
    )
else:
    # 使用现有分类训练
    model_all = train_production_lightgbm(...)
```

### 方案 2：完全迁移到 Rank IC 训练

将 dim-compare 完全改为使用 Rank IC 训练（回归），但这需要：
- 修改标签生成逻辑
- 修改评估逻辑
- 修改报告生成逻辑
- 可能影响现有工作流

---

## 当前 dim-compare 的实际效果

### 分类训练 + Rank IC 评估

**优点**：
- ✅ 可以评估分类模型的预测能力
- ✅ Rank IC 可以帮助判断模型是否有预测能力
- ✅ 不影响现有流程

**局限性**：
- ❌ 模型不是直接优化 Rank IC
- ❌ 没有使用波动率标准化、历史分位数等新功能
- ❌ 分类损失函数（multi_logloss）和 Rank IC 目标不完全一致

---

## 建议

1. **当前使用**：dim-compare 的分类训练 + Rank IC 评估
   - 可以快速评估模型预测能力
   - 适合对比降维前后的效果

2. **新项目**：使用新的 Rank IC 训练模块
   - 直接优化 Rank IC
   - 使用所有新功能（波动率标准化、历史分位数等）

3. **未来**：可以考虑在 dim-compare 中添加 `--use-rank-ic-training` 参数
   - 让用户选择使用哪种训练方式
   - 可以对比两种方法的效果

