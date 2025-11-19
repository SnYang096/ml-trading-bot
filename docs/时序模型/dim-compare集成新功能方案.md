# dim-compare 集成新 Rank IC 训练功能方案

## 当前 dim-compare 的状态

### 现有功能
1. **标签类型**：使用 `create_labels_multi_horizon` 创建分类标签（Long/Short/Hold）
2. **训练方式**：使用 `train_production_lightgbm` 进行多分类训练
3. **评估指标**：R², RMSE, Accuracy, F1-score 等
4. **数据可用性**：
   - ✅ 有 `future_return` 原始数据（在 `df_features` 中）
   - ✅ 有价格数据（`close`）
   - ✅ 支持多资产（`_symbol` 列）

### 可以集成的功能

## 方案 1：添加 Rank IC 评估（最简单，推荐）

在现有训练流程中添加 Rank IC 评估，不改变训练方式。

### 实现方式

在 `train_production_lightgbm` 调用时传入 `future_return` 数据：

```python
# 在 dimensionality_comparison.py 中
# 1. 保存原始 future_return 数据
future_return_train = df_features.loc[df_features.index[train_indices], f"future_return_{horizon}"].values
future_return_val = df_features.loc[df_features.index[val_indices], f"future_return_{horizon}"].values

# 2. 训练时传入 future_return
model_all = train_production_lightgbm(
    X_train_all,
    y_train,
    X_val_all,
    y_val,
    feature_names=keep_all,
    y_train_true_return=future_return_train,  # 新增
    y_val_true_return=future_return_val,      # 新增
)
```

### 优势
- ✅ 最小改动，不影响现有流程
- ✅ 立即获得 Rank IC 评估
- ✅ 向后兼容

### 需要修改的文件
- `dimensionality_comparison.py`: 在训练调用时传入 `future_return`

---

## 方案 2：添加可选的 Rank IC 训练模式（中等复杂度）

添加命令行参数，允许选择使用 Rank IC 训练或传统分类训练。

### 实现方式

```python
# 在 dimensionality_comparison.py 中添加参数
parser.add_argument(
    "--use-rank-ic-training",
    action="store_true",
    help="Use Rank IC-optimized training (regression with volatility normalization)",
)

# 在训练部分
if args.use_rank_ic_training:
    # 使用新的 Rank IC 训练流程
    from time_series_model.pipeline.training.rank_ic_trainer import (
        prepare_rank_ic_labels,
        train_rank_ic_model,
        generate_ensemble_signals,
    )
    
    # 准备 Rank IC 标签
    df_with_labels = prepare_rank_ic_labels(
        df_features,
        price_col="close",
        asset_col="_symbol" if "_symbol" in df_features.columns else None,
        date_col=df_features.index.name if isinstance(df_features.index, pd.DatetimeIndex) else None,
        hold_period=horizon,
    )
    
    # 训练 Rank IC 模型
    models, avg_ic, cv_results = train_rank_ic_model(
        df_with_labels,
        feature_cols=feature_names,
        date_col=...,
        n_splits=5,
    )
else:
    # 使用传统分类训练（现有流程）
    model_all = train_production_lightgbm(...)
```

### 优势
- ✅ 保留现有功能
- ✅ 可以选择使用新功能
- ✅ 可以对比两种方法的效果

### 需要修改的文件
- `dimensionality_comparison.py`: 添加参数和条件分支

---

## 方案 3：完全迁移到 Rank IC 训练（最大改动）

将 dim-compare 完全迁移到 Rank IC 训练流程。

### 实现方式

1. 替换标签准备：使用 `prepare_rank_ic_labels`
2. 替换训练：使用 `train_rank_ic_model`
3. 替换评估：使用 `evaluate_model_performance`
4. 添加信号生成：使用 `generate_ensemble_signals`

### 优势
- ✅ 使用最新的最佳实践
- ✅ 更好的评估指标（Rank IC）
- ✅ 置信度过滤信号

### 劣势
- ❌ 需要大量修改
- ❌ 可能影响现有报告生成
- ❌ 需要重新测试

---

## 推荐方案：方案 1 + 方案 2 的组合

### 阶段 1：立即添加 Rank IC 评估（方案 1）
- 在现有训练中添加 Rank IC 计算
- 在报告中显示 Rank IC 指标
- 不影响现有流程

### 阶段 2：添加可选 Rank IC 训练模式（方案 2）
- 添加 `--use-rank-ic-training` 参数
- 允许用户选择使用新功能
- 可以对比两种方法

### 阶段 3：根据效果决定是否完全迁移（方案 3）
- 如果 Rank IC 训练效果明显更好，考虑完全迁移
- 否则保持两种模式可选

---

## 具体实现步骤（方案 1）

### 1. 修改 `dimensionality_comparison.py`

在 `run_dimensionality_comparison` 函数中：

```python
# 在数据加载后，保存 future_return 数据
future_return_col = f"future_return_{horizon}"
if future_return_col in df_features.columns:
    future_return_all = df_features[future_return_col].values
    future_return_train = future_return_all[train_indices]
    future_return_val = future_return_all[val_indices]
    future_return_test = future_return_all[test_indices]
else:
    future_return_train = None
    future_return_val = None
    future_return_test = None

# 在训练时传入
model_all = train_production_lightgbm(
    X_train_all,
    y_train,
    X_val_all,
    y_val,
    feature_names=keep_all,
    y_train_true_return=future_return_train,
    y_val_true_return=future_return_val,
)
```

### 2. 在报告中显示 Rank IC

在 `evaluate_model_performance` 或报告生成中添加 Rank IC 显示。

---

## 检查清单

### 数据可用性
- [x] `future_return` 数据可用（在 `df_features` 中）
- [x] 价格数据可用（`close`）
- [x] 支持多资产（`_symbol`）
- [x] 时间索引可用（`df_features.index`）

### 功能兼容性
- [x] `train_production_lightgbm` 已支持 Rank IC 评估（已添加 `y_train_true_return` 参数）
- [x] 新训练模块支持多资产
- [x] 新训练模块支持时间序列交叉验证

### 需要确认
- [ ] `df_features` 中是否有 `close` 列
- [ ] 时间索引是否为 `DatetimeIndex`
- [ ] 是否需要保持向后兼容（现有报告格式）

---

## 使用建议

1. **立即实施**：方案 1（添加 Rank IC 评估）
   - 最小改动，立即获得 Rank IC 指标
   - 可以评估现有模型的实际预测能力

2. **短期实施**：方案 2（可选 Rank IC 训练）
   - 允许用户选择使用新功能
   - 可以对比两种方法的效果

3. **长期考虑**：根据效果决定是否完全迁移

