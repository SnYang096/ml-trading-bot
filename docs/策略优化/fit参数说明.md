# `fit` 参数说明

## `fit=True` 的作用

`fit` 参数在特征工程中遵循标准的 **fit/transform 模式**（类似 scikit-learn）：

### 1. **当前实现中的使用情况**

**大部分特征计算函数不使用 `fit` 参数**：
- 如 `extract_wpt_features`, `extract_order_flow_features`, `compute_rsi` 等
- 这些函数只是基于当前和历史数据计算特征，不需要"拟合"

**少数特征可能需要 `fit` 参数**：
- 归一化特征（`normalize_features_unified`）：需要 `fit=True` 计算统计量（均值、标准差）
- DL序列特征（`DeepLearningSequenceExtractor`）：有 `fit()` 和 `transform()` 方法
- 未来可能需要的特征（如 PCA、特征选择等）

### 2. **为什么不能简单去掉？**

虽然当前大部分特征不使用 `fit` 参数，但保留它是有意义的：

1. **接口一致性**：符合标准的 fit/transform 模式，便于理解和维护
2. **未来扩展性**：为需要拟合的特征预留接口（如标准化、PCA等）
3. **缓存控制**：虽然当前缓存不依赖 `fit`，但未来可能需要区分 fit/transform 的缓存

### 3. **正确的使用方式**

```python
# ✅ 正确：先划分训练/测试集，然后分别处理
split_idx = int(len(df_raw) * (1 - args.test_size))
df_raw_train = df_raw.iloc[:split_idx].copy()
df_raw_test = df_raw.iloc[split_idx:].copy()

# 训练集：fit=True（计算统计量、初始化状态等）
df_train = run_feature_pipeline(
    df_raw_train,
    feature_loader=feature_loader,
    pipeline_cfg=strategy_cfg.features,
    fit=True,  # 在训练集上拟合
)

# 测试集：fit=False（使用训练集学到的统计量）
df_test = run_feature_pipeline(
    df_raw_test,
    feature_loader=feature_loader,
    pipeline_cfg=strategy_cfg.features,
    fit=False,  # 不在测试集上拟合！
)
```

### 4. **错误的使用方式（会导致数据泄漏）**

```python
# ❌ 错误：在整个数据集上fit，然后才划分
df_features = run_feature_pipeline(
    df_raw,  # 包含训练集和测试集
    feature_loader=feature_loader,
    pipeline_cfg=strategy_cfg.features,
    fit=True,  # 使用了测试集数据来拟合！
)

# 然后才划分
split_idx = int(len(df_features) * (1 - args.test_size))
df_train = df_features.iloc[:split_idx].copy()
df_test = df_features.iloc[split_idx:].copy()
```

## 总结

- **`fit` 参数不能去掉**：虽然当前大部分特征不使用，但保留它是为了接口一致性和未来扩展
- **问题不在 `fit` 参数本身**：问题在于**使用方式**（在划分训练/测试集之前使用了 `fit=True`）
- **正确的做法**：先划分数据集，然后在训练集上 `fit=True`，在测试集上 `fit=False`

## 相关文件

- 特征计算: `src/features/loader/parallel_computer.py`
- 修复后的代码: `scripts/diagnostics/sr_reversal_model_comparison.py` (第1348-1361行)

