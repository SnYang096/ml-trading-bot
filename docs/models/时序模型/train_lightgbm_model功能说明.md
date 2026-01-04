# train_lightgbm_model 功能说明

## 函数复杂性分析

### 当前功能（200行代码）

1. **自动任务类型检测**（~60行）
   - 检测3类分类 → 自动转换为2类分类（过滤Hold）
   - 检测2类分类 → 使用二分类
   - 检测连续值 → 使用回归

2. **分类特征处理**（~20行）
   - 支持多种输入格式（列表、单个值、索引、名称）
   - 自动映射特征名称到索引

3. **GPU/CPU 回退**（~15行）
   - 自动处理 CUDA 错误
   - 回退到 CPU 训练

4. **LightGBM 训练封装**（~50行）
   - Dataset 创建
   - 验证集处理
   - Early stopping
   - 日志输出

### 你只需要的功能

1. **三分类模型**（多空Hold）
2. **波动率模型**（回归）
3. **收益率模型**（回归）

## 简化使用方式

### 方式1：明确指定 task_type（推荐）

```python
# 三分类模型
model = train_lightgbm_model(
    X_train, y_train, X_val=X_val, y_val=y_val,
    task_type="multiclass",  # 明确指定，跳过自动检测
    params={"num_class": 3},
)

# 波动率/收益率模型（回归）
model = train_lightgbm_model(
    X_train, y_train, X_val=X_val, y_val=y_val,
    task_type="regression",  # 明确指定，跳过自动检测
    params={"objective": "regression", "metric": "rmse"},
)
```

### 方式2：直接使用 LightGBM（最简单）

如果你觉得函数太复杂，可以直接使用 LightGBM：

```python
import lightgbm as lgb

# 三分类模型
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
model = lgb.train(
    {"objective": "multiclass", "num_class": 3, "metric": "multi_logloss"},
    train_data,
    num_boost_round=4000,
    valid_sets=[train_data, val_data],
    callbacks=[lgb.early_stopping(400)],
)

# 回归模型（波动率/收益率）
model = lgb.train(
    {"objective": "regression", "metric": "rmse"},
    train_data,
    num_boost_round=4000,
    valid_sets=[train_data, val_data],
    callbacks=[lgb.early_stopping(400)],
)
```

## 为什么这么复杂？

1. **历史兼容性**：为了兼容旧代码，默认将3类转换为2类
2. **自动检测**：试图自动判断任务类型，但增加了复杂性
3. **错误处理**：GPU/CPU 回退、各种边界情况处理
4. **灵活性**：支持多种输入格式（特征名称、索引等）

## 建议

如果你只需要这三种模型，建议：

1. **使用 `task_type` 参数**：明确指定任务类型，避免自动检测
2. **或者直接使用 LightGBM**：如果函数太复杂，直接调用 `lgb.train()` 更简单

## 当前代码中的使用

在 `model_training.py` 中，我们已经为多类分类使用了直接调用 LightGBM 的方式（绕过 `train_lightgbm_model`），这样可以：
- 确保3类分类被正确保留
- 支持样本权重
- 避免自动转换的复杂性

