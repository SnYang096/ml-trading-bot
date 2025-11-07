# 参数搜索原理：tune_q50_params

## 📋 概述

`tune_q50_params` 是一个**预训练参数搜索脚本**，用于找到满足 Q50 约束的最优 LightGBM 参数。

**核心目标**：找到参数使得 `Q50 loss <= max(Q10, Q90) loss`（允许 5% 容差）

---

## 🕐 使用什么时间范围的数据？

### 1. **数据来源**
- **数据目录**：`--data-dir` 指定的目录（默认：`data/parquet_data`）
- **文件格式**：`SYMBOL_YYYY-MM.parquet`（如 `BTC-USD_2024-11.parquet`）

### 2. **时间范围过滤**
```python
# 从命令行参数获取
--start YYYY-MM  # 起始月份（包含）
--end YYYY-MM    # 结束月份（包含）
```

**示例**：
```bash
make tune-q50-params \
  START_DATE=2024-11-01 \
  END_DATE=2025-04-30
```
- 会加载 `2024-11` 到 `2025-04` 之间的所有文件
- 默认最多加载 **10 个文件**（`--max-files 10`）

### 3. **数据量**
- **默认**：最多 10 个文件（约 2-3 个月数据）
- **目的**：快速搜索，不需要全部历史数据
- **建议**：使用**最近 2-3 个月**的数据，代表当前市场状态

---

## 🔧 使用什么特征？

### 1. **特征类型：Baseline Features**

```python
# 使用 BaselineFeatureEngineer
engineer = BaselineFeatureEngineer()
feat_df = engineer.engineer_features(df, fit=True)
feature_cols = get_baseline_feature_columns(feat_df)
```

**包含的特征**：
- **价格特征**：OHLC、收益率、波动率
- **技术指标**：MA、RSI、MACD、布林带等
- **时间特征**：`hour_sin`, `hour_cos`, `is_weekend`, `minutes_since_last_trade` 等
- **成交量特征**：成交量、买卖量比、CVD 等
- **约 40-50 个特征**（取决于数据列）

### 2. **特征清洗**

```python
# 使用 clean_features_train_test 进行清洗
feat_df_clean = clean_features_train_test(
    feat_df[feature_cols], 
    feat_df[feature_cols], 
    k=4.0  # Winsorize 阈值
)[0]
```

**清洗步骤**：
1. 替换 `±inf` 为 `NaN`
2. 用训练集**中位数**填充 `NaN`
3. 使用 **MAD-based Winsorize**（k=4.0）裁剪极端值

### 3. **目标变量**

```python
# 构建未来收益率
future_return = (close.shift(-forward_bars) / close - 1)
```

- **forward_bars**：预测未来多少根 K 线（默认 5）
- **对齐**：只保留特征和目标都有效的样本

---

## 🔍 搜索原理（两阶段优化）

### **阶段 1：Q50 模型优化（快速筛选）**

```python
# 使用 Optuna 搜索，只训练 Q50 模型
for trial in range(n_trials):  # 默认 50 次
    # 1. Optuna 建议一组参数
    params = {
        "num_leaves": trial.suggest_int(15, 127),
        "learning_rate": trial.suggest_float(0.001, 0.1, log=True),
        "n_estimators": trial.suggest_int(500, 2000),
        "min_data_in_leaf": trial.suggest_int(20, 200),
        # ... 其他参数
    }
    
    # 2. 使用 TimeSeriesSplit 交叉验证
    for train_idx, val_idx in cv_splits:
        # 3. 训练 Q50 模型
        model = train_q50_model(params, X_train, y_train)
        
        # 4. 计算 Q50 loss
        q50_loss = calculate_quantile_loss(y_val, y_pred, alpha=0.5)
    
    # 5. 返回平均 Q50 loss（Optuna 最小化）
    return -avg_q50_loss
```

**优化目标**：最小化 Q50 loss（不检查约束）

**搜索参数范围**：
- `num_leaves`: 15-127
- `learning_rate`: 0.001-0.1 (log scale)
- `n_estimators`: 500-2000
- `min_data_in_leaf`: 20-200
- `lambda_l1`, `lambda_l2`: 1e-8 到 10.0 (log scale)
- `feature_fraction`, `bagging_fraction`: 0.5-1.0
- `bagging_freq`: 0-7

### **阶段 2：Q50 约束验证（最终确认）**

```python
# 用找到的最佳参数，训练 Q10/Q50/Q90 三个模型
best_params = study.best_trial.params

# 训练并评估 Q10, Q50, Q90
for alpha in [0.1, 0.5, 0.9]:
    model = train_quantile_model(best_params, alpha=alpha)
    loss = calculate_quantile_loss(y_val, y_pred, alpha=alpha)
    losses[alpha].append(loss)

# 检查约束
q50_ratio = avg_q50_loss / max(avg_q10_loss, avg_q90_loss)
if q50_ratio > 1.05:  # 5% 容差
    return None  # 不满足约束，返回 None
else:
    return best_params  # 满足约束，返回参数
```

**验证条件**：
- `Q50 loss <= 1.05 × max(Q10 loss, Q90 loss)`
- 如果不满足，返回 `None`（搜索失败）

---

## 📊 交叉验证策略

### **1. 时间序列交叉验证**

```python
# 使用 TimeSeriesSplit（保持时间顺序）
cv = TimeSeriesSplit(n_splits=3)  # 默认 3 折

# 或 GroupTimeSeriesSplit（多资产时）
cv = GroupTimeSeriesSplit(n_splits=3, drop_same_group=True)
```

**特点**：
- ✅ 保持时间顺序（训练集 < 验证集）
- ✅ 避免未来信息泄露
- ✅ 多资产时，验证集的 symbol 不会出现在训练集

### **2. CV 折数**

- **默认**：3 折（快速搜索）
- **训练时**：5 折（更严格评估）

---

## ⚡ 为什么分两阶段？

### **问题**：如果每次 trial 都训练 Q10/Q50/Q90，计算量太大

**原方案**（已废弃）：
```python
# 每个 trial 训练 3 个模型 × 3 折 = 9 次训练
for trial in range(50):
    for fold in range(3):
        train_q10()  # 1 次
        train_q50()  # 1 次
        train_q90()  # 1 次
# 总计：50 × 3 × 3 = 450 次训练！太慢！
```

**新方案**（当前）：
```python
# 阶段 1：只训练 Q50（快速筛选）
for trial in range(50):
    for fold in range(3):
        train_q50()  # 1 次
# 总计：50 × 3 = 150 次训练

# 阶段 2：最终验证（只做 1 次）
train_q10()  # 3 折
train_q50()  # 3 折
train_q90()  # 3 折
# 总计：3 × 3 = 9 次训练

# 总计：150 + 9 = 159 次训练（比原方案快 2.8 倍！）
```

**优势**：
- ✅ 快速筛选：只优化 Q50，找到候选参数
- ✅ 最终验证：用候选参数训练 Q10/Q50/Q90，确认约束
- ✅ 如果验证失败，返回 `None`（不会返回不满足约束的参数）

---

## 🎯 搜索策略总结

| 阶段 | 目标 | 训练模型 | 评估指标 | 计算量 |
|------|------|----------|----------|--------|
| **阶段 1** | 最小化 Q50 loss | 只训练 Q50 | Q50 loss | 50 trials × 3 folds = 150 次 |
| **阶段 2** | 验证 Q50 约束 | 训练 Q10/Q50/Q90 | Q50 <= max(Q10, Q90) | 1 次 × 3 models × 3 folds = 9 次 |

**总计算量**：约 159 次模型训练（相比原方案 450 次，快 2.8 倍）

---

## 💡 使用建议

### **1. 数据选择**
```bash
# 使用最近 2-3 个月数据（代表当前市场状态）
make tune-q50-params \
  START_DATE=2024-11-01 \
  END_DATE=2025-01-31 \
  MAX_FILES=10
```

### **2. 搜索次数**
```bash
# 快速搜索（20-30 trials，约 10-15 分钟）
make tune-q50-params TUNE_TRIALS=30

# 深度搜索（50-100 trials，约 30-60 分钟）
make tune-q50-params TUNE_TRIALS=100
```

### **3. 参数复用**
```bash
# 搜索一次，多次使用
make tune-q50-params ...  # 生成 results/params/q50_params_*.json

# 后续训练直接使用
make train PARAMS_FILE=results/params/q50_params_*.json AUTO_TUNE=0
```

---

## ⚠️ 注意事项

1. **数据量**：默认最多 10 个文件，如果数据太少可能搜索失败
2. **时间范围**：建议使用**最近 2-3 个月**数据，不要用太老的数据
3. **特征一致性**：搜索时用 `baseline` 特征，训练时也要用 `baseline` 特征
4. **forward_bars**：搜索时的 `forward_bars` 必须与训练时一致
5. **约束验证**：如果所有 trial 都不满足约束，会返回 `None`（需要调整搜索范围或数据）

---

## 🔬 技术细节

### **1. Optuna 搜索算法**
- 使用 **TPE (Tree-structured Parzen Estimator)** 算法
- 自动学习参数分布，高效搜索

### **2. Early Stopping**
- 每个模型训练时使用 early stopping（50 轮无改善则停止）
- 防止过拟合，加速训练

### **3. 参数转换**
```python
# Optuna 返回 n_estimators，需要转换为 num_boost_round
n_est = best_params.pop("n_estimators")
best_params["num_boost_round"] = n_est
```

---

## 📈 输出示例

```
✅ Found optimal parameters:
   num_leaves: 127
   learning_rate: 0.03
   num_boost_round: 2000
   min_data_in_leaf: 10
   lambda_l2: 0.1
   ...

✅ Updated parameters: {'num_leaves': 127, 'learning_rate': 0.03, 'num_boost_round': 2000}
     Avg losses -> Q10: 0.002088, Q50: 0.000661, Q90: 0.000389, ratio=0.32

💾 Saved parameters to: results/params/q50_params_BTCUSDT_ETHUSDT_SOLUSDT_5min_5bars.json
```

**ratio < 1.0** 表示满足约束（Q50 loss < max(Q10, Q90) loss）

