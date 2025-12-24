# 自适应R/R标签训练方案

## 📊 问题背景

用户问：**重新训练ML模型（使用自适应R/R标签）要怎么做，感觉没法做啊**

确实，这是一个**挑战**，但不是完全不可能。下面详细说明问题和解决方案。

---

## ⚠️ 核心挑战

### 问题1：训练时需要"未来波动率"

**当前情况**：
- 训练时：需要计算标签，但标签依赖于**未来波动率**（未来N期的实际波动率）
- 回测时：使用**预测波动率**（波动率模型的预测值）

**矛盾**：
- 训练时用"未来波动率" → 有未来信息，可以计算准确标签
- 回测时用"预测波动率" → 没有未来信息，只能用预测值

### 问题2：信号与标签不匹配

**当前流程**：
1. ML模型在**固定R/R**的标签上训练
2. 回测时使用**动态R/R**（基于预测波动率）

**问题**：
- ML模型学习的是"在固定R/R下，哪些信号会成功"
- 但回测时使用的是"在动态R/R下，哪些信号会成功"
- 这可能导致模型预测失效

---

## ✅ 解决方案

### 方案A：使用"未来波动率"训练（推荐，但需要小心）

**核心思想**：
- 训练时使用**未来波动率**计算标签（有未来信息，但仅用于标签）
- 回测时使用**预测波动率**计算标签（无未来信息，用预测值）

**实现步骤**：

#### 步骤1：训练时计算自适应R/R标签

```python
from src.time_series_model.pipeline.training.label_utils import (
    compute_adaptive_rr_label_with_future_vol,
)

# 训练时：使用未来波动率计算标签
labels_train = compute_adaptive_rr_label_with_future_vol(
    df_train,
    signal_col="signal",
    price_col="close",
    atr_col="atr",
    max_holding_bars=50,
    stop_loss_multiplier=1.0,
    take_profit_multiplier=2.0,
    volatility_window=10,  # 使用未来10期的波动率
    use_breakeven_stop=True,
    entry_price_col="open",
    entry_offset=1,
)
```

**关键点**：
- ✅ 使用`compute_adaptive_rr_label_with_future_vol`（有未来信息）
- ✅ 标签反映"在动态R/R下，哪些信号会成功"
- ⚠️ 标签有未来信息，但**仅用于训练**，不用于特征

#### 步骤2：训练ML模型

```python
# 训练ML模型（使用自适应R/R标签）
ml_model.fit(X_train, labels_train)
```

#### 步骤3：训练波动率模型

```python
# 训练波动率模型（预测未来波动率）
from src.time_series_model.pipeline.training.label_utils import (
    future_volatility_label,
)

# 计算未来波动率标签
future_vol_labels = future_volatility_label(
    df_train["close"],
    horizon=10,  # 预测未来10期的波动率
)

# 训练波动率模型
vol_model.fit(X_train, future_vol_labels)
```

#### 步骤4：回测时使用预测波动率

```python
# 回测时：使用预测波动率计算标签
from src.time_series_model.diagnostics.compute_adaptive_rr_with_predicted_vol import (
    compute_adaptive_rr_label_with_predicted_vol,
)

# 预测波动率
pred_vol = vol_model.predict(X_test)

# 使用预测波动率计算标签（用于评估，不用于训练）
labels_test = compute_adaptive_rr_label_with_predicted_vol(
    df_test,
    predicted_vol=pred_vol,
    signal_col="signal",
    stop_loss_multiplier=1.0,
    take_profit_multiplier=2.0,
    atr_lower_bound=0.8,
    atr_upper_bound=1.5,
    use_breakeven_stop=True,
)
```

**优点**：
- ✅ 训练和回测都使用自适应R/R，信号与标签匹配
- ✅ 模型学习的是"在动态R/R下，哪些信号会成功"

**缺点**：
- ⚠️ 训练时使用未来信息（但仅用于标签，不用于特征）
- ⚠️ 需要波动率模型预测准确

---

### 方案B：使用"预测波动率"训练（更保守，但可能不准确）

**核心思想**：
- 训练时也使用**预测波动率**（通过时间序列交叉验证）
- 回测时也使用**预测波动率**

**实现步骤**：

#### 步骤1：时间序列交叉验证

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for train_idx, val_idx in tscv.split(X):
    X_train_fold = X.iloc[train_idx]
    X_val_fold = X.iloc[val_idx]
    
    # 在训练集上训练波动率模型
    vol_model.fit(X_train_fold, future_vol_labels[train_idx])
    
    # 在验证集上预测波动率
    pred_vol_val = vol_model.predict(X_val_fold)
    
    # 使用预测波动率计算标签
    labels_val = compute_adaptive_rr_label_with_predicted_vol(
        df.iloc[val_idx],
        predicted_vol=pred_vol_val,
        ...
    )
    
    # 训练ML模型
    ml_model.fit(X_train_fold, labels_train_fold)
```

**优点**：
- ✅ 训练时也不使用未来信息（更保守）
- ✅ 更接近实盘情况

**缺点**：
- ⚠️ 训练时预测波动率可能不准确，导致标签不准确
- ⚠️ 需要更多数据（时间序列交叉验证）

---

### 方案C：混合方案（推荐用于生产）

**核心思想**：
- 训练时使用**固定R/R**（简单可靠）
- 回测时使用**动态R/R**（基于预测波动率）
- 通过**保本止损**和**动态调整**来提升效果

**实现步骤**：

#### 步骤1：训练ML模型（固定R/R）

```python
# 训练时使用固定R/R
labels_train = compute_rr_label(
    df_train,
    signal_col="signal",
    stop_loss_r=1.0,
    take_profit_r=2.0,
    use_breakeven_stop=True,  # 启用保本止损
)
```

#### 步骤2：训练波动率模型

```python
# 训练波动率模型
future_vol_labels = future_volatility_label(df_train["close"], horizon=10)
vol_model.fit(X_train, future_vol_labels)
```

#### 步骤3：回测时使用动态R/R

```python
# 回测时使用动态R/R（基于预测波动率）
pred_vol = vol_model.predict(X_test)
labels_test = compute_adaptive_rr_label_with_predicted_vol(
    df_test,
    predicted_vol=pred_vol,
    ...
)
```

**优点**：
- ✅ 训练简单可靠（固定R/R）
- ✅ 回测时动态适应市场波动
- ✅ 通过保本止损提升效果

**缺点**：
- ⚠️ 信号与标签可能不完全匹配（但通过保本止损可以缓解）

---

## 🛠️ 实际实现建议

### 推荐方案：方案A（使用未来波动率训练）

**理由**：
1. ✅ 训练和回测都使用自适应R/R，信号与标签匹配
2. ✅ 模型学习的是"在动态R/R下，哪些信号会成功"
3. ✅ 虽然训练时使用未来信息，但**仅用于标签**，不用于特征

**实现位置**：
- 修改 `scripts/train_strategy_pipeline.py` 中的标签生成逻辑
- 添加 `use_adaptive_rr` 参数，控制是否使用自适应R/R标签

**代码示例**：

```python
# 在 train_strategy_pipeline.py 中
if use_adaptive_rr:
    # 使用自适应R/R标签（训练时用未来波动率）
    labels = compute_adaptive_rr_label_with_future_vol(
        df,
        signal_col="signal",
        stop_loss_multiplier=1.0,
        take_profit_multiplier=2.0,
        volatility_window=10,
        use_breakeven_stop=True,
    )
else:
    # 使用固定R/R标签
    labels = compute_rr_label(
        df,
        signal_col="signal",
        stop_loss_r=1.0,
        take_profit_r=2.0,
        use_breakeven_stop=True,
    )
```

---

## 📌 总结

### 为什么"感觉没法做"？

1. ⚠️ **训练时需要未来信息**：计算自适应R/R标签需要未来波动率
2. ⚠️ **回测时只能用预测值**：实盘时没有未来信息，只能用预测波动率
3. ⚠️ **信号与标签不匹配**：训练时用固定R/R，回测时用动态R/R

### 解决方案

1. ✅ **方案A**：训练时用未来波动率（仅用于标签），回测时用预测波动率
2. ✅ **方案B**：训练时也用预测波动率（时间序列交叉验证）
3. ✅ **方案C**：训练时用固定R/R，回测时用动态R/R（当前方案，通过保本止损提升效果）

### 建议

- **短期**：继续使用方案C（当前方案），通过保本止损提升效果
- **长期**：尝试方案A（使用未来波动率训练），确保信号与标签匹配

---

## 📚 参考资料

- `src/time_series_model/pipeline/training/label_utils.py`
- `src/time_series_model/diagnostics/compute_adaptive_rr_with_predicted_vol.py`
- `docs/策略优化/ML模型止损止盈机制详解.md`

