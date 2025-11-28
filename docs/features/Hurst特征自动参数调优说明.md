# Hurst 特征自动参数调优说明

## 📋 概述

Hurst 特征现在支持**自动参数调整**，可以根据数据频率和品种特性自动优化 `update_freq` 和 `clip_pct` 参数。

## 🔧 自动参数调整

### 1. `update_freq` 自动调整

根据数据频率自动设置更新频率，提升计算效率：

| 数据频率 | 自动设置 | 说明 |
|---------|---------|------|
| < 15分钟 | `update_freq=5` | 高频数据，每5根K线更新一次 |
| 15分钟-1小时 | `update_freq=3` | 中频数据，每3根K线更新一次 |
| >= 1小时 | `update_freq=1` | 低频数据，每根K线更新 |

**使用方法：**
```python
# 自动模式（推荐）
df = extract_hurst_features(df, update_freq="auto")

# 手动指定
df = extract_hurst_features(df, update_freq=5)
```

### 2. `clip_pct` 自动调整

根据品种波动特性自动调整极端值裁剪阈值：

| 波动特性 | 自动设置 | 说明 |
|---------|---------|------|
| 高波动（>3%） | `clip_pct=0.8-1.0` | 放宽到80-100%，适应高波动品种 |
| 中波动（1-3%） | `clip_pct=0.5-0.7` | 默认50-70%，平衡稳健性 |
| 低波动（<1%） | `clip_pct=0.3` | 收紧到30%，低波动品种更敏感 |

**波动特性检测逻辑：**
- 计算历史滚动波动率（100根K线窗口）
- 检测最大单日收益率
- 统计极端收益率比例（>10%）
- 综合判断品种波动等级

**使用方法：**
```python
# 自动模式（推荐）
df = extract_hurst_features(df, clip_pct="auto")

# 手动指定
df = extract_hurst_features(df, clip_pct=0.5)

# 禁用裁剪
df = extract_hurst_features(df, clip_pct=None)
```

## 🎯 阈值调优策略

### 一、特征阈值 vs 模型参数

特征阈值和模型参数是**两个不同层面**的调优：

| 层面 | 内容 | 调优方式 | 调优时机 |
|-----|------|---------|---------|
| **特征阈值** | `update_freq`, `clip_pct`, `rolling_window` | 经验性指定 → 网格搜索 | 特征工程阶段 |
| **模型参数** | LightGBM 的 `learning_rate`, `max_depth` 等 | 超参数优化（Optuna/Bayesian） | 模型训练阶段 |

### 二、调优流程建议

#### 阶段 1：经验性指定（必须）

**目的：** 建立基准配置，确保特征计算稳定

**步骤：**
1. **数据频率分析**
   ```python
   # 检查数据频率
   freq = df.index.inferred_freq
   print(f"数据频率: {freq}")
   ```

2. **波动特性分析**
   ```python
   # 计算历史波动率
   returns = df['close'].pct_change()
   rolling_vol = returns.rolling(100).std()
   print(f"平均波动率: {rolling_vol.mean():.4f}")
   print(f"最大单日收益率: {returns.abs().max():.4f}")
   ```

3. **初始配置**
   - 使用 `update_freq="auto"` 和 `clip_pct="auto"`（推荐）
   - 或根据经验手动指定

#### 阶段 2：网格搜索优化（可选）

**目的：** 在基准配置基础上，精细调优关键参数

**可调优参数：**
- `rolling_window`: 滚动窗口大小（影响 Hurst 的平滑度）
- `update_freq`: 更新频率（如果自动模式不理想）
- `clip_pct`: 裁剪阈值（如果自动模式不理想）

**网格搜索示例：**
```python
from itertools import product

# 定义参数网格
param_grid = {
    'rolling_window': [30, 50, 70, 100],
    'update_freq': [1, 3, 5],
    'clip_pct': [0.3, 0.5, 0.7, 1.0],
}

# 评估指标：特征稳定性（Hurst 值的方差）
best_params = None
best_score = float('inf')

for params in product(*param_grid.values()):
    rolling_window, update_freq, clip_pct = params
    
    # 提取特征
    df_features = extract_hurst_features(
        df,
        rolling_window=rolling_window,
        update_freq=update_freq,
        clip_pct=clip_pct,
    )
    
    # 评估特征质量（例如：Hurst 值的稳定性）
    hurst_values = df_features['hurst_price_rolling'].dropna()
    if len(hurst_values) > 100:
        # 计算稳定性（方差越小越稳定）
        stability = hurst_values.std()
        
        if stability < best_score:
            best_score = stability
            best_params = params

print(f"最佳参数: {best_params}")
```

**评估指标建议：**
1. **特征稳定性**：Hurst 值的标准差（越小越好）
2. **特征覆盖率**：非 NaN 值的比例（越大越好）
3. **计算效率**：特征提取耗时（越小越好）
4. **预测能力**：与目标变量的相关性（越大越好）

#### 阶段 3：模型参数优化（独立进行）

**目的：** 在固定特征配置下，优化模型超参数

**工具：**
- Optuna（推荐）
- Hyperopt
- scikit-learn GridSearchCV

**示例：**
```python
import optuna
from lightgbm import LGBMClassifier

def objective(trial):
    # 固定特征配置（使用阶段2的最佳参数）
    df_features = extract_hurst_features(
        df,
        rolling_window=50,  # 固定
        update_freq="auto",  # 固定
        clip_pct="auto",  # 固定
    )
    
    # 模型参数（可调优）
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        # ... 其他参数
    }
    
    model = LGBMClassifier(**params)
    # 训练和评估
    score = cross_val_score(model, X, y, cv=5).mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

### 三、调优优先级

**高优先级（必须调优）：**
1. ✅ `rolling_window`: 直接影响 Hurst 特征质量
2. ✅ `update_freq`: 影响计算效率和特征粒度
3. ✅ `clip_pct`: 影响极端值处理

**中优先级（建议调优）：**
4. ⚠️ 模型超参数（LightGBM）
5. ⚠️ 特征选择（哪些特征组合效果最好）

**低优先级（可选）：**
6. ℹ️ DFA 内部参数（`min_window`, `max_window`）
7. ℹ️ 其他特征工程参数

## 📊 实际应用建议

### 场景 1：多品种统一模型

**策略：** 使用自动参数，让每个品种自适应

```python
# 配置文件：每个品种自动适应
for symbol in ['BTC', 'ETH', 'SOL']:
    df = load_data(symbol)
    
    # 自动适应参数
    df_features = extract_hurst_features(
        df,
        update_freq="auto",  # 自动检测频率
        clip_pct="auto",     # 自动检测波动
    )
```

### 场景 2：单一品种精细调优

**策略：** 先自动，再手动微调

```python
# 步骤1：自动模式建立基准
df_features_auto = extract_hurst_features(
    df,
    update_freq="auto",
    clip_pct="auto",
)

# 步骤2：分析自动参数效果
print(f"自动 update_freq: {实际使用的值}")
print(f"自动 clip_pct: {实际使用的值}")

# 步骤3：如果效果不理想，手动微调
df_features_manual = extract_hurst_features(
    df,
    update_freq=3,  # 手动调整
    clip_pct=0.6,   # 手动调整
)
```

### 场景 3：回测验证

**策略：** 使用滚动窗口验证参数稳定性

```python
# 在不同时间段验证参数效果
for year in [2020, 2021, 2022, 2023]:
    df_year = df[df.index.year == year]
    
    df_features = extract_hurst_features(
        df_year,
        update_freq="auto",
        clip_pct="auto",
    )
    
    # 评估特征质量
    evaluate_features(df_features)
```

## ⚠️ 注意事项

1. **特征阈值 vs 模型参数**
   - 特征阈值在**特征工程阶段**调优
   - 模型参数在**模型训练阶段**调优
   - **不要混在一起调优**（会导致过拟合）

2. **自动参数的限制**
   - 自动参数基于**历史数据**推断，可能不适用于未来
   - 建议在回测中验证自动参数的稳定性

3. **调优顺序**
   - 先调特征阈值（建立稳定的特征）
   - 再调模型参数（在稳定特征上优化模型）

4. **避免过拟合**
   - 特征阈值调优应该在**训练集**上进行
   - 使用**验证集**评估效果
   - 避免在测试集上反复调优

## 🔍 调试技巧

### 检查自动参数效果

```python
# 添加调试输出
import logging
logging.basicConfig(level=logging.INFO)

df_features = extract_hurst_features(
    df,
    update_freq="auto",
    clip_pct="auto",
)

# 检查特征质量
print(f"特征覆盖率: {df_features['hurst_price_rolling'].notna().sum() / len(df_features):.2%}")
print(f"特征稳定性: {df_features['hurst_price_rolling'].std():.4f}")
```

### 对比自动 vs 手动参数

```python
# 自动参数
df_auto = extract_hurst_features(df, update_freq="auto", clip_pct="auto")

# 手动参数
df_manual = extract_hurst_features(df, update_freq=1, clip_pct=0.5)

# 对比效果
compare_features(df_auto, df_manual)
```

## 📚 参考

- [Hurst 特征实现文档](../src/features/time_series/utils_hurst_features.py)
- [特征工程最佳实践](../features/特征工程最佳实践.md)
- [模型超参数优化指南](../策略优化/超参数优化指南.md)

