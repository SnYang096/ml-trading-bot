# 🔍 数据泄露分析报告

## 📋 执行摘要

根据文档要求进行深入检查，**发现了严重的数据泄露问题**。虽然代码层面的特征生成逻辑（滑动窗口、rolling操作）看起来是安全的，但在特征工程的流程中存在**关键的数据泄露点**。

## 🚨 核心问题

### 问题 1: dl_seq_f 特征的 Adaptive Normalization 泄露 ⚠️ **高危**

**泄露位置**: `src/data_tools/comprehensive_feature_engineering.py` 和 `src/time_series_model/pipeline/training/train_rank_ic_standalone.py`

**问题描述**:
1. 在 `train_rank_ic_standalone.py` 的 `load_data()` 函数中：
   ```python
   df_features = engineer.engineer_all_features(df, fit=True, required_features=None)
   ```
   这里 `fit=True` 是在**整个数据集**（包括训练集和测试集）上执行的。

2. `add_dl_sequence_features()` 每次调用都会创建新的 `DeepLearningSequenceExtractor`，并在 `add_to_dataframe()` 中调用 `fit_transform()`。

3. Adaptive normalization 的全局 scaler 使用全样本统计：
   ```python
   # 在 fit 时计算全局统计
   self.scaler_mean = np.mean(data, axis=0, keepdims=True)  # 包含测试集！
   self.scaler_std = np.std(data, axis=0, keepdims=True)    # 包含测试集！
   ```

**影响**: 
- 测试集的特征值包含了未来信息（因为 scaler 是用包含测试集的全样本统计的）
- 这解释了为什么 Feature-Future Correlation 检测到高相关性（0.18）
- 这解释了为什么 OOS IC (0.2785) 异常高

### 问题 2: 长窗口 Z-score 的潜在问题 🟡 **中危**

**检查结果**: 
- ✅ Rolling zscore 使用 `center=False`（默认，安全）
- ✅ 使用 `min_periods=10`，合理
- ⚠️ 但长窗口（w288, w500）的高相关性可能来自：
  1. 边界效应（早期样本的 NaN 处理）
  2. 与 dl_seq_f 特征的交互效应

## 📊 证据链

### 1. Feature-Future Correlation 检测结果
```
24/47 features |corr| > 0.1
最高相关性: dl_seq_f43 = 0.1840
```

### 2. OOS 表现异常
```
TSCV Average Rank IC: 0.1294 ± 0.1501
OOS Test Rank IC: 0.2785  ← 异常高！
```

### 3. 代码检查结果
- ✅ dl_seq_f 对齐逻辑正确（特征 i 对应时间点 i+seq_length-1）
- ✅ Rolling 操作安全（center=False）
- ❌ **Adaptive normalization 在测试集上重新 fit scaler**

## 🛠️ 修复方案

### 方案 1: 修复特征工程流程（推荐）🔴 **立即执行**

**修改 `train_rank_ic_standalone.py`**:

```python
def load_data(...):
    # ... 加载数据 ...
    
    # ❌ 错误：在整个数据集上 fit
    # df_features = engineer.engineer_all_features(df, fit=True)
    
    # ✅ 正确：先分割，再分别处理
    df_train_raw, df_test_raw = split_train_test(df, test_size=0.15)
    
    # 在训练集上 fit
    engineer = ComprehensiveFeatureEngineer(feature_types=feature_type)
    df_train_features = engineer.engineer_all_features(df_train_raw, fit=True)
    
    # 在测试集上只 transform（使用训练集的 scaler）
    df_test_features = engineer.engineer_all_features(df_test_raw, fit=False)
    
    # 合并
    df_features = pd.concat([df_train_features, df_test_features])
    
    return df_features, feature_cols
```

**但注意**: `ComprehensiveFeatureEngineer` 需要支持 `fit=False` 模式，并且需要保存 scaler 状态。

### 方案 2: 修改 dl_sequence_features 支持 fit/transform 分离

**修改 `dl_sequence_features.py`**:

```python
class DeepLearningSequenceExtractor:
    def __init__(self, ...):
        # ... 现有代码 ...
        self.extractor = None  # 保存 extractor 实例
    
    def fit(self, df, feature_columns):
        # 只在训练集上调用
        if self.extractor is None:
            self.extractor = DeepLearningSequenceExtractor(...)
        self.extractor.fit(df, feature_columns)
        return self
    
    def transform(self, df, feature_columns):
        # 在测试集上调用
        return self.extractor.transform(df, feature_columns)
```

### 方案 3: 隔离高相关特征，验证泄露（临时方案）🟡 **立即执行**

创建一个测试脚本，移除所有 `corr > 0.1` 的特征，重新训练：

```python
suspicious_features = [
    'dl_seq_f43', 'atr_zscore_w288', 'volatility_zscore_w288',
    'atr_percentile', 'dl_seq_f55', 'dl_seq_f18', 'dl_seq_f38',
    'atr_compression_ratio', 'dl_seq_f7', 'volatility_zscore_w500',
    # ... 其他 14 个特征
]

# 从 top_factors.json 中移除这些特征
# 重新运行训练
# 如果 OOS IC 大幅下降（如从 0.28 → <0.05），证实存在泄露
```

## 📋 立即行动清单

### 🔴 高优先级（立即执行）

1. **验证泄露**: 移除高相关特征，重新测试 OOS 表现
   ```bash
   # 创建测试版本的 top_factors.json，移除 corr > 0.1 的特征
   python3 scripts/remove_suspicious_features.py
   make ts-r-rank-ic-train RANK_IC_TOP_FACTORS=results/feature_evaluation/top_factors_clean.json
   ```

2. **修复特征工程流程**: 实现 fit/transform 分离
   - 修改 `ComprehensiveFeatureEngineer` 支持保存状态
   - 修改 `load_data()` 先分割再特征工程
   - 确保测试集不使用自己的数据 fit scaler

### 🟡 中优先级（本周内）

3. **Forward-Walk Simulation**: 从 t=0 开始逐步加入数据，实时计算特征
4. **审查所有 scaler**: 确保没有其他特征使用全样本统计

### 🟢 低优先级（持续监控）

5. **持续监控 OOS 表现**: 修复后，OOS IC 应该下降到合理范围（0.05-0.15）
6. **文档化**: 记录修复过程和验证结果

## 🎯 预期结果

### 修复前
- OOS Rank IC: 0.2785（异常高，可能由泄露驱动）
- Feature-Future Correlation: 24/47 features > 0.1

### 修复后（预期）
- OOS Rank IC: 0.05-0.15（合理范围）
- Feature-Future Correlation: < 5 features > 0.1
- TSCV 和 OOS IC 差距缩小

## 📝 结论

**确认存在数据泄露**，主要来源是：
1. ✅ **dl_seq_f 特征的 adaptive normalization 在测试集上重新 fit scaler**
2. ⚠️ 长窗口 zscore 的高相关性需要进一步调查

**强烈建议**:
- 🚨 **暂停实盘部署**
- 🔴 **立即修复特征工程流程**
- ✅ **验证修复后的 OOS 表现**

修复后，如果 OOS IC 仍然 > 0.05，说明模型有真实的预测能力；如果归零，说明之前的表现完全由泄露驱动。

