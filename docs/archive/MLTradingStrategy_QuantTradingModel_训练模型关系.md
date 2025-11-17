# MLTradingStrategy、QuantTradingModel 和训练模型的关系

## 📋 概述

这三个组件在训练和部署流程中扮演不同的角色，它们的关系如下：

```
训练阶段：
  ClassificationModelTrainer / QuantileModelTrainer
    ↓ 训练
  LightGBMModel (实际训练的模型)
    ↓ 封装
  QuantTradingModel (封装预处理+模型+后处理)
    ↓ 保存
  .pkl 文件 (rolling 格式)

或者：

  MLTradingStrategy
    ↓ 使用
  MultiTimeframePipeline
    ↓ 训练
  LightGBMModel (多个模型，每个时间框架一个)
    ↓ 保存
  .pkl 文件 (legacy 格式，包含整个 MLTradingStrategy)
```

---

## 🔍 三个组件的定义

### 1. LightGBMModel - 实际训练的模型

**位置**: `src/time_series_model/models/lightgbm_model.py`

**作用**: 
- 实际训练的 LightGBM 模型
- 封装了 LightGBM 的训练、预测、超参数优化等功能

**特点**:
- 底层模型类
- 直接使用 LightGBM 库
- 支持 quantile、regression、classification 三种类型

**示例**:
```python
model = LightGBMModel(model_type="classification", use_gpu=True)
metrics, preprocess_params = model.train(X_train, y_train, n_splits=5)
predictions = model.predict(X_test)
```

---

### 2. QuantTradingModel - 模型管道封装

**位置**: `src/time_series_model/models/quant_trading_model.py`

**作用**:
- 封装 "预处理 + 模型 + 后处理" 的完整流程
- 确保训练和部署时数据流完全一致
- 支持保存和加载整个管道

**特点**:
- 自包含：预处理参数、模型、配置都在一个类中
- 版本一致：使用 joblib 保存，确保训练和部署使用相同版本
- 易部署：只需加载一个 .pkl 文件即可使用

**内部结构**:
```python
class QuantTradingModel:
    def __init__(self):
        self.preprocessor: RobustWinsorizer  # 预处理
        self.model: lgb.Booster              # LightGBM 模型（内部使用）
        self.feature_cols: List[str]         # 特征列
        self.preprocess_params: Dict         # 预处理参数
```

**示例**:
```python
# 训练和保存
model = QuantTradingModel(
    model_type="classification",
    forward_bars=5,
    feature_cols=feature_list
)
model.fit(X_train, y_train, current_returns=current_returns_train)
model.save("models/classification_pipeline.pkl")

# 加载和预测
model = QuantTradingModel.load("models/classification_pipeline.pkl")
predictions = model.predict(X_test)
```

---

### 3. MLTradingStrategy - 策略封装类

**位置**: `src/time_series_model/strategies/ml_strategy.py`

**作用**:
- 整合数据加载、特征工程、模型训练、信号生成的完整流程
- 高层封装，适合完整的策略系统

**特点**:
- 包含多个组件：`MarketDataLoader`、`FeatureEngineer`、`MultiTimeframePipeline` 等
- 支持多时间框架训练
- 包含风险管理 (`RiskManager`)

**内部结构**:
```python
class MLTradingStrategy:
    def __init__(self):
        self.data_loader = MarketDataLoader()
        self.feature_engineer = FeatureEngineer()
        self.pipeline = MultiTimeframePipeline()  # 包含多个 LightGBMModel
        self.strategy_handler = ClassificationStrategyHandler()
        self.risk_manager = RiskManager()
```

---

## 🔗 它们之间的关系

### 关系图

```
┌─────────────────────────────────────────────────────────────┐
│                 训练阶段 (make rolling)                      │
│                                                             │
│  ClassificationModelTrainer                                │
│    ↓ train_models()                                         │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  LightGBMModel (classification)                      │ │
│  │  LightGBMModel (return regression)                    │ │
│  │  LightGBMModel (volatility)                           │ │
│  └───────────────────────────────────────────────────────┘ │
│    ↓ 封装为 QuantTradingModel                              │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  QuantTradingModel (classification_pipeline.pkl)      │ │
│  │  QuantTradingModel (return_pipeline.pkl)              │ │
│  │  QuantTradingModel (vol_pipeline.pkl)                 │ │
│  └───────────────────────────────────────────────────────┘ │
│    ↓ 保存到文件                                             │
│  results/rolling_*/latest/                                 │
│    - classification_pipeline.pkl                            │
│    - return_pipeline.pkl                                   │
│    - vol_pipeline.pkl                                       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                 训练阶段 (make train)                       │
│                                                             │
│  MLTradingStrategy                                          │
│    ↓ train_strategy()                                       │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  MultiTimeframePipeline                               │ │
│  │    ↓ train_pipeline()                                 │ │
│  │  ┌─────────────────────────────────────────────────┐ │ │
│  │  │  LightGBMModel (classification_models["5T"])     │ │ │
│  │  │  LightGBMModel (return_models["5T"])            │ │ │
│  │  │  LightGBMModel (volatility_models["5T"])        │ │ │
│  │  │  ... (其他时间框架)                               │ │ │
│  │  └─────────────────────────────────────────────────┘ │ │
│  └───────────────────────────────────────────────────────┘ │
│    ↓ 保存整个对象                                            │
│  models/trained_model_*.pkl                                 │
│    - 包含完整的 MLTradingStrategy 对象                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 详细对比

| 特性 | LightGBMModel | QuantTradingModel | MLTradingStrategy |
|------|--------------|------------------|------------------|
| **层级** | 底层模型 | 中层管道 | 高层策略 |
| **包含内容** | LightGBM 模型 | 预处理 + 模型 + 后处理 | 数据加载 + 特征工程 + 模型 + 信号生成 |
| **训练方式** | 直接训练 | 封装训练 | 使用 MultiTimeframePipeline |
| **保存格式** | 不单独保存 | .pkl (rolling 格式) | .pkl (legacy 格式) |
| **使用场景** | 训练阶段 | Rolling 训练/部署 | Legacy 训练/回测 |
| **依赖关系** | 无依赖 | 依赖 LightGBMModel | 依赖 MultiTimeframePipeline → LightGBMModel |

---

## 🔄 在训练流程中的使用

### 场景 1: make rolling（使用 QuantTradingModel）

**流程**:
```python
# 1. 使用 ClassificationModelTrainer 训练
trainer = ClassificationModelTrainer()
models_dict, metrics_dict, preprocess_params_dict = trainer.train_models(...)

# models_dict 包含：
#   - model_classification: LightGBMModel
#   - model_return: LightGBMModel
#   - model_vol: LightGBMModel

# 2. 封装为 QuantTradingModel
cls_pipeline = QuantTradingModel(
    model_type="classification",
    feature_cols=feature_cols,
    preprocess_params=classification_preprocess_params
)
cls_pipeline.model = model_classification.model  # 设置 LightGBM 模型
cls_pipeline.save("classification_pipeline.pkl")

# 3. 保存三个管道
# - classification_pipeline.pkl
# - return_pipeline.pkl
# - vol_pipeline.pkl
```

**代码位置**: `rolling.py` 第 600-800 行

---

### 场景 2: make train（使用 MLTradingStrategy）

**流程**:
```python
# 1. 创建 MLTradingStrategy
strategy = MLTradingStrategy(model_type="classification")

# 2. 训练策略（内部使用 MultiTimeframePipeline）
metrics = strategy.train_strategy()
# 内部调用：
#   - self.pipeline.train_pipeline(engineered_data)
#   - 训练多个 LightGBMModel（每个时间框架一个）

# 3. 保存整个策略对象
model_data = {
    "strategy": strategy,  # 包含所有组件
    "data_loader": strategy.data_loader,
    "feature_engineer": strategy.feature_engineer,
    "engineered_data": engineered_data,
    "metrics": metrics
}
pickle.dump(model_data, open("trained_model.pkl", "wb"))
```

**代码位置**: `train.py` 第 405-3000 行

---

## 🎯 关键区别

### 1. 模型封装方式

**QuantTradingModel**:
- 封装单个模型（classification、return 或 vol）
- 包含预处理和后处理
- 适合滚动训练（每个模型单独保存）

**MLTradingStrategy**:
- 封装整个策略系统
- 包含多个时间框架的模型
- 适合一次性训练（整个策略保存为一个文件）

### 2. 训练方式

**QuantTradingModel**:
- 使用 `ClassificationModelTrainer` 训练 `LightGBMModel`
- 然后封装为 `QuantTradingModel`

**MLTradingStrategy**:
- 使用 `MultiTimeframePipeline` 训练多个 `LightGBMModel`
- 然后封装为 `MLTradingStrategy`

### 3. 保存格式

**QuantTradingModel**:
- 保存为独立的 .pkl 文件
- 每个模型一个文件（classification、return、vol）

**MLTradingStrategy**:
- 保存为单个 .pkl 文件
- 包含整个策略对象

---

## 💡 为什么有两个方式？

### 历史原因

1. **Legacy 格式**（MLTradingStrategy）:
   - 早期的训练方式
   - 保存完整的策略对象
   - 适合一次性训练和评估

2. **Rolling 格式**（QuantTradingModel）:
   - 新的训练方式
   - 只保存模型管道
   - 适合滚动训练和生产环境

### 设计考虑

- **QuantTradingModel**: 轻量级，适合批量训练和部署
- **MLTradingStrategy**: 完整封装，适合完整的策略系统

---

## 📝 总结

1. **LightGBMModel**: 底层模型，实际训练的 LightGBM 模型
2. **QuantTradingModel**: 中层管道，封装预处理+模型+后处理（用于 rolling）
3. **MLTradingStrategy**: 高层策略，整合完整流程（用于 legacy）

4. **关系**:
   - `QuantTradingModel` 内部使用 `LightGBMModel`
   - `MLTradingStrategy` 使用 `MultiTimeframePipeline`，后者包含多个 `LightGBMModel`
   - 两者都是对 `LightGBMModel` 的封装，但封装方式不同

5. **使用场景**:
   - **Rolling 训练**: 使用 `QuantTradingModel`
   - **Legacy 训练**: 使用 `MLTradingStrategy`
   - **回测**: 根据格式选择加载方式

