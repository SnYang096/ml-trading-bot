# MLTradingStrategy 与回测的关系

## 📋 概述

`MLTradingStrategy` 是一个**策略封装类**，它整合了训练和信号生成的完整流程。它与回测系统的关系如下：

---

## 🔗 关系图

```
┌─────────────────────────────────────────────────────────────┐
│                    MLTradingStrategy                        │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  • MarketDataLoader (数据加载)                        │  │
│  │  • FeatureEngineer (特征工程)                        │  │
│  │  • MultiTimeframePipeline (模型管道)                 │  │
│  │  • ClassificationStrategyHandler (信号生成)           │  │
│  │  • RiskManager (风险管理)                            │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  主要功能：                                                  │
│  • train_strategy() - 训练策略                             │
│  • generate_signals() - 生成交易信号                       │
│  • optimize_strategy() - 优化策略参数                      │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ 使用
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              VectorBotBacktest (回测系统)                   │
│                                                             │
│  使用场景：                                                  │
│  1. Legacy 格式模型（make train 生成的 .pkl 文件）         │
│     - 加载 MLTradingStrategy 对象                         │
│     - 使用 strategy.generate_signals() 生成信号           │
│     - 执行回测                                              │
│                                                             │
│  2. Rolling 格式模型（make rolling 生成的目录）            │
│     - 不使用 MLTradingStrategy                             │
│     - 直接加载 QuantTradingModel 管道                     │
│     - 自己生成信号                                          │
│     - 执行回测                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 MLTradingStrategy 的作用

### 1. 策略封装

`MLTradingStrategy` 是一个**高层封装**，整合了以下组件：

```python
class MLTradingStrategy:
    def __init__(self):
        self.data_loader = MarketDataLoader()        # 数据加载
        self.feature_engineer = FeatureEngineer()   # 特征工程
        self.pipeline = MultiTimeframePipeline()    # 模型管道
        self.strategy_handler = ClassificationStrategyHandler()  # 信号生成
        self.risk_manager = RiskManager()           # 风险管理
```

### 2. 主要功能

| 方法 | 功能 | 用途 |
|------|------|------|
| `train_strategy()` | 训练完整的策略 | 训练阶段使用 |
| `generate_signals()` | 生成交易信号 | 回测和生产使用 |
| `optimize_strategy()` | 优化策略参数 | 参数调优使用 |

---

## 🔄 在回测中的使用

### 场景 1: Legacy 格式（make train）

**流程**:
```python
# 1. make train 训练并保存
strategy = MLTradingStrategy()
strategy.train_strategy()
# 保存为 .pkl 文件

# 2. vectorbot.py 加载并使用
backtest = VectorBotBacktest(model_path="trained_model.pkl")
# 内部加载：
#   - self.strategy = MLTradingStrategy (从 pickle 加载)
#   - signals = self.strategy.generate_signals(data)
#   - 执行回测
```

**代码位置**: `vectorbot.py` 第 89-97 行
```python
# Legacy format: single pickle file
with open(self.model_path, "rb") as f:
    model_data = pickle.load(f)

self.strategy: MLTradingStrategy = model_data["strategy"]
self.data_loader = model_data["data_loader"]
self.feature_engineer = model_data["feature_engineer"]
```

---

### 场景 2: Rolling 格式（make rolling）

**流程**:
```python
# 1. make rolling 训练并保存
# 保存为 QuantTradingModel 管道（.pkl 文件）

# 2. vectorbot.py 加载并使用
backtest = VectorBotBacktest(model_path="rolling_results/latest")
# 内部加载：
#   - self.cls_pipeline = QuantTradingModel.load(...)
#   - self.return_pipeline = QuantTradingModel.load(...)
#   - self.vol_pipeline = QuantTradingModel.load(...)
#   - 自己生成信号（不使用 MLTradingStrategy）
#   - 执行回测
```

**代码位置**: `vectorbot.py` 第 103-127 行
```python
def _load_rolling_models(self, model_dir: str) -> None:
    """Load models from rolling training directory."""
    self.cls_pipeline = QuantTradingModel.load(cls_path)
    self.return_pipeline = QuantTradingModel.load(return_path)
    self.vol_pipeline = QuantTradingModel.load(vol_path)
    # 不使用 MLTradingStrategy
```

---

## 📊 对比：MLTradingStrategy vs 直接使用 Handler

### 使用 MLTradingStrategy（Legacy 格式）

```python
# 优点：
✅ 完整的封装，包含数据加载、特征工程、信号生成、风险管理
✅ 适合生产环境，功能完整
✅ 支持多时间框架

# 缺点：
❌ 依赖较多组件（MarketDataLoader, FeatureEngineer 等）
❌ 不适合简单的回测场景
```

### 直接使用 Handler（Rolling 格式）

```python
# 优点：
✅ 轻量级，只使用必要的组件
✅ 适合快速回测
✅ 灵活性高

# 缺点：
❌ 需要自己管理数据加载和特征工程
❌ 功能相对简单
```

---

## 🔍 关键区别

| 特性 | MLTradingStrategy | 直接使用 Handler |
|------|------------------|-----------------|
| **使用场景** | Legacy 格式（make train） | Rolling 格式（make rolling） |
| **数据加载** | 内置 `MarketDataLoader` | 需要自己加载 |
| **特征工程** | 内置 `FeatureEngineer` | 需要自己工程 |
| **信号生成** | 调用 `generate_signals()` | 直接调用 `handler.generate_signals()` |
| **风险管理** | 内置 `RiskManager` | 需要自己实现 |
| **复杂度** | 高（完整封装） | 低（轻量级） |

---

## 💡 为什么有两个方式？

### 历史原因

1. **Legacy 格式**（使用 MLTradingStrategy）:
   - 早期的训练方式（`make train`）
   - 保存完整的策略对象
   - 适合一次性训练和评估

2. **Rolling 格式**（不使用 MLTradingStrategy）:
   - 新的训练方式（`make rolling`）
   - 只保存模型管道（QuantTradingModel）
   - 适合滚动训练和生产环境

### 设计考虑

- **MLTradingStrategy**: 适合**完整的策略系统**，包含所有组件
- **直接使用 Handler**: 适合**轻量级回测**，只需要信号生成

---

## 🎯 总结

1. **MLTradingStrategy** 是一个**策略封装类**，整合了训练和信号生成的完整流程
2. **VectorBotBacktest** 在 Legacy 格式中使用 `MLTradingStrategy`，在 Rolling 格式中不使用
3. **MLTradingStrategy** 主要用于：
   - `make train` 训练流程
   - Legacy 格式的回测
   - 生产环境的信号生成

4. **Rolling 格式**不使用 `MLTradingStrategy`，因为：
   - 只需要模型管道，不需要完整策略
   - 更轻量级，适合批量回测
   - 灵活性更高

---

## 📝 代码位置总结

| 文件 | 用途 | MLTradingStrategy 使用 |
|------|------|----------------------|
| `ml_strategy.py` | 定义 MLTradingStrategy 类 | ✅ 定义 |
| `vectorbot.py` (Legacy) | Legacy 格式回测 | ✅ 使用 |
| `vectorbot.py` (Rolling) | Rolling 格式回测 | ❌ 不使用 |
| `rolling.py` | Rolling 训练 | ❌ 不使用 |
| `dim-compare` | 特征选择 | ❌ 不使用 |

