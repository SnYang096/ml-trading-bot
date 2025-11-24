# 回测机制统一到 vectorbot.py

## 📋 变更总结

所有回测相关的函数和类已统一到 `src/time_series_model/backtesting/vectorbot.py` 中，实现代码集中管理和一致性。

---

## ✅ 已完成的迁移

### 1. 迁移的函数

从 `src/time_series_model/utils/training.py` 迁移：
- ✅ `evaluate_signal_performance` - 评估信号性能（简化回测）
- ✅ `print_backtest_results` - 打印回测结果

从 `src/time_series_model/pipeline/dimensionality/backtest_evaluator.py` 迁移：
- ✅ `calculate_strategy_returns_from_predictions` - 从预测计算策略收益
- ✅ `calculate_financial_metrics_from_returns` - 从收益计算金融指标
- ✅ `backtest_classification_model` - 分类模型回测

### 2. 更新的引用

以下文件已更新导入：
- ✅ `src/time_series_model/pipeline/training/rolling.py`
- ✅ `src/time_series_model/pipeline/rolling/auto_rolling_update.py`
- ✅ `src/time_series_model/pipeline/dimensionality/dimensionality_comparison.py`
- ✅ `src/time_series_model/pipeline/dimensionality/evaluation.py`

### 3. 向后兼容

旧文件已标记为 deprecated，但仍保留转发功能：
- ⚠️ `src/time_series_model/utils/training.py` - `evaluate_signal_performance` 和 `print_backtest_results` 已标记为 deprecated
- ⚠️ `src/time_series_model/pipeline/dimensionality/backtest_evaluator.py` - 所有函数已标记为 deprecated

---

## 📦 统一的回测接口

### 导入方式

```python
from time_series_model.backtesting.vectorbot import (
    # 高级回测类（带风险管理和完整交易逻辑）
    VectorBotBacktest,
    
    # 简化回测函数（用于快速评估）
    evaluate_signal_performance,
    print_backtest_results,
    
    # 维度比较回测函数
    calculate_strategy_returns_from_predictions,
    calculate_financial_metrics_from_returns,
    backtest_classification_model,
)
```

### 使用场景

#### 1. 完整回测（推荐用于生产）

```python
from time_series_model.backtesting.vectorbot import VectorBotBacktest

backtest = VectorBotBacktest(
    model_path="results/rolling_btcusdt_ethusdt_20251115_001813/latest",
    symbol="BTCUSDT",
    initial_capital=100000
)

backtest.run_backtest(
    start_date="2024-01-01",
    end_date="2024-12-31",
    output_dir="results/backtest_output"
)
```

**特点**：
- ✅ 完整的风险管理（止损、止盈、仓位控制）
- ✅ 逐笔交易记录
- ✅ HTML 报告生成
- ✅ 支持滚动训练模型（QuantTradingModel）

#### 2. 快速信号评估（用于训练阶段）

```python
from time_series_model.backtesting.vectorbot import (
    evaluate_signal_performance,
    print_backtest_results,
)

# 在 rolling.py 中使用
bt_results = evaluate_signal_performance(
    signals_df,
    y_return_test,
    initial_capital=100000.0
)
print_backtest_results(bt_results, "Monthly Backtest")
```

**特点**：
- ✅ 快速评估信号质量
- ✅ 使用 signal_strength 直接作为仓位大小
- ✅ 适合训练阶段的快速反馈

#### 3. 维度比较回测（用于 ts-dim-compare）

```python
from time_series_model.backtesting.vectorbot import (
    backtest_classification_model,
    calculate_strategy_returns_from_predictions,
    calculate_financial_metrics_from_returns,
)

# 在 dimensionality_comparison.py 中使用
metrics = backtest_classification_model(
    model,
    X_test,
    y_test,
    price_data,
    horizon=1,
    risk_free_rate=0.0
)
```

**特点**：
- ✅ 基于分类预测（0=Hold, 1=Long, 2=Short）
- ✅ 计算金融指标（Sharpe Ratio, Max Drawdown 等）
- ✅ 适合特征维度比较

---

## 🔄 迁移路径

### 旧代码

```python
# ❌ 旧方式
from time_series_model.utils.training import (
    evaluate_signal_performance,
    print_backtest_results,
)

from time_series_model.pipeline.dimensionality.backtest_evaluator import (
    backtest_classification_model,
    calculate_strategy_returns_from_predictions,
)
```

### 新代码

```python
# ✅ 新方式（统一导入）
from time_series_model.backtesting.vectorbot import (
    evaluate_signal_performance,
    print_backtest_results,
    backtest_classification_model,
    calculate_strategy_returns_from_predictions,
    calculate_financial_metrics_from_returns,
)
```

---

## 📁 文件结构

```
src/time_series_model/backtesting/
├── vectorbot.py          # ✅ 统一回测引擎（所有回测功能）
│   ├── evaluate_signal_performance()
│   ├── print_backtest_results()
│   ├── calculate_strategy_returns_from_predictions()
│   ├── calculate_financial_metrics_from_returns()
│   ├── backtest_classification_model()
│   └── VectorBotBacktest (class)
│
└── nautilus_dim.py        # Nautilus Trader 回测（独立，不迁移）

src/time_series_model/utils/
└── training.py            # ⚠️ DEPRECATED（保留转发）

src/time_series_model/pipeline/dimensionality/
└── backtest_evaluator.py  # ⚠️ DEPRECATED（保留转发）
```

---

## 🎯 优势

### 1. 代码集中管理
- ✅ 所有回测逻辑在一个文件中
- ✅ 易于维护和扩展
- ✅ 减少代码重复

### 2. 一致性
- ✅ 统一的接口和返回值格式
- ✅ 一致的指标计算方式
- ✅ 统一的错误处理

### 3. 向后兼容
- ✅ 旧代码仍可工作（通过 deprecated 转发）
- ✅ 逐步迁移，不影响现有功能
- ✅ 清晰的迁移路径

---

## 📝 注意事项

1. **旧文件仍可用**：`training.py` 和 `backtest_evaluator.py` 中的函数已标记为 deprecated，但仍会转发到新位置，不会立即破坏现有代码。

2. **建议迁移**：虽然旧代码仍可用，但建议尽快更新导入路径，因为：
   - 旧文件可能会在未来版本中删除
   - 新位置提供更好的文档和类型提示
   - 统一的位置便于维护

3. **功能不变**：所有函数的签名和行为保持不变，只是位置改变。

---

## 🔍 验证

运行以下命令验证迁移是否成功：

```bash
# 测试导入
python3 -c "from time_series_model.backtesting.vectorbot import *; print('✅ All imports successful')"

# 运行回测
make ts-vectorbot-backtest BACKTEST_MODEL=results/rolling_btcusdt_ethusdt_20251115_001813/latest
```

---

## 📅 变更日期

- **迁移完成**：2025-01-XX
- **向后兼容保留期**：至少 3 个月
- **计划删除旧文件**：待所有引用更新后

