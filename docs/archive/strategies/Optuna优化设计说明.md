# Optuna 优化设计说明

## 当前实现：只优化决策阈值

### 设计选择

当前 `ts_sr_reversal_optuna.py` **只优化决策阈值**（`long_entry_threshold`, `long_exit_threshold` 等），而不是模型超参数。

**原因：**
1. **关注点分离**：模型训练和阈值优化是两个不同的阶段
2. **计算效率**：优化阈值比优化超参数快得多（不需要重新训练模型）
3. **实际需求**：在模型已训练好的情况下，找到最优的决策阈值更重要
4. **业务目标**：直接优化最终交易信号，而不是模型内部参数

### 当前优化流程

```
1. 加载已训练模型（使用固定超参数）
2. 对每个 trial：
   - 采样不同的阈值参数
   - 使用相同模型预测
   - 应用不同阈值生成交易信号
   - 评估回测结果
3. 找到最优阈值组合
```

**优点：**
- ✅ 快速：不需要重新训练模型
- ✅ 直接：优化最终业务目标（交易信号）
- ✅ 灵活：可以快速测试不同阈值组合

## 联合优化：模型超参数 + 决策阈值

### 概念

可以同时优化：
1. **模型超参数**（num_leaves, learning_rate, max_depth 等）
2. **决策阈值**（long_entry_threshold, short_entry_threshold 等）

### 实现方式

```python
def objective(trial: optuna.Trial) -> float:
    # 1. 优化模型超参数
    model_params = {
        "num_leaves": trial.suggest_int("num_leaves", 20, 255),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        # ... 其他超参数
    }
    
    # 2. 优化决策阈值
    threshold_params = {
        "long_entry_threshold": trial.suggest_float("long_entry_threshold", 0.4, 0.8),
        "long_exit_threshold": trial.suggest_float("long_exit_threshold", 0.2, 0.5),
        "short_entry_threshold": trial.suggest_float("short_entry_threshold", 0.2, 0.6),
        "short_exit_threshold": trial.suggest_float("short_exit_threshold", 0.5, 0.8),
    }
    
    # 3. 使用这些参数训练模型和评估
    # - 用 model_params 训练模型
    # - 用 threshold_params 生成交易信号
    # - 评估回测结果
    
    return backtest_metric
```

### 优缺点对比

| 特性 | 只优化阈值 | 联合优化 |
|------|-----------|---------|
| **计算成本** | 低（不重新训练） | 高（每个 trial 都要训练） |
| **优化速度** | 快 | 慢 |
| **优化范围** | 有限（只优化阈值） | 全面（模型+阈值） |
| **适用场景** | 模型已训练好 | 从零开始优化 |
| **业务目标** | 直接优化交易信号 | 端到端优化 |

### 推荐使用场景

#### 场景 1：只优化阈值（当前实现）✅

**适用情况：**
- 模型已经训练好，性能满意
- 需要快速找到最优交易阈值
- 计算资源有限
- 需要频繁调整阈值

**示例：**
```python
# 模型已训练，只需优化阈值
study.optimize(optimize_thresholds_only, n_trials=50)  # 快速
```

#### 场景 2：联合优化

**适用情况：**
- 模型性能不满意，需要同时优化模型和阈值
- 有充足的计算资源
- 需要端到端优化业务目标
- 模型和阈值有强耦合关系

**示例：**
```python
# 同时优化模型和阈值
study.optimize(optimize_model_and_thresholds, n_trials=200)  # 较慢但全面
```

## 实现建议

### 选项 1：保持当前设计（推荐）

**优点：**
- 简单清晰
- 计算效率高
- 符合实际工作流程（先训练模型，再优化阈值）

**适用：**
- 大多数情况
- 模型训练和阈值优化分开进行

### 选项 2：添加联合优化脚本

**创建新脚本：** `ts_sr_reversal_optuna_joint.py`

**功能：**
- 同时优化模型超参数和决策阈值
- 端到端优化业务目标（如回测收益）

**使用场景：**
- 模型性能需要大幅提升
- 有充足计算资源
- 需要探索模型和阈值的交互

### 选项 3：分阶段优化

**阶段 1：** 优化模型超参数（使用 CV 指标）
**阶段 2：** 使用最优模型，优化决策阈值（使用回测指标）

**优点：**
- 兼顾效率和效果
- 每个阶段优化不同目标

## 当前实现验证

### ✅ 正确性

当前实现**完全正确**：
- Optuna 可以优化任意参数（包括阈值）
- 只优化阈值是合理的设计选择
- 代码实现符合 Optuna 最佳实践

### ✅ 测试覆盖

已添加单元测试和集成测试：
- `tests/test_ts_sr_reversal_optuna.py` - 单元测试
- `tests/integration/test_ts_sr_reversal_optuna_integration.py` - 集成测试

## 总结

1. **当前设计合理**：只优化阈值是高效且实用的选择
2. **Optuna 灵活性**：如果需要，可以轻松扩展为联合优化
3. **按需选择**：根据实际需求选择优化策略
4. **测试完善**：已有完整的测试覆盖

## 相关文件

- `src/time_series_model/optimization/ts_sr_reversal_optuna.py` - 当前实现（只优化阈值）
- `tests/test_ts_sr_reversal_optuna.py` - 单元测试
- `tests/integration/test_ts_sr_reversal_optuna_integration.py` - 集成测试
- `docs/archive/strategies/Optuna优化脚本问题分析.md` - 问题分析文档

