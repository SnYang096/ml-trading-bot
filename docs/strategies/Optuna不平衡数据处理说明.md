# Optuna 不平衡数据处理说明

## 核心原则

**不要用"准确率"作为优化目标！**

对于不平衡数据（如金融场景中上涨/下跌样本比例悬殊），应该使用对不平衡数据鲁棒的业务指标或评估指标。

## 当前实现

### 1. 优化目标选择

**默认使用业务指标（推荐）：**

- ✅ **`sharpe`** (默认) - 夏普比率
  - 天然不受类别比例影响
  - 直接反映风险调整后的收益
  - 适合金融/交易场景

- ✅ **`total_return`** - 总收益百分比
  - 直接优化实际盈亏
  - 不受数据平衡性影响

- ✅ **`sharpe_with_cv_fallback`** - 优先夏普比率，回退到 CV 指标
  - 如果有回测结果，使用夏普比率
  - 如果没有回测结果，使用 CV 指标

- ⚠️ **`cv_metric`** - 交叉验证指标（原始行为）
  - 可能受数据不平衡影响
  - 仅在需要时使用

### 2. 不平衡数据约束

**最小交易次数约束：**
```python
--min-trades 10  # 至少需要 10 笔交易
```

**作用：**
- 防止阈值过高导致零交易（过拟合）
- 确保有足够的样本评估策略效果
- 对不平衡数据特别重要

**最小胜率约束：**
```python
--min-win-rate 0.5  # 至少 50% 胜率
```

**作用：**
- 防止策略虽然交易次数多但胜率过低
- 可以设置为 0.0 禁用此约束

### 3. 目标函数实现

```python
def objective(trial: optuna.Trial) -> float:
    # ... 运行训练和回测 ...
    
    # 检查约束（处理不平衡数据）
    if backtest_results:
        n_trades = backtest_results["debug"]["trades_meta"].get("n_trades", 0)
        if n_trades < args.min_trades:
            raise optuna.TrialPruned("Insufficient trades")
        
        win_rate = backtest_results.get("win_rate", 0.0) / 100.0
        if win_rate < args.min_win_rate:
            raise optuna.TrialPruned("Win rate too low")
    
    # 选择优化目标
    if args.objective == "sharpe":
        return backtest_results["sharpe"]  # 业务指标，不受不平衡影响
    elif args.objective == "total_return":
        return backtest_results["total_return_pct"]  # 实际收益
    # ...
```

## 使用示例

### 场景 1：不平衡数据（正样本稀少）

```bash
# 使用夏普比率优化（默认，推荐）
python src/time_series_model/optimization/ts_sr_reversal_optuna.py \
    --strategy-config config/strategies/sr_reversal \
    --symbol BTCUSDT \
    --objective sharpe \
    --min-trades 20 \
    --min-win-rate 0.45 \
    --n-trials 50
```

**说明：**
- `--objective sharpe`：使用夏普比率（不受不平衡影响）
- `--min-trades 20`：至少需要 20 笔交易
- `--min-win-rate 0.45`：至少 45% 胜率

### 场景 2：极端不平衡（正样本 < 5%）

```bash
# 使用总收益优化，放宽交易次数要求
python src/time_series_model/optimization/ts_sr_reversal_optuna.py \
    --strategy-config config/strategies/sr_reversal \
    --symbol BTCUSDT \
    --objective total_return \
    --min-trades 5 \
    --min-win-rate 0.0 \
    --n-trials 100
```

**说明：**
- `--objective total_return`：直接优化收益
- `--min-trades 5`：极端不平衡时降低要求
- `--min-win-rate 0.0`：不限制胜率（让模型学习）

### 场景 3：联合优化（模型 + 阈值）

```bash
# 联合优化，使用夏普比率
python src/time_series_model/optimization/ts_sr_reversal_optuna_joint.py \
    --strategy-config config/strategies/sr_reversal \
    --symbol BTCUSDT \
    --objective sharpe \
    --min-trades 15 \
    --min-win-rate 0.5 \
    --n-trials 50
```

## 为什么业务指标更适合不平衡数据？

### 1. 准确率的问题

**准确率受类别比例影响：**
- 如果正样本只占 1%，模型可以简单地预测所有样本为负，获得 99% 准确率
- 但这对于交易策略毫无意义

### 2. 业务指标的优势

**夏普比率：**
- 基于实际收益和波动率
- 不受标签分布影响
- 直接反映策略质量

**总收益：**
- 直接优化"钱"
- 不关心样本数量
- 只关心实际盈亏

### 3. 对比示例

| 指标 | 正样本 1% | 正样本 50% | 是否适合不平衡数据 |
|------|----------|-----------|------------------|
| 准确率 | 99% (无意义) | 50% | ❌ 不适合 |
| F1-score | 可能很低 | 可能很高 | ⚠️ 需要调整 |
| 夏普比率 | 基于实际收益 | 基于实际收益 | ✅ 适合 |
| 总收益 | 基于实际盈亏 | 基于实际盈亏 | ✅ 适合 |

## 最佳实践

### 1. 选择优化目标

**金融/交易场景（推荐）：**
```bash
--objective sharpe  # 或 total_return
```

**原因：**
- 直接优化业务目标（收益）
- 不受数据不平衡影响
- 天然鲁棒

### 2. 设置合理约束

**最小交易次数：**
- 平衡数据：`--min-trades 20-50`
- 不平衡数据：`--min-trades 10-20`
- 极端不平衡：`--min-trades 5-10`

**最小胜率：**
- 保守策略：`--min-win-rate 0.5-0.6`
- 一般策略：`--min-win-rate 0.4-0.5`
- 探索阶段：`--min-win-rate 0.0`（不限制）

### 3. 阈值搜索范围

**根据正样本比例调整：**

如果正样本稀少（< 5%），可能需要：
- 更低的阈值范围：`[0.3, 0.6]` 而不是 `[0.4, 0.8]`
- 或更高的阈值范围：`[0.7, 0.95]`（取决于模型输出分布）

**当前实现：**
- `long_entry_threshold`: 0.4-0.8
- `short_entry_threshold`: 0.2-0.6

如果发现最优阈值总是在边界，可以调整搜索范围。

## 注意事项

### 1. AUC 不能用于阈值优化

❌ **错误：**
```python
# AUC 与阈值无关，不能用于优化阈值
return roc_auc_score(y_true, y_prob)
```

✅ **正确：**
```python
# 使用业务指标
return backtest_results["sharpe"]
```

### 2. 验证集需要包含少数类

确保验证集中有足够的正样本，否则无法评估阈值效果。

### 3. 动态调整搜索范围

如果发现最优阈值总是在边界，说明搜索范围不合理，需要调整。

## 相关文件

- `src/time_series_model/optimization/ts_sr_reversal_optuna.py` - 阈值优化（已支持不平衡数据）
- `src/time_series_model/optimization/ts_sr_reversal_optuna_joint.py` - 联合优化（已支持不平衡数据）

## 总结

✅ **当前实现已优化：**
1. 默认使用业务指标（夏普比率）而非准确率
2. 支持最小交易次数约束（防止零交易）
3. 支持最小胜率约束（防止低质量策略）
4. 支持多种优化目标（sharpe, total_return, cv_metric）

✅ **推荐使用：**
- 金融场景：`--objective sharpe`（默认）
- 不平衡数据：设置合理的 `--min-trades` 和 `--min-win-rate`
- 探索阶段：可以禁用约束（设置为 0）

