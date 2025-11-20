# Sharpe Ratio 分析

## 📊 当前结果

从你的输出来看：
- **Total Trades**: 25
- **Win Rate**: 64.0%
- **Total Return**: 25.91%
- **Sharpe Ratio**: 0.282
- **Avg Return per Trade**: 1.18%
- **Max Drawdown**: -8.26%

## 🔍 Sharpe Ratio 计算方式

### 当前计算

```python
avg_return = trade_pnl.mean()  # 每个交易的平均收益
std_return = trade_pnl.std()   # 每个交易收益的标准差
sharpe_ratio = (avg_return - risk_free_rate) / std_return
```

### 为什么这么低？

**Sharpe Ratio = 0.282 意味着：**
- 平均收益 = 1.18%
- 标准差 ≈ 1.18% / 0.282 ≈ **4.18%**

**问题分析：**

1. **样本数量少（25 个交易）**
   - 标准差估计不稳定
   - 小样本会导致 Sharpe Ratio 被低估
   - 建议：至少需要 30+ 个交易才能有稳定的估计

2. **收益波动性大**
   - 标准差（4.18%）远大于平均收益（1.18%）
   - 这意味着交易收益的波动性很大
   - 即使 Win Rate 高（64%），但亏损交易的损失可能较大

3. **没有年化处理**
   - 当前计算的是交易级别的 Sharpe Ratio
   - 如果每个交易持续 `hold_period=24` 期，应该考虑年化
   - 年化 Sharpe Ratio = Sharpe Ratio * sqrt(年交易数)

## 💡 改进建议

### 1. 增加样本数量

**方法**：降低置信度阈值

```python
# 当前：confidence_threshold = 0.85
# 建议：confidence_threshold = 0.7 或 0.75
```

这样可以增加交易数量，使 Sharpe Ratio 估计更稳定。

### 2. 使用稳健的统计量

**方法**：使用中位数和 MAD（Median Absolute Deviation）

```python
median_return = trade_pnl.median()
mad = (trade_pnl - median_return).abs().median()
robust_std = mad * 1.4826  # 转换为标准差
robust_sharpe = median_return / robust_std
```

对于小样本，这更稳健。

### 3. 年化 Sharpe Ratio

**方法**：考虑每个交易的持续时间

```python
# 假设每个交易持续 hold_period 期
# 一年有 periods_per_year 个期
annualized_sharpe = sharpe_ratio * sqrt(periods_per_year / hold_period)
```

### 4. 分析收益分布

**检查**：
- 是否有极端值（outliers）？
- 亏损交易的损失是否特别大？
- 盈利交易的收益是否集中？

## 📈 预期改善

如果采取改进措施：

1. **增加样本数量到 50+**
   - Sharpe Ratio 估计会更稳定
   - 可能从 0.282 提升到 0.4-0.6

2. **使用稳健统计量**
   - 减少极端值的影响
   - 可能提升到 0.3-0.5

3. **年化处理**
   - 如果年化，Sharpe Ratio 可能会更高
   - 例如：0.282 * sqrt(252*26/24) ≈ 0.282 * 16.5 ≈ 4.65（但这可能不准确）

## 🔧 已实现的改进

1. **添加了稳健的 Sharpe Ratio 计算**
   - 对于小样本（<30），使用中位数和 MAD
   - 使用更保守的估计

2. **添加了诊断信息**
   - 在统计输出中添加 `sharpe_diagnostic`
   - 包括：交易数量、平均收益、标准差、收益波动比等

## 📝 相关文件

- **诊断脚本**：`scripts/analyze_sharpe_ratio.py`
- **修复代码**：`src/time_series_model/pipeline/training/evaluation_utils.py`

## 🚀 下一步

1. 运行诊断脚本分析 Sharpe Ratio：
```bash
python3 scripts/analyze_sharpe_ratio.py \
  --results results/rank_ic_training/rank_ic_results_*.json \
  --hold-period 24
```

2. 尝试降低置信度阈值，增加交易数量：
```bash
# 在训练时使用更低的置信度阈值
# 修改 confidence_threshold 从 0.85 到 0.7
```

3. 检查收益分布，看是否有极端值影响标准差

