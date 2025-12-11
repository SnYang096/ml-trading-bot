# 代码清理总结 - src/time_series_model

## ✅ 已删除的文件

1. **`pipeline/training/evaluation_utils.py`** - 未被使用（之前被已删除的 rank_ic_trainer 使用）
2. **`pipeline/training/rank_ic_utils_improved.py`** - 未被使用（之前被已删除的 rank_ic_trainer 使用）
3. **`utils/drift.py`** - 未被使用
4. **`utils/logger.py`** - 未被使用
5. **`analysis/timeframe_forward_correlation.py`** - 已标记为 deprecated，未被使用

## ⚠️ 保留的文件（需要进一步评估）

### strategies/backtesting/ 目录
这些文件是策略特定的回测脚本，可能包含策略特定的回测逻辑。

**保留原因**：不同策略可能需要不同的回测方法（入场/出场逻辑、止损止盈规则等），统一的 `run_vectorbt_backtest` 可能无法满足所有需求。

**保留文件**：
- `sr_reversal_backtest.py` - SR 反转策略回测
- `sr_breakout_backtest.py` - SR 突破策略回测
- `compression_breakout_backtest.py` - 压缩突破策略回测
- `trend_following_backtest.py` - 趋势跟随策略回测

**后续评估**：
- 检查这些脚本是否包含策略特定的回测逻辑
- 评估是否可以整合到配置驱动的回测系统中
- 如果确认不再需要，可以删除

### strategies/evaluation/ 目录
这些文件是策略特定的评估脚本，可能包含策略特定的评估指标。

**保留原因**：不同策略可能需要不同的评估指标（胜率、R/R、Profit Factor 等），统一的评估流程可能无法满足所有需求。

**保留文件**：
- `sr_reversal_evaluation.py` - SR 反转策略评估
- `sr_breakout_evaluation.py` - SR 突破策略评估
- `compression_breakout_evaluation.py` - 压缩突破策略评估
- `trend_following_evaluation.py` - 趋势跟随策略评估
- `strategy_feature_compare.py` - 特征比较（被 Makefile 和 optuna 使用）

**后续评估**：
- 检查这些脚本是否包含策略特定的评估指标
- 评估是否可以整合到配置驱动的评估系统中
- 如果确认不再需要，可以删除

## 📊 清理统计

- **已删除**：5 个文件
- **保留待评估**：8 个文件（backtesting 和 evaluation 目录下的策略特定脚本）

## ✅ 清理完成

已删除确认未使用的文件，策略特定的回测和评估脚本已保留，等待后续详细评估。

## 🔍 检查方法

要确认 backtesting 和 evaluation 脚本是否仍在使用，可以：
1. 检查 Makefile 中是否有相关命令
2. 检查诊断脚本是否使用这些函数
3. 检查是否有文档说明这些脚本的用途

