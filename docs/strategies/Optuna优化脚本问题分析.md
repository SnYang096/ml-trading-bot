# Optuna 优化脚本问题分析

## 当前问题

### 1. Optuna 脚本在优化什么？

**当前行为：**
- `ts_sr_reversal_optuna.py` 设置环境变量 `SR_SIGNAL_MIN_STRENGTH` 等
- 这些环境变量原本用于**信号生成的阈值**（过滤哪些 SR 区域可以生成信号）

**问题：**
- ❌ 这些环境变量**不起作用**，因为：
  - 标签生成使用 `compute_sr_reversal_label_full_scan`（全量扫描）
  - 不依赖 `_generate_sr_reversal_signals` 函数
  - `_apply_env_overrides` 函数已被删除，且从未被调用

### 2. 应该优化什么？

**应该优化的是：最终信号给出的阈值**

在 `backtest.yaml` 中配置的阈值：
```yaml
long_entry_threshold: 0.6    # 模型预测概率 >= 0.6 时才做多
long_exit_threshold: 0.45     # 模型预测概率 <= 0.45 时平多仓
short_entry_threshold: 0.4    # 模型预测概率 <= 0.4 时才做空
short_exit_threshold: 0.55    # 模型预测概率 >= 0.55 时平空仓
```

**实际使用逻辑**（`run_vectorbt_backtest`）：
```python
if use_signal_direction and signal_col in df.columns:
    # SR reversal：方向由 signal 决定，preds 只控制"是否参与这笔 SR 反转交易"
    signal_series = df[signal_col].fillna(0).astype(float)
    
    base_long_entries = preds_series >= long_entry_threshold
    base_short_entries = preds_series <= short_entry_threshold
    
    long_entries = (signal_series > 0) & base_long_entries
    short_entries = (signal_series < 0) & base_short_entries
```

**关键点：**
- `signal` 列决定**方向**（做多/做空）
- `preds`（模型预测概率）决定**是否参与**这笔交易
- `long_entry_threshold` 和 `short_entry_threshold` 才是应该优化的参数！

## 解决方案

### ✅ 已实现：修改 Optuna 脚本优化预测阈值

已修改 `ts_sr_reversal_optuna.py` 来优化 `backtest.yaml` 中的阈值：

**主要改动：**
1. **删除了环境变量相关代码**：不再使用 `SR_SIGNAL_*` 环境变量
2. **修改 `sample_params` 函数**：优化预测阈值而不是信号生成阈值
3. **修改 `objective` 函数**：临时修改 `strategy_cfg.backtest.params` 来设置不同的阈值

**新的优化参数：**
- `long_entry_threshold`: 0.4-0.8（模型预测 >= 此值才做多）
- `long_exit_threshold`: 0.2-0.5（模型预测 <= 此值平多仓）
- `short_entry_threshold`: 0.2-0.6（模型预测 <= 此值才做空）
- `short_exit_threshold`: 0.5-0.8（模型预测 >= 此值平空仓）

**约束检查：**
- `long_entry_threshold > long_exit_threshold`（避免开仓后立即平仓）
- `short_exit_threshold > short_entry_threshold`（避免开仓后立即平仓）

### 方案 2：优化信号生成阈值（如果仍需要）

如果确实需要优化信号生成阈值（用于诊断/分析），应该：
1. 在诊断脚本中直接传递 `SRSignalConfig` 参数
2. 不要使用环境变量（因为不起作用）

## 当前流程分析

### 标签生成流程（已优化）
```
全量扫描 → 对所有 K 线计算标签 → 模型训练
         ↑ 不依赖信号过滤
```

### 信号生成流程（诊断用）
```
_generate_sr_reversal_signals → 生成 signal 列（+1/-1/0）
         ↑ 用于诊断/分析，不影响标签生成
```

### 最终交易信号流程
```
模型预测 (preds) + signal 列 → 阈值判断 → 开仓/平仓
         ↑                    ↑
    模型输出概率           backtest.yaml 中的阈值
```

## 建议

1. **修改 Optuna 脚本**：优化 `long_entry_threshold` 等预测阈值
2. **删除环境变量相关代码**：因为不起作用
3. **更新文档**：说明应该优化什么参数

## 相关文件

- `src/time_series_model/optimization/ts_sr_reversal_optuna.py` - 需要修改
- `config/strategies/sr_reversal/backtest.yaml` - 包含应该优化的阈值
- `scripts/train_strategy_pipeline.py` - `run_vectorbt_backtest` 函数使用这些阈值

