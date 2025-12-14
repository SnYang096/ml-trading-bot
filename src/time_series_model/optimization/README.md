# 参数优化 (Optimization)

超参数优化工具，使用 Optuna 进行自动化参数搜索。

## 脚本说明

### 1. `ts_sr_reversal_optuna.py` - SR反转预测阈值优化

**用途**: 优化**SR反转策略的预测阈值**（决定何时开仓/平仓）

**优化参数** (在 `backtest.yaml` 中配置):
- `long_entry_threshold`: 0.4-0.8（模型预测 >= 此值才做多）
- `long_exit_threshold`: 0.2-0.5（模型预测 <= 此值平多仓）
- `short_entry_threshold`: 0.2-0.6（模型预测 <= 此值才做空）
- `short_exit_threshold`: 0.5-0.8（模型预测 >= 此值平空仓）

**工作流程**:
1. 加载策略配置
2. 准备训练/测试数据
3. 对每个 trial 临时修改 `strategy_cfg.backtest.params` 中的阈值
4. 运行完整的训练和回测流程
5. 优化目标：最大化交叉验证指标 (`avg_cv_metric`)

**使用场景**: 
- 模型已训练完成，需要优化最终交易信号的阈值
- 这些阈值决定模型预测概率达到多少时才开仓/平仓
- **注意**: 不优化信号生成参数（标签生成使用全量扫描，不依赖信号过滤）
- **特别适合不平衡数据**：默认使用业务指标（夏普比率）而非准确率

**优化目标** (通过 `--objective` 参数):
- `sharpe` (默认) - 夏普比率，对不平衡数据鲁棒
- `total_return` - 总收益百分比，直接优化实际盈亏
- `cv_metric` - 交叉验证指标（可能受不平衡影响）
- `sharpe_with_cv_fallback` - 优先夏普比率，回退到 CV 指标

**不平衡数据约束**:
- `--min-trades N` - 最小交易次数（默认 10），防止零交易
- `--min-win-rate X` - 最小胜率（默认 0.0），防止低质量策略

**约束**:
- `long_entry_threshold > long_exit_threshold`（避免开仓后立即平仓）
- `short_exit_threshold > short_entry_threshold`（避免开仓后立即平仓）

**使用示例**:
```bash
# 基本使用（默认使用夏普比率，适合不平衡数据）
python src/time_series_model/optimization/ts_sr_reversal_optuna.py \
    --strategy-config config/strategies/sr_reversal \
    --symbol BTCUSDT \
    --data-path data/parquet_data \
    --timeframe 240T \
    --n-trials 30 \
    --output-dir results/sr_reversal_optuna

# 不平衡数据场景（正样本稀少）
python src/time_series_model/optimization/ts_sr_reversal_optuna.py \
    --strategy-config config/strategies/sr_reversal \
    --symbol BTCUSDT \
    --objective sharpe \
    --min-trades 20 \
    --min-win-rate 0.45 \
    --n-trials 50

# 使用总收益优化
python src/time_series_model/optimization/ts_sr_reversal_optuna.py \
    --strategy-config config/strategies/sr_reversal \
    --symbol BTCUSDT \
    --objective total_return \
    --min-trades 10 \
    --n-trials 30
```

---

---

### 2. `ts_sr_reversal_optuna_joint.py` - SR反转联合优化（模型超参数 + 预测阈值）

**用途**: 同时优化**模型超参数**和**预测阈值**，端到端优化业务目标。

**优化参数**:
- **模型超参数** (XGBoost/LightGBM):
  - XGBoost: `max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, `reg_alpha`, `reg_lambda`
  - LightGBM: `num_leaves`, `max_depth`, `learning_rate`, `min_data_in_leaf`, `feature_fraction`, `bagging_fraction`, `lambda_l1`, `lambda_l2` 等
- **预测阈值**:
  - `long_entry_threshold`, `long_exit_threshold`
  - `short_entry_threshold`, `short_exit_threshold`

**工作流程**:
1. 加载策略配置
2. 准备训练/测试数据
3. 对每个 trial：
   - 采样模型超参数和预测阈值
   - 使用这些参数训练模型
   - 应用阈值生成交易信号
   - 评估回测结果
4. 优化目标：最大化交叉验证指标 (`avg_cv_metric`)

**使用场景**:
- 需要同时优化模型和阈值
- 有充足的计算资源（每个 trial 都需要重新训练模型）
- 需要端到端优化业务目标

**使用示例**:
```bash
# 联合优化（模型 + 阈值）
python src/time_series_model/optimization/ts_sr_reversal_optuna_joint.py \
    --strategy-config config/strategies/sr_reversal \
    --symbol BTCUSDT \
    --data-path data/parquet_data \
    --timeframe 240T \
    --n-trials 50 \
    --output-dir results/sr_reversal_optuna_joint

# 只优化模型超参数
python src/time_series_model/optimization/ts_sr_reversal_optuna_joint.py \
    --strategy-config config/strategies/sr_reversal \
    --optimize-model-only \
    --n-trials 50

# 只优化阈值（等同于 ts_sr_reversal_optuna.py）
python src/time_series_model/optimization/ts_sr_reversal_optuna_joint.py \
    --strategy-config config/strategies/sr_reversal \
    --optimize-thresholds-only \
    --n-trials 30
```

**注意**:
- ⚠️ 计算成本高：每个 trial 都需要完整训练模型
- ⚠️ 建议先用 `ts_sr_reversal_optuna.py` 优化阈值，再用此脚本联合优化
- ✅ 适合需要大幅提升模型性能的场景

---

## 主要区别

| 特性 | `ts_sr_reversal_optuna.py` | `ts_sr_reversal_optuna_joint.py` |
|------|---------------------------|--------------------------------|
| **优化对象** | 预测阈值（开仓/平仓阈值） | 模型超参数 + 预测阈值 |
| **数据来源** | 策略配置 + Parquet数据 | 策略配置 + Parquet数据 |
| **评估方式** | 交叉验证指标 + 回测结果 | 交叉验证指标 + 回测结果 |
| **参数传递** | 临时修改策略配置对象 | 临时修改策略配置对象 |
| **适用阶段** | 模型训练后（优化交易阈值） | 模型训练阶段（端到端优化） |
| **计算成本** | 低（不训练模型） | 高（每个 trial 训练模型） |
| **输出** | 阈值参数JSON + CSV历史 + 回测结果 | 模型参数 + 阈值参数 + CSV历史 + 回测结果 |

## 其他策略优化需求

目前项目中有以下策略类型：
- **SR Reversal** (已有优化工具 ✅)
- **SR Breakout** (可能需要类似优化)
- **Trend Following** (可能需要类似优化)
- **Compression Breakout** (可能需要类似优化)

如果其他策略也有类似的信号参数或风险参数需要优化，可以考虑：
1. 复用 `ts_sr_reversal_optuna.py` 的模式（优化预测阈值）
2. 复用 `ts_sr_reversal_optuna_joint.py` 的模式（联合优化模型和阈值）
3. 创建策略特定的优化脚本

