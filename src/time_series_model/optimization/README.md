# 参数优化 (Optimization)

超参数优化工具，使用 Optuna 进行自动化参数搜索。

## 脚本说明

### 1. `optuna_risk_search.py` - 风险参数优化

**用途**: 优化已训练模型的**风险管理和交易执行参数**

**优化参数**:
- 止损/止盈比例 (`sl`, `tp`)
- 信号阈值 (`sig`)
- 风险百分比 (`risk`)
- 杠杆倍数 (`lev`)
- 加仓参数 (`adds`, `add_frac`)
- ATR 追踪止损参数 (`atr_trail`, `atr_k`)
- 最大并发持仓 (`max_cc`)

**工作流程**:
1. 加载已训练的模型和特征工程器
2. 准备回测数据（从ZIP文件提取）
3. 对每个 trial 运行回测
4. 优化目标：最大化 `return - 0.5 * drawdown`（最大回撤限制在10%）

**使用场景**: 模型已训练完成，需要优化交易执行和风险管理参数

**使用示例**:
```bash
python src/time_series_model/optimization/optuna_risk_search.py
```

---

### 2. `ts_sr_reversal_optuna.py` - SR反转信号参数优化

**用途**: 优化**SR反转策略的信号生成参数**（通过环境变量配置）

**优化参数** (通过环境变量 `SR_SIGNAL_*`):
- `SR_SIGNAL_MIN_STRENGTH`: 最小支撑/阻力强度
- `SR_SIGNAL_MIN_SUPPORT`: 最小支撑分数
- `SR_SIGNAL_MIN_RESISTANCE`: 最小阻力分数
- `SR_SIGNAL_TOLERANCE_MULT`: 容差倍数
- `SR_SIGNAL_MIN_TOLERANCE_PCT`: 最小容差百分比
- `SR_SIGNAL_REQUIRE_FIRST_TOUCH`: 是否需要首次触碰
- `SR_SIGNAL_MAX_TOUCHES`: 最大触碰次数
- `SR_SIGNAL_ZONE_PRECISION`: 价格精度

**工作流程**:
1. 加载策略配置
2. 准备训练/测试数据
3. 对每个 trial 设置环境变量，运行策略特征对比
4. 优化目标：最大化交叉验证指标 (`avg_cv_metric`)

**使用场景**: 需要优化SR反转信号的识别参数，影响信号生成质量

**使用示例**:
```bash
python src/time_series_model/optimization/ts_sr_reversal_optuna.py \
    --strategy-config config/strategies/sr_reversal \
    --symbol BTCUSDT \
    --data-path data/parquet_data \
    --timeframe 240T \
    --n-trials 30 \
    --output-dir results/sr_reversal_optuna
```

---

## 主要区别

| 特性 | `optuna_risk_search.py` | `ts_sr_reversal_optuna.py` |
|------|------------------------|---------------------------|
| **优化对象** | 风险管理和交易执行参数 | 信号生成参数 |
| **数据来源** | 已训练模型 + ZIP数据文件 | 策略配置 + Parquet数据 |
| **评估方式** | 回测结果（收益/回撤） | 交叉验证指标 |
| **参数传递** | 函数参数 | 环境变量 |
| **适用阶段** | 模型训练后 | 策略开发阶段 |
| **输出** | 风险参数JSON | 信号参数JSON + CSV历史 |

## 其他策略优化需求

目前项目中有以下策略类型：
- **SR Reversal** (已有优化工具 ✅)
- **SR Breakout** (可能需要类似优化)
- **Trend Following** (可能需要类似优化)
- **Compression Breakout** (可能需要类似优化)

如果其他策略也有类似的信号参数或风险参数需要优化，可以考虑：
1. 复用 `ts_sr_reversal_optuna.py` 的模式（如果使用环境变量配置）
2. 复用 `optuna_risk_search.py` 的模式（如果优化风险参数）
3. 创建策略特定的优化脚本

