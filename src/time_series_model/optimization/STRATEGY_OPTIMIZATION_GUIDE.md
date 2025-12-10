# 策略参数优化指南

本文档说明各策略的参数优化需求和实现建议。

## 当前优化工具状态

### ✅ SR Reversal 策略
- **优化脚本**: `ts_sr_reversal_optuna.py`
- **优化参数**: 通过环境变量 `SR_SIGNAL_*` 配置
- **参数类型**: 信号生成参数（支撑/阻力识别、容差等）
- **状态**: 已实现

---

## 其他策略优化需求分析

### 1. Trend Following 策略

**当前参数** (`trend_following_label.py`):
- `horizon`: 未来收益率窗口（默认 50）
- `rank_window`: 滚动排名窗口（默认 200）
- `min_periods`: 最小周期数（默认 50）

**优化建议**:
- 这些参数影响标签质量，可以考虑优化
- 可以创建类似 `ts_trend_following_optuna.py` 的脚本
- 优化目标：最大化交叉验证指标或信息系数

**实现方式**:
```python
# 可以通过函数参数或配置文件传递
# 不需要环境变量（与SR Reversal不同）
```

---

### 2. SR Breakout 策略

**当前参数** (`sr_breakout_label.py`):
- `max_holding_bars`: 最大持仓周期（默认 50）
- `max_rr`: 最大R/R上限（默认 3.0）
- `stop_loss_r`: 止损R倍数（默认 1.0）
- `atr_window`: ATR窗口（默认 14）

**优化建议**:
- 这些参数影响标签质量和交易逻辑
- 可以创建 `ts_sr_breakout_optuna.py`
- 优化目标：最大化R/R标签质量或回测表现

**实现方式**:
```python
# 可以通过函数参数传递
# 或者创建配置类（类似 SRSignalConfig）
```

---

### 3. Compression Breakout 策略

**当前参数** (`compression_breakout_label.py`):
- `lookback_window`: 压缩检测窗口（默认 10）
- `confirmation_bars`: 突破确认K线数（默认 3）
- `volume_lookback`: 成交量平均窗口（默认 20）
- `min_volume_ratio`: 最小成交量比率（默认 1.0）
- `breakout_threshold`: 突破阈值（ATR倍数，默认 1.5）
- `atr_window`: ATR窗口（默认 14）

**优化建议**:
- 这些参数影响突破识别的准确性
- 可以创建 `ts_compression_breakout_optuna.py`
- 优化目标：最大化突破识别准确率或标签质量

**实现方式**:
```python
# 通过函数参数传递
# 可以创建配置类统一管理
```

---

## 通用优化框架建议

### 方案A: 函数参数优化（适用于 Trend Following, SR Breakout）

```python
# 示例：ts_trend_following_optuna.py
def sample_params(trial: optuna.Trial):
    return {
        "horizon": trial.suggest_int("horizon", 30, 100),
        "rank_window": trial.suggest_int("rank_window", 100, 300),
        "min_periods": trial.suggest_int("min_periods", 30, 100),
    }

def objective(trial: optuna.Trial):
    params = sample_params(trial)
    # 使用参数运行策略评估
    result = execute_strategy_with_params(params)
    return result["metric"]
```

### 方案B: 环境变量优化（适用于 SR Reversal 类型）

```python
# 示例：如果策略支持环境变量配置
ENV_KEYS = ["TREND_FOLLOWING_HORIZON", "TREND_FOLLOWING_RANK_WINDOW"]

@contextmanager
def strategy_env(params: Dict[str, str]):
    # 设置环境变量
    # 运行策略
    # 恢复环境变量
    pass
```

### 方案C: 配置文件优化

```python
# 创建临时配置文件，运行策略，评估结果
def objective(trial: optuna.Trial):
    params = sample_params(trial)
    config_path = create_temp_config(params)
    result = run_strategy_with_config(config_path)
    return result["metric"]
```

---

## 风险参数优化（通用）

所有策略都可以使用 `optuna_risk_search.py` 的模式来优化：
- 止损/止盈比例
- 风险百分比
- 杠杆倍数
- 加仓参数
- ATR追踪止损参数

**建议**: 为每个策略创建专门的风险参数优化脚本，或创建通用的风险优化框架。

---

## 实施优先级

1. **高优先级**: 
   - ✅ SR Reversal（已完成）
   - SR Breakout（参数较多，影响大）

2. **中优先级**:
   - Trend Following（参数相对简单）
   - Compression Breakout（需要先检查参数）

3. **低优先级**:
   - 通用风险参数优化框架
   - 多策略联合优化

---

## 测试建议

为每个新的优化脚本创建集成测试：
- 参考 `tests/integration/test_optimization_integration.py`
- 测试参数采样函数
- 测试目标函数
- 测试完整流程（使用mock）

