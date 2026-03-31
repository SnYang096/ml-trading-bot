# 实盘归因工作流程

## 概述

当实盘出现连续亏损或Sharpe下降时，通过分层诊断定位问题层，并提供修复建议。

## 检测层级

1. **Layer 1: NN Path Head** - IC、Rank IC、Calibration
2. **Layer 2: Gate** - ΔSharpe、Precision@Trade、False Allow Rate
3. **Layer 3: Archetype** - Stability、Hit-rate
4. **Layer 4: Execution** - R-multiple、MAE control、Slippage PnL
5. **Layer 5: PCM (Portfolio Capital Management)** - Slot allocation、Risk budgeting、Position sizing
6. **Layer 6: Outcome/Attribution** - Realized PnL、NN drift

## 触发条件

- 连续亏损次数 > N (默认5)
- Sharpe下降幅度 > Δ (默认-0.5)
- 交易数下降 > M% (默认20%)

## 工作流程

### 综合归因分析

```bash
mlbot diagnose production-attribution \
  --production-logs results/production_logs.parquet \
  --baseline-logs results/baseline_smoke_test/logs_baseline.parquet \
  --output-dir results/diagnostics/production_attribution \
  --alert-thresholds '{"consecutive_losses": 5, "sharpe_drop": -0.5, "trade_count_drop": -0.2}'
```

### 分层诊断

#### Layer 1: NN Path Head

```bash
python scripts/diagnose_nn_path_head.py \
  --preds results/predictions \
  --logs results/production_logs.parquet \
  --output results/diagnostics/nn_path_head.md
```

**检测指标**:
- IC(dir)下降
- Rank IC(mfe/mae)下降
- Calibration curve漂移

**修复方法**:
- 检查特征质量（coverage、latency）
- 检查模型输入特征是否变化
- 考虑重新训练或校准

#### Layer 2: Gate

```bash
python scripts/diagnose_gate_performance.py \
  --logs results/production_logs.parquet \
  --baseline results/baseline_smoke_test/logs_baseline.parquet \
  --output results/diagnostics/gate_performance.md
```

**检测指标**:
- Gate on vs off的Sharpe差异变化
- False Allow Rate上升
- Rule-level attribution变化

**修复方法**:
- 检查gate规则是否过拟合
- 调整gate规则阈值（使用平坦高原方法）
- 检查物理特征是否漂移

#### Layer 3: Archetype

```bash
python scripts/diagnose_archetype_stability.py \
  --logs results/production_logs.parquet \
  --baseline results/baseline_smoke_test/logs_baseline.parquet \
  --output results/diagnostics/archetype_stability.md
```

**检测指标**:
- 各archetype的稳定性变化
- Hit-rate下降
- Archetype分布变化

**修复方法**:
- 检查archetype选择逻辑
- 检查多archetype冲突处理
- 调整archetype优先级

#### Layer 4: Execution

```bash
python scripts/diagnose_execution_performance.py \
  --logs results/production_logs.parquet \
  --output results/diagnostics/execution_performance.md
```

**检测指标**:
- R-multiple下降
- MAE控制失效
- Slippage-adjusted PnL下降

**修复方法**:
- 检查stop-loss/take-profit配置
- 检查position sizing
- 检查execution timing

#### Layer 5: PCM

```bash
mlbot diagnose pcm-performance \
  --logs results/production_logs.parquet \
  --baseline results/baseline_smoke_test/logs_baseline.parquet \
  --output results/diagnostics/pcm_performance.md
```

**检测指标**:
- Slot allocation效率
- Risk budgeting执行情况
- Position sizing合理性
- Slot rotation频率

**修复方法**:
- 检查PCM policy配置（capacity_limit、risk_release_threshold等）
- 调整slot rotation逻辑
- 检查archetype兼容性规则
- 调整风险预算分配

#### Layer 6: Outcome/Attribution

```bash
mlbot diagnose outcome-attribution \
  --logs results/production_logs.parquet \
  --baseline results/baseline_smoke_test/logs_baseline.parquet \
  --output results/diagnostics/outcome_attribution.md
```

**检测指标**:
- Realized PnL vs Predicted
- NN drift (pred vs real)
- Gate规则过拟合

**修复方法**:
- 分析attribution，找出问题层
- 根据attribution结果，回到对应层修复

## PCM层说明

PCM层独立存在，负责：
- Slot管理（capacity_limit=2，slot rotation逻辑）
- Archetype兼容性检查
- Risk release检查
- Position replacement（基于ppath dominance）

**位置**: 在Execution层之后，Outcome层之前

**与Execution层的关系**: 
- Execution层决定单个archetype的执行（stop-loss/take-profit）
- PCM层决定多个archetype的仓位分配和风险预算

## 使用建议

1. **定期监控**: 设置定期运行归因分析
2. **及时响应**: 发现degradation后立即进行分层诊断
3. **记录修复**: 记录每次修复的变更和效果
