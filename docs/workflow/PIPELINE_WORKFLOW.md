# Pipeline Workflow

本文档描述完整的交易系统工作流命令序列。

> **注意**: 本文档是 `README_CN.md` 中工作流部分的详细扩展版本。  
> 基础命令和快速开始请参考 `README_CN.md`。

## 完整工作流命令序列

### 1. 生成预测 (NN Multi-head Inference)

```bash
mlbot nnmultihead predict \
  --feature-store-root feature_store \
  --layer tier0 \
  --timeframe 240T \
  --model-path results/runs/.../model.pt \
  --config-dir config/nnmultihead/path_primitives_4h_80h_min \
  --output-dir results/preds/
```

**输出**: `preds_*.parquet` (每个 symbol 一个文件)

---

### 2. 生成 Regime 分类 (Physics/Regime Classifier)

```bash
mlbot rule physics-regime \
  --preds results/preds/ \
  --output /tmp/physics_regime.parquet \
  --stats-output /tmp/physics_regime_stats.json
```

**输出**: `physics_regime.parquet` (包含 `regime`: TC_REGIME/TE_REGIME/MEAN_REGIME/NO_TRADE)

**说明**:
- Regime 分类器输出 `regime` (TC_REGIME/TE_REGIME/MEAN_REGIME/NO_TRADE) 用于 Gate 层
- 这是 Physics/Regime 分类器的输出，基于市场物理可行性判断
- **注意**: Router 模块已被移除。Execution 层现在直接根据 archetype 选择 ret_mean 或 ret_trend

---

### 3. 构建 Execution 日志 (Build Execution Logs)

```bash
mlbot rl build-execution-logs \
  --preds results/preds/ \
  --output /tmp/logs_execution.parquet \
  --returns-source rr_execution
```

**输出**: `logs_execution.parquet` (包含 `ret_mean`, `ret_trend` 等)

**说明**:
- 计算 counterfactual execution returns (`ret_mean`, `ret_trend`)
- 这些 returns 已经包含了止损止盈的执行逻辑（在 `rr_execution` 模式下）
- Execution 层根据 archetype 选择使用哪个 return：
  - TC/TE → 使用 `ret_trend` (趋势风格执行)
  - FR/ET → 使用 `ret_mean` (均值回归风格执行)
- **注意**: 不再输出 `mode` 列，因为 Router 模块已被移除

---

### 4. 应用 Gate 过滤 ⭐ (必需，自动执行)

```bash
mlbot rule apply-tree-gate \
  --logs /tmp/logs_execution.parquet \
  --out /tmp/logs_execution_gated.parquet \
  --live-config config/nnmultihead/live/meta_router_live_config.yaml
```

**输出**: `logs_execution_gated.parquet` (包含 `gate_decision`, `gate_archetype`, `gate_ok`)

**说明**:
- 这一步应用 Gate 规则过滤交易
- 输入 `logs` 必须包含 `regime` 列（来自 physics-regime 步骤）
- 根据 `regime` 和 `live_config` 中的 `enabled_archetypes` 决定哪些交易可以执行
- 过滤掉 NO_TRADE regime 中的交易
- 应用 semantic score thresholds（如果提供）
  - TC_REGIME: 使用上限阈值（p95），veto 高分毒区
  - TE_REGIME: 使用下限阈值（p10），veto 低分噪声
- 输出 `gate_archetype` 列（TC/TE/FR/ET/NO_TRADE），用于 Execution 层选择 ret_mean 或 ret_trend

---

### 5. 生成 E2E KPI 报告

```bash
mlbot rule diagnose-e2e-kpi \
  --logs /tmp/logs_execution_gated.parquet \
  --regime /tmp/physics_regime.parquet \
  --gate /tmp/logs_execution_gated.parquet \
  --output-json results/e2e_kpi/e2e_kpi_report.json \
  --output-md results/e2e_kpi/e2e_kpi_report.md \
  --no-regime-filter
```

**输出**: 
- `e2e_kpi_report.json`: JSON 格式的 KPI 数据
- `e2e_kpi_report.md`: Markdown 格式的报告

**说明**:
- `--gate` 参数提供 archetype 信息（从 Gate 输出中读取）
- `--no-regime-filter` 生成对比报告（有/无 regime 过滤）

**最新报告位置**:
- 默认输出到 `results/e2e_kpi/e2e_kpi_report.md` 和 `results/e2e_kpi/e2e_kpi_report.json`
- 可以通过 `--output-md` 和 `--output-json` 参数指定输出路径

---

## 可选诊断命令

### Gate 过滤诊断

```bash
mlbot rule diagnose-gate-filtering \
  --logs /tmp/logs_execution.parquet \
  --regime /tmp/physics_regime.parquet \
  --live-config config/nnmultihead/live/meta_router_live_config.yaml \
  --output-md /tmp/gate_filtering_diagnosis.md
```

**输出**: Gate 过滤效果分析报告

---

### TC Regime 执行诊断

```bash
mlbot rule diagnose-tc-regime-execution \
  --logs /tmp/logs_execution_gated.parquet \
  --regime /tmp/physics_regime.parquet \
  --output-json /tmp/tc_execution.json \
  --output-md /tmp/tc_execution.md
```

**输出**: TC_REGIME 子集内的执行 KPI 分析

---

## 工作流集成建议

### 自动化脚本

可以创建一个脚本来自动执行完整工作流：

```bash
#!/bin/bash
# run_full_pipeline.sh

# 1. Predict
mlbot nnmultihead predict --output-dir results/preds/ ...

# 2. Regime classification
mlbot rule physics-regime --preds results/preds/ --output /tmp/physics_regime.parquet

# 3. Build execution logs
mlbot rl build-execution-logs \
  --preds results/preds/ \
  --output /tmp/logs_execution.parquet \
  --returns-source rr_execution

# 4. Gate filtering (默认执行)
mlbot rule apply-tree-gate \
  --logs /tmp/logs_execution.parquet \
  --out /tmp/logs_execution_gated.parquet \
  --live-config config/nnmultihead/live/meta_router_live_config.yaml

# 5. E2E report
mlbot rule diagnose-e2e-kpi \
  --logs /tmp/logs_execution_gated.parquet \
  --regime /tmp/physics_regime.parquet \
  --gate /tmp/logs_execution_gated.parquet \
  --output-md results/e2e_kpi/e2e_kpi_report.md
```

---

## 关键点

1. **Gate 过滤是必需的**: 步骤 4 应该默认执行，因为：
   - 过滤掉 NO_TRADE regime 中的交易
   - 应用 archetype 级别的 Gate 规则
   - 提供准确的 archetype 分类（TC/TE/FR/ET）
   - **输出带 archetype 信息的日志**，供 E2E 报告使用

2. **E2E 报告需要 Gate 输出**: 使用 `--gate` 参数提供准确的 archetype 信息

3. **ET/FR 交易缺失原因**:
   - **主要原因**: MEAN_REGIME 中没有交易（MEAN_REGIME 分类条件可能过于严格）
   - **次要原因**: Gate 过滤可能过于严格，导致 FR/ET archetype 被 veto
   - **检查方法**: 
     ```bash
     # 检查 MEAN_REGIME 分布
     python3 -c "import pandas as pd; df = pd.read_parquet('/tmp/physics_regime.parquet'); print(df['regime'].value_counts())"
     
     # 检查 Gate 输出中的 archetype 分布
     python3 -c "import pandas as pd; logs = pd.read_parquet('/tmp/logs_execution_gated.parquet'); print(logs['gate_archetype'].value_counts())"
     ```

---

## 配置说明

### Live Config (`meta_router_live_config.yaml`)

控制哪些 archetype 在哪些 regime 中启用：

```yaml
enabled_archetypes:
  TREND:
    - TrendContinuationTC
    - TrendExpansionTE
  MEAN:
    - FailureReversionFR
    - ExhaustionTurnET
  NO_TRADE: []
```

### Execution Archetypes (`execution_archetypes.yaml`)

定义每个 archetype 的 Gate 规则和证据要求。

---

## 常见问题

### Q: 为什么 ET/FR 交易没有？

**A**: 可能的原因：
1. **MEAN_REGIME 中没有交易**: MEAN_REGIME 分类条件可能过于严格
2. **MEAN_REGIME 中的交易被 Gate 过滤**: 即使有 MEAN_REGIME 的交易，也可能被 Gate 规则 veto
3. **Execution 层选择**: Execution 层根据 archetype 选择 ret_mean 或 ret_trend，如果 archetype 不正确，可能导致交易缺失

**检查方法**:
```bash
# 1. 检查 MEAN_REGIME 的分布
mlbot rule diagnose-fr-et-filtering \
  --preds results/preds/ \
  --regime /tmp/physics_regime.parquet \
  --output-md /tmp/fr_et_diagnosis.md

# 2. 检查 Gate 输出中的 archetype 分布
python3 -c "import pandas as pd; df = pd.read_parquet('/tmp/logs_execution_gated.parquet'); print(df['gate_archetype'].value_counts())"
```

### Q: 为什么 gate_plan 有两个开关（enabled 和 kind）？

**A**: 
- **`enabled`**: 是否启用 Gate 过滤（总开关）
- **`kind`**: Gate 类型（`tree_gate_veto` 表示使用树规则否决机制）
- 设计原因：未来可能支持多种 Gate 类型（如 `tree_gate_veto`, `score_floor`, `evidence_based` 等），所以需要 `kind` 来指定类型
- 当前只有 `tree_gate_veto` 一种类型，但架构预留了扩展性

### Q: 为什么 apply-tree-gate 是单独命令？

**A**: 
- `mlbot nnmultihead pipeline-3action-e2e` 会自动执行 Gate 过滤（如果 gate_plan 配置启用）
- 单独命令主要用于：
  - 调试 Gate 过滤效果
  - 重新应用 Gate 规则到已有的 logs 文件
  - 生成带 archetype 信息的日志供 E2E 报告使用

## 更新历史

- 2025-01-21: 添加 `apply-tree-gate` 命令到 CLI，作为工作流的默认步骤 4
