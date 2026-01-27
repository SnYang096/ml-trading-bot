# Pipeline Workflow

**状态**: ✅ 当前版本  
**最后更新**: 2026-01  
**相关文档**: [主文档索引](../README.md), [README_CN.md](../../README_CN.md)

本文档描述完整的交易系统工作流命令序列。

> **注意**: 本文档是 `README_CN.md` 中工作流部分的详细扩展版本。  
> 基础命令和快速开始请参考 `README_CN.md`。

## 完整工作流命令序列

> **重要**: 每一步都有明确的输入输出和日志文件，支持独立执行和断点续传。

### 0. FeatureStore构建（如果需要新特征）

```bash
mlbot nnmultihead build-feature-store \
  --task-spec config/tasks/task_spec_highcap6_2024_202510.yaml \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31 \
  --feature-store-root feature_store \
  --layer nnmh_highcap6_240T_2024_2025_test \
  --warmup-months 1 \
  --no-docker
```

**输出**: 
- `feature_store/<layer>/<symbol>/240T/*.parquet` (按月存储的特征文件)

**日志**: `results/featurestore_build.log`

**说明**:
- 如果FeatureStore已存在且包含所需特征，可跳过此步骤
- 包含反身性特征（OFCI, SHD）的FeatureStore需要tick数据，构建时间较长

---

### 1. 生成预测 (NN Multi-head Inference)

```bash
mlbot nnmultihead predict \
  --task-spec config/tasks/task_spec_highcap6_2024_202510.yaml \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --model results/nnmultihead/.../model.pt \
  --feature-store-layer nnmh_highcap6_240T_2024_with_reflexivity \
  --feature-store-root feature_store \
  --output-dir results/pipeline_<run_id>/preds \
  --no-docker
```

**输出**: 
- `results/pipeline_<run_id>/preds/preds_*.parquet` (每个 symbol 一个文件)

**日志**: `results/pipeline_<run_id>/predict.log`

**说明**:
- `<run_id>` 建议使用时间戳或描述性名称，如 `2024_reflexivity_validation`
- 预测结果包含 `pred_dir_prob`, `pred_mfe_atr`, `pred_mae_atr`, `pred_t_to_mfe`

---

### 2. 构建 Execution 日志 (Build Execution Logs)

---

```bash
mlbot nnmultihead build-execution-logs \
  --preds results/pipeline_<run_id>/preds \
  --model results/nnmultihead/.../model.pt \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --data-path data/parquet_data \
  --returns-source rr_execution \
  --output results/pipeline_<run_id>/logs_execution.parquet \
  --no-docker
```

**输出**: 
- `logs_execution.parquet` (包含 `ret_mean`, `ret_trend`, `drawdown` 等)

**日志**: `results/pipeline_<run_id>/build_logs.log`

**说明**:
- 计算 counterfactual execution returns (`ret_mean`, `ret_trend`)
- 这些 returns 已经包含了止损止盈的执行逻辑（在 `rr_execution` 模式下）
- Execution 层根据 archetype 选择使用哪个 return：
  - TC/TE → 使用 `ret_trend` (趋势风格执行)
  - FR/ET → 使用 `ret_mean` (均值回归风格执行)
- **注意**: 不再输出 `mode` 列（旧 regime/router 已移除）

---

### 3. 应用 Gate 过滤 ⭐ (必需，自动执行)

**(a) 提取 quantile keys（建议）**
```bash
mlbot rule extract-evidence-keys --no-docker \
  --config config/nnmultihead/execution_archetypes.yaml
```

**(b) 生成 evidence_quantiles.json（quantile_* 规则必需）**
```bash
mlbot rule build-evidence-quantiles --no-docker \
  --feature-store-root feature_store \
  --layer nnmh_highcap6_240T_2024_with_reflexivity \
  --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 --end-date 2024-12-31 \
  --keys <COMMA_SEPARATED_KEYS> \
  --out results/pipeline_<run_id>/evidence_quantiles.json
```

**(c) 应用 gate**

```bash
python scripts/apply_archetype_gate.py \
  --logs results/pipeline_<run_id>/logs_execution.parquet \
  --out results/pipeline_<run_id>/logs_execution_gated.parquet \
  --features-store-layer nnmh_highcap6_240T_2024_with_reflexivity \
  --features-store-root feature_store \
  --live-config config/nnmultihead/live/meta_router_live_config.yaml \
  --evidence-quantiles results/pipeline_<run_id>/evidence_quantiles.json \
  --timeframe 240T \
  --start-date 2024-01-01 --end-date 2024-12-31 \
  --no-docker
```

**输出**: 
- `logs_execution_gated.parquet` (包含 `gate_ok`, `gate_decision`, `gate_reasons`, `gate_archetype`)

**日志**: `results/pipeline_<run_id>/gate.log`

**说明**:
- 这一步应用 Gate 规则过滤交易
- Gate 完全基于 `when_then_rules` 与 `evidence_quantiles.json`
- 输出 `gate_archetype` 列（6 archetypes）用于后续 KPI 和执行归因

---

### 4. 添加反身性特征到Stage Logs（可选，如果FeatureStore包含反身性特征）

```bash
python scripts/add_reflexivity_features_to_logs.py \
  --preds results/pipeline_<run_id>/preds \
  --logs results/pipeline_<run_id>/logs_execution.parquet \
  --out-dir results/pipeline_<run_id>/exec_logs \
  --feature-store-dir feature_store \
  --feature-store-layer nnmh_highcap6_240T_2024_with_reflexivity \
  --data-path data/parquet_data \
  --timeframe 240T \
  --run-id <run_id> \
  --strategy-name pipeline-3action-e2e
```

**输出**: 
- `exec_logs/features/*.jsonl` (包含 `ofci_pct`, `shd_pct` 特征)

**日志**: `results/pipeline_<run_id>/add_reflexivity.log`

**说明**:
- 如果FeatureStore已包含反身性特征，此步骤会从FeatureStore加载
- 如果FeatureStore不包含，会重新计算（需要tick数据，较慢）

---

### 5. 构建Stage Logs（包含gate和execution）

```bash
python scripts/build_execution_log_stages.py \
  --preds results/pipeline_<run_id>/preds \
  --logs results/pipeline_<run_id>/logs_execution.parquet \
  --gated-logs results/pipeline_<run_id>/logs_execution_gated.parquet \
  --out-dir results/pipeline_<run_id>/exec_logs \
  --run-id <run_id> \
  --timeframe 240T \
  --strategy-name pipeline-3action-e2e
```

**输出**: 
- `exec_logs/{preds,router,gate,execution,returns,features}/*.jsonl` (分stage的日志文件)

**日志**: `results/pipeline_<run_id>/build_stages.log`

**说明**:
- `--gated-logs` 参数用于从gated logs中提取gate和execution信息
- 生成的gate stage包含：`blocked`, `decisions`, `reasons`, `archetype`
- 生成的execution stage包含：`intent`, `submit_order`, `gate_blocked`, `archetype`

---

### 6. 聚合Canonical Log

```bash
python scripts/aggregate_execution_log_stages.py \
  --stage-dir results/pipeline_<run_id>/exec_logs \
  --out results/pipeline_<run_id>/execution_log.jsonl
```

**输出**: 
- `execution_log.jsonl` (聚合后的canonical格式日志，包含所有stage)

**日志**: `results/pipeline_<run_id>/aggregate.log`

**说明**:
- Canonical log包含所有stage的信息，便于后续分析和验证

---

### 8. 生成 E2E KPI 报告

```bash
mlbot rule diagnose-e2e-kpi \
  --logs results/pipeline_<run_id>/logs_execution_gated.parquet \
  --regime results/pipeline_<run_id>/physics_regime.parquet \
  --gate results/pipeline_<run_id>/logs_execution_gated.parquet \
  --output-json results/pipeline_<run_id>/e2e_kpi_report.json \
  --output-md results/pipeline_<run_id>/e2e_kpi_report.md \
  --no-regime-filter \
  --no-docker
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

可以创建一个脚本来自动执行完整工作流（见 `scripts/run_full_pipeline.py`）：

```bash
#!/bin/bash
# run_full_pipeline.sh

RUN_ID="pipeline_$(date +%Y%m%d_%H%M%S)"

# 0. FeatureStore构建（如果需要）
mlbot nnmultihead build-feature-store \
  --task-spec config/tasks/task_spec_highcap6_2024_202510.yaml \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --feature-store-root feature_store \
  --layer nnmh_highcap6_240T_2024_with_reflexivity \
  --warmup-months 1 \
  --no-docker 2>&1 | tee results/${RUN_ID}/featurestore_build.log

# 1. Predict
mlbot nnmultihead predict \
  --task-spec config/tasks/task_spec_highcap6_2024_202510.yaml \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --model results/nnmultihead/.../model.pt \
  --feature-store-layer nnmh_highcap6_240T_2024_with_reflexivity \
  --output-dir results/${RUN_ID}/preds \
  --no-docker 2>&1 | tee results/${RUN_ID}/predict.log

# 2. Regime classification
mlbot rule physics-regime \
  --preds results/${RUN_ID}/preds \
  --output results/${RUN_ID}/physics_regime.parquet \
  --stats-output results/${RUN_ID}/physics_regime_stats.json \
  --no-docker 2>&1 | tee results/${RUN_ID}/regime.log

# 3. Build execution logs
mlbot nnmultihead build-execution-logs \
  --preds results/${RUN_ID}/preds \
  --model results/nnmultihead/.../model.pt \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --data-path data/parquet_data \
  --returns-source rr_execution \
  --output results/${RUN_ID}/logs_execution.parquet \
  --no-docker 2>&1 | tee results/${RUN_ID}/build_logs.log

# 4. Gate filtering
mlbot rule apply-tree-gate \
  --logs results/${RUN_ID}/logs_execution.parquet \
  --regime results/${RUN_ID}/physics_regime.parquet \
  --out results/${RUN_ID}/logs_execution_gated.parquet \
  --features-store-layer nnmh_highcap6_240T_2024_with_reflexivity \
  --features-store-root feature_store \
  --execution-archetypes config/nnmultihead/execution_archetypes.yaml \
  --live-config config/nnmultihead/live/meta_router_live_config.yaml \
  --no-docker 2>&1 | tee results/${RUN_ID}/gate.log

# 5. Add reflexivity features (optional)
python scripts/add_reflexivity_features_to_logs.py \
  --preds results/${RUN_ID}/preds \
  --logs results/${RUN_ID}/logs_execution.parquet \
  --out-dir results/${RUN_ID}/exec_logs \
  --feature-store-dir feature_store \
  --feature-store-layer nnmh_highcap6_240T_2024_with_reflexivity \
  --data-path data/parquet_data \
  --timeframe 240T \
  --run-id ${RUN_ID} \
  --strategy-name pipeline-3action-e2e 2>&1 | tee results/${RUN_ID}/add_reflexivity.log

# 6. Build stage logs
python scripts/build_execution_log_stages.py \
  --preds results/${RUN_ID}/preds \
  --logs results/${RUN_ID}/logs_execution.parquet \
  --gated-logs results/${RUN_ID}/logs_execution_gated.parquet \
  --out-dir results/${RUN_ID}/exec_logs \
  --run-id ${RUN_ID} \
  --timeframe 240T \
  --strategy-name pipeline-3action-e2e 2>&1 | tee results/${RUN_ID}/build_stages.log

# 7. Aggregate canonical log
python scripts/aggregate_execution_log_stages.py \
  --stage-dir results/${RUN_ID}/exec_logs \
  --out results/${RUN_ID}/execution_log.jsonl 2>&1 | tee results/${RUN_ID}/aggregate.log

# 8. E2E report
mlbot rule diagnose-e2e-kpi \
  --logs results/${RUN_ID}/logs_execution_gated.parquet \
  --regime results/${RUN_ID}/physics_regime.parquet \
  --gate results/${RUN_ID}/logs_execution_gated.parquet \
  --output-md results/${RUN_ID}/e2e_kpi_report.md \
  --output-json results/${RUN_ID}/e2e_kpi_report.json \
  --no-regime-filter \
  --no-docker 2>&1 | tee results/${RUN_ID}/e2e_kpi.log
```

---

## 关键点

1. **Gate 过滤是必需的**: 步骤 4 应该默认执行，因为：
   - 过滤掉 NO_TRADE regime 中的交易
   - 应用 archetype 级别的 Gate 规则
   - 提供准确的 archetype 分类（TC/TE/FR/ET）
   - **输出带 archetype 信息的日志**，供 E2E 报告使用

2. **E2E 报告需要 Gate 输出**: 使用 `--gate` 参数提供准确的 archetype 信息

3. **Safety 硬停机优先于下游流程**：
   - Live enforcement 在下单前先做 Safety 判定（kill-switch + 日内限制 + 冷却恢复）
   - 日亏损触发后仅跨日恢复（当日回归也不恢复）
   - EVT 极端风险仅作软告警，不触发硬停机

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
