# E2E KPI 指标解释与划分/分桶对下游的影响

## 一、Sharpe (Trades Only) vs Sharpe (E2E) 的区别

### 1.1 定义

**Sharpe (E2E)** (`sharpe_e2e`):
- **计算范围**: 所有时间点（包括 `NO_TRADE`）
- **收益序列**: 
  - `TREND` 模式 → 使用 `ret_trend`
  - `MEAN` 模式 → 使用 `ret_mean`
  - `NO_TRADE` 模式 → 收益为 0
- **公式**: `Sharpe = mean(returns) / std(returns) * sqrt(6 * 365)`
- **含义**: 衡量**整个时间序列**的风险调整收益（包含空仓期）

**Sharpe (Trades Only)** (`sharpe_trades_only`):
- **计算范围**: 只包含实际交易的时间点（`TREND` 或 `MEAN`）
- **收益序列**: 只包含 `ret_trend` 或 `ret_mean`（排除所有 `NO_TRADE`）
- **公式**: 同上，但只对交易行计算
- **含义**: 衡量**实际交易**的风险调整收益（排除空仓期）

### 1.2 为什么有两个 Sharpe？

**Sharpe (E2E)** 回答：
> "如果我全程持有这个策略（包括空仓期），我的风险调整收益是多少？"

**Sharpe (Trades Only)** 回答：
> "如果我只看实际交易的部分，我的风险调整收益是多少？"

### 1.3 实际例子

假设有 1000 个时间点：
- 400 个 `TREND` 交易，平均收益 0.001，标准差 0.002
- 100 个 `MEAN` 交易，平均收益 0.0005，标准差 0.0015
- 500 个 `NO_TRADE`，收益为 0

**Sharpe (E2E)**:
- 包含所有 1000 个点
- 平均收益 = (400 * 0.001 + 100 * 0.0005 + 500 * 0) / 1000 = 0.00045
- 标准差 = 计算所有 1000 个点的标准差（包含 500 个 0）
- **结果**: 较低（因为包含大量 0 值，降低了波动率）

**Sharpe (Trades Only)**:
- 只包含 500 个交易点
- 平均收益 = (400 * 0.001 + 100 * 0.0005) / 500 = 0.0009
- 标准差 = 只计算 500 个交易点的标准差
- **结果**: 较高（因为排除了空仓期的 0 值）

### 1.4 在你的报告中

```
Sharpe (E2E): 2.79
Sharpe (Trades Only): 4.31
```

**解读**:
- 实际交易的 Sharpe 是 4.31（很高）
- 但包含空仓期后，整体 Sharpe 降到 2.79
- **结论**: 交易质量很好，但交易频率可能偏低（42.21% 交易率）

---

## 二、划分方式和分桶如何影响下游

### 2.1 整体流程

```
NN Predictions (dir/mfe/mae/mtt)
    ↓
Physics/Regime Classifier
    ↓ 输出: regime (TC_REGIME/TE_REGIME/MEAN_REGIME/NO_TRADE)
    ↓ 输出: semantic_score (tc_semantic_score/te_semantic_score/fr_semantic_score/et_semantic_score)
    ↓
Build Execution Logs (build-execution-logs)
    ↓ 计算: ret_mean, ret_trend (counterfactual execution returns)
    ↓ 注意: 这些 returns 已包含止损止盈的执行逻辑（在 rr_execution 模式下）
    ↓
Gate (apply-tree-gate)
    ↓ 使用: regime + semantic_score + live_config
    ↓ 输出: gate_ok, gate_archetype (TC/TE/FR/ET/NO_TRADE)
    ↓
Execution
    ↓ 使用: gate_archetype 选择 ret_mean 或 ret_trend
    ↓ TC/TE → ret_trend, FR/ET → ret_mean
    ↓ 输出: 实际交易
```

**注意**: Router 模块已被移除。Execution 层现在直接根据 archetype 选择 ret_mean 或 ret_trend。

### 2.2 Regime 划分如何影响下游

#### 2.2.1 Regime 分类（Physics/Regime Classifier）

**输入**:
- Features: `atr`, `atr_percentile`, `jump_risk`, `dir_sign_consistency`, `path_length`, `range_expansion`, `deviation_z`
- NN 输出: `pred_dir_prob`

**输出**:
- `regime`: `TC_REGIME` / `TE_REGIME` / `MEAN_REGIME` / `NO_TRADE`
- `tc_semantic_score`: TC 语义分数（0-1）
- `te_semantic_score`: TE 语义分数（0-1）
- `fr_semantic_score`: FR 语义分数（0-1）⚠️ 待实现
- `et_semantic_score`: ET 语义分数（0-1）⚠️ 待实现

**语义分数计算方式**:

所有语义分数都使用 `np.nanmin`（取最小值）的保守策略，确保所有条件都满足：

1. **TC 语义分数** (`tc_semantic_score`):
   ```python
   tc_semantic_score = min(
       1.0 - atr_slope_pct,        # 波动率不扩张（越低越好）
       path_length_pct,            # 路径长度（越高越好）
       1.0 - dir_conf_std_pct,     # 方向稳定性（std 越低越好）
       dir_sign_consistency_pct    # 方向一致性（越高越好）
   )
   ```
   **含义**: 衡量"稳定趋势延续"的语义强度，分数越高表示越适合 TC 执行。

2. **TE 语义分数** (`te_semantic_score`):
   ```python
   te_semantic_score = min(
       atr_slope_pct,              # 波动率扩张（越高越好）
       range_expansion_pct,        # 区间扩张（越高越好）
       path_length_pct,            # 路径长度（越高越好）
       1.0 - dir_conf_std_pct,     # 方向稳定性（std 越低越好）
       dir_sign_consistency_pct    # 方向一致性（越高越好）
   )
   ```
   **含义**: 衡量"趋势扩张"的语义强度，分数越高表示越适合 TE 执行。

3. **FR 语义分数** (`fr_semantic_score`) ⚠️ **待实现**:
   ```python
   # 建议实现
   fr_semantic_score = min(
       deviation_z_abs / 5.0,       # 极端偏离（z-score 越高越好，归一化到 [0,1]）
       1.0 - dir_sign_consistency_pct,  # 方向不稳定（consistency 越低越好）
       path_length_pct,            # 路径过度延伸（越高越好）
       atr_percentile               # 波动率尖峰（越高越好）
   )
   ```
   **含义**: 衡量"失败反转"的语义强度，分数越高表示越适合 FR 执行。

4. **ET 语义分数** (`et_semantic_score`) ⚠️ **待实现**:
   ```python
   # 建议实现
   et_semantic_score = min(
       atr_percentile,             # 波动率尖峰（越高越好）
       path_length_pct,            # 路径过度延伸（越高越好）
       1.0 - dir_sign_consistency_pct,  # 方向不稳定（consistency 越低越好）
       deviation_z_abs / 5.0        # 极端偏离（z-score 越高越好，归一化到 [0,1]）
   )
   ```
   **含义**: 衡量"衰竭反转"的语义强度，分数越高表示越适合 ET 执行。

**注意**: FR 和 ET 的语义分数已在 `regime.py` 中实现。

**分类逻辑**（`multi_dim` 策略）:
```python
# TC_REGIME: 稳定趋势延续
tc_conditions = [
    dir_sign_consistency_pct >= 0.6,  # 方向稳定
    atr_slope_pct < 0.6,              # 波动率不扩张
    path_length_pct in [0.3, 0.7],    # 路径长度合理
    jump_risk_pct < 0.85              # 跳空风险可控
]
# 满足 3/4 条件 → TC_REGIME

# TE_REGIME: 趋势扩张
te_conditions = [
    dir_sign_consistency_pct >= 0.5,  # 方向仍明确
    atr_slope_pct >= 0.6,             # 波动率扩张
    range_expansion_pct >= 0.6,      # 区间突破
    jump_risk_pct in [0.6, 0.9]      # 跳空风险中等
]
# 满足 3/4 条件 → TE_REGIME

# MEAN_REGIME: 极端均值回归
mean_conditions = [
    deviation_z_abs >= 2.5,           # 极端偏离
    path_length_pct >= 0.8,           # 路径过度延伸
    dir_sign_consistency_pct <= 0.4,  # 方向不稳定
    atr_percentile >= 0.9            # 波动率尖峰
]
# 满足 3/4 条件 → MEAN_REGIME
```

#### 2.2.2 Build Execution Logs

**位置**: `mlbot rl build-execution-logs`

**功能**:
- 计算 counterfactual execution returns (`ret_mean`, `ret_trend`)
- 这些 returns 已经包含了止损止盈的执行逻辑（在 `rr_execution` 模式下）
- Execution 层根据 archetype 选择使用哪个 return：
  - TC/TE → 使用 `ret_trend` (趋势风格执行)
  - FR/ET → 使用 `ret_mean` (均值回归风格执行)
- **注意**: Router 模块已被移除，不再输出 `mode` 列

**输出**:
- `ret_mean`: MEAN 风格的 counterfactual execution return（已包含止损止盈逻辑）
- `ret_trend`: TREND 风格的 counterfactual execution return（已包含止损止盈逻辑）

---

#### 2.2.3 Regime → Gate 的影响

**Gate 使用 Regime 做以下判断**:

1. **NO_TRADE Regime 直接过滤**:
   ```python
   if regime == "NO_TRADE":
       gate_ok = False
       return
   ```

2. **Semantic Score Floor Veto**:
   ```python
   if regime == "TC_REGIME":
       if tc_semantic_score < tc_semantic_score_p05:  # P05 分位数
           gate_ok = False  # Veto
   
   if regime == "TE_REGIME":
       if te_semantic_score < te_semantic_score_p10:  # P10 分位数
           gate_ok = False  # Veto
   ```

3. **Archetype 候选选择**:
   ```python
   # Regime → Archetype 映射
   if regime == "TC_REGIME":
       candidates = ["TrendContinuationTC", ...]  # 优先 TC
   
   if regime == "TE_REGIME":
       candidates = ["TrendExpansionTE", ...]  # 优先 TE
   
   if regime == "MEAN_REGIME":
       candidates = ["FailureReversionFR", "ExhaustionTurnET"]  # MEAN archetypes
   ```

4. **Live Config 过滤**:
   ```python
   # 从 meta_router_live_config.yaml 读取
   enabled_archetypes = {
       "TREND": ["TrendContinuationTC", "TrendExpansionTE"],
       "MEAN": ["FailureReversionFR", "ExhaustionTurnET"]
   }
   
   # 只允许 enabled_archetypes 中的 archetype
   candidates = [c for c in candidates if c in enabled_archetypes[regime]]
   ```

#### 2.2.3 Gate → Execution 的影响

**Gate 输出**:
- `gate_ok`: 是否通过 Gate
- `gate_arch`: 选择的 Archetype（如 `TrendContinuationTC`）

**Execution 使用**:
- 从 `execution_archetypes.yaml` 读取该 Archetype 的:
  - `required_conditions`: 必需条件（如 `atr_percentile > 0.3`）
  - `gate_rules`: Gate 规则（如 `deny_if atr_percentile < 0.2`）
  - `execution_params`: 执行参数（如 `stop_loss_pct`, `take_profit_pct`）

---

### 2.3 Semantic Buckets 如何影响下游

#### 2.3.1 什么是 Semantic Buckets？

**Semantic Buckets** 是对 `tc_semantic_score` 或 `te_semantic_score` 的分位数分桶（通常 5 个桶）。

**目的**:
- 诊断不同语义分数区间的表现
- 识别"甜点区"（高 Sharpe）和"毒区"（负 Sharpe）
- 用于设置 Gate 的语义分数阈值

#### 2.3.2 分桶计算（在 E2E 报告中）

```python
# 按 tc_semantic_score 分 5 个桶
buckets = pd.qcut(tc_semantic_score, q=5, duplicates="drop")

# 每个桶计算 KPI
for bucket in buckets:
    kpi = {
        "sharpe_e2e": ...,
        "sharpe_trades_only": ...,
        "trade_rate": ...,
        "ret_mean_e2e": ...
    }
```

#### 2.3.3 分桶结果示例（来自你的报告）

**TC_REGIME Semantic Buckets**:
```
bucket                    | sharpe_e2e | sharpe_trades_only | trade_rate
(-0.000656, 0.0677]       | 5.811       | 9.351              | 0.3913  ← 甜点区
(0.0677, 0.127]           | 5.183       | 8.149              | 0.4074  ← 甜点区
(0.127, 0.225]            | 0.063       | 0.096              | 0.4234  ← 毒区
(0.225, 0.321]            | 3.348       | 6.578              | 0.2574
(0.321, 0.645]            | -3.869      | -6.402             | 0.3650  ← 毒区
```

**解读**:
- **低分桶（-0.0007 到 0.0677）**: Sharpe 最高（5.811），是甜点区
- **中分桶（0.127-0.225）**: Sharpe 接近 0（0.063），是毒区
- **高分桶（0.321-0.645）**: Sharpe 为负（-3.869），是最差区域

**结论（TC_REGIME）**: 
- **低分桶表现最好**，高分桶表现最差
- 不是"分数越高越好"
- 应该**保留低分桶，过滤高分桶**

**注意**: TE_REGIME 的分桶模式不同：
- **高分桶（0.308-0.761）**: Sharpe 最高（6.215）
- 说明 `tc_semantic_score` 和 `te_semantic_score` 的含义可能不同

#### 2.3.4 分桶如何影响 Gate

**Gate 使用分桶结果设置阈值**:

1. **计算分位数阈值**:
   ```bash
   python3 scripts/compute_semantic_score_floors.py \
     --physics-regime physics_regime.parquet \
     --output semantic_score_floors.json \
     --tc-quantile 0.95 \
     --te-quantile 0.10
   ```
   **注意**: TC 使用 p95（上限），TE 使用 p10（下限），因为它们的语义分数含义不同。

2. **Gate 应用阈值**（已修正）:
   ```python
   # 从 semantic_score_floors.json 读取
   thresholds = {
       "tc_semantic_score_p95": 0.321,  # P95 分位数（上限，veto 高分毒区）
       "te_semantic_score_p10": 0.0443  # P10 分位数（下限，veto 低分噪声）
   }
   
   # Gate 过滤（修正后的逻辑）
   if regime == "TC_REGIME":
       # Veto 高分毒区，保留低分甜点区
       if tc_semantic_score > thresholds["tc_semantic_score_p95"]:
           gate_ok = False  # Veto 高分桶
   
   if regime == "TE_REGIME":
       # Veto 低分噪声，保留高分信号
       if te_semantic_score < thresholds["te_semantic_score_p10"]:
           gate_ok = False  # Veto 低分桶
   ```

**修正说明**:

根据 E2E 分桶分析结果：
- **TC_REGIME**: 低分桶（0-0.127）Sharpe 最高（5.811），高分桶（>0.321）Sharpe 为负（-3.869）
  - ✅ **修正**: 使用上限阈值（p95），veto 高分毒区，保留低分甜点区
- **TE_REGIME**: 高分桶（0.308-0.761）Sharpe 最高（6.215）
  - ✅ **保持**: 使用下限阈值（p10），veto 低分噪声，保留高分信号

**关键发现**:
- `tc_semantic_score` 和 `te_semantic_score` 的含义不同
- TC 的"好"是低分（稳定、不扩张），TE 的"好"是高分（扩张、突破）
- 必须根据实际分桶结果调整 Gate 逻辑，不能假设"分数越高越好"

---

### 2.4 完整影响链总结

```
1. Regime 划分 (Physics/Regime Classifier)
   ↓ 决定: 哪些 Archetype 可用
   ↓ 影响: Gate 的候选 Archetype 列表
   ↓ 输出: regime (TC_REGIME/TE_REGIME/MEAN_REGIME/NO_TRADE)

2. Build Execution Logs
   ↓ 使用: NN Predictions + Raw OHLCV
   ↓ 输出: ret_mean, ret_trend (counterfactual execution returns)
   ↓ 影响: 为 Execution 层提供收益数据（已包含止损止盈逻辑）

3. Semantic Score
   ↓ 计算: tc_semantic_score / te_semantic_score
   ↓ 分桶: 识别甜点区和毒区
   ↓ 影响: Gate 的语义分数阈值（当前实现可能有问题）

4. Gate 过滤
   ↓ 使用: Regime + Semantic Score + Live Config
   ↓ 输出: gate_ok, gate_archetype
   ↓ 影响: 哪些交易可以进入 Execution

5. Execution
   ↓ 使用: gate_archetype 选择 ret_mean 或 ret_trend
   ↓ TC/TE → ret_trend, FR/ET → ret_mean
   ↓ 输出: 实际交易参数（SL/TP/hold/trail）
   ↓ 影响: 最终 PnL
```

---

## 三、关键发现与建议

### 3.1 关于 Semantic Score

**发现**:
- 低分桶（0-0.127）Sharpe 最高（5.811）
- 高分桶（0.321-0.645）Sharpe 为负（-3.869）

**问题**:
- 当前 Gate 实现是"低于阈值则 veto"，这会过滤掉甜点区
- 需要确认 `semantic_score` 的含义：分数越高是否代表"质量越好"？

**建议**:
1. 检查 `tc_semantic_score` 和 `te_semantic_score` 的计算逻辑
2. 如果分数越高越好，则应该设置**上限阈值**（veto 低分桶）
3. 如果分数越低越好，则应该设置**下限阈值**（veto 高分桶）

### 3.2 关于 Regime 划分

**当前实现**:
- 使用 `multi_dim` 策略（多维度投票）
- 满足 3/4 条件即可分类

**建议**:
- 继续优化 Regime 分类的 recall（覆盖率）
- 确保 Regime 不替 Gate 做选择（只负责"可行性"，不负责"质量"）

### 3.3 关于 E2E KPI

**当前指标**:
- `sharpe_e2e`: 2.79（包含空仓期）
- `sharpe_trades_only`: 4.31（只计算交易）

**解读**:
- 交易质量很好（4.31）
- 但交易频率偏低（42.21%）
- 可以考虑提高交易频率（降低 Gate 阈值）或保持当前质量

---

## 四、相关命令

### 4.1 完整工作流命令

```bash
# 1. 生成预测
mlbot nnmultihead predict --output-dir results/preds/ ...

# 2. 生成 Regime 分类
mlbot rule physics-regime \
  --preds results/preds/ \
  --output /tmp/physics_regime.parquet

# 3. 构建 Execution 日志
mlbot rl build-execution-logs \
  --preds results/preds/ \
  --output /tmp/logs_execution.parquet \
  --returns-source rr_execution

# 4. 应用 Gate 过滤
mlbot rule apply-tree-gate \
  --logs /tmp/logs_execution.parquet \
  --regime /tmp/physics_regime.parquet \
  --out /tmp/logs_execution_gated.parquet

# 5. 生成 E2E 报告
mlbot rule diagnose-e2e-kpi \
  --logs /tmp/logs_execution_gated.parquet \
  --regime /tmp/physics_regime.parquet \
  --gate /tmp/logs_execution_gated.parquet \
  --output-md results/e2e_kpi/e2e_kpi_report.md \
  --output-json results/e2e_kpi/e2e_kpi_report.json \
  --no-regime-filter
```

**注意**: Router 模块已被移除。Execution 层现在直接根据 archetype 选择 ret_mean 或 ret_trend。

### 4.2 计算 Semantic Score Thresholds

```bash
python3 scripts/compute_semantic_score_floors.py \
  --physics-regime /tmp/physics_regime.parquet \
  --output /tmp/semantic_score_floors.json \
  --tc-quantile 0.95 \
  --te-quantile 0.10 \
  --fr-quantile 0.05 \
  --et-quantile 0.05
```

**输出**: JSON 文件包含 TC/TE/FR/ET 的语义分数阈值
- `tc_semantic_score_p95`: TC 上限阈值（veto 高分毒区）
- `te_semantic_score_p10`: TE 下限阈值（veto 低分噪声）
- `fr_semantic_score_p05`: FR 下限阈值（veto 低分）
- `et_semantic_score_p05`: ET 下限阈值（veto 低分）

**注意**: TC 使用 p95（上限），因为低分桶表现最好；TE/FR/ET 使用 p05/p10（下限），因为高分桶表现更好。

### 4.3 应用 Gate（带 Semantic Score Floors）

```bash
mlbot rule apply-tree-gate \
  --logs /tmp/logs_execution.parquet \
  --out /tmp/logs_execution_gated.parquet \
  --features-store-layer tier0 \
  --physics-regime /tmp/physics_regime.parquet \
  --semantic-score-floors /tmp/semantic_score_floors.json
```

---

## 五、参考资料

- `scripts/diagnose_e2e_kpi.py`: E2E KPI 计算逻辑
- `scripts/apply_tree_gate_3action.py`: Gate 过滤逻辑
- `src/time_series_model/rule/regime.py`: Regime 分类逻辑
- `src/time_series_model/rl/build_execution_logs.py`: Build Execution Logs 逻辑
- `src/time_series_model/rule/regime.py`: Regime 分类逻辑
- `docs/ARCHITECTURE.md`: 系统架构文档
- `docs/workflow/PIPELINE_WORKFLOW.md`: 工作流文档

**注意**: Router 模块已被移除。Execution 层现在直接根据 archetype 选择 ret_mean 或 ret_trend。相关代码在 `src/time_series_model/rl/build_execution_logs.py`。
