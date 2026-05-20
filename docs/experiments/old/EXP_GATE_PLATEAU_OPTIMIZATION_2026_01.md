# Gate平台高原阈值搜索实施报告

**测试时间**: 2026-01-22  
**目的**: 实施Gate平台高原阈值搜索，使用Robustness Score (min Sharpe)作为优化目标

---

## 实施内容

### 1. 移除VolMeanCompressionExpansionReversion ✅

**修改文件**: `scripts/apply_archetype_gate.py`

```python
arches = load_execution_archetypes_registry(str(args.execution_archetypes))
# Filter out VolMeanCompressionExpansionReversion (not suitable for current framework)
arches = {k: v for k, v in arches.items() if k != "VolMeanCompressionExpansionReversion"}
```

**效果**:
- 之前: 5种archetype（包含VolMeanCompressionExpansionReversion）
- 现在: 4种archetype（TC, TE, FR, ET）
- 移除VolMean后，archetype从5种减少到4种

### 2. 增强Gate规则 - 添加价格轨迹特征 ✅

**修改文件**: `config/nnmultihead/execution_archetypes.yaml`

#### TC (TrendContinuationTC)
添加了以下规则：
- `tc_not_tc_regime_atr_slope_too_high`: `atr_slope_pct <= 0.6` (低波动扩张)
- `tc_not_tc_regime_path_efficiency_too_low`: `path_efficiency_pct >= 0.6` (高效率)
- `tc_not_tc_regime_path_length_too_low`: `path_length_pct >= 0.4` (足够路径长度)
- `tc_not_tc_regime_dir_consistency_too_low`: `price_dir_consistency_pct >= 0.6` (方向稳定性)

#### TE (TrendExpansionTE)
添加了以下规则：
- `te_not_te_regime_atr_slope_too_low`: `atr_slope_pct >= 0.6` (高波动扩张)
- `te_not_te_regime_range_expansion_too_low`: `range_expansion_pct >= 0.6` (范围扩张)

### 3. 创建平台高原阈值搜索脚本 ✅

**文件**: `scripts/optimize_gate_plateau.py`

**功能**:
- 使用Robustness Score (min Sharpe across buckets)作为优化目标
- 分桶维度: World (TREND/MEAN) × Archetype (TC/TE/FR/ET) × Vol (low/mid/high)
- 约束条件: `trade_rate >= R_min`, `coverage_per_bucket >= N_min`
- 找到"Sharpe ≥ S_min 的最大阈值区间"（平台高原）

**优化方法**:
1. 对每个Gate rule单独扫描阈值
2. 画「Sharpe–Threshold 曲线」
3. 选「最宽高原」的中位数作为推荐阈值

---

## 当前Gate规则使用的价格轨迹特征

### TC (TrendContinuationTC)
- ✅ `jump_risk_pct`: [0.3, 0.6]
- ✅ `atr_slope_pct`: <= 0.6
- ✅ `path_efficiency_pct`: >= 0.6
- ✅ `path_length_pct`: >= 0.4
- ✅ `price_dir_consistency_pct`: >= 0.6

### TE (TrendExpansionTE)
- ✅ `jump_risk_pct`: [0.6, 0.9]
- ✅ `atr_slope_pct`: >= 0.6
- ✅ `range_expansion_pct`: >= 0.6

### FR (FailureReversionFR)
- ✅ `path_efficiency_pct`: <= 0.5
- ✅ `price_dir_consistency_pct`: <= 0.5
- ✅ `deviation_z_abs_pct`: >= 0.5
- ✅ `path_length_pct`: >= 0.5
- ✅ `atr_percentile`: >= 0.5
- ✅ `jump_risk_pct`: <= 0.4

### ET (ExhaustionTurnET)
- ✅ `jump_risk_pct`: [0.2, 0.5]
- ✅ `atr_percentile`: >= 0.85
- ✅ `path_efficiency_pct`: [0.55, 0.7]
- ✅ `path_length_pct`: >= 0.6

---

## 优化目标函数

### Robustness Score定义

```python
Robustness(θ) = min_over_(w,a,v) Sharpe(w,a,v | θ)
```

其中：
- `w`: World bucket (TREND / MEAN)
- `a`: Archetype (TC / TE / FR / ET)
- `v`: Vol bucket (low / mid / high)

### 约束条件

```python
trade_rate(θ) ≥ R_min        # 默认 0.5%
coverage_per_bucket ≥ N_min  # 默认 10 trades
```

### 优化问题

```python
maximize_θ   Robustness(θ)
subject to:
    trade_rate(θ) ≥ R_min
    coverage_bucket ≥ N_min
```

---

## 搜索顺序（推荐）

1. **结构存在类**（path_efficiency / consistency）
2. **稳定性 veto**（jump_risk）
3. **极端 veto**（deviation_z）

每一类 **冻结后再动下一类**。

---

## 下一步行动

1. ✅ **已完成**: 移除VolMean
2. ✅ **已完成**: 增强Gate规则（添加价格轨迹特征）
3. ✅ **已完成**: 创建优化脚本
4. ⏳ **待办**: 运行优化脚本，找到最佳阈值组合
5. ⏳ **待办**: 验证优化后的gate规则效果
6. ⏳ **待办**: 使用2025年数据重新测试，与以前regime时期直接对比

---

## 使用方法

### 运行优化脚本

```bash
python scripts/optimize_gate_plateau.py \
  --gated-logs results/e2e_kpi/logs_3action_2024_enhanced_gate.parquet \
  --raw-logs results/e2e_kpi/logs_3action_2024.parquet \
  --output results/gate_optimization.json \
  --min-trade-rate 0.005 \
  --min-trades-per-bucket 10 \
  --min-sharpe-threshold 0.5 \
  --threshold-step 0.05
```

### 输出格式

```json
{
  "TrendContinuationTC_tc_not_tc_regime_path_efficiency_too_low": {
    "archetype": "TrendContinuationTC",
    "rule_name": "tc_not_tc_regime_path_efficiency_too_low",
    "feature_key": "path_efficiency_pct",
    "rule_kind": "quantile_lt",
    "current_threshold": 0.6,
    "plateau_start": 0.55,
    "plateau_end": 0.65,
    "recommended_threshold": 0.6,
    "robustness_score": 0.8,
    "trade_rate": 0.45,
    "min_coverage": 15
  }
}
```

---

## 相关文件

- `scripts/optimize_gate_plateau.py` - 优化脚本
- `config/nnmultihead/execution_archetypes.yaml` - Gate规则配置
- `docs/architecture/FINAL_SIMPLIFIED_ARCHITECTURE_2026_01.md` - 最终架构文档
- `docs/experiments/EXP_GATE_ENHANCEMENT_COMPARISON_2026_01.md` - 对比报告
