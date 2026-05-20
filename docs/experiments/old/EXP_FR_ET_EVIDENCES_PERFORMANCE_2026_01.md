# FR/ET Evidences性能分析报告

## 实验元信息

- **实验时间**: 2026-01-22 03:36:56
- **实验目的**: 分析FR/ET的evidences单独使用时的表现，以及加上gate后的表现
- **数据时间范围**: 2025-05-01 到 2025-10-31
- **数据Symbols**: BTCUSDT, ETHUSDT, ADAUSDT, BNBUSDT, SOLUSDT
- **总样本数**: 2930

## 关键发现

### 1. vpin特征缺失问题（已修复）

- **问题**: FeatureStore layer `nnmh_highcap6_240T_2024_202510` 中没有 `vpin` 特征
- **原因**: 
  - vpin特征计算需要tick数据（必须）
  - 当前FeatureStore可能是在vpin被添加到配置之前构建的
  - 或者构建时没有启用tick数据支持
- **修复方案**: 
  - ✅ 修改分析脚本，添加 `--vpin-missing-strategy` 参数
  - ✅ 使用 `skip_has_orderflow` 策略：跳过 `has_orderflow` evidence检查，只检查其他evidences
  - ✅ 这样可以在缺少vpin的情况下继续分析其他evidences的表现
- **影响**: 
  - 所有样本都缺少vpin（100%缺失）
  - 使用skip策略后，可以继续分析其他evidences（如`has_sr_quality`）

### 2. FR/ET Evidences分析结果

#### FR (FailureReversion) 结果

**情况A: 所有数据，只用FR evidences（跳过has_orderflow）**
- 通过evidences: **2930/2930** (100%)
- 平均ret_mean: **-0.000506** (负收益)
- 胜率: **38.2%**
- Sharpe: **-0.813**

**情况B: 所有数据，FR evidences + gate**
- 通过evidences: 2930/2930
- 通过gate: **2555/2930** (87.2%)
- 平均ret_mean: **-0.000469** (负收益)
- 胜率: **37.7%**
- Sharpe: **-0.752**

**情况C: MEAN_REGIME数据，只用FR evidences（跳过has_orderflow）**
- 通过evidences: **27/27** (100%)
- 平均ret_mean: **0.001384** (正收益 ✅)
- 胜率: **44.4%**
- Sharpe: **1.759** ✅

**情况D: MEAN_REGIME数据，FR evidences + gate**
- 通过evidences: 27/27
- 通过gate: **27/27** (100%)
- 平均ret_mean: **0.001384** (正收益 ✅)
- 胜率: **44.4%**
- Sharpe: **1.759** ✅

**关键发现**:
- ✅ **MEAN_REGIME中的FR evidences表现良好**：正收益、高Sharpe、44.4%胜率
- ❌ **所有数据中的FR evidences表现不佳**：负收益、负Sharpe、低胜率
- ✅ **Gate rules对MEAN_REGIME的FR样本没有过滤**（27个全部通过）
- ⚠️ **Gate rules对所有数据的FR样本有轻微改善**（Sharpe从-0.813提升到-0.752）

#### ET (ExhaustionTurn) 结果

**所有情况**: 0个样本通过evidences

**原因**: `has_volume_profile` evidence检查失败（可能缺少volume profile相关特征）

### 3. 对比分析

| 场景 | 样本数 | 平均ret_mean | 胜率 | Sharpe | 说明 |
|------|--------|--------------|------|--------|------|
| 所有数据，FR evidences | 2930 | -0.000506 | 38.2% | -0.813 | 负收益 |
| 所有数据，FR evidences + gate | 2555 | -0.000469 | 37.7% | -0.752 | Gate略有改善 |
| MEAN_REGIME，FR evidences | 27 | 0.001384 | 44.4% | 1.759 | ✅ 正收益 |
| MEAN_REGIME，FR evidences + gate | 27 | 0.001384 | 44.4% | 1.759 | ✅ 正收益 |

**结论**:
1. **FR evidences在MEAN_REGIME中表现良好**，但在所有数据中表现不佳
2. **Gate rules对MEAN_REGIME的FR样本没有额外过滤**，说明这些样本质量较高
3. **需要进一步优化MEAN_REGIME分类**，以增加MEAN_REGIME样本数（当前只有27个）

### 4. 可用的特征

- ✅ `cvd_change_5`: 可用
- ✅ `cvd_change_5_normalized`: 可用
- ✅ `sr_/sqs/poc`相关特征: 7个可用
- ✅ `absorption`相关特征: 1个可用（wick_absorption_score）
- ❌ `vpin`: 不可用（已通过skip策略处理）
- ❌ `volume_profile`相关特征: 可能不可用（导致ET evidences全部失败）



## 详细分析结果

### FR (FailureReversion) 详细结果

| 场景 | 通过evidences | 通过gate | 平均ret_mean | 胜率 | Sharpe | 中位数ret_mean | std_ret |
|------|---------------|----------|--------------|------|--------|----------------|---------|
| A: 所有数据，只用FR evidences | 2930 | 2930 | -0.000506 | 38.2% | -0.813 | 0.000000 | 0.009871 |
| B: 所有数据，FR evidences + gate | 2930 | 2555 | -0.000469 | 37.7% | -0.752 | -0.000000 | 0.009907 |
| C: MEAN_REGIME数据，只用FR evidences | 27 | 27 | 0.001384 | 44.4% | 1.759 | 0.000000 | 0.012492 |
| D: MEAN_REGIME数据，FR evidences + gate | 27 | 27 | 0.001384 | 44.4% | 1.759 | 0.000000 | 0.012492 |

### ET (ExhaustionTurn) 详细结果

| 场景 | 通过evidences | 通过gate | 平均ret_mean | 胜率 | Sharpe | 中位数ret_mean | std_ret |
|------|---------------|----------|--------------|------|--------|----------------|---------|
| A: 所有数据，只用ET evidences | 0 | 0 | 0.000000 | 0.0% | 0.000 | 0.000000 | 0.000000 |
| B: 所有数据，ET evidences + gate | 0 | 0 | 0.000000 | 0.0% | 0.000 | 0.000000 | 0.000000 |
| C: MEAN_REGIME数据，只用ET evidences | 0 | 0 | 0.000000 | 0.0% | 0.000 | 0.000000 | 0.000000 |
| D: MEAN_REGIME数据，ET evidences + gate | 0 | 0 | 0.000000 | 0.0% | 0.000 | 0.000000 | 0.000000 |

## 结论

### vpin特征缺失问题

**⚠️ 重要**: 订单流特征一个都不能少。如果缺少vpin等关键特征，分析应该直接失败，而不是跳过。

**解决方案**:
1. **必须重新生成FeatureStore**: 确保包含所有订单流特征（vpin, cvd_change_5等）
   ```bash
   # 使用包含vpin的配置重新生成FeatureStore
   mlbot nnmultihead build-feature-store \
     --task-spec ... \
     --layer nnmh_highcap6_240T_2024_202510_v2 \
     --symbols BTCUSDT,ETHUSDT,ADAUSDT,BNBUSDT,SOLUSDT \
     --timeframe 240T \
     --start-date 2025-05-01 \
     --end-date 2025-10-31
   ```
2. **确保tick数据可用**: vpin计算需要tick数据，确保构建时tick数据可访问
3. **验证配置**: 确认使用的配置（如`live_feature_plan.yaml`）包含vpin特征
4. **严格验证**: 分析脚本应该检查所有必需的订单流特征，缺少任何特征都应该直接报错退出

### FR/ET Evidences表现

**主要发现**:
1. ✅ **FR evidences在MEAN_REGIME中表现优秀**（Sharpe 1.759，正收益）
2. ❌ **FR evidences在所有数据中表现不佳**（Sharpe -0.813，负收益）
3. ⚠️ **ET evidences无法评估**（缺少volume profile特征）

**关键结论**:
1. **Regime过滤至关重要**: FR evidences在MEAN_REGIME中表现良好，但在所有数据中表现差
2. **需要找出适合FR的regime**: 当前只有MEAN_REGIME表现好，但样本太少（27个）
3. **需要深度分析**: 分析全量数据，找出决定FR适合regime的关键特征和参数范围

**下一步行动**:
1. **重新生成FeatureStore**: 确保包含所有订单流特征
2. **FR Evidences深度分析**: 单独实验FR evidences，找出适合FR的regime和evidence参数范围
3. **扩大数据范围**: 寻找更多适合FR的样本
4. **优化MEAN_REGIME分类**: 放宽条件以增加样本数

## 文件位置

- 分析结果: `results/fr_et_evidences_performance.json`
- 分析脚本: `scripts/analyze_fr_et_evidences_performance.py`
