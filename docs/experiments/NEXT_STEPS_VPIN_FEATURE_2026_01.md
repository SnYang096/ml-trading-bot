# 下一步行动：修复vpin特征缺失问题

## 当前状态

### 问题
- FeatureStore layer `nnmh_highcap6_240T_2024_202510` 中缺少 `vpin` 特征
- 所有依赖vpin的分析无法运行（FR Evidences深度分析、FR/ET Evidences性能分析等）
- 订单流特征一个都不能少，缺少vpin会导致分析失败

### 影响范围
1. **FR Evidences深度分析** (`analyze_fr_evidences_regime_optimization.py`): 无法运行
2. **FR/ET Evidences性能分析** (`analyze_fr_et_evidences_performance.py`): 无法运行
3. **所有依赖`has_orderflow` evidence的分析**: 无法运行

## 解决方案

### 方案1: 重新生成FeatureStore（推荐）

**步骤**:

1. **确认配置包含vpin特征**:
   - 检查 `config/nnmultihead/live_feature_plan.yaml` 或相关配置文件
   - 确认包含 `vpin_block_features_f` 或 `vpin_features_f` 特征组
   - 确认包含 `extract_order_flow_features` 特征计算函数

2. **准备tick数据**:
   - vpin计算需要tick数据（必需）
   - 确认tick数据可用且覆盖所需的时间范围（2025-05-01 到 2025-10-31）
   - 确认tick数据包含必需的列：`price`, `volume`, `side`

3. **重新构建FeatureStore**:
   ```bash
   mlbot nnmultihead build-feature-store \
     --task-spec config/tasks/task_spec_xxx.yaml \
     --layer nnmh_highcap6_240T_2024_202510_v2 \
     --symbols BTCUSDT,ETHUSDT,ADAUSDT,BNBUSDT,SOLUSDT \
     --timeframe 240T \
     --start-date 2025-05-01 \
     --end-date 2025-10-31 \
     --feature-plan config/nnmultihead/live_feature_plan.yaml
   ```

4. **验证vpin特征**:
   ```bash
   # 检查FeatureStore中是否包含vpin
   python3 -c "
   from src.feature_store import FeatureStore, FeatureStoreSpec
   store = FeatureStore('feature_store')
   spec = FeatureStoreSpec(
       layer='nnmh_highcap6_240T_2024_202510_v2',
       symbol='BTCUSDT',
       timeframe='240T'
   )
   df = store.read_range(spec, start=pd.Timestamp('2025-05-01'), end=pd.Timestamp('2025-05-02'))
   print('vpin' in df.columns)
   print(df.columns.tolist())
   "
   ```

### 方案2: 检查现有FeatureStore是否有其他layer包含vpin

**检查结果** (2026-01-22):
- ✅ 已检查现有layers: `nnmh_highcap6_240T_2024_202510` 和 `nnmh_highcap6_240T_2024_202510_ma_adx_cvd_vwap_v1`
- ❌ **两个layers都缺少vpin特征**
- ✅ 两个layers都包含 `cvd_change_5` 和 `cvd_change_5_normalized`
- **结论**: 必须重新生成FeatureStore才能获得vpin特征

**配置确认**:
- ✅ `config/nnmultihead/live_feature_plan.yaml` 已包含 `vpin` 和 `vpin_features_f`
- ✅ 配置正确，问题在于FeatureStore构建时未包含vpin

## 配置检查清单

### 1. 特征配置检查

检查以下配置文件，确认包含vpin相关特征：

- `config/nnmultihead/live_feature_plan.yaml`: 确认包含 `vpin_block_features_f` 或相关特征组
- `config/feature_dependencies.yaml`: 确认vpin特征依赖关系正确

### 2. Tick数据检查

确认tick数据：
- 数据源可用（如Binance tick数据）
- 时间范围覆盖：2025-05-01 到 2025-10-31
- 包含必需的列：`timestamp`, `price`, `volume`, `side`
- 数据格式正确（side为1/-1或'buy'/'sell'）

### 3. 特征计算函数检查

确认 `extract_order_flow_features` 函数：
- 在特征配置中被正确引用
- 参数配置正确（`vpin_bucket_volume`, `vpin_n_buckets`, `vpin_adaptive`等）
- 能够访问tick数据

## 重新生成FeatureStore后的操作

### 1. 更新分析脚本配置

更新以下脚本的默认layer参数：
- `scripts/analyze_fr_evidences_regime_optimization.py`
- `scripts/analyze_fr_et_evidences_performance.py`
- `scripts/experiment_regime_gate.py`

### 2. 重新运行分析

```bash
# 1. FR Evidences深度分析
python3 scripts/analyze_fr_evidences_regime_optimization.py \
  --logs results/e2e_kpi/logs_3action_regime_optimized.parquet \
  --feature-store-root feature_store \
  --feature-store-layer nnmh_highcap6_240T_2024_202510_v2 \
  --timeframe 240T \
  --start-date 2025-05-01 \
  --end-date 2025-10-31

# 2. FR/ET Evidences性能分析
python3 scripts/analyze_fr_et_evidences_performance.py \
  --logs results/e2e_kpi/logs_3action_regime_optimized.parquet \
  --feature-store-root feature_store \
  --feature-store-layer nnmh_highcap6_240T_2024_202510_v2 \
  --timeframe 240T \
  --start-date 2025-05-01 \
  --end-date 2025-10-31
```

### 3. 更新实验报告

更新以下报告：
- `docs/experiments/EXP_FR_EVIDENCES_REGIME_OPTIMIZATION_2026_01.md`: 添加实际分析结果
- `docs/experiments/EXP_FR_ET_EVIDENCES_PERFORMANCE_2026_01.md`: 更新为使用完整特征的结果

## 临时方案（不推荐）

如果需要快速验证其他部分，可以考虑：
1. 暂时修改evidence rules，移除`has_orderflow`要求
2. 但这会降低分析质量，不推荐用于最终分析

## 关键文件

- 特征计算: `src/features/time_series/utils_order_flow_features.py`
- 特征配置: `config/feature_dependencies.yaml`
- 分析脚本: `scripts/analyze_fr_evidences_regime_optimization.py`
- 实验报告: `docs/experiments/EXP_FR_EVIDENCES_REGIME_OPTIMIZATION_2026_01.md`

---

**最后更新**: 2026-01-22  
**状态**: 待执行（需要重新生成FeatureStore）
