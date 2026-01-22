# FR Evidences Regime和Evidence参数优化分析报告（结果）

## 实验元信息

- **实验时间**: 2026-01-22
- **实验目的**: 
  1. 分析不同regime下FR evidences的表现
  2. 找出适合FR的regime特征范围
  3. 优化evidence参数（quantile阈值等）
  4. 扩大数据范围寻找更多样本

- **数据时间范围**: 2025-05-01 到 2025-10-31
- **数据Symbols**: BTCUSDT, ETHUSDT, ADAUSDT, BNBUSDT, SOLUSDT
- **总样本数**: 2930
- **FeatureStore Layer**: `nnmh_highcap6_240T_2024_202510_v2`（包含vpin特征 ✅）
- **Timeframe**: 240T (4H)

## 关键发现

### 1. 当前配置问题

**问题**: 所有regime下FR evidences的通过数都是0

**原因分析**:
- FR的`has_orderflow` evidence要求 `vpin > quantile(0.6)`
- 当前配置中quantile阈值设置为0.6
- 分析结果显示，使用quantile=0.6时，没有样本通过evidences
- 只有当quantile降低到0.55或0.65时才有样本通过

**vpin分布分析**:
- 需要进一步分析vpin的实际分布，确定合适的quantile阈值

### 2. Evidence参数优化结果

| has_orderflow quantile | 通过evidences | 平均ret_mean | 胜率 | Sharpe |
|----------------------|---------------|--------------|------|--------|
| 0.55 | 1317 | -0.000769 | 37.5% | -1.246 |
| 0.65 | 1024 | -0.000570 | 37.8% | -0.966 |

**发现**:
- ⚠️ **两个quantile阈值下的表现都不佳**（负收益、负Sharpe）
- ⚠️ **quantile=0.65的表现略好**（Sharpe -0.966 vs -1.246）
- ⚠️ **胜率都较低**（37.5-37.8%）

### 3. Regime表现分析

| Regime | 总样本数 | 通过evidences | 平均ret_mean | 胜率 | Sharpe |
|--------|----------|---------------|--------------|------|--------|
| MEAN_REGIME | 27 | 0 | 0.000000 | 0.0% | 0.000 |
| NO_TRADE | 1448 | 0 | 0.000000 | 0.0% | 0.000 |
| TC_REGIME | 711 | 0 | 0.000000 | 0.0% | 0.000 |
| TE_REGIME | 744 | 0 | 0.000000 | 0.0% | 0.000 |

**关键发现**:
- ❌ **所有regime下FR evidences的通过数都是0**
- ⚠️ **需要调整has_orderflow evidence的quantile阈值**

### 4. 适合FR的Regime特征范围

**分析结果**: 由于没有样本通过evidences，无法分析适合FR的regime特征范围

**下一步**: 需要先解决evidence通过率问题，然后才能分析特征范围

## 问题诊断

### 问题1: has_orderflow evidence阈值过高

**当前配置**:
```yaml
- name: has_orderflow
  kind: quantile_gt
  key: vpin
  quantile: 0.6
  on_missing: error
```

**问题**: quantile=0.6的阈值导致所有样本都无法通过

**可能原因**:
1. vpin的实际分布可能比预期低
2. quantile计算可能有问题（需要按symbol计算，而不是全局）
3. 需要检查vpin的实际分位数分布

### 问题2: 即使调整quantile，表现仍然不佳

**发现**: 即使将quantile降低到0.55或0.65，通过evidences的样本仍然表现不佳（负收益、负Sharpe）

**可能原因**:
1. FR evidences本身可能不够强
2. 需要结合regime过滤（MEAN_REGIME）才能表现好
3. 需要进一步优化evidence规则组合

## 建议的优化方向

### 1. 调整has_orderflow evidence阈值

**建议**: 
- 分析vpin的实际分布，确定合适的quantile阈值
- 考虑按symbol分别计算quantile（而不是全局）
- 测试不同的quantile值（0.5, 0.55, 0.6, 0.65, 0.7）

### 2. 结合Regime过滤

**发现**: 之前的分析显示，FR在MEAN_REGIME中表现良好（Sharpe 1.759）

**建议**:
- 优先在MEAN_REGIME中应用FR evidences
- 分析MEAN_REGIME中FR evidences的表现
- 可能需要放宽MEAN_REGIME条件以增加样本数

### 3. 优化Evidence规则组合

**建议**:
- 分析哪些evidence规则最重要
- 考虑调整required_evidence的组合
- 测试不同的evidence规则组合

## 下一步行动

1. **分析vpin分布**: 确定合适的quantile阈值
2. **测试MEAN_REGIME中的FR**: 在MEAN_REGIME中单独测试FR evidences
3. **优化evidence参数**: 根据分析结果调整quantile阈值
4. **扩大数据范围**: 如果可能，扩大数据范围寻找更多样本

## 文件位置

- 分析结果: `results/fr_evidences_regime_optimization.json`
- 分析脚本: `scripts/analyze_fr_evidences_regime_optimization.py`
- 配置文件: `config/nnmultihead/execution_archetypes.yaml`

---

**最后更新**: 2026-01-22  
**状态**: 分析完成，需要进一步优化evidence参数
