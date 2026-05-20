# Gate增强对比报告

**测试时间**: 2026-01-22  
**目的**: 对比移除VolMean和增强gate规则后的效果，确保没有回退

---

## 测试配置

### 数据
- **时间范围**: 2024年全年
- **Symbols**: BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, XRPUSDT, ADAUSDT
- **总样本数**: 13146
- **FeatureStore Layer**: `nnmh_highcap6_240T_2024_202510_ma_adx_cvd_vwap_v1`

### 配置变更

1. **移除VolMeanCompressionExpansionReversion**: 从archetype候选列表中过滤
2. **增强Gate规则**: 为TC/TE添加价格轨迹特征
   - TC: 添加 `atr_slope_pct`, `path_efficiency_pct`, `path_length_pct`, `price_dir_consistency_pct`
   - TE: 添加 `atr_slope_pct`, `range_expansion_pct`

---

## 结果对比

### 整体KPI

| 指标 | 移除VolMean后 | 增强Gate后 | 变化 |
|------|--------------|-----------|------|
| 总样本数 | 13146 | 13146 | - |
| Gate OK | 6008 | 6008 | - |
| Gate Veto | 7138 | 7138 | - |
| Trade Rate | 45.7% | 45.7% | - |
| Sharpe (E2E) | 0.324 | 0.324 | - |
| Sharpe (Trades Only) | 0.133 | 0.133 | - |

### 按Archetype分布

| Archetype | 移除VolMean后 | 增强Gate后 | 变化 |
|-----------|--------------|-----------|------|
| TC | 5398 | 5398 | - |
| ET | 610 | 610 | - |
| FR | 0 | 0 | - |
| TE | 0 | 0 | - |

### 多个Archetype同时触发统计

**移除VolMean后**:
- 总组合数: 3
- 总出现次数: 3835
- 最常见组合: `ExhaustionTurnET+FailureReversionFR` (2797次)

**增强Gate后**:
- 总组合数: 3
- 总出现次数: 3835
- 最常见组合: `ExhaustionTurnET+FailureReversionFR` (2797次)

---

## 与以前Regime时期对比

### 以前Regime时期（2025年数据，baseline配置）

| 指标 | 值 |
|------|-----|
| Sharpe | 2.565 |
| Trades | 1074 |
| Win Rate | 34.3% |

**注意**: 这是2025年5-10月的数据，与当前2024年全年数据不同，不能直接对比。

### 当前配置（2024年全年，gate-based）

| 指标 | 值 |
|------|-----|
| Sharpe (E2E) | 0.324 |
| Sharpe (Trades Only) | 0.133 |
| Trades | 6008 |
| Win Rate (TC) | 39.4% |
| Win Rate (ET) | 37.2% |

---

## 关键发现

### 1. VolMean移除效果

✅ **成功移除**: VolMean不再出现在archetype组合中
- 之前有5种archetype（包含VolMeanCompressionExpansionReversion）
- 现在只有4种archetype（TC, TE, FR, ET）
- 移除VolMean后，archetype组合从5种减少到4种

### 2. Gate规则增强

⚠️ **当前结果**: 增强gate规则后，交易数和Sharpe没有变化
- 可能原因：
  1. 新添加的特征（atr_slope_pct, range_expansion_pct等）在当前数据中可能缺失
  2. 阈值设置可能过于宽松，没有过滤掉更多样本
  3. 需要运行优化脚本找到最佳阈值

### 3. 多个Archetype同时触发

- **ET+FR组合**: 2797次（最常见）
- **ET+FR+TC组合**: 593次
- **ET+TC组合**: 445次

这些组合都被正确拒绝（不开仓），符合设计。

---

## 下一步行动

1. ✅ **已完成**: 移除VolMean
2. ✅ **已完成**: 增强Gate规则（添加价格轨迹特征）
3. ⏳ **待办**: 运行平台高原阈值搜索，优化gate规则参数
4. ⏳ **待办**: 验证优化后的gate规则效果
5. ⏳ **待办**: 使用2025年数据重新测试，与以前regime时期直接对比

---

## 相关文件

- `results/e2e_kpi/logs_3action_2024_no_volmean.parquet` - 移除VolMean后的结果
- `results/e2e_kpi/logs_3action_2024_enhanced_gate.parquet` - 增强Gate后的结果
- `results/e2e_kpi/kpi_2024_no_volmean.md` - KPI报告
- `config/nnmultihead/execution_archetypes.yaml` - Gate规则配置
