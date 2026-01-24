# 反身性特征和ET对冲有效性测试

本文档说明如何运行反身性特征和ET对冲有效性验证测试。

## 测试目标

证明以下功能的有效性：

1. **反身性特征（OFCI, SHD）**：能够识别高风险场景，gate规则正确阻止/减少高风险交易
2. **ET对冲机制**：能够减少left-tail风险，配对机制正确工作，成本在可接受范围内

## 测试脚本

### 1. 反身性特征Gate规则触发验证

**脚本**: `scripts/test_reflexivity_gate_effectiveness.py`

**功能**:
- 验证当 `ofci_pct > 0.9` 时，soft veto（仓位减少60%）触发率
- 验证当 `shd_pct > 0.9` 时，hard veto（拒绝交易）触发率
- 验证当 `ofci_pct > 0.95` 时，hard veto触发率
- 检测false positives和false negatives

**使用方法**:
```bash
# 使用stage logs
python scripts/test_reflexivity_gate_effectiveness.py \
    --logs results/live_logs \
    --output results/reflexivity_gate_analysis.json

# 使用canonical log文件
python scripts/test_reflexivity_gate_effectiveness.py \
    --logs results/execution_log.jsonl \
    --canonical \
    --output results/reflexivity_gate_analysis.json
```

**输出指标**:
- `ofci_soft_veto_rate`: OFCI soft veto触发率
- `shd_hard_veto_rate`: SHD hard veto触发率
- `ofci_extreme_hard_veto_rate`: OFCI极端情况hard veto触发率
- `false_positives_count`: 误报数量
- `false_negatives_count`: 漏报数量

### 2. ET对冲配对机制验证

**脚本**: `scripts/test_et_hedge_pairing.py`

**功能**:
- 验证ET订单只在有TC/TE仓位时创建
- 验证ET订单与TC/TE仓位的配对关系
- 验证TC/TE仓位关闭时ET对冲也正确关闭
- 分析ET对冲成本

**使用方法**:
```bash
# 使用stage logs
python scripts/test_et_hedge_pairing.py \
    --logs results/live_logs \
    --output results/et_pairing_analysis.json

# 使用canonical log文件
python scripts/test_et_hedge_pairing.py \
    --logs results/execution_log.jsonl \
    --canonical \
    --output results/et_pairing_analysis.json
```

**输出指标**:
- `et_pairing_rate`: ET订单配对率（有TC/TE仓位时创建ET订单的比例）
- `tc_te_hedged_rate`: TC/TE订单被对冲的比例
- `cost_rate`: ET对冲成本率（ET成本/总收益）
- `cost_acceptable`: 成本是否可接受（成本率 <= 5%）

### 3. 综合测试脚本

**脚本**: `scripts/run_reflexivity_et_effectiveness_tests.py`

**功能**:
- 运行所有有效性测试
- 生成综合报告
- 提供改进建议

**使用方法**:
```bash
# 运行所有测试
python scripts/run_reflexivity_et_effectiveness_tests.py \
    --logs results/live_logs \
    --output results/reflexivity_et_effectiveness_report.json

# 跳过某些测试
python scripts/run_reflexivity_et_effectiveness_tests.py \
    --logs results/live_logs \
    --skip-gate-test \
    --output results/reflexivity_et_effectiveness_report.json
```

## 测试数据要求

### Stage Logs格式

如果使用stage logs，日志目录结构应为：
```
results/live_logs/
├── features/
│   └── YYYY-MM.jsonl
├── gate/
│   └── YYYY-MM.jsonl
├── execution/
│   └── YYYY-MM.jsonl
└── returns/
    └── YYYY-MM.jsonl
```

### Canonical Log格式

如果使用canonical log，文件应为JSONL格式，每行一个完整的执行记录：
```json
{
  "schema_version": "v1",
  "source": "live",
  "symbol": "BTCUSDT",
  "timestamp": "2024-01-01T00:00:00Z",
  "features": {
    "ofci_pct": 0.95,
    "shd_pct": 0.85,
    ...
  },
  "gate": {
    "blocked": false,
    "decisions": [...],
    ...
  },
  "execution": {
    "submit_order": true,
    "execution_strategy": "TC",
    ...
  },
  ...
}
```

## 预期结果

### 反身性特征Gate规则

- **OFCI soft veto触发率**: >= 80%
- **SHD hard veto触发率**: >= 90%
- **False negatives**: < 10个
- **False positives**: < 5个

### ET对冲配对机制

- **ET配对率**: >= 90%（ET订单创建时应有TC/TE仓位）
- **TC/TE对冲率**: >= 70%（TC/TE订单创建后应有ET对冲，如果风险条件满足）
- **成本率**: <= 5%（ET对冲成本不超过总收益的5%）

## 故障排查

### 问题1: 日志格式不匹配

**症状**: 脚本报错"无法解析日志"

**解决**:
- 检查日志格式是否符合要求
- 使用`--canonical`标志如果使用canonical log格式
- 确保日志包含必要的字段（features, gate, execution等）

### 问题2: 触发率过低

**症状**: Gate规则触发率 < 预期值

**可能原因**:
- Gate规则配置不正确
- 反身性特征计算有误
- 日志中缺少必要的特征值

**解决**:
- 检查`config/nnmultihead/execution_archetypes.yaml`中的gate规则配置
- 验证反身性特征计算逻辑
- 检查日志中的特征值是否完整

### 问题3: ET配对率过低

**症状**: ET配对率 < 90%

**可能原因**:
- ET订单创建逻辑不正确
- 仓位跟踪机制有误
- 日志中缺少配对信息

**解决**:
- 检查`src/time_series_model/live/meta_router_strategy.py`中的ET对冲逻辑
- 验证仓位跟踪机制
- 确保日志记录了配对关系

## 相关文档

- [反身性特征实现](../features/reflexivity_features.md)
- [ET对冲机制实现](../execution/et_hedge.md)
- [执行日志架构](../guides/EXECUTION_LOG_SCHEMA_CN.md)
