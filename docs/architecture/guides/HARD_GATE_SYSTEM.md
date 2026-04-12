# Hard-Gate System 规则调优协议

> **仓库同步（2026）**：本文中的 `scripts/optimize_gate_plateau_hard_gate.py`、`scripts/optimize_gate_plateau.py` 等入口**已不存在**。当前 Gate 阈值扫描与高原/稳健性逻辑集中在 **`scripts/optimize_gate_unified.py`**（`python scripts/optimize_gate_unified.py --help`）。`mlbot optimize gate-plateau` 仍指向缺失的 `optimize_gate_plateau.py`，若需 CLI 级封装需在代码中恢复或改指向统一脚本。

## 概述

Hard-Gate System 是一个严格的规则调优协议，确保规则按照语义优先级顺序逐一优化，已优化的规则参数被冻结，后续优化基于前序规则过滤后的数据集进行。

## 核心原则

1. **规则按照语义优先级排序**：安全性 → 市场状态 → 执行策略
2. **规则按顺序逐一优化**：不允许联合优化
3. **规则参数冻结**：每个规则一旦优化完成，其参数就被冻结
4. **基于过滤数据集优化**：后续规则的调优基于前序规则生成的过滤数据集进行
5. **Plateau评估考虑上游规则**：Plateau评估必须考虑所有上游固定的规则条件

## 优先级定义

在 `config/nnmultihead/execution_archetypes.yaml` 中，每个规则可以定义 `priority` 字段：

- **Priority 1**: 安全性规则（最高优先级）
  - Reflexivity风险控制（SHD, OFCI）
  - 系统性风险检测

- **Priority 2**: 市场状态规则（中等优先级）
  - 结构存在类：path_efficiency, path_length, dir_consistency
  - 稳定性veto：jump_risk, atr_slope
  - 极端veto：deviation_z, cvd

- **Priority 3**: 执行策略规则（较低优先级）
  - Volume, bb_width, adx
  - Quality, score
  - Orderflow continuation

## 使用方法

### 方法1: 统一优化脚本（当前推荐）

```bash
python scripts/optimize_gate_unified.py \
  --strategy bpc \
  --logs path/to/trade_logs_with_features.parquet \
  --output results/gate_optimization_bpc.json
```

按需追加：`--gate-path`、`--promote`、`--prefilter`、`--cutoff-date` 等（见 `--help`）。

### 方法2: 历史 `mlbot optimize gate-plateau` / `optimize_gate_plateau.py`

上述 CLI 与旧脚本依赖的 parquet 形态与统一脚本**不完全相同**；在新管线未把 `scripts/optimize_gate_plateau.py` 恢复前，**优先使用方法 1**。

## 工作流程

1. **加载规则优先级**：从 `execution_archetypes.yaml` 读取每个规则的 `priority` 字段
2. **按优先级排序**：将所有规则按优先级（从小到大）排序
3. **逐一优化**：
   - 对每个规则，先应用所有已冻结的规则，得到过滤后的数据集
   - 在过滤后的数据集上进行阈值扫描和Plateau优化
   - 优化完成后，将规则参数冻结
4. **输出结果**：保存所有优化后的规则阈值

## 示例输出

```json
{
  "TrendContinuationTC_tc_reflexivity_shd_too_high": {
    "archetype": "TrendContinuationTC",
    "rule_name": "tc_reflexivity_shd_too_high",
    "feature_key": "shd_pct",
    "rule_kind": "value_gt",
    "current_threshold": 0.9,
    "recommended_threshold": 0.92,
    "robustness_score": 1.234,
    "trade_rate": 0.0234,
    "priority": 1
  },
  ...
}
```

## 与渐进式优化的区别

- **渐进式优化**：先大幅放宽所有规则增加交易数，再优化，最后收紧
- **Hard-Gate System**：按优先级顺序逐一优化，每个规则优化后立即冻结

## 注意事项

1. 确保 `execution_archetypes.yaml` 中的规则都定义了 `priority` 字段
2. 优先级数字越小，越先优化
3. 如果规则没有定义 `priority`，默认优先级为 999（最后优化）
4. 已优化的规则参数会被冻结，后续优化必须考虑这些冻结的规则

## 配置文件示例

在 `execution_archetypes.yaml` 中定义优先级：

```yaml
gate_rules:
  rules:
    # Priority 1: Safety rules
    - name: tc_reflexivity_shd_too_high
      kind: value_gt
      key: shd_pct
      threshold: 0.9
      priority: 1
      on_missing: false
    
    # Priority 2: Market state rules
    - name: tc_not_tc_regime_path_efficiency_too_low
      kind: value_lt
      key: path_efficiency_pct
      threshold: 0.6
      priority: 2
      on_missing: false
    
    # Priority 3: Execution strategy rules
    - name: tc_volume_too_low
      kind: value_lt
      key: volume_ratio_pct
      threshold: 0.2
      priority: 3
      on_missing: false
```
