# SR Reversal RR Reg 样本权重使用说明

## 概述

`sr_reversal_rr_reg` 是回归模型策略，通过预测连续 RR 值，使用 `top_quantile` 选择交易信号。相比分类模型，具有以下优势：

1. **不需要调整 entry_threshold**：通过 top_quantile 自适应选择信号
2. **更精确的预测**：模型预测连续 RR 值，而非二元分类
3. **更灵活的交易逻辑**：可以根据预测的 RR 值进行更精细的仓位管理

## 使用样本权重

### 1. 配置文件

使用 `labels_with_weights.yaml` 配置文件，启用 `result_based_rr` 权重策略：

```yaml
label_generator:
  function: compute_sr_reversal_rr_continuous_label_with_weights
  params:
    compute_weights: true
    weight_strategy: result_based_rr
    weight_config:
      logic_mode: triple_resonance
      # ... 其他配置
```

### 2. 权重策略说明

对于回归任务，`result_based_rr` 策略会：
- **直接使用计算好的 RR 值**：避免重复计算，提高效率
- **基于未来 RR 反向赋能**：让模型更关注"逻辑成立且利润丰厚"的样本
- **权重公式**：`Logic_Score * log(1 + RR)`

### 3. 回归任务的特殊处理

在回归任务中，标签本身就是 RR 值（连续值），因此：
- 可以直接使用标签值作为未来 RR
- 不需要重新计算 RR（通过 `rr_col` 参数指定）
- 权重计算更高效

### 4. 配置示例

完整配置请参考 `labels_with_weights.yaml`：

```yaml
weight_config:
  logic_mode: triple_resonance
  vpin_col: vpin
  vpin_threshold: 0.7
  cvd_slope_col: cvd_slope_5_f
  sr_strength_col: sr_strength_max
  logic_base: 1.0
  logic_boost: 1.5
  min_rr_threshold: 1.0
  loss_weight: 0.05
  normalize_weights: true
```

### 5. 与分类模型的区别

| 特性 | 分类模型 (sr_reversal_long) | 回归模型 (sr_reversal_rr_reg) |
|------|---------------------------|----------------------------|
| 标签类型 | 二元 (0/1) | 连续 RR 值 |
| 交易信号 | entry_threshold | top_quantile |
| 权重计算 | 需要计算未来 RR | 直接使用标签值 |
| 模型输出 | 概率 (0-1) | 连续 RR 值 |

### 6. 优势总结

1. **自适应信号选择**：top_quantile 根据市场条件自动调整
2. **更精确的预测**：连续 RR 值比二元分类更信息丰富
3. **高效的权重计算**：直接使用标签值，无需重复计算
4. **更好的风险控制**：可以根据预测 RR 值进行仓位管理

## 使用建议

1. **主力使用回归模型**：`sr_reversal_rr_reg` 更适合实际交易
2. **启用样本权重**：使用 `result_based_rr` 策略提升模型质量
3. **调整 top_quantile**：根据回测结果调整 `top_quantile` 参数（默认 0.1）
4. **监控权重分布**：确保权重分布合理，避免极端值

## 注意事项

1. **向后兼容**：如果不设置 `compute_weights=True`，行为与原有函数完全一致
2. **权重归一化**：建议启用 `normalize_weights: true`，保持权重分布稳定
3. **特征依赖**：确保所需特征（VPIN、CVD、SR 强度）存在于 DataFrame 中

