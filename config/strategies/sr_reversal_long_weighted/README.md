# SR Reversal Long Weighted 策略

## 概述

这是 `sr_reversal_long` 的带权重版本，用于对比测试样本权重对模型性能的影响。

**策略方向**：只做多（long_only）  
**模型类型**：分类模型（binary classification）  
**标签类型**：二元标签（0/1）

## 主要区别

### 原策略 (sr_reversal_long)
- 不使用样本权重
- 所有样本权重 = 1.0
- 函数：`compute_sr_reversal_label_full_scan`
- 特点：基线版本，不使用样本权重

### 带权重策略 (sr_reversal_long_weighted)
- 使用 `result_based_rr` 样本权重策略
- 权重公式：`Logic_Score * log(1 + RR)`
- 函数：`compute_sr_reversal_label_with_weights`
- 特点：
  - 三重共振样本：逻辑分加成 1.5 倍
  - 亏损样本（RR < 1）：权重降至 0.05
  - 权重归一化：已启用

## 配置说明

### 权重策略配置

```yaml
compute_weights: true
weight_strategy: result_based_rr
weight_config:
  logic_mode: triple_resonance
  vpin_threshold: 0.7
  cvd_slope_col: cvd_slope_5_f
  sr_strength_threshold: 0.5
  logic_base: 1.0
  logic_boost: 1.5
  min_rr_threshold: 1.0
  loss_weight: 0.05
  normalize_weights: true
```

## 使用方法

### 训练模型

```bash
python scripts/train_strategy_pipeline.py \
  --config config/strategies/sr_reversal_long_weighted
```

### 对比测试

可以同时训练两个版本进行对比：

```bash
# 训练原版本
python scripts/train_strategy_pipeline.py \
  --config config/strategies/sr_reversal_long

# 训练带权重版本
python scripts/train_strategy_pipeline.py \
  --config config/strategies/sr_reversal_long_weighted
```

## 预期效果

使用样本权重后，预期会看到：

1. **Precision（精准率）提升**：模型更关注高质量信号
2. **高质量样本权重更高**：三重共振样本权重约为普通样本的 5-10 倍
3. **亏损样本权重降低**：RR < 1 的样本权重降至 0.05，减少对模型的负面影响

## 与回归模型的区别

| 特性 | 分类模型 (sr_reversal_long) | 回归模型 (sr_reversal_rr_reg_long) |
|------|---------------------------|----------------------------------|
| 标签类型 | 二元 (0/1) | 连续 RR 值 |
| 交易信号 | entry_threshold | top_quantile |
| 模型输出 | 概率 (0-1) | 连续 RR 值 |
| 权重计算 | 需要计算未来 RR | 直接使用标签值 |

## 注意事项

1. **确保特征完整**：带权重版本需要 VPIN、CVD 斜率、SR 强度等特征
2. **数据一致性**：两个版本使用相同的数据集
3. **随机种子**：建议设置相同的随机种子以确保可复现性
4. **训练时间**：带权重版本可能训练时间稍长（需要计算权重）

## 文件说明

- `labels.yaml` - 带权重的标签配置（已从 labels_with_weights.yaml.example 复制）
- `labels_with_weights.yaml.example` - 带权重配置的原始示例文件
- 其他配置文件与原策略相同

