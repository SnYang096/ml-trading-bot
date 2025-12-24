# SR Reversal RR Reg Long Weighted 策略

## 概述

这是 `sr_reversal_rr_reg_long` 的带权重版本，用于对比测试样本权重对模型性能的影响。

**策略方向**：只做多（long_only）

## 主要区别

### 原策略 (sr_reversal_rr_reg_long)
- 不使用样本权重
- 所有样本权重 = 1.0
- 函数：`compute_sr_reversal_rr_continuous_label`

### 带权重策略 (sr_reversal_rr_reg_long_weighted)
- 使用 `result_based_rr` 样本权重策略
- 权重公式：`Logic_Score * log(1 + RR)`
- 函数：`compute_sr_reversal_rr_continuous_label_with_weights`
- 三重共振样本：逻辑分加成 1.5 倍
- 亏损样本（RR < 1）：权重降至 0.05

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
  --config config/strategies/sr_reversal_rr_reg_weighted
```

### 对比测试

可以同时训练两个版本进行对比：

```bash
# 训练原版本
python scripts/train_strategy_pipeline.py \
  --config config/strategies/sr_reversal_rr_reg_long

# 训练带权重版本
python scripts/train_strategy_pipeline.py \
  --config config/strategies/sr_reversal_rr_reg_long_weighted
```

## 预期效果

使用样本权重后，预期会看到：

1. **Precision（精准率）提升**：模型更关注高质量信号
2. **高质量样本权重更高**：三重共振样本权重约为普通样本的 5-10 倍
3. **亏损样本权重降低**：RR < 1 的样本权重降至 0.05，减少对模型的负面影响

## 注意事项

1. 确保所需特征存在：VPIN、CVD 斜率、SR 强度等
2. 权重已归一化：`normalize_weights: true` 确保权重分布稳定
3. 回归任务优化：直接使用标签值（RR值）作为未来RR，避免重复计算

## 文件说明

- `labels.yaml` - 带权重的标签配置（已从 labels_with_weights.yaml 复制）
- `labels.yaml.backup` - 原无权重配置的备份
- `labels_with_weights.yaml` - 带权重配置的原始文件
- 其他配置文件与原策略相同

