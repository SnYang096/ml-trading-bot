# 样本权重策略 - RR分级处理

## 概述

带权重版本的 `result_based_rr` 策略现在支持**RR分级处理**，根据未来RR比率的不同级别给予不同的权重。

## RR分级策略

### 分级标准

1. **高RR样本**（RR >= 2.0）：
   - 权重公式：`log(1 + RR) * high_rr_boost`
   - 默认 `high_rr_boost = 1.5`
   - 示例：RR = 2.0 → 权重 = log(3) * 1.5 ≈ 1.65

2. **中等RR样本**（1.0 <= RR < 2.0）：
   - 权重公式：`log(1 + RR)`
   - 示例：RR = 1.5 → 权重 = log(2.5) ≈ 0.92

3. **低RR样本**（RR < 1.0）：
   - 权重：`loss_weight`（默认 0.05）
   - 这些是亏损样本，给予最低权重

### 配置参数

```yaml
weight_config:
  # RR分级参数
  medium_rr_threshold: 1.0    # 中等RR阈值（1.0 <= RR < high_rr_threshold）
  high_rr_threshold: 2.0       # 高RR阈值（RR >= 2.0 时使用高权重）
  high_rr_boost: 1.5           # 高RR加成倍数
  loss_weight: 0.05            # 亏损样本的权重（RR < medium_rr_threshold）
```

## 权重计算公式

最终权重 = `Logic_Score * Result_Weight`

其中：
- **Logic_Score**：基于 VPIN/CVD/SR 的先验逻辑分
  - 三重共振：`logic_base * logic_boost`（默认 1.0 * 1.5 = 1.5）
  - 普通样本：`logic_base`（默认 1.0）

- **Result_Weight**：根据RR分级计算
  - RR >= 2.0：`log(1 + RR) * high_rr_boost`
  - 1.0 <= RR < 2.0：`log(1 + RR)`
  - RR < 1.0：`loss_weight`

## 示例

假设有一个样本：
- RR = 2.5（高RR）
- 满足三重共振条件（Logic_Score = 1.5）

权重计算：
1. Result_Weight = log(1 + 2.5) * 1.5 = log(3.5) * 1.5 ≈ 1.88
2. Final_Weight = 1.5 * 1.88 = 2.82

归一化后（假设平均权重为1.0），最终权重 ≈ 2.82

## 测试用例

测试文件：`tests/strategies/test_weighted_vs_unweighted_labels.py`

验证内容：
1. ✅ 带权重版本和无权重、SR过滤版本的标签数量相同
2. ✅ 权重根据RR值分级处理
3. ✅ 高RR样本权重 >= 中等RR样本权重
4. ✅ 中等RR样本权重 > 低RR样本权重
5. ✅ 三重共振样本的权重合理

## 优势

1. **更精细的权重分配**：根据RR值分级，而不是简单的二分（盈利/亏损）
2. **突出高RR样本**：RR >= 2.0 的样本获得更高权重，让模型更关注高收益机会
3. **平滑过渡**：使用对数函数平滑权重，避免极端值主导

