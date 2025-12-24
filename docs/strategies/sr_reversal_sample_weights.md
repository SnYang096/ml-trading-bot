# SR 反转策略样本权重系统

## 概述

样本权重系统允许你为不同的训练样本分配不同的权重，让模型更关注高质量的反转信号，同时保留普通样本的学习价值。

## 核心思想

在树模型（如 LightGBM）中，你可以给不同的样本赋予不同的权重：
- **普通反转**：Weight = 1.0
- **高质量反转**（如 VPIN+CVD+SR 三重共振）：Weight = 5.0

意义：让树模型在分裂时，哪怕损失 5 个普通样本，也要保住这 1 个"教科书样本"的准确性。

## 使用方法

### 1. 在 YAML 配置中启用权重

```yaml
label_generator:
  module: src.time_series_model.strategies.labels.sr_reversal_label
  function: compute_sr_reversal_label_with_weights
  params:
    max_holding_bars: 50
    take_profit_r: 4.0
    stop_loss_r: 1.0
    combine_mode: long_only
    
    # 启用样本权重
    compute_weights: true
    weight_strategy: triple_resonance  # 选择权重策略
    
    # 权重策略配置
    weight_config:
      vpin_col: vpin
      vpin_threshold: 0.7
      cvd_slope_col: cvd_slope_5_f
      cvd_slope_threshold: 0.0
      sr_strength_col: sr_strength_max
      sr_strength_threshold: 0.5
      resonance_weight: 5.0
```

### 2. 在训练代码中使用权重

如果使用 `return_weights=True`，标签生成函数会返回 `(labels, weights)` 元组：

```python
from src.time_series_model.strategies.labels.sr_reversal_label import (
    compute_sr_reversal_label_with_weights
)

# 生成标签和权重
labels, weights = compute_sr_reversal_label_with_weights(
    df,
    compute_weights=True,
    return_weights=True,
    weight_strategy="triple_resonance",
    weight_config={
        "vpin_col": "vpin",
        "vpin_threshold": 0.7,
        "cvd_slope_col": "cvd_slope_5_f",
        "cvd_slope_threshold": 0.0,
        "sr_strength_col": "sr_strength_max",
        "sr_strength_threshold": 0.5,
        "resonance_weight": 5.0,
    }
)

# 在训练时传入权重
import lightgbm as lgb

train_data = lgb.Dataset(
    X_train,
    label=y_train,
    weight=weights_train,  # 传入样本权重
)
```

## 支持的权重策略

### 1. uniform（统一权重）

所有样本权重为 1.0，相当于不使用权重。

```yaml
weight_strategy: uniform
weight_config: {}
```

### 2. result_based_rr（基于未来 RR 的结果驱动加权）⭐ 推荐

**这是最高阶且最具实战意义的权重策略**，属于"基于结果的样本重要性重采样（Results-based Importance Resampling）"。

**核心思想**：用未来 RR 反向赋能，让模型更关注"逻辑成立且利润丰厚"的样本。

**权重公式**：`Logic_Score * log(1 + RR)`
- **Logic_Score**：基于 VPIN/CVD/SR 的先验逻辑分
- **log(1 + RR)**：对未来收益进行对数平滑，避免极端值主导
- **亏损处理**：如果 RR < 1（亏损），权重降到最低（如 0.05）

**为什么有效？**
- 在低信噪比环境中，很多信号虽然符合逻辑（如 VPIN+CVD 完美），但由于宏观扰动或随机性，最终并没有走出来
- 如果不赋权：模型会花大量精力纠结这些"符合逻辑但结果失败"的样本
- 反向赋权：告诉模型"不仅要看逻辑，更要看结果。那些逻辑成立且利润丰厚的样本才是'真神'"

**配置示例**：

```yaml
weight_strategy: result_based_rr
weight_config:
  # 逻辑分模式：triple_resonance, sr_strength, cvd_only, none
  logic_mode: triple_resonance
  
  # 三重共振逻辑分配置
  vpin_col: vpin
  vpin_threshold: 0.7
  cvd_slope_col: cvd_slope_5_f
  cvd_slope_threshold: 0.0
  sr_strength_col: sr_strength_max
  sr_strength_threshold: 0.5
  logic_base: 1.0      # 基础逻辑分
  logic_boost: 1.5     # 三重共振时的加成
  
  # RR 计算参数（如果未提供 rr_col）
  max_holding_bars: 50
  stop_loss_r: 1.0
  take_profit_r: 2.0
  rr_ratio: 2.0
  
  # 权重计算参数
  min_rr_threshold: 1.0    # RR 低于此值视为亏损
  loss_weight: 0.05       # 亏损样本的权重
  normalize_weights: true  # 是否归一化权重
  
  # 可选：如果已有 RR 列，直接使用
  # rr_col: realized_rr
```

**注意事项**：
- 这不是标签泄露！训练时用未来数据告诉模型哪些样本重要，推理时模型已经学会了特征组合
- 比规则过滤强：模型能看到失败样本但权重低，学会"虽然像，但不是"
- 建议对比"不加权重"和"加权重"后的模型 Precision（精准率）

### 3. sr_strength（SR 强度分级加权）

根据 SR 强度分级分配权重，强度越高，权重越大。

```yaml
weight_strategy: sr_strength
weight_config:
  sr_strength_col: sr_strength_max
  strength_thresholds: [0.3, 0.5, 0.7]
  strength_weights: [1.0, 2.0, 3.0, 5.0]  # 对应 [<0.3, 0.3-0.5, 0.5-0.7, >=0.7]
```

### 4. triple_resonance（三重共振）

当 VPIN、CVD 和 SR 三个条件同时满足时，给予高权重。

```yaml
weight_strategy: triple_resonance
weight_config:
  vpin_col: vpin
  vpin_threshold: 0.7
  cvd_slope_col: cvd_slope_5_f
  cvd_slope_threshold: 0.0
  sr_strength_col: sr_strength_max
  sr_strength_threshold: 0.5
  resonance_weight: 5.0
```

**三重共振条件**：
- VPIN >= threshold（高买压/卖压）
- CVD 斜率 > threshold（资金流入/流出确认）
- SR 强度 >= threshold（强支撑/阻力）

### 5. cvd_confirmation（CVD 确认加权）

当 CVD 斜率与反转方向一致时，给予更高权重。

```yaml
weight_strategy: cvd_confirmation
weight_config:
  cvd_slope_col: cvd_slope_5_f
  cvd_slope_threshold: 0.1
  cvd_weight: 3.0
```

### 6. distance_based（距离 SR 加权）

距离 SR 越近，权重越高（反转更可靠）。

```yaml
weight_strategy: distance_based
weight_config:
  dist_col: dist_to_nearest_sr
  dist_atr_mult: 1.5
  near_weight: 3.0
  far_weight: 1.0
```

### 7. composite（组合策略）

组合多种策略，权重可以相乘或相加。

```yaml
weight_strategy: composite
weight_config:
  combine_mode: multiply  # 或 "add"
  sub_strategies:
    - name: sr_strength
      config:
        sr_strength_col: sr_strength_max
        strength_thresholds: [0.5, 0.7]
        strength_weights: [1.0, 2.0, 3.0]
    - name: cvd_confirmation
      config:
        cvd_slope_col: cvd_slope_5_f
        cvd_weight: 2.0
```

## 独立使用权重函数

你也可以单独调用权重计算函数：

```python
from src.time_series_model.strategies.labels.sr_reversal_label import (
    compute_sr_reversal_sample_weights
)

# 先计算标签
labels = compute_sr_reversal_label_full_scan(df, ...)

# 再计算权重
weights = compute_sr_reversal_sample_weights(
    df,
    labels,
    weight_strategy="triple_resonance",
    weight_config={
        "vpin_col": "vpin",
        "vpin_threshold": 0.7,
        "cvd_slope_col": "cvd_slope_5_f",
        "cvd_slope_threshold": 0.0,
        "sr_strength_col": "sr_strength_max",
        "sr_strength_threshold": 0.5,
        "resonance_weight": 5.0,
    }
)
```

## 设计思路

### 为什么需要样本权重？

1. **样本质量不均**：不是所有反转信号都同等重要
2. **高质量信号稀缺**：三重共振等高质量信号较少，但价值更高
3. **模型学习导向**：通过权重引导模型关注高质量信号

### 权重策略选择建议

- **uniform**：基线，不使用权重
- **result_based_rr** ⭐：**强烈推荐**，基于未来 RR 反向赋能，让模型关注高质量信号
- **sr_strength**：适合 SR 强度特征明显的场景
- **triple_resonance**：适合需要多重确认的高质量反转
- **cvd_confirmation**：适合资金流确认重要的场景
- **distance_based**：适合距离 SR 远近影响反转可靠性的场景
- **composite**：适合需要组合多种因素的复杂场景

### 权重值设置建议

- **基础权重**：1.0（普通样本）
- **中等权重**：2.0-3.0（有一定确认的样本）
- **高权重**：5.0-10.0（多重确认的高质量样本）

注意：权重过大可能导致模型过度关注少数样本，建议从 3.0-5.0 开始尝试。

## 注意事项

1. **向后兼容**：如果不设置 `compute_weights=True`，行为与原有函数完全一致
2. **特征依赖**：不同权重策略依赖不同的特征，确保这些特征存在于 DataFrame 中
3. **训练代码修改**：如果使用 `return_weights=True`，需要修改训练代码以接收元组
4. **权重归一化**：LightGBM 会自动处理权重，无需手动归一化

## 示例配置

完整示例配置请参考：
- `config/strategies/sr_reversal_long/labels_with_weights.yaml.example`
- `config/strategies/sr_reversal_long/labels_weight_examples.yaml`

