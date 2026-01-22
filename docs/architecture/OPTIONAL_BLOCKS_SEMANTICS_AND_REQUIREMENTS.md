# Optional Blocks语义和需求说明

## 问题背景

`optional_blocks_enabled`的语义存在混淆，导致gate rules和regime classification的需求没有被正确满足。

## `optional_blocks`的两个作用

`optional_blocks`可以包含两类特征：

1. **定义额外特征**（需要额外计算）：
   - tier0~1 没有的特征，需要通过 `optional_blocks_enabled` 启用对应的 feature nodes
   - 这些 nodes 会被计算，然后输入模型
   - 例如：`vpin_block`、`volume_profile_block`（需要 tick 数据）

2. **定义 mask 范围**（用于训练鲁棒性）：
   - 即使特征已经在 tier0~2 中（作为 required features），也可以标记为 `optional_blocks`
   - `missingness_policy` 只作用于 `optional_blocks` 中定义的特征
   - 这样可以让模型训练时对这些特征进行 mask（`block_dropout_p`），提高鲁棒性
   - 例如：将 tier1 中的某些特征标记为 optional，训练时随机 mask 以提高鲁棒性

**关键点**：
- `optional_blocks_enabled` 控制**是否计算**这些 blocks 的特征（materialize 阶段）
- `missingness_policy` 控制**模型训练时如何 mask**已计算的特征（训练阶段）
- 两者是独立的：即使特征已在 tier0~2 中，也可以标记为 `optional_blocks` 以便 mask

## 当前语义

### 官方文档说明

根据`docs/architecture/NNMULTIHEAD_FEATURE_CONTRACT_BLOCK_GATING.md`：

- **启用block时**：
  - 这些block对应的feature nodes会被写入派生`features.yaml.requested_features.optional_blocks`
  - 同时派生`features.yaml.feature_contract.optional_blocks`会被自动生成
  - 训练时可使用`missingness_policy.block_dropout_p`做**block级dropout**

- **不启用block时**：
  - 这组nodes**不会被计算/不会被喂给模型**（派生config里不出现）
  - 对应的列级block mask/缺失策略也不参与

### 关键问题

**`optional_blocks_enabled`控制的是"是否计算"，而不仅仅是"是否喂给模型"**。

这意味着：
- 如果gate rules需要`vpin`特征，但`optional_blocks_enabled`中没有`vpin_block`，`vpin`特征**不会被计算**
- 如果regime classification需要某些特征，但这些特征属于optional blocks且未启用，这些特征**不会被计算**

## Gate/Regime需求分析

### Gate Rules需要的特征

通过分析`config/nnmultihead/execution_archetypes.yaml`，gate rules需要以下特征：

| 特征 | 所属Block | 是否必需 |
|------|----------|---------|
| `vpin` | `vpin_block` | ✅ 必需（用于has_orderflow evidence） |
| `cvd_change_5` | 不属于optional blocks（在live_feature_plan.yaml中通过add_features添加） | ✅ 必需 |
| `cvd_change_5_normalized` | 不属于optional blocks | ✅ 必需 |
| `vp_absorption_score` | `volume_profile_block` | ⚠️ 需要确认 |

### Regime Classification需要的特征

Regime classification主要使用物理特征（path_efficiency, price_dir_consistency, deviation_z等），这些特征不属于optional blocks，属于required features。

### 当前TaskSpec状态

```yaml
feature_plan_overrides:
  optional_blocks_enabled:
    - vpin_block  # ✅ 已启用
```

**问题**：
- `vpin_block`已启用 ✅
- `volume_profile_block`未启用（如果gate rules需要`vp_absorption_score`，需要启用）

## 解决方案

### 方案1：分离"计算需求"和"模型输入需求"（推荐，但需要较大改动）

在TaskSpec中区分：
- `feature_compute_requirements`：哪些blocks必须被计算（用于gate/regime/FeatureStore）
- `model_input_blocks`：哪些blocks会被喂给模型（用于训练/推理）

**优点**：语义清晰，完全分离计算和模型输入需求
**缺点**：需要修改materialize逻辑，改动较大

### 方案2：扩展optional_blocks_enabled语义（简单，推荐当前采用）

**明确语义**：
- `optional_blocks_enabled`控制"是否计算"（不仅仅是"是否喂给模型"）
- 如果gate/regime需要某个block，**必须**在`optional_blocks_enabled`中启用它
- 模型训练时，可以通过`feature_contract.optional_blocks`和`append_block_mask`来控制哪些blocks被喂给模型

**实施**：
1. 在TaskSpec中添加注释说明gate/regime需求
2. 确保所有gate/regime需要的blocks都被启用
3. 更新文档明确语义

### 方案3：自动推导计算需求（最智能，但实现复杂）

从gate rules和regime配置自动推导需要哪些blocks：
- 扫描`execution_archetypes.yaml`，找出所有需要的特征
- 映射特征到blocks
- 自动添加到计算需求中

## 当前推荐方案（方案2）

### 实施步骤

1. **更新TaskSpec**：
   - 在`feature_plan_overrides.optional_blocks_enabled`中添加所有gate/regime需要的blocks
   - 添加注释说明为什么需要这些blocks

2. **更新文档**：
   - 明确`optional_blocks_enabled`控制计算
   - 说明gate/regime需求与模型需求的区别

3. **验证**：
   - 确保所有gate/regime需要的特征都被计算
   - 验证FeatureStore包含所有需要的特征

### TaskSpec示例

```yaml
feature_plan_overrides:
  # optional_blocks_enabled控制"是否计算"（不仅仅是"是否喂给模型"）
  # 如果gate/regime需要某个block，必须在这里启用它
  optional_blocks_enabled:
    - vpin_block  # 必需：gate rules需要vpin（has_orderflow evidence）
    # - volume_profile_block  # 可选：如果gate rules需要vp_absorption_score，需要启用
```

## 最佳实践

1. **明确需求来源**：
   - 列出所有gate rules和regime需要的特征
   - 确认哪些属于optional blocks
   - 在TaskSpec中明确启用这些blocks

2. **分离关注点**：
   - 计算需求：由`optional_blocks_enabled`控制
   - 模型输入需求：由`feature_contract.optional_blocks`和`append_block_mask`控制
   - 训练时block dropout：由`block_dropout_p`控制

3. **文档化**：
   - 在TaskSpec中添加注释说明每个block的用途
   - 记录gate/regime对特征的依赖关系

## 未来改进方向

考虑实施方案1，完全分离"计算需求"和"模型输入需求"：
- 添加`feature_compute_requirements`字段
- 保持`optional_blocks_enabled`仅控制模型输入
- 这样可以让模型训练和gate/regime的需求完全解耦

---

**最后更新**: 2026-01-22  
**状态**: 文档化完成，推荐采用方案2（简单且有效）
