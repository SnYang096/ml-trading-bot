# nnmultihead：Feature Contract 的 block gating（可选特征块）完整说明

本文档完整解释 **optional blocks** 在 `nnmultihead`（path-primitives MLP）里的语义、需求、工程实现和最佳实践。

## 核心结论（先说清楚）

- **线上不要"删列/不传列"**：模型输入维度必须固定。  
  正确做法是：缺失的 block 对应列保持为缺失（NaN）→ 进入 torch 前会被填成 0，并且同时提供 **block availability mask**。
- **`append_block_mask` 不是 drop 特征**：它是 **给模型额外拼一个 block 是否可用的 0/1 标记**（输入维度增加 `n_blocks`）。
- **`block_dropout_p` 是训练时的"整块随机 drop"**：按样本、按 block 随机把该 block 的所有特征置 0，并把对应 mask 置 0（只在训练阶段生效）。
- 当前实现是 **block 级别**，不是"block 内随机 mask 某些特征"。（后续如需 per-feature masking，可以再扩展。）

---

## `optional_blocks` 的两个作用

`optional_blocks` 可以包含两类特征：

### 1. 定义额外特征（需要额外计算）

- tier0~1 没有的特征，需要通过 `optional_blocks_enabled` 启用对应的 feature nodes
- 这些 nodes 会被计算，然后输入模型
- 例如：`vpin_block`、`volume_profile_block`（需要 tick 数据）

### 2. 定义 mask 范围（用于训练鲁棒性）

- 即使特征已经在 tier0~2 中（作为 required features），也可以标记为 `optional_blocks`
- `missingness_policy` 只作用于 `optional_blocks` 中定义的特征
- 这样可以让模型训练时对这些特征进行 mask（`block_dropout_p`），提高鲁棒性
- 例如：将 tier1 中的某些特征标记为 optional，训练时随机 mask 以提高鲁棒性

**关键点**：
- `optional_blocks_enabled` 控制**是否计算**这些 blocks 的特征（materialize 阶段）
- `missingness_policy` 控制**模型训练时如何 mask**已计算的特征（训练阶段）
- 两者是独立的：即使特征已在 tier0~2 中，也可以标记为 `optional_blocks` 以便 mask

---

## 配置（Feature Contract）

位置：`config/nnmultihead/<task>/features.yaml`（已合并，推荐）

> **注意**：`feature_contract` 现在合并到 `features.yaml` 中，作为 `feature_pipeline` 的同级字段。  
> 如果存在独立的 `feature_contract.yaml`，代码仍会读取（向后兼容），但推荐使用合并格式。

示例（节选）：

```yaml
# features.yaml
feature_pipeline:
  requested_features:
    - atr_f
    - trend_r2_20_f
    # ... 所有特征都会被计算

feature_contract:
  minimal_required_cols:
    - open
    - high
    - low
    - close
    - volume
    - atr

  # 新格式：block_name -> patterns（fnmatch）
  optional_blocks:
    ticks_orderflow_semantic_blocks:
      - "*vpin*"
    trade_cluster_semantic_blocks:
      - "*trade_cluster*"
      - "*TradeCluster*"

  missingness_policy:
    # 可选：线上缺失时如何处理（当前实现以 NaN->0 + mask 为主）
    optional_blocks_on_missing: skip
    # 开启：在输入尾部拼上每个 block 的可用性 mask
    append_block_mask: true
    # 训练时：整块随机 drop 的概率（建议从 0.01~0.10 起步）
    block_dropout_p: 0.05
```

### TaskSpec 中的配置

在 `task_spec_*.yaml` 中，通过 `feature_plan_overrides.optional_blocks_enabled` 控制哪些 blocks 被计算：

```yaml
feature_plan_overrides:
  # optional_blocks_enabled控制"是否计算"（不仅仅是"是否喂给模型"）
  # 如果gate/regime需要某个block，必须在这里启用它
  optional_blocks_enabled:
    - vpin_block  # 必需：gate rules需要vpin（has_orderflow evidence）
    # - volume_profile_block  # 可选：如果gate rules需要vp_absorption_score，需要启用
```

**重要语义**：
- `optional_blocks_enabled` 控制**是否计算**这些 blocks 的特征（materialize 阶段）
- 如果 gate/regime 需要某个 block，**必须**在 `optional_blocks_enabled` 中启用它
- 模型训练时，可以通过 `feature_contract.optional_blocks` 和 `append_block_mask` 来控制哪些 blocks 被喂给模型

---

## 算法与数据流（训练 / 推理）

### 1) 解析 block → 列集合（pattern match）

代码：`src/time_series_model/models/nn/path_primitives_dataset.py`

- `resolve_block_cols_by_name(feature_cols, optional_blocks=...)`
  - 输入：训练用 `feature_cols`（固定列集合）+ 配置里的 patterns
  - 输出：`block_cols_by_name: Dict[str, List[str]]`

### 2) 计算每行的 block availability mask（0/1）

同文件：

- `_compute_block_availability_mask(...)`
  - 规则：某一行只要 block 内 **任意列**是 finite（notna 且不是 inf），该 block mask=1，否则 0。

### 3) 构建 torch 输入矩阵（NaN→0，并拼 mask）

同文件：

- `build_feature_matrix(..., append_block_mask=True, block_cols_by_name=...)`
  - 先把所有 feature cols 转数值、inf→NaN、NaN→0
  - 再把 `block_mask` 拼到输入尾部

结果：模型实际输入维度变成：

\[
d_{in} = |feature\_cols| + |blocks|
\]

### 4) 训练时 block-dropout（随机整块 drop）

代码：`src/time_series_model/models/nn/path_primitives_trainer.py`

在每个训练 batch 上：
- 对每个样本、每个 block 生成随机数 `rand ~ U(0,1)`
- 若该 block 本来可用（mask=1）且 `rand < block_dropout_p`
  - 将该 block 对应的所有 feature dims 置 0
  - 将该 block 的 mask dim 置 0

这相当于对"线上 block 开/关、缺失、成本 gating"的情形做 Monte-Carlo augmentation。

### 5) 推理阶段（不做 dropout，但保留 mask）

推理/评估代码：`src/time_series_model/models/nn/path_primitives_reporting.py`

- `predict_path_primitives(..., append_block_mask=True, block_cols_by_name=...)`
  - 不会做 dropout
  - 仍然会构建相同维度的输入（NaN→0 + mask）

---

## Gate/Regime 需求分析

### 问题背景

`optional_blocks_enabled` 的语义存在混淆，导致 gate rules 和 regime classification 的需求没有被正确满足。

**关键问题**：
- `optional_blocks_enabled` 控制的是"是否计算"，而不仅仅是"是否喂给模型"
- 如果 gate rules 需要 `vpin` 特征，但 `optional_blocks_enabled` 中没有 `vpin_block`，`vpin` 特征**不会被计算**
- 如果 regime classification 需要某些特征，但这些特征属于 optional blocks 且未启用，这些特征**不会被计算**

### Gate Rules 需要的特征

通过分析 `config/nnmultihead/execution_archetypes.yaml`，gate rules 需要以下特征：

| 特征 | 所属Block | 是否必需 |
|------|----------|---------|
| `vpin` | `vpin_block` | ✅ 必需（用于has_orderflow evidence） |
| `cvd_change_5` | 不属于optional blocks（在live_feature_plan.yaml中通过add_features添加） | ✅ 必需 |
| `cvd_change_5_normalized` | 不属于optional blocks | ✅ 必需 |
| `vp_absorption_score` | `volume_profile_block` | ⚠️ 需要确认 |

### Regime Classification 需要的特征

Regime classification 主要使用物理特征（path_efficiency, price_dir_consistency, deviation_z等），这些特征不属于 optional blocks，属于 required features。

### 当前 TaskSpec 状态

```yaml
feature_plan_overrides:
  optional_blocks_enabled:
    - vpin_block  # ✅ 已启用
```

**问题**：
- `vpin_block` 已启用 ✅
- `volume_profile_block` 未启用（如果 gate rules 需要 `vp_absorption_score`，需要启用）

---

## 解决方案

### 方案1：分离"计算需求"和"模型输入需求"（推荐，但需要较大改动）

在 TaskSpec 中区分：
- `feature_compute_requirements`：哪些 blocks 必须被计算（用于 gate/regime/FeatureStore）
- `model_input_blocks`：哪些 blocks 会被喂给模型（用于训练/推理）

**优点**：语义清晰，完全分离计算和模型输入需求  
**缺点**：需要修改 materialize 逻辑，改动较大

### 方案2：扩展 optional_blocks_enabled 语义（简单，推荐当前采用）

**明确语义**：
- `optional_blocks_enabled` 控制"是否计算"（不仅仅是"是否喂给模型"）
- 如果 gate/regime 需要某个 block，**必须**在 `optional_blocks_enabled` 中启用它
- 模型训练时，可以通过 `feature_contract.optional_blocks` 和 `append_block_mask` 来控制哪些 blocks 被喂给模型

**实施**：
1. 在 TaskSpec 中添加注释说明 gate/regime 需求
2. 确保所有 gate/regime 需要的 blocks 都被启用
3. 更新文档明确语义

### 方案3：自动推导计算需求（最智能，但实现复杂）

从 gate rules 和 regime 配置自动推导需要哪些 blocks：
- 扫描 `execution_archetypes.yaml`，找出所有需要的特征
- 映射特征到 blocks
- 自动添加到计算需求中

### 当前推荐方案（方案2）

#### 实施步骤

1. **更新 TaskSpec**：
   - 在 `feature_plan_overrides.optional_blocks_enabled` 中添加所有 gate/regime 需要的 blocks
   - 添加注释说明为什么需要这些 blocks

2. **更新文档**：
   - 明确 `optional_blocks_enabled` 控制计算
   - 说明 gate/regime 需求与模型需求的区别

3. **验证**：
   - 确保所有 gate/regime 需要的特征都被计算
   - 验证 FeatureStore 包含所有需要的特征

#### TaskSpec 示例

```yaml
feature_plan_overrides:
  # optional_blocks_enabled控制"是否计算"（不仅仅是"是否喂给模型"）
  # 如果gate/regime需要某个block，必须在这里启用它
  optional_blocks_enabled:
    - vpin_block  # 必需：gate rules需要vpin（has_orderflow evidence）
    # - volume_profile_block  # 可选：如果gate rules需要vp_absorption_score，需要启用
```

---

## 最佳实践

### 1. 明确需求来源

- 列出所有 gate rules 和 regime 需要的特征
- 确认哪些属于 optional blocks
- 在 TaskSpec 中明确启用这些 blocks

### 2. 分离关注点

- **计算需求**：由 `optional_blocks_enabled` 控制
- **模型输入需求**：由 `feature_contract.optional_blocks` 和 `append_block_mask` 控制
- **训练时 block dropout**：由 `block_dropout_p` 控制

### 3. 文档化

- 在 TaskSpec 中添加注释说明每个 block 的用途
- 记录 gate/regime 对特征的依赖关系

---

## 用现有命令验证

训练命令（例）：

```bash
python scripts/train_path_primitives_mlp.py \
  --config config/nnmultihead/path_primitives_4h_80h_min \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 240T \
  --horizon-hours 80 \
  --output-dir results/nn_path_primitives
```

训练产物中 `meta.json` / `report.html` 的 Raw meta 会包含：
- `feature_contract`
- `block_cols_by_name`
- `append_block_mask`
- `block_dropout_p`
- `block_mask_names`

---

## 常见误解澄清

- **Q：append_block_mask 是"删除整个 block 特征"吗？**  
  不是。它是"增加一个 mask 输入"，让模型识别 block 是否可用。

- **Q：block_dropout_p 是"mask block 里某些特征"吗？**  
  不是。当前实现是"整块 drop"（block 内全部特征一起 drop）。

- **Q：随机 drop 足够覆盖各种线上情况吗？**  
  它只能覆盖"缺失/关闭"的分布，但不能替代分组 ablation 的"准入证据"。  
  推荐：`block_dropout_p` 小比例做鲁棒性正则；是否把某块默认 on/off 仍要靠 add-back + slice 评估决定。

- **Q：optional_blocks_enabled 只控制模型输入吗？**  
  不是。它控制**是否计算**这些 blocks 的特征。如果 gate/regime 需要某个 block，必须启用它。

- **Q：特征已经在 tier0~2 中，还需要在 optional_blocks_enabled 中启用吗？**  
  如果只是为了模型训练时的 mask（block_dropout_p），不需要在 `optional_blocks_enabled` 中启用，只需要在 `feature_contract.optional_blocks` 中定义即可。  
  但如果 gate/regime 需要这些特征，且它们属于某个 optional block，则需要启用。

---

## 未来改进方向

考虑实施方案1，完全分离"计算需求"和"模型输入需求"：
- 添加 `feature_compute_requirements` 字段
- 保持 `optional_blocks_enabled` 仅控制模型输入
- 这样可以让模型训练和 gate/regime 的需求完全解耦

---

**最后更新**: 2026-01-22  
**状态**: 文档合并完成，推荐采用方案2（简单且有效）