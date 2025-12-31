# Semantic Groups 单例展开

## 状态

✅ **已实现**：通过 `--expand-semantic-singletons` 选项启用

## 问题

当前 `feature-group-search` 将 semantic groups 作为整体选择，但同一个 semantic feature node 可能包含多个语义（如 `trade_cluster_scene_semantic_scores_f` 包含 compression/ignition/absorption/exhaustion），这些语义可能对策略有相反的作用。

## 当前行为

- **Semantic groups**：一组一组选择（如 `trade_cluster_scene: [trade_cluster_scene_semantic_scores_f]`）
- **Pool B**：每个特征单独作为一个候选组（singleton groups）

## 问题分析

### 示例：`trade_cluster_scene_semantic_scores_f`

该特征输出 4 个列：
- `trade_cluster_compression_score`
- `trade_cluster_ignition_score`
- `trade_cluster_absorption_scene_score`
- `trade_cluster_exhaustion_scene_score`

**问题**：
- `ignition` 对突破策略好，`exhaustion` 对反转策略好
- 如果作为一组选择，可能同时加入两个相反的语义，导致冲突

### 分组依据

**当前分组方式**：根据 **raw data 分组**（即 feature node 分组）
- `trade_cluster_scene: [trade_cluster_scene_semantic_scores_f]` 是一个组
- 但该 node 输出多个语义列

**建议分组方式**：根据 **语义分组**（即 output_columns 分组）
- 每个语义列单独作为一个候选组
- 例如：`trade_cluster_ignition: [trade_cluster_ignition_score]`（需要从 node 中提取）

## 解决方案

### 方案 1：展开 semantic groups 为单例（推荐）

将 semantic groups 中的每个 feature node 展开为其 `output_columns`，每个输出列单独作为一个候选组。

**优点**：
- 与 Pool B 一致（都是单例）
- 避免相反语义冲突
- 更精细的选择

**缺点**：
- 需要解析 `feature_dependencies.yaml` 获取 `output_columns`
- 候选组数量增加（但仍在可接受范围内）

**实现**：
1. 加载 `feature_dependencies.yaml`
2. 对于 semantic groups 中的每个 feature node，获取其 `output_columns`
3. 为每个 `output_column` 创建一个单例组（需要确保依赖的 feature node 也被包含）

### 方案 2：保持当前分组，但添加冲突检测

保持当前的分组方式，但在选择时检测冲突。

**优点**：
- 不需要修改太多代码
- 保持分组逻辑简单

**缺点**：
- 仍然可能选择相反的语义
- 需要人工定义冲突规则

## 性能分析

### Pool B 单例展开的性能

**复杂度**：
- 候选组数量：Pool B 特征数（例如 100-200 个）
- 每步评估：`O(candidates × seeds)`
- 总复杂度：`O(steps × candidates × seeds)`

**实际性能**：
- 如果 Pool B 有 100 个特征，5 个 seeds，6 步：
  - 每步需要评估 100 个候选 × 5 seeds = 500 次训练+回测
  - 总评估次数：6 × 500 = 3000 次
- **可能很慢**，但可以通过以下方式优化：
  1. 减少 seeds（例如 3 个而不是 5 个）
  2. 减少 max_steps（例如 4 步而不是 6 步）
  3. 使用更快的回测（例如简化回测逻辑）
  4. 并行化（但当前是顺序执行）

### Semantic groups 单例展开的性能

**复杂度**：
- 如果每个 semantic feature node 平均输出 4 个列：
  - 候选组数量：semantic groups 数 × 4（例如 10 个 groups × 4 = 40 个）
- 总复杂度：`O(steps × (poolb_candidates + semantic_candidates) × seeds)`

**实际性能**：
- 如果 Pool B 有 100 个特征，semantic 有 40 个列，5 个 seeds，6 步：
  - 每步需要评估 140 个候选 × 5 seeds = 700 次训练+回测
  - 总评估次数：6 × 700 = 4200 次
- **会更慢**，但语义特征数量通常较少（10-20 个 groups），所以影响有限

## 实现状态

✅ **已实现**：功能已添加到 `feature-group-search` 中

### 使用方法

功能已实现，代码位于 `src/time_series_model/diagnostics/feature_group_search.py`：

- `_expand_semantic_groups_to_singletons()` 函数：展开 semantic groups 为单例
- `--expand-semantic-singletons` 选项：启用展开功能

## 使用方式

```bash
# 默认行为（保持向后兼容）
mlbot diagnose feature-group-search \
  --groups-yaml config/feature_groups_sr_reversal_semantic.yaml \
  --pool-b-yaml results/pools/sr_reversal_rr_reg_long/pool_b/features_pool_b.yaml \
  ...

# 展开 semantic groups 为单例（推荐）
mlbot diagnose feature-group-search \
  --groups-yaml config/feature_groups_sr_reversal_semantic.yaml \
  --pool-b-yaml results/pools/sr_reversal_rr_reg_long/pool_b/features_pool_b.yaml \
  --expand-semantic-singletons \
  ...
```

## 注意事项

1. **依赖处理**：展开时需要确保依赖的 feature node 也被包含（例如 `trade_cluster_scene_semantic_scores_f` 需要 `trade_cluster_semantic_scores_f`）
2. **命名冲突**：确保展开后的组名唯一
3. **性能**：展开后候选组数量增加，评估时间可能增加
4. **向后兼容**：默认不展开，保持当前行为

## 后续优化

1. **并行化**：并行评估候选组（当前是顺序执行）
2. **早停**：如果某个候选组明显很差，提前停止评估
3. **缓存**：缓存特征计算结果，避免重复计算
4. **分批评估**：将候选组分批评估，减少内存占用

