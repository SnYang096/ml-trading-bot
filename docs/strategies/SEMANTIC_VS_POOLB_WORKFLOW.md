# 语义特征 vs Pool B 工作流指南

> **注意**：本文档已被 `docs/strategies/RECOMMENDED_FEATURE_WORKFLOW.md` 替代。
> 推荐使用新的稳定工作流。

## 问题分析

### 关键发现

通过对比分析发现：

1. **语义 groups 覆盖率很低（1.8%）**
   - 全量特征数: 650
   - 语义 groups 特征数: 12
   - 覆盖率: 1.8%
   - **结论**: 语义 groups 确实可能遗漏重要特征

2. **语义特征内部可能的冲突**
   - 发现同源但不同语义的组：`liquidity_void` vs `liquidity_void_scene`
   - 建议测试同时加入是否会冲突

### 两个核心担忧

1. **怕全量特征里面，经过 Pool B 还有一些特征有用**
   - ✅ 确实如此：语义 groups 只覆盖了 1.8% 的特征
   - ✅ Pool B 可能发现未被语义化的有效特征

2. **害怕语义特征直接内部还有打架的**
   - ✅ 发现可能的冲突：`liquidity_void` vs `liquidity_void_scene`
   - ✅ 需要测试组合验证

## 解决方案

### 方案 1: 两阶段工作流（推荐）

#### 阶段 1: Pool B 过滤（发现遗漏特征）

```bash
# 1. 运行 factor-eval 生成 Pool B
mlbot analyze factor-eval \
  -c config/strategies/sr_reversal_rr_reg_long/features_all.yaml \
  -s BTCUSDT \
  -t 240T \
  --start-date 2023-01-01 \
  --end-date 2025-10-31 \
  --output-dir results/pools/sr_reversal_rr_reg_long/pool_b \
  --export-yaml results/pools/sr_reversal_rr_reg_long/pool_b/features_pool_b.yaml \
  --remove-correlated \
  --filter-by-best-lag \
  --no-docker
```

#### 阶段 2: 分析 Pool B 结果

```bash
# 2. 分析 Pool B 中是否有语义 groups 未覆盖的重要特征
python scripts/analyze_semantic_vs_all_features.py \
  --strategy sr_reversal_rr_reg_long \
  --semantic-groups config/feature_groups_sr_reversal_semantic.yaml \
  --all-features results/pools/sr_reversal_rr_reg_long/pool_b/features_pool_b.yaml \
  --output-dir results/feature_analysis
```

#### 阶段 3: 检测语义特征冲突

```bash
# 3. 检测语义特征内部可能的冲突
python scripts/detect_semantic_conflicts.py \
  --strategy sr_reversal_rr_reg_long \
  --semantic-groups config/feature_groups_sr_reversal_semantic.yaml \
  --test-combinations
```

#### 阶段 4: 测试冲突组合

```bash
# 4. 测试可能的冲突组合（如 liquidity_void + liquidity_void_scene）
mlbot diagnose feature-group-search \
  -c config/strategies/sr_reversal_rr_reg_long \
  -s BTCUSDT \
  -t 240T \
  --start-date 2023-01-01 \
  --end-date 2025-10-31 \
  --seeds 1,2,3 \
  --max-steps 2 \
  --groups-yaml <(cat <<EOF
groups:
  liquidity_void:
    - liquidity_void_f
  liquidity_void_scene:
    - liquidity_void_scene_semantic_scores_f
EOF
) \
  --output-dir results/conflict_test/liquidity_void_vs_scene
```

#### 阶段 5: 合并 Pool B 和语义 groups

```bash
# 5. 运行 feature-group-search，同时使用语义 groups 和 Pool B
mlbot diagnose feature-group-search \
  -c config/strategies/sr_reversal_rr_reg_long \
  -s BTCUSDT \
  -t 240T \
  --start-date 2023-01-01 \
  --end-date 2025-10-31 \
  --seeds 1,2,3,4,5 \
  --groups-yaml config/feature_groups_sr_reversal_semantic.yaml \
  --pool-b-yaml results/pools/sr_reversal_rr_reg_long/pool_b/features_pool_b.yaml \
  --max-steps 6 \
  --writeback-yaml config/strategies/sr_reversal_rr_reg_long/features_suggested.yaml \
  --output-dir results/feature_group_search/sr_reversal_best_combo \
  --no-docker
```

### 方案 2: 只用语义 groups（快速验证）

如果 Pool B 分析显示没有重要遗漏，可以直接用语义 groups：

```bash
mlbot diagnose feature-group-search \
  -c config/strategies/sr_reversal_rr_reg_long \
  -s BTCUSDT \
  -t 240T \
  --start-date 2023-01-01 \
  --end-date 2025-10-31 \
  --seeds 1,2,3,4,5 \
  --groups-yaml config/feature_groups_sr_reversal_semantic.yaml \
  --max-steps 6 \
  --writeback-yaml config/strategies/sr_reversal_rr_reg_long/features_suggested.yaml \
  --output-dir results/feature_group_search/sr_reversal_semantic_only \
  --no-docker
```

## 工具说明

### 1. `analyze_semantic_vs_all_features.py`

分析语义特征 vs 全量特征的覆盖情况。

**功能**:
- 统计总体覆盖率
- 按特征类型分析覆盖情况
- 检测语义特征内部冲突
- 生成 JSON 报告

**输出**:
- `results/feature_analysis/{strategy}_feature_analysis.json`

### 2. `detect_semantic_conflicts.py`

检测语义特征内部可能的冲突。

**功能**:
- 检测重复特征
- 检测同源但不同语义的组
- 生成测试组合命令

**输出**:
- 冲突检测报告
- 测试组合命令建议

## 判断标准

### 何时只用语义 groups？

- ✅ Pool B 分析显示没有重要遗漏
- ✅ 语义 groups 已经覆盖了主要特征类型
- ✅ 当前最佳配置（如 `vpin_scene` + `kline_core`）表现很好
- ✅ 需要快速验证

### 何时同时使用 Pool B 和语义 groups？

- ✅ Pool B 分析显示有重要遗漏（如 DTW、Hilbert 等）
- ✅ 语义 groups 覆盖率很低（< 5%）
- ✅ 需要最全面的特征搜索
- ✅ 有充足的计算资源

### 如何处理语义冲突？

1. **测试冲突组合**：运行 feature-group-search 测试同时加入是否会冲突
2. **选择语义化版本**：通常语义化版本（`*_scene`）应该替代原始版本
3. **移除冲突组**：如果确认冲突，从语义 groups 中移除原始版本

## 当前状态

### SR Reversal 分析结果

- **覆盖率**: 1.8% (12/650)
- **主要遗漏**:
  - DTW 模式匹配: 175 个特征，0% 覆盖
  - Trade Cluster: 67 个特征，0% 覆盖
  - K线技术指标: 63 个特征，0% 覆盖
  - VPIN 相关: 53 个特征，0% 覆盖
  - WPT 小波: 22 个特征，0% 覆盖

- **可能的冲突**:
  - `liquidity_void` vs `liquidity_void_scene`: 同源但不同语义

### 建议

1. **等待 Pool B 完成**：检查是否有重要遗漏
2. **测试冲突组合**：验证 `liquidity_void` + `liquidity_void_scene` 是否冲突
3. **合并搜索**：如果 Pool B 有重要遗漏，同时使用 Pool B 和语义 groups
4. **移除冲突组**：如果确认冲突，从语义 groups 中移除原始版本

