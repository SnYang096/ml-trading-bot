# features_all.yaml 自包含修复总结

## 修复概述

修复了 `features_all.yaml` 的生成和使用方式，确保它是自包含的，不依赖 `features.yaml`。

## 重要概念

### Pool B：数学/数值特征的"海选池"

**Pool B** 是从大量原始特征中通过 IC/IR 筛选出的候选特征集合，包含：

- **数学特征**：DTW、EVT、GARCH、Hilbert、Hurst、频谱特征等
- **TA-Lib 特征**：MACD、RSI、BBands、ATR 等技术指标
- **数值特征**：各种统计量、相关性、波动率等

**特点**：
- 数量庞大（数百个特征），适合做"海选"
- 通过 `factor-eval` 自动筛选（基于 IC/IR）
- 数据驱动，客观评估

### Semantic 特征：人类可理解的语义因子

**Semantic 特征**是经过人工深度加工、具有明确市场逻辑语义的特征，例如：

- `vpin_scene_semantic_scores_f`：VPIN 场景语义（compression/ignition/absorption/exhaustion）
- `wpt_scene_semantic_scores_f`：WPT 场景语义
- `trade_cluster_scene_semantic_scores_f`：交易聚类场景语义

**特点**：
- **人类可理解**：每个特征都有明确的语义含义
- **需要人工维护**：必须由人类根据市场逻辑设计和维护
- **从 Pool B 深度加工而来**：语义特征通常是从 Pool B 中的原始特征经过语义化转换得到的

## 修复内容

### 1. 修复 `features_all.yaml` 生成（自包含）

**文件**: `scripts/generate_all_features_yaml.py`

**修改**:
- 使用 `resolve_dependencies()` 递归解析所有依赖特征
- 只包含特征计算函数名（带 `_f` 后缀），不包含输出列
- 确保所有依赖都被自动包含

**结果**:
- 从 841 个项目（特征名 + 输出列）精简到 205 个特征名（包含所有依赖）
- 所有依赖自动解析，文件完全自包含

### 2. 修复 `factor-eval` 使用 `features_all.yaml`

**文件**: `src/time_series_model/diagnostics/factor_ts_eval.py`

**修改**:
- 使用 `features_all.yaml` 时，直接覆盖 `strategy_cfg.features`
- 不再加载 `features.yaml`
- 移除 append 模式的自动切换
- 使用 strategy 模式（因为 `features_all.yaml` 已自包含）

**关键变化**:
```python
# 之前：使用 append 模式，加载 features.yaml + features_all.yaml
# 现在：直接覆盖 strategy_cfg.features，只使用 features_all.yaml
strategy_cfg.features = FeaturePipelineConfig(...)
args.feature_mode = "strategy"  # 不再使用 append
```

### 3. 修复 `feature-group-search` 的 base_features

**文件**: `src/time_series_model/diagnostics/feature_group_search.py`

**修改**:
- 使用 Pool B 或 semantic groups 时，`base_features = []`（从空开始）
- 不再从 `features.yaml` 加载 base_features（避免污染）
- 只有在没有 Pool B 和 semantic groups 时，才使用 `features.yaml` 作为 base

**关键变化**:
```python
# 之前：总是从 features.yaml 加载 base_features
# 现在：使用 Pool B 或 semantic groups 时，base_features = []
if has_pool_b or has_semantic_groups:
    base_features = []  # 从空开始
```

## 测试验证

创建了 `tests/unit/test_features_all_self_contained.py` 来验证修复：

### 测试结果

✅ **所有测试通过** (4/4):

1. `test_generate_all_features_yaml_includes_all_dependencies`: PASSED
   - 验证生成的 `features_all.yaml` 包含所有依赖

2. `test_features_all_yaml_self_contained_for_real_strategy`: PASSED
   - 验证所有策略的 `features_all.yaml` 都是自包含的
   - sr_breakout: 205 个特征，所有依赖已解析
   - compression_breakout: 205 个特征，所有依赖已解析
   - trend_following: 205 个特征，所有依赖已解析

3. `test_factor_eval_overrides_features_config`: PASSED
   - 验证 `factor-eval` 正确覆盖 `strategy_cfg.features`

4. `test_feature_group_search_empty_base_with_pool_b`: PASSED
   - 验证 `feature-group-search` 使用空 base_features 当 Pool B 或 semantic groups 存在时

## 重新生成的文件

已重新生成所有策略的 `features_all.yaml`:

- ✅ `config/strategies/sr_breakout/features_all.yaml` (205 个特征)
- ✅ `config/strategies/compression_breakout/features_all.yaml` (205 个特征)
- ✅ `config/strategies/trend_following/features_all.yaml` (205 个特征)

## 使用方式

### 1. 生成 `features_all.yaml`

```bash
python scripts/generate_all_features_yaml.py --strategy-config config/strategies/sr_breakout
```

### 2. 使用 `factor-eval` 评估所有特征

```bash
mlbot analyze factor-eval \
  -c config/strategies/sr_breakout/features_all.yaml \
  -s BTCUSDT -t 240T \
  --start-date 2023-01-01 --end-date 2025-10-31 \
  --output-dir results/pools/sr_breakout/pool_b \
  --export-yaml results/pools/sr_breakout/pool_b/features_pool_b.yaml \
  --remove-correlated --filter-by-best-lag --no-docker
```

**注意**: `features_all.yaml` 是自包含的，不需要 `features.yaml`。

### 3. 使用 `feature-group-search` 找到最优组合

```bash
mlbot diagnose feature-group-search \
  -c config/strategies/sr_breakout \
  -s BTCUSDT -t 240T \
  --start-date 2023-01-01 --end-date 2025-10-31 \
  --pool-b-yaml results/pools/sr_breakout/pool_b/features_pool_b.yaml
```

**注意**: 当使用 Pool B 或 semantic groups 时，`base_features` 为空，从空开始搜索最优组合。

## 验证清单

- [x] `features_all.yaml` 包含所有特征和依赖
- [x] `features_all.yaml` 只包含特征计算函数名（带 `_f`）
- [x] `factor-eval` 使用 `features_all.yaml` 时不加载 `features.yaml`
- [x] `factor-eval` 不使用 append 模式
- [x] `feature-group-search` 使用 Pool B 时 base_features 为空
- [x] 所有测试通过

## 后续工作

1. 等待 `factor-eval` 完成，生成 Pool B YAML
2. 运行 `feature-group-search` 找到最优特征组合
3. 为其他策略（compression_breakout, trend_following）也运行 `factor-eval`

## 相关文档

- `docs/strategies/RECOMMENDED_FEATURE_WORKFLOW.md`: 推荐的特征工作流
- `docs/strategies/POOLB_SEMANTIC_OVERLAP_REPORT.md`: Pool B 和语义特征重叠报告

