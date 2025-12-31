# 推荐的特征工作流（稳定版本）

## 核心概念

### Pool B：数学/数值特征的"海选池"

**Pool B** 是从大量原始特征中通过 IC/IR 筛选出的候选特征集合，包含：

- **数学特征**：DTW、EVT、GARCH、Hilbert、Hurst、频谱特征等
- **TA-Lib 特征**：MACD、RSI、BBands、ATR 等技术指标
- **数值特征**：各种统计量、相关性、波动率等

**特点**：
- 数量庞大（数百个特征），适合做"海选"
- 通过 `factor-eval` 自动筛选（基于 IC/IR）
- 数据驱动，客观评估
- **不包含语义特征**（语义特征需要人工深度加工）

### Semantic 特征：人类可理解的语义因子

**Semantic 特征**是经过人工深度加工、具有明确市场逻辑语义的特征，例如：

- `vpin_scene_semantic_scores_f`：VPIN 场景语义（compression/ignition/absorption/exhaustion）
- `wpt_scene_semantic_scores_f`：WPT 场景语义
- `trade_cluster_scene_semantic_scores_f`：交易聚类场景语义

**特点**：
- **人类可理解**：每个特征都有明确的语义含义（如"压缩"、"点火"、"吸收"、"衰竭"）
- **需要人工维护**：必须由人类根据市场逻辑设计和维护
- **从 Pool B 深度加工而来**：语义特征通常是从 Pool B 中的原始特征（如 VPIN、TradeCluster）经过语义化转换得到的
- **经过验证**：已经知道它们对特定策略有效（如 SR Reversal 中的 `vpin_scene`）

## 核心原则

1. **保持 features_all.yaml 的原始性**：只包含原始特征（数学/数值特征），不包含语义特征
2. **语义特征单独管理**：通过 `config/feature_groups_<strategy>_semantic.yaml` 定义，由人类维护
3. **让 Pool B 和语义 groups 在 feature-group-search 中竞争**：找到最佳组合

---

## 完整工作流（推荐）

### 阶段 1: 生成 Pool B（原始特征）

```bash
# 1. 运行 factor-eval 生成 Pool B（只包含原始特征）
mlbot analyze factor-eval \
  -c config/strategies/<strategy>/features_all.yaml \
  -s BTCUSDT -t 240T \
  --start-date 2023-01-01 --end-date 2025-10-31 \
  --output-dir results/pools/<strategy>/pool_b \
  --export-yaml results/pools/<strategy>/pool_b/features_pool_b.yaml \
  --remove-correlated \
  --filter-by-best-lag \
  --no-docker
```

**说明**：
- `features_all.yaml` 只包含原始特征（数学特征、TA-Lib 特征、数值特征）
  - 数学特征：DTW、EVT、GARCH、Hilbert、Hurst、频谱特征等
  - TA-Lib 特征：MACD、RSI、BBands、ATR 等
  - 数值特征：统计量、相关性、波动率等
- **不包含语义特征**（如 `vpin_scene_semantic_scores_f`）
- Pool B 会包含经过 IC/IR 筛选的原始特征（适合"海选"）

### 阶段 2: 准备语义 groups（人类维护）

语义 groups 定义在 `config/feature_groups_<strategy>_semantic.yaml` 中，**需要人类维护和添加**。

**说明**：
- **语义特征是人类可理解的特征因子**，必须由人类根据市场逻辑设计和维护
- 语义特征通常是从 Pool B 中的原始特征经过深度加工得到的
  - 例如：从 `vpin_base_aligned_features_f`（原始 VPIN 特征）加工成 `vpin_scene_semantic_scores_f`（VPIN 场景语义）
  - 例如：从 `trade_cluster_block_features_f`（原始交易聚类特征）加工成 `trade_cluster_scene_semantic_scores_f`（交易聚类场景语义）
- 语义 groups 包含的是经过语义化的特征（compression/ignition/absorption/exhaustion）
- 不需要运行 factor-eval 评估语义特征（因为已经知道它们有效，且语义特征需要人类理解其含义）
- **如何添加新的语义特征**：
  1. 从 Pool B 中选择有潜力的原始特征
  2. 根据市场逻辑设计语义转换（如"压缩"、"点火"、"吸收"、"衰竭"）
  3. 实现语义特征计算函数
  4. 添加到 `config/feature_groups_<strategy>_semantic.yaml`

### 阶段 3: 运行 feature-group-search（同时使用 Pool B 和语义 groups）

```bash
# 3. 运行 feature-group-search，同时使用 Pool B 和语义 groups
mlbot diagnose feature-group-search \
  -c config/strategies/<strategy> \
  -s BTCUSDT -t 240T \
  --start-date 2023-01-01 \
  --end-date 2025-10-31 \
  --seeds 1,2,3,4,5 \
  --groups-yaml config/feature_groups_<strategy>_semantic.yaml \
  --pool-b-yaml results/pools/<strategy>/pool_b/features_pool_b.yaml \
  --max-steps 6 \
  --objective Sharpe_mean \
  --min-trades 10 \
  --writeback-yaml config/strategies/<strategy>/features_suggested.yaml \
  --output-dir results/feature_group_search/<strategy>_best_combo \
  --deterministic \
  --no-docker
```

**说明**：
- `--groups-yaml`：指定语义 groups（经过人工筛选的语义特征）
- `--pool-b-yaml`：指定 Pool B（经过 IC/IR 筛选的原始特征）
- feature-group-search 会将 Pool B 中未在语义 groups 中的特征转换为 singleton groups
- 两者一起竞争，找到最佳组合

---

## 工作流优势

### 1. 职责分离

- **Pool B**：发现未被语义化的有效原始特征（数据驱动，"海选"）
  - 包含大量数学特征、TA-Lib 特征、数值特征
  - 通过 IC/IR 自动筛选
  - 适合发现新的有效特征
- **语义 groups**：提供经过语义化的特征（人类维护，可理解）
  - 从 Pool B 中的原始特征深度加工而来
  - 具有明确的语义含义（如"压缩"、"点火"、"吸收"、"衰竭"）
  - 需要人类根据市场逻辑设计和维护

### 2. 互补性强

- **Pool B 发现的特征**：DTW、EVT、GARCH、Hilbert、Hurst、TA-Lib 指标等原始数学/数值特征
  - 数量庞大，适合"海选"
  - 数据驱动，客观评估
- **语义 groups 提供的特征**：vpin_scene、wpt_scene、trade_cluster_scene 等语义特征
  - 从 Pool B 中的原始特征深度加工而来
  - 人类可理解，具有明确的语义含义
  - 经过验证，对特定策略有效
- **关系**：语义特征是从 Pool B 中原始特征的深度加工，两者是"原料"和"成品"的关系

### 3. 效率高

- 不需要运行两次 factor-eval
- 不需要维护两个 Pool B
- 语义 groups 已经定义好了，直接使用

### 4. 可解释性强

- **Pool B 中的特征**：经过 IC/IR 筛选，有数据支持（客观评估）
- **语义 groups 中的特征**：经过人工深度加工和语义化，有逻辑支持（人类可理解）
- **feature-group-search 的结果**：结合了两者的优势
  - Pool B 提供数据驱动的特征发现
  - 语义 groups 提供人类可理解的语义特征

---

## 可选：验证语义特征的 IC/IR

如果需要验证语义特征的 IC/IR，可以创建一个 `features_semantic.yaml`：

```bash
# 创建 features_semantic.yaml（只包含语义特征）
python scripts/create_semantic_features_yaml.py \
  --strategy <strategy> \
  --semantic-groups config/feature_groups_<strategy>_semantic.yaml \
  --output config/strategies/<strategy>/features_semantic.yaml

# 运行 factor-eval 评估语义特征
mlbot analyze factor-eval \
  -c config/strategies/<strategy>/features_semantic.yaml \
  -s BTCUSDT -t 240T \
  --start-date 2023-01-01 --end-date 2025-10-31 \
  --output-dir results/pools/<strategy>/pool_b_semantic \
  --export-yaml results/pools/<strategy>/pool_b_semantic/features_pool_b_semantic.yaml \
  --no-docker
```

**说明**：
- 这是**可选的验证步骤**，不是必需的
- 主要用于验证语义特征的 IC/IR 是否足够好
- 主要工作流仍然使用语义 groups（更高效）

---

## 文件结构

```
config/strategies/<strategy>/
├── features.yaml              # 当前使用的特征配置
├── features_all.yaml          # 所有原始特征（用于生成 Pool B）
├── features_semantic.yaml     # 所有语义特征（可选，用于验证）
└── features_suggested.yaml    # feature-group-search 的建议配置

config/
├── feature_groups_<strategy>_semantic.yaml  # 语义 groups 定义

results/pools/<strategy>/
└── pool_b/
    └── features_pool_b.yaml   # Pool B（原始特征，经过 IC/IR 筛选）
```

---

## 工作流对比

| 方案 | Pool B 来源 | 语义特征来源 | 优点 | 缺点 |
|------|------------|------------|------|------|
| **推荐方案** | features_all.yaml（原始特征） | feature_groups_<strategy>_semantic.yaml | 简洁、高效、职责分离 | 语义特征没有 IC/IR 评估 |
| 方案 A（用户建议） | features_all.yaml + features_semantic.yaml | features_semantic.yaml 的 Pool B | 可以验证语义特征的 IC/IR | 需要维护两个 Pool B，运行两次 factor-eval |
| 方案 C（混合） | features_all.yaml | feature_groups + 可选的语义 Pool B | 结合两者优点 | 复杂度较高 |

---

## 推荐理由

1. **简洁高效**：不需要运行两次 factor-eval，不需要维护两个 Pool B
2. **职责分离**：Pool B 发现原始特征，语义 groups 提供语义特征
3. **互补性强**：两者覆盖不同的特征类型，互补性强
4. **可解释性强**：Pool B 有数据支持，语义 groups 有逻辑支持
5. **已经支持**：feature-group-search 已经支持同时使用 Pool B 和语义 groups

---

## 工具

- `scripts/analyze_poolb_semantic_overlap.py`：分析 Pool B 与语义特征的交叉情况
- `scripts/create_semantic_features_yaml.py`（待创建）：创建 features_semantic.yaml（可选）

---

## 下一步

1. **采用推荐工作流**：使用 Pool B + 语义 groups
2. **可选验证**：如果需要，创建 features_semantic.yaml 并运行 factor-eval 验证语义特征的 IC/IR
3. **持续优化**：根据 feature-group-search 的结果，不断优化语义 groups 和 Pool B

