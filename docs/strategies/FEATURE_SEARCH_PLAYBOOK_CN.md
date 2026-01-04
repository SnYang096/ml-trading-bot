## 特征搜索 Playbook（树模型）

本文档用于沉淀本项目的 **特征工程/特征组合搜索** 的标准工作流（面向树模型：LightGBM 等），覆盖四个核心策略：
**SR 反转**、**SR 突破**、**压缩突破**、**趋势跟随**。

同时解释“基线稳定”的判定标准，并对下一阶段准备引入的搜索算法（Successive Halving / Beam Search / SFFS）做概念说明与落地建议。

---

## 关键概念（术语表）

### Feature node（特征节点） vs Feature column（特征列）

- **特征节点**：特征计算函数（通常以 `*_f` 结尾），例如 `trade_cluster_scene_semantic_scores_f`。
- **特征列**：节点产出的具体列，例如 `trade_cluster_absorption_scene_score`。

默认情况下，大多数工具按 **节点级（node-level）** 做选择；一个节点可能输出多列。

### Base features（Pool A，基础必需特征）

每个策略可以定义 `config/strategies/<strategy>/features_base.yaml`，作为 **标签/回测必需特征**：

- 它们 **始终包含**，且 **不参与搜索优化**（例如 `atr_f`、`poc_hal_features_close_f`）。

### Pool B（数据驱动候选池）

**Pool B** 是 `mlbot analyze factor-eval` 导出的候选 YAML（IC/IR 筛选 + 去相关），用于 wrapper 搜索：

- 默认输出：`results/pools/<strategy>/pool_b/<tag>/features_pool_b.yaml`

### Semantic groups（语义候选组，人类维护）

语义 groups 通过 YAML 管理（优先级）：

- `config/feature_groups_<strategy_dir>_semantic.yaml`（存在则优先）
- 否则 `config/feature_groups.yaml`

语义组承载“路径故事/场景语义”（compression / ignition / absorption / exhaustion）与人工维护的特征块。

### `--expand-semantic-singletons`（语义单列展开）

某些语义节点会同时输出多列场景语义（并且不同策略可能存在“相反语义”冲突）。

启用 `--expand-semantic-singletons` 后：

- 语义块会被展开为 **每个输出列一个候选**（更细粒度选择）。
- 代价是候选变多 → 评估更慢。

---

## “基线稳定”是什么意思？什么时候算稳定？

我们会先把 4 个策略在同一实验面下跑出稳定基线，然后再上更工业化的搜索算法。

**基线稳定（可升级算法）的判定标准：**

- **实验面固定**：symbol/timeframe/date range/test_size/seeds/max_steps/min_trades 等不再频繁变化。
- **可复现**：同一配置重复跑时：
  - `Sharpe_mean` 不大幅漂移，
  - selected groups / 核心特征不会频繁翻车，
  - stop_reason（例如“无进一步提升”）行为一致。
- **四策略完整闭环**：每个策略都产出：
  - `feature_group_search_result.json`
  - `features_pool_b.yaml`
  - writeback `features_suggested_<tag>.yaml`（如果开启）
  - 一份统一汇总报告（用于横向对比）。

---

## 标准工作流（一步一步怎么做）

### Step 0：先保证特征契约正确

在跑大规模搜索前，先做 contract 检查与关键特征测试，避免“语义事故”（例如 ATR 语义被误改导致所有结果不可比）。

### Step 1：生成 Pool B（factor-eval）

```bash
mlbot analyze factor-eval \
  --strategy-config config/strategies/<strategy> \
  --symbol BTCUSDT --timeframe 240T \
  --start-date 2023-01-01 --end-date 2025-12-31 \
  --remove-correlated --filter-by-best-lag \
  --output-dir results/pools/<strategy>/pool_b/<tag> \
  --export-yaml results/pools/<strategy>/pool_b/<tag>/features_pool_b.yaml \
  --no-docker
```

### Step 2：组合搜索（semantic groups + Pool B）

```bash
mlbot diagnose feature-group-search \
  --base-strategy-config config/strategies/<strategy> \
  --symbol BTCUSDT --timeframe 240T \
  --start-date 2023-01-01 --end-date 2025-12-31 \
  --seeds 1,2,3,4,5 \
  --objective Sharpe_mean \
  --min-trades 10 --max-steps 10 \
  --pool-b-yaml results/pools/<strategy>/pool_b/<tag>/features_pool_b.yaml \
  --writeback-yaml config/strategies/<strategy>/features_suggested_<tag>.yaml \
  --output-dir results/feature_group_search/<strategy>_greedy_poolb_semantic_<tag> \
  --no-docker
```

### Step 3：语义单列展开对照实验（可选，但推荐做一次）

```bash
mlbot diagnose feature-group-search \
  ... \
  --expand-semantic-singletons
```

### Step 4：读结果/对比（你应该看什么）

每个策略至少对齐比较：

- baseline `Sharpe_mean` vs final `Sharpe_mean`
- greedy 每一步选择了什么 group（history）
- 最终入模 requested features（node 或 column）
- 候选被拒的原因分布（如 `min_trades`）

---

## 这些“更工业化的算法”是什么？

### 现有基线：Greedy Forward Selection（贪心前向）

每一步都评估：

- “当前已选组合 + 一个候选组”的 multi-seed 表现
- 选择能让目标（如 `Sharpe_mean`）提升最大的那个
- 若没有任何候选能严格提升 → 停止

优点：简单、成本较低、可解释。  
缺点：容易陷入 **局部最优**，对 **协同效应（A+B 才有效）**不友好。

### 现在已经可以在 `feature-group-search` 里直接用

`mlbot diagnose feature-group-search` 已支持：

- `--search-algo greedy`（默认）
- `--search-algo halving`（Successive Halving，逐级减半）
- `--search-algo beam`（Beam Search）
- `--search-algo sffs`（SFFS）

### Successive Halving（第一优先级升级）

核心思想：把预算分级。

- 用 **小预算** 先评估大量候选（例如更少 seeds / 更短时间窗 / 更小 max_steps）
- 只保留表现最好的前一部分
- 再用 **大预算** 对幸存者复核

为什么适合我们：

- 我们已经有天然的“预算维度”（seeds、max_steps、时间范围），改造成本最低，节省算力最明显。

怎么跑（优先用于加速）：

```bash
mlbot diagnose feature-group-search \
  ... \
  --search-algo halving \
  --halving-stages 1,3,5 \
  --halving-top-fraction 0.25 \
  --halving-min-survivors 5
```

### Beam Search（第二优先级升级）

核心思想：每一步不只保留 1 条路径，而是保留 top‑K 条候选路径。

- step 1 保留 top‑K 个“加一组”的方案
- step 2 对每条路径继续扩展，再保留 top‑K

收益：

- 专治“协同效应/局部最优”：某些组单独不强，但和另一个组组合后很强；Beam 能把这种路径保留下来。

怎么跑（优先用于找强组合）：

```bash
mlbot diagnose feature-group-search \
  ... \
  --search-algo beam \
  --beam-width 3
```

### SFFS（第三优先级升级）

SFFS = Sequential Floating Forward Selection（前进 + 后退）。

流程：

- forward add（像贪心一样加组）
- 然后尝试 **backward remove**：把已选组逐个移除，只要能提升目标就移除

收益：

- 修正贪心“早期加错组”的问题（后续变成冗余/有害，也能被删掉）。

代价：

- 需要更多评估（尤其是 backward 阶段），所以放在 baseline 稳定之后做。

怎么跑（适合“先加再删”）：

```bash
mlbot diagnose feature-group-search \
  ... \
  --search-algo sffs \
  --sffs-max-backward-per-step 2
```

---

## 实战提示：为什么“单列展开”结果看起来会很少？

这其实是预期行为之一：

- 协同效应往往发生在“块级/多列”层面，单列 greedy 容易错过 A+B 的组合。
- 单列候选变多，贪心更容易陷入局部最优。
- 有的策略确实只需要极少数核心列（加上 scale 列如 `atr`）就能达到最优/次优。

因此我们推荐的路线是：

先建立 **四策略稳定 baseline**（greedy：non-singletons + singletons 对照）→ 再上 Successive Halving / Beam / SFFS。

---

## nnmultihead（Path Primitives）怎么办？

本 playbook 主要面向“策略/树模型”的特征搜索。  
如果你在做 **nn 多头路径原语**（`dir/mfe/mae/t_to_mfe`），请使用对应的 playbook：

- `docs/strategies/NNMULTIHEAD_FEATURE_SEARCH_PLAYBOOK_CN.md`
- 命令速查：`docs/guides/NNMULTIHEAD_COMMANDS_CN.md`


