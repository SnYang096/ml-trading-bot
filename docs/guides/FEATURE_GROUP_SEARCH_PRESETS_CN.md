# Feature-group-search 预算预设（A / B / C）与命令工作流（中文）

本文档回答两个问题：

1) **A/B/C 各自到底调用了什么命令、覆盖了哪些参数、差异是什么？**  
2) **如何用一个可复现的工作流最终找到特征组合（A → shortlist → B/C）？**

> 入口链接：见 `README_CN.md` 的 “A/B/C 预算预设（推荐工作流：A → B → C）”。

---

## 1) 你只需要一个“最稳流程”：Pool‑B → A → B → C（自动 shortlist）

你现在只需要用 `mlbot diagnose poolb-semantic-search` 这一条命令，它会固定执行：

- 生成 Pool‑B：factor-eval → `features_pool_b.yaml`
- Stage A：preset A（快筛）→ 自动导出 shortlist groups.yaml
- Stage B：preset B（收敛）→ 自动导出 shortlist groups.yaml
- Stage C：preset C（最终验收）→ 写回最终 features YAML（以 `_C.yaml` 结尾）

这让你**只有一个最佳工作流**，不用再手动拼命令/担心漏步骤。

---

## 2) 一条命令跑最稳流程（推荐、默认）

```bash
mlbot diagnose poolb-semantic-search \
  --strategies sr_reversal_rr_reg_long,sr_breakout,compression_breakout,trend_following \
  --tag <TAG> \
  --symbol BTCUSDT --timeframe 240T \
  --start-date 2023-01-01 --end-date 2025-12-31 \
  --search-algo pipeline \
  --expand-semantic-singletons \
  --regen-poolb --rerun-search
```

---

## 3) 算法说明：Pool‑B 怎么来？Semantic 怎么来？怎么合在一起搜？

这一节回答你关心的 4 个点：

- **Pool‑B 怎么来的**
- **Semantic groups 怎么来的**
- **是“展开找”还是“合在一起找”**
- **A/B/C 的参数差异**

### 3.1 Pool‑B（候选池）是怎么来的？

Pool‑B 由 `factor-eval` 生成，落盘成：

- `results/pools/<strategy>/pool_b/<TAG>/features_pool_b.yaml`

它本质是一次“因子评估 → 过滤/去相关 → 导出候选特征节点集合”的过程，主要特点：

- **输入**：策略自己的 factor 列表（来自 strategy config），按时间窗与 symbol 取数据
- **评估**：做 IC/稳定性等统计（并支持 forward-lag IC 衰减分析 `--ic-decay-lags 1,3,5,10,20`）
- **过滤**：
  - `--remove-correlated`：对高相关候选做去冗余（默认阈值 0.9）
  - `--filter-by-best-lag`：按“最佳 lag”对齐策略目标 lag（可由标签配置推断）
- **导出**：
  - `feature_pipeline.requested_features`: **特征“节点名”（feature compute function，*_f）**的列表
  - `feature_pipeline.invert_features`: **强负向因子**会进这里（供后续 `invert_candidates` 使用）

> 关键点：Pool‑B 导出的是 **feature 节点**，不是单列。一个节点可能对应多个输出列；导出阶段会把“合格的列”映射回它的源节点并做合并，这是预期行为。

### 3.2 Semantic groups 是怎么来的？

Semantic groups 是“人定义的候选分组空间”，feature-group-search 会按如下优先级自动加载：

1) `--groups-yaml / --groups-json`（显式传入）  
2) `config/feature_groups_<strategy>_semantic.yaml`（策略专属语义组）  
3) `config/feature_groups.yaml`（全局组）  
4) 代码 fallback `_default_groups()`

每个 group 的 value 是 **若干特征节点名（*_f）**，表示“这个语义主题/路径故事”对应的一组 features。

### 3.3 `--expand-semantic-singletons`：是怎么“展开找”的？

当开启 `--expand-semantic-singletons` 时，feature-group-search 会先把 semantic group 里的“多输出节点”展开成**按输出列的 singleton group**（一列一个 group），用于更细粒度地挑选“同一语义块里的某个子 score”。

实现要点（高层含义）：

- semantic 原本是：group → [`scene_semantic_scores_f`]  
- 展开后变成：  
  - `scene__compression` → [`scene_compression_score`]  
  - `scene__ignition` → [`scene_ignition_score`]  
  - ...

> 注意：这不会丢掉依赖。训练时实际会通过依赖解析把输出列映射回源节点来计算，只是“选择粒度”变成了输出列。

### 3.4 Pool‑B 与 Semantic：是合在一起搜，还是分开搜？

在 `feature-group-search` 里，它们是 **合在一起成为同一个 candidate groups 空间**来搜索的：

- 先加载（并可展开）semantic groups
- 再把 Pool‑B 里导出的每个节点补成一个 singleton group：`poolb__<feature_node>`  
  - 只对“还没出现在 groups 里”的 Pool‑B 节点补 singleton，避免重复

所以最终 candidate groups 类似：

- `semantic__compression` / `trade_cluster_scene__ignition` / ...（语义组或语义 singleton）
- `poolb__trend_r2_20_f` / `poolb__rsi_f` / ...（Pool‑B singleton）

然后 `search-algo`（默认 pipeline）在这个“合并后的 groups 空间”里做选择。

### 3.5 A/B/C 的参数差异（以代码为准）

Preset 的定义在：`src/time_series_model/diagnostics/feature_group_search.py::_apply_preset()`。

| preset | objective | seeds | halving_stages | top_fraction | min_survivors | pipeline_survivors | beam_width | max_steps | sffs_backward |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| A | `CV_mean` | `1,2` | `1,2` | 0.35 | 20 | 25 | 3 | 4 | 1 |
| B | `CV_mean` | `1,2,3` | `1,2,3` | 0.5 | 30 | 40 | 4 | 5 | 1 |
| C | `Sharpe_mean` | `1,2,3,4,5` | `1,3,5` | 0.6 | 40 | 60 | 5 | 6 | 2 |

- **A**：proxy + 少 seeds → 快速压缩候选空间  
- **B**：更稳一些，仍用 proxy → 收敛到更可信的 shortlist  
- **C**：最终目标 + 全 seeds → 最终验收并写回 `_C.yaml`

---

## 4) 产物清单（跑完你会得到什么）

对每个策略，会生成：

- **Pool‑B**：`results/pools/<strategy>/pool_b/<TAG>/features_pool_b.yaml`
- **Stage A/B/C 结果目录**：
  - `results/feature_group_search/<strategy>_<algo>_poolb_semantic_<TAG>_A/`
  - `results/feature_group_search/<strategy>_<algo>_poolb_semantic_<TAG>_B/`
  - `results/feature_group_search/<strategy>_<algo>_poolb_semantic_<TAG>_C/`
- **最终写回 YAML（以 C 为准）**：
  - `config/strategies/<strategy>/features_suggested_<algo>_poolb_semantic_<TAG>_C.yaml`
- **报告**：
  - `docs/architecture/reports/feature_group_search_summary_<TAG>_poolb_semantic.md`

（补充）报告会把每个策略的 Stage A/B/C 都列出来，但最终以 **Stage C** 的 writeback YAML 作为“最终特征组合”。


