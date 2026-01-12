# Pool-B 反向特征（`invert_features`）在 `feature-group-search` 中如何处理（已修复版）

本文件解释：当 Pool-B（factor-eval 产物）里包含 `invert_features` 时，`feature-group-search` 如何把它们当作“一等公民”处理，避免早期版本“没处理反向/反向裁剪错命名空间”的问题。

## 1. 背景：两种命名空间（这是所有 bug 的根源）

在本项目里，`features.yaml` 中有两个经常被混用但**必须严格区分**的字段：

- **`feature_pipeline.requested_features`**：特征**计算节点**（通常以 `*_f` 结尾）。例：`wick_ratios_f`、`trade_cluster_scene_semantic_scores_f`。
- **`feature_pipeline.invert_features`**：特征**输出列名（output column）**，表示在训练/推理前对该列乘以 `-1`。例：`macd_signal`、`trade_cluster_net_runs_zscore_20`。

这两者不在同一个命名空间：`requested_features` 是“算什么”，`invert_features` 是“算出来的某一列怎么变号”。因此：

- **不能**用 `requested_features` 去裁剪 `invert_features`（会把合法的 output column 当成“不在 requested_features 里”的垃圾误删）。
- **可以**在写回时保留“多余的 invert_features”（训练器只会对实际存在的列应用反向，不存在的列不会产生影响）。

相关代码位于：`src/time_series_model/diagnostics/feature_group_search.py` 的 `_writeback_features_yaml()`。

## 2. Pool-B 里 `invert_features` 是什么？从哪来？

Pool-B YAML 通常位于：`results/pools/<strategy>/pool_b/<tag>/features_pool_b.yaml`，结构形如：

```yaml
feature_pipeline:
  requested_features: [...]
  invert_features: [...]
```

其中：

- `requested_features`：候选特征（可能是节点名或列名，取决于产生者）。
- `invert_features`：候选“需要反向”的 output column 名单（来自 factor-eval 的方向性判断/负相关因子修正）。

## 3. `feature-group-search` 如何把 Pool-B 注入候选空间？

`feature-group-search` 会把 Pool-B 产物合并进 groups，使其成为可搜索的候选集合：

1. **Pool-B requested_features**：注入成 singleton group：`poolb__<f>`，内容为 `[f]`。
2. **Pool-B invert_features（output columns）**：注入成 singleton group：`poolb_invcol__<col>`，内容为 `[col]`。

这一步的意图是：把“反向后的列”当成搜索空间中的**可选项**，让搜索过程有机会验证“反向是否真的有利”。

相关代码位于：`src/time_series_model/diagnostics/feature_group_search.py` 解析参数并合并 Pool-B groups 的逻辑（`poolb_invcol__` 命名空间）。

## 4. 反向验证策略：`--invert-eval {none|conservative|all}`

搜索过程在 prefilter（successive halving）阶段支持对“可反向的列候选”做 A/B 对照：

- **none**：不做反向验证（保持 base `invert_features`，不会尝试把 Pool-B invert candidates 加进来）。
- **conservative（默认）**：仅当满足以下条件才选择反向版本：
  - raw 版本是 valid，且 raw 分数“明显为负”（阈值：`invert_min_negative_score=-0.05`）
  - inverted 版本 valid，且相对 raw 有“显著改进”（阈值：`invert_min_improvement=0.05`）
- **all**：只要 inverted 比 raw 更好（且两者 valid），就选择 inverted（更激进，适合离线研究/探索）。

实现逻辑位于：`src/time_series_model/diagnostics/feature_group_search.py` 的 `successive_halving_prefilter()`。

反向验证的工作方式是：

- 先评估 raw：`invert_features = base_inv`
- 再评估 inverted：`invert_features = base_inv + inv_cols_for_group`
- 根据策略（conservative/all）决定是否 pick inverted
- 若 pick 了 inverted，则会记录 `invert_by_group[group_name] = [col1, col2, ...]`

## 5. 写回（writeback）语义：最终到底写回什么？

`feature-group-search` 的 writeback 目标是生成一个可直接用于训练/回测的 `features_suggested*.yaml`（或覆盖写回到某个路径）。

关键规则：

- 写回时**只写最终 `invert_features` 列表**（去重、稳定顺序），**不写**“一大坨未验证候选列表”。  
- `invert_features` 不会被 `requested_features` 裁剪（因为命名空间不同）。  

这能保证：

- 复现时你拿到的是“可执行”的特征配置
- 反向逻辑不会因为 YAML 结构/命名空间差异而 silently 失效

## 6. 你在跑 `scripts/run_poolb_semantic_search.py` 时会发生什么？

该脚本在调用 tree 侧 `feature-group-search` 时，默认会：

- 传 `--pool-b-yaml <pool_b_yaml>`
- 传 `--invert-candidates-yaml <pool_b_yaml>`（同一个 YAML，工具会自动读取其中的 `feature_pipeline.invert_features`）
- 传 `--invert-eval all`（让反向验证更激进，直接挑更优版本）

相关代码位于：`scripts/run_poolb_semantic_search.py` 的 `run_feature_group_search()` 组装命令部分。

## 7. 推荐实践（避免“反向特征”再次成为自由度黑洞）

- **研究阶段**：可以用 `--invert-eval all` 做探索；但要保留 `invert_by_group` 记录，避免“反向到底发生没发生”不可追溯。
- **上线前**：建议切回 `--invert-eval conservative` 或者固定 `invert_features`，把反向列当成“已验证的硬规则”，避免过度炼丹。
- **评估口径**：所有反向决策必须出现在报告/产物里（例如 meta.json / summary.md）并可回放。

