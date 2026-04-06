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

---

## 5) B/C 输出能不能直接用？（以及为什么还要“验收/门禁”）

### 5.1 可以这么理解：B/C 的 writeback YAML = “可直接喂给模型的候选特征配置”

当你看到这些文件存在时：

- `config/strategies/<strategy>/features_suggested_<algo>_poolb_semantic_<TAG>_B.yaml`
- `config/strategies/<strategy>/features_suggested_<algo>_poolb_semantic_<TAG>_C.yaml`

你就可以把它当作**一个可执行的特征配置**：其中 `feature_pipeline.requested_features` 与 `feature_pipeline.invert_features` 是“要传给模型”的输入列（节点级/列级取决于是否启用了 singleton）。

> 实操上：B 更适合日常迭代（快），C 更适合准备合并/上线前的最终候选（更稳、更贴近最终目标）。

### 5.2 但要注意：可用 ≠ 可上线（feature-group-search 不做所有“上线门禁”）

feature-group-search 在过程里已经做了 CV 与回测（按当前时间切分 + 多 seeds），并把结果落盘；
但它默认**不承担**以下“上线门禁”（因为成本高、口径因团队而异）：

- **Holdout / OOS 验收**：固定 6 个月（或你指定窗口）只验收不调参
- **Rolling 稳定性**：walk-forward/多窗口一致性（验证长期有效性与不跳跃）
- **多标的泛化**：不仅 BTC 单标的（尤其是你要做 multi-asset）
- **事件驱动一致性**：用 Nautilus 做“回测=实盘假设一致性”复核（撮合/滑点/延迟等）
- **特征成本与可得性**：tick 重特征、频域/DTW 等是否能实时稳定产出

因此推荐的工业化节奏是：

- **研发迭代**：A → B，直接拿 B 的 YAML 继续训练/对比
- **合并/上线前**：对 B 的 top 3–5（或直接 C）做更严格验收，通过后再固化为上线版本

### 5.3 后续命令（现成可用）：把 B/C 作为候选，做“验收/门禁”

#### (1) Ablation：baseline vs B vs C（可选 rolling）

用 `mlbot analyze strategy-feature-compare` 一次对比多个 feature 配置（并支持 rolling）：

```bash
mlbot analyze strategy-feature-compare \
  --strategy-config config/strategies/<strategy> \
  --symbol BTCUSDT --timeframe 240T \
  --start-date 2023-01-01 --end-date 2025-12-31 \
  --test-size 0.30 \
  --feature-overrides \
    base=config/strategies/<strategy>/features.yaml \
    B=config/strategies/<strategy>/features_suggested_<algo>_poolb_semantic_<TAG>_B.yaml \
    C=config/strategies/<strategy>/features_suggested_<algo>_poolb_semantic_<TAG>_C.yaml \
  --run-rolling \
  --rolling-train-bars 1000 --rolling-test-bars 200 --rolling-step-bars 100 --rolling-max-windows 10 \
  --no-docker
```

> 说明：`features.yaml` 是当前“基线/主配置”；B/C 是候选。你也可以把 base 换成 `features_base.yaml` 或你自己的对照版本。

#### (2) 事件驱动复核：Nautilus（更贴近执行假设）

当你已经训练出最终模型（或有 model artifact）后，用 Nautilus 做事件驱动复核：

```bash
mlbot backtest strategy \
  --strategy <strategy> \
  --symbol BTCUSDT --timeframe 240T \
  --start-date 2024-01-01 --end-date 2025-10-31 \
  --mode event-driven \
  --output-dir results/backtest/nautilus_check \
  --no-docker
```

> 说明：Nautilus 是“执行一致性/事件驱动撮合”的门禁；它不是用来在大候选空间里做搜索的。

---

## 6) 性能加速（搜索阶段推荐）：Fast Mode + 频谱拆分 + 月度并行

> 你如果感觉“orderflow/数学特征太慢，流程跑不通”，建议先看：
> - `docs/architecture/guides/FEATURE_COMPLEXITY_LAYERS_CN.md`（按计算复杂性分层：先易后难，逐层解锁）

### 6.1 月度并行（opt-in）：并行计算“月度缓存 miss”的月份

特征计算器在“特征级别”仍然是顺序执行（避免依赖图/缓存一致性问题），但现在支持 **按月并行**：

- 只对 **monthly cache miss** 的月份并行计算（cache hit 仍然直接读）
- 并行粒度是“单月切片”，不会把整段大 DataFrame 在进程间来回拷贝（更稳定）

启用方式（环境变量）：

```bash
export FEATURE_MONTHLY_WORKERS=4          # >1 开启月度并行
export FEATURE_MONTHLY_BACKEND=process    # process 或 thread
```

> 常见疑问：会不会导致“状态不连续/无法把上月状态传到下个月”？  
> 不会额外变差：月度缓存本身就是按“单月切片”计算，边界处的滚动特征会自然出现 warmup 缺失（一般表现为月初 NaN/更弱信号）。  
> 如果你需要严格连续的滚动状态（跨月 warmup），需要进一步做“跨月 overlap warmup”切片（这属于正确性增强，不是并行的副作用）。

**补充确认（你关心的跨月 state）：**

- **VPIN**：有跨月 bucket state（用于保证 bucket 续接），但这个 state 在 tick 计算函数内部维护，并且有“带 state 的月度缓存键”。按月并行不会破坏它。
- **Trade Cluster**：有跨月 state（如 run-length 窗口），同样在 tick 计算函数内部维护；当 state 为空时还会从前一个月缓存里加载 final_state 来 warm-start。按月并行不会破坏它。
- **Footprint**：通常是“每根 bar 当期 ticks 的统计聚合”，没有类似 VPIN/trade_cluster 的跨月 state 需要继承，月度分块主要是为缓存与 I/O。

### 6.2 Fast Mode（用于搜索阶段 A/B）：关 DTW random templates + 降频 spectrum 计算

Fast Mode 会在搜索阶段自动启用（A/B），在最终验收阶段默认关闭（C）。

- **DTW**：禁用 random templates（随机模板特征），减少 DTW 的模板数 → 直接降耗  
  - random templates 是为了“对比学习/负样本对照”加入的随机形态模板（例如 `random_15/random_20/...`）
- **Spectrum**：对频谱特征做降频计算（例如每 4 根 bar 才算一次并 forward-fill），把 O(n) 的滚动 Welch 计算成本显著降低

### 6.3 频谱拆分：把 `spectrum_features_f` 拆成三路（price / volume / cvd）

原来的 `spectrum_features_f` 会把 **price + volume + cvd** 三路一起算（对搜索很贵）。
现在支持拆分节点（你可以按需选其中一路）：

- `spectrum_price_features_f`：只依赖 `close`
- `spectrum_volume_features_f`：只依赖 `volume`
- `spectrum_cvd_features_f`：只依赖 `cvd`

### 6.4 这些增强哪些流程都能用上？

- **月度并行（FEATURE_MONTHLY_WORKERS）**：只要走的是同一个 `StrategyFeatureLoader -> FeatureComputer` 特征栈，就都能用上（树模型训练/回测、rolling 验证、以及“自动 materialize FeatureStore”）。  
  - 开关是环境变量，所以对 tree/nn 都是“全局生效”（谁用 FeatureComputer 谁受益）。
- **Fast Mode（FEATURE_FAST_MODE）**：目前只在 DTW / Spectrum 相关的特征函数里读这个环境变量。  
  - 树模型 `feature-group-search` 的 preset A/B 会自动开启；C 默认关闭。  
  - 其他流程（包括 nn 多头模型/feature store 构建）如果你也想快，可以在启动命令前手动 `export FEATURE_FAST_MODE=1`（只影响 DTW/Spectrum，不影响其他特征）。

### 6.5 Pool‑B 的“反向特征”还要不要再验证？（`--invert-eval`）

你会在 Pool‑B 的 `features_pool_b.yaml` 里看到 `feature_pipeline.invert_features`（一串需要取反的**输出列名**）。  
这些“反向”是基于 **单因子 IC/ICIR/t-stat** 的符号判断：从单列视角看，它更像负相关因子。

但 feature-group-search 优化的是 **多特征 + 模型 + CV/回测目标（CV_mean / Sharpe_mean）**，因此符号不一定完全一致：

- **交互/共线**：某个列在组合里与其它列交互后，“有效符号”可能变化
- **目标函数不同**：ICIR 与 CV_mean/Sharpe_mean 的最优符号可能不一致
- **噪声/不稳定**：取反与否可能没差，你可能更希望“最终写回 YAML 里别塞一长串未复核的 invert 列”

因此我们支持对 Pool‑B 反向列做“raw vs inverted”的显式对照验证（只针对这些输出列）：

- `--invert-eval none`：不做对照，完全信任 Pool‑B 的符号（最快）
- `--invert-eval conservative`：只在“raw 明显更差（偏负）且 inverted 明显提升”时才选取反（更保守）
- `--invert-eval all`：对候选反向列一律做 raw vs inverted 对照，**选更好的符号**（最严格、也更慢）

推荐策略（兼顾速度与可信度）：

- **A/B（筛选阶段）**：`none` 或 `conservative`（快跑、先筛方向）
- **C（最终验收阶段）**：`all`（用真实目标函数复核符号后再写回最终配置）

> 当前 `poolb-semantic-search` 工作流默认会对 Pool‑B 反向列启用更严格的验证（等价于 `--invert-eval all`）。如果你更偏向“先快后严”，可以在 C 阶段再开启 all。
