## nnmultihead 常用命令速查（CLI）

这份文档的目标：把 **nn 多头（Path Primitives）** 相关命令集中到一个地方，方便日常训练/评估/产物定位。

### TL;DR：当前推荐“主链路”

**主链路（推荐，先把 Rule + Gate + Execution 做硬）**：

```text
前置：如果还没有可用的 model.pt，先跑 mlbot nnmultihead train 训练出模型

（推荐一键）mlbot nnmultihead pipeline-3action-e2e

（等价展开）nnmultihead predict → rule mode-3action → nnmultihead build-logs-3action → nnmultihead run-e2e-3action
（可选插入）nnmultihead shadow-eval-3action（单独跑 shadow 报告）

说明（分层归属）：
- **v0 主链路评估**：`nnmultihead build-logs-3action`、`nnmultihead run-e2e-3action`  
  用于系统级 counterfactual/shadow 评估，不属于 BC/RL 训练。
- **BC/RL 独有**：`mlbot rl ...` 下的影子研究命令（可选研究，v1 影子）。
```

> 说明：BC/RL/FSM 目前都当作 **将来可选模块**。现在主力是 **Rule Router +（将来的 Gate）+ Execution 假设版本化**，通过 counterfactual 报告做复盘与迭代。

### 约定：核心配置目录

默认配置示例：
- `config/nnmultihead/path_primitives_4h_80h_min/`

---

## 0) 主链路（Rule + Gate + Execution）：命令清单（建议照抄）

下面用 `<SYMS>` / `<CFG>` / `<MODEL>` / `<LAYER>` / `<OOS_START>` 这种占位符，避免你每次手改漏参数。

### 0.1 nnmultihead 推理：产出 primitives（preds）

**它干啥？**
- **输入**：FeatureStore 里的特征（月分区），+ 训练好的 `model.pt`
- **输出**：每个 symbol 一个 `preds_<SYMBOL>.parquet`，包含 4 个 head：
  - `pred_dir_prob`：方向概率
  - `pred_mfe_atr`：未来最大有利波动（ATR 标准化，通常是 log1p 空间）
  - `pred_mae_atr`：未来最大不利波动（ATR 标准化，通常是 log1p 空间）
  - `pred_t_to_mfe`：到达 MFE 的时间（bars，通常是 log1p 空间）
- **作用**：这是主链路的“模型输出层”，后面的 Router/Gate/Execution 都只吃这些 preds（或从中派生）。

**关键参数怎么选？**
- **`--config <CFG>`**：决定“特征定义/特征 contract/训练时的列选择逻辑”。必须和训练时一致（否则列对不上）。
- **`--feature-store-layer <LAYER>`**：必须和训练/特征生成对应的 layer 一致（否则读不到列或尺度不一致）。
- **`--start-date/--end-date`**：建议按 **OOS window** 跑（例如 2025-05~2025-10）。

**常见坑**
- **FeatureStore 月份缺失**：会导致某些 symbol 输出为空或报错（先跑 coverage audit）。
- **config 不一致**：会导致特征列不匹配（预测失败或隐性错位）。

```bash
mlbot nnmultihead predict --no-docker \
  --config <CFG> \
  --symbols <SYMS> \
  --timeframe 240T \
  --start-date <OOS_START> --end-date <OOS_END> \
  --model <MODEL> \
  --output <PREDS_DIR> \
  --feature-store-root feature_store \
  --feature-store-layer <LAYER>
```

产物（示例）：
- `<PREDS_DIR>/preds_BTCUSDT.parquet`（含 `pred_dir_prob/pred_mfe_atr/pred_mae_atr/pred_t_to_mfe`）

### 0.2 Rule Router：把 preds → mode_3action.parquet

**它干啥？**
- **输入**：`preds_*.parquet` + 一组 Router 阈值（可解释旋钮）
- **输出**：`mode_3action.parquet`，每行给出当前 bar 的 Router 决策：
  - `NO_TRADE` / `MEAN` / `TREND`
- **作用**：这是你当前路线的“参与决策层”（是否交易、交易用哪种模式），也是最重要的可控旋钮集合。

**你调的阈值到底在控制什么？**
- `mfe_min`：要求“潜在利润空间”足够大才参与
- `eff_min`：要求“性价比”足够好（常见定义是 `mfe/(mae+eps)` 的下限）
- `dir_conf_trend_min`：趋势模式需要更高方向置信

**常见坑**
- 只传 3 个阈值但你实际上有 7 个 tuned 阈值：会导致口径不一致（tuner vs e2e 对不上）。

#### 0.2.1 基础版（只写 3 个“最常用旋钮”）
> 适用：你还没做 tuning，只想先跑通主链路 / 先粗略验证。

```bash
mlbot rule mode-3action --no-docker \
  --preds <PREDS_DIR> \
  --model <MODEL> \
  --output <MODE_PARQUET> \
  --mfe-min <MFE_MIN> \
  --eff-min <EFF_MIN> \
  --dir-conf-trend-min <DIR_CONF_MIN>
```

#### 0.2.2 完整 tuned 版（推荐：用于复现 tuner / 对齐口径）
> 适用：你已经在 tuner 里找到 best 阈值，想让 `mode_3action.parquet` 与 tuner 完全一致。
> 说明：除了 3 个全局阈值，还要把 TREND/MEAN 分支的阈值也传进去，否则会落回默认值。

```bash
mlbot rule mode-3action --no-docker \
  --preds <PREDS_DIR> \
  --model <MODEL> \
  --output <MODE_PARQUET> \
  --mfe-min <MFE_MIN> \
  --eff-min <EFF_MIN> \
  --dir-conf-trend-min <DIR_CONF_MIN> \
  --mfe-trend-min <MFE_TREND_MIN> \
  --ttm-trend-min <TTM_TREND_MIN> \
  --eff-mean-min <EFF_MEAN_MIN> \
  --ttm-mean-max <TTM_MEAN_MAX>
```

#### 0.2.3 默认 tuned‑threshold 流程（带启发式约束）
> 目的：让阈值搜索“尊重真实分布”，避免 TREND 被阈值卡死。

```bash
mlbot diagnose threshold-plateau --no-docker \
  --preds <PREDS_DIR> \
  --logs <LOGS_PARQUET> \
  --model <MODEL> \
  --baseline-json <BASELINE_JSON> \
  --out <OUT_DIR> \
  --n-candidates 300 \
  --n-windows 6 --min-days-per-window 25 \
  --n-bootstrap 30 \
  --trend-rate-min 0.005 --trend-rate-penalty 2.0 \
  --heuristic-bounds --heuristic-qmin 0.05 --heuristic-qmax 0.95
```

产物：
- `<OUT_DIR>/router_thresholds_best.json`（用于 0.2.2 的完整 tuned 版）
- `<OUT_DIR>/summary.json` / `report.html`（复盘口径）

协议解释见：`docs/guides/THRESHOLD_PLATEAU_TUNING_PROTOCOL_CN.md`

**举例（EXP_006 Top9 best 阈值）**：
- `--mfe-min 0.1259208384`
- `--eff-min 1.0434303531`
- `--dir-conf-trend-min 0.0641852109`
- `--mfe-trend-min 0.4572347656`
- `--ttm-trend-min 5.6310524663`
- `--eff-mean-min 1.2011240134`
- `--ttm-mean-max 29.0044411469`


### 0.3（可选，但建议）Gate：树模型规则（将来模块，不影响主链路）

当前 repo 的 Gate(tree rules) 还在规划/待实现阶段（见 `EXP_007` 与 Gate TODO）。你现在可以把 Gate 理解为：
- 输入：features + path primitives + router 输出
- 输出：`allow/deny + reason`
- 位置：介于 Router 与 Execution 之间（只做过滤，不改仓位）

> 现在先把 Router 阈值 + Execution 假设版本化跑通；Gate 可以后续补进主链路。[树模型在多头模型下游的角色.md](docs/architecture/树模型在多头模型下游的角色.md)

### 0.4 build-logs：把 (preds + mode + raw) 组装成 logs（并定义 execution 假设）

**它干啥？**
- **输入**：
  - `preds_*.parquet`（NN 输出）
  - `mode_3action.parquet`（Rule Router 的 action）
  - 原始 OHLCV（用于对齐 timestamp，以及计算 counterfactual returns）
- **输出**：`logs_3action.parquet`（统一日志表），关键列：
  - `symbol,timestamp,mode`
  - `head_dir_score, head_mfe_atr, head_mae_atr, head_t_to_mfe`（用于 RL/BC 的 state，也用于诊断）
  - `ret_mean, ret_trend`（**counterfactual 执行回报**：如果这一 bar 采用 MEAN/TREND 执行，下一步的理论收益）
- **作用**：把“模型输出 + Router 决策 + 执行假设”统一落地成可复盘数据，后面所有评估只依赖 logs。

**关键参数怎么选？**
- **`--returns-source`**：这就是你说的 execution 假设入口：
  - `rr_execution`：用你当前的 rr_execution 模拟（更贴近“用 primitives 做执行”的路线）
  - `momentum_proxy`：更像“execution 解耦”的对照口径（隔离 execution 影响，用来评估 Router 切分本身）
  - 详细说明见：[`docs/guides/NNMULTIHEAD_RETURNS_SOURCE_CN.md`](docs/guides/NNMULTIHEAD_RETURNS_SOURCE_CN.md)

**常见坑**
- timestamp 不对齐会导致 `n_rows=0`（我们之前已经修过）；如果再出现，优先检查 preds/mode/raw 的时间范围与 timezone。

```bash
mlbot nnmultihead build-logs-3action --no-docker \
  --preds <PREDS_DIR> \
  --mode <MODE_PARQUET> \
  --model <MODEL> \
  --symbols <SYMS> \
  --timeframe 240T \
  --start-date <OOS_START> --end-date <OOS_END> \
  --data-path data/parquet_data \
  --returns-source rr_execution \
  --output <LOGS_PARQUET>
```

说明：
- **`returns-source`** 就是你“Execution 假设”的核心开关（后续要版本化/做稳健性扫描）。

### 0.5 e2e：产出 counterfactual（PnL）与 Router-aligned diagnostics（更适合读）

**它干啥？**
- **输入**：`logs_3action.parquet`
- **输出**：`<E2E_DIR>/` 下的一组报告（主看 counterfactual）：
  - `counterfactual/report.html`：PnL 结论导向报告（Rule vs BC Router 的对照）
  - `counterfactual/metrics.json`：核心数值
  - `counterfactual/router_diag_*`：Router-aligned 诊断（trade slice / AUC/AP / rolling drift）
- **作用**：
  - 把“当前 Rule Router + 当前 execution 假设”在 OOS 上的表现做成统一口径的报告
  - 把你关心的“按 Router trade slice 看 head 是否有用”也同步进同一份报告（不再靠读一堆 raw metrics）

**关键参数怎么选？**
- `entry_delay/cost/slippage`：用于稳健性扫描（建议后续做网格；现在先从 0 假设跑通）
- `--preds-in-log1p`：告诉报告如何还原 head 的真实量纲（用于 Router-aligned 诊断）
- `--router-*-min`：把你本次阈值写进报告，保证可复盘与口径一致

**怎么读报告？**
- 第一屏只看：**Rule Sharpe / MaxDD / Trade rate** 是否在可接受区间
- 然后看：Router-aligned 的 **AUC/AP 是否明显 > 0.5**（否则就是“看起来赚钱但 head 没排序力”，容易不稳）
- 再看：per-symbol 是否被单一币驱动（集中风险）

```bash
mlbot nnmultihead run-e2e-3action --no-docker \
  --logs <LOGS_PARQUET> \
  --out <E2E_DIR> \
  --entry-delay 0 \
  --cost-per-turnover 0 \
  --slippage-bps 0 \
  --preds-in-log1p \
  --router-mfe-min <MFE_MIN> \
  --router-eff-min <EFF_MIN> \
  --router-dir-conf-trend-min <DIR_CONF_MIN>
```

你主要看：
- `<E2E_DIR>/counterfactual/report.html`（我们已经做成“结论导向”的版本）

**这份 report 在回答什么？（非常重要）**
- **回答**：在你选定的 execution 假设（`returns_source` + `entry_delay/cost/slippage`）下，**Rule Router（+未来 Gate）作为系统能不能赚钱/是否稳健**（Sharpe/DD/Trade rate/单币贡献）。
- **不回答**：head 本身“预测是否好”（那是 `nnmultihead train/eval` 的 report）。
- **提示**：如果你用 `momentum_proxy` 做 build-logs，这份 report 更偏“execution 解耦对照”；如果用 `rr_execution`，这份 report 更贴近“落地执行假设”，必须做成本/延迟稳健性扫描。

>这里的“execution 解耦对照”意思是：当你用 --returns-source momentum_proxy 时，logs_3action.parquet 里的 ret_mean/ret_trend 不是用真实的执行规则算出来的，而是用一个极简、与 execution 细节无关的公式“伪造”出来的回报，用来当 baseline。

#### 它在对照什么？
你在 counterfactual/report.html 里看到的 Sharpe/DD，本质是：
Router 给每根 bar 一个 action（MEAN/TREND/NO_TRADE）
系统用 ret_mean/ret_trend 去“结算”每根 bar 的收益
当 momentum_proxy 时：
ret_trend = sign(过去动量) * 下一根收益
ret_mean = -sign(过去动量) * 下一根收益
也就是说：你评估的是“Router 的 action 切分在一个非常简化的回报世界里是否有用”，而不是“在你真实的 RR/ATR 执行规则里是否赚钱”。
#### 为什么叫“解耦”？
因为它把下面这些 execution 自由度都拿掉了（或几乎不体现）：
止损/止盈、持仓时间、滑点、手续费、入场延迟
用 head 去调整 TP/SL/Time stop 的逻辑
所以它更像是在问：
Router 在“只看方向/反向”这种粗糙执行假设下有没有 edge？
如果这里都没 edge，说明 Router 切分/阈值/样本切片本身可能就有问题
#### 你应该怎么用它？
用它做 sanity check / baseline：先验证 Router 切分是否“站得住”
真正决定落地时，还是要用 rr_execution（再做 cost/slippage/entry_delay 敏感性）
一句话：
momentum_proxy 评估的是 Router 的“切分能力”（弱执行假设下）
rr_execution 评估的是 Router+执行规则的“落地赚钱能力”（强依赖执行假设）
---

### 1) 训练：`mlbot nnmultihead train`

训练并自动生成训练报告（`report.html` + `metrics_summary.md`）。

**train 的 report vs e2e(counterfactual) 的 report 区别**
- **train report（模型层）**：评估 **head 是否有信息量/是否稳定**（AUC/Rank-IC/rolling/阈值一致评估）。不直接等价于赚钱能力。
- **e2e counterfactual report（系统层）**：评估 **Router + execution 假设** 的 PnL（Sharpe/DD）。强依赖 `returns_source` 与成本/延迟假设。

**另外：train report 和 `momentum_proxy` 不等价**
- `momentum_proxy` 是 build-logs 时生成 `ret_mean/ret_trend` 的一种 **执行回报构造口径**（系统层对照）。
- train report 是在数据上现算 labels、评估 head 预测质量的 **模型层指标**。

```bash
mlbot nnmultihead train --no-docker \
  --config config/nnmultihead/path_primitives_4h_80h_min \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 240T \
  --start-date 2023-01-01 \
  --end-date 2024-12-31 \
  --features-store-root feature_store \
  --features-store-layer features_83f12ecc5e \
  --output-dir results/nnmultihead/my_run_name
```

产物（典型）：
- `results/nnmultihead/<run>/model.pt`
- `results/nnmultihead/<run>/meta.json`
- `results/nnmultihead/<run>/metrics.json`
- `results/nnmultihead/<run>/report.html`
- `results/nnmultihead/<run>/metrics_summary.md`

### 2) 预测：`mlbot nnmultihead predict`

对指定数据段产出 primitives 预测（parquet/csv 等）。

```bash
mlbot nnmultihead predict --no-docker \
  --config config/nnmultihead/path_primitives_4h_80h_min \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 240T \
  --start-date 2025-07-01 \
  --end-date 2025-12-31 \
  --features-store-root feature_store \
  --features-store-layer features_83f12ecc5e \
  --model results/nnmultihead/my_run_name/model.pt \
  --output results/nnmultihead/my_run_name/preds_2025H2
```

### 2.5) Router 可视化：`mlbot rule plot-router-modes-kline`

基于 `mode_3action` 画 K 线 + Router 模式点（支持 gate filter）。

```bash
mlbot rule plot-router-modes-kline --no-docker \
  --mode results/nnmultihead/my_run_name/mode_3action_2024.parquet \
  --feature-store-root feature_store \
  --feature-store-layer features_83f12ecc5e \
  --all-symbols \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --out results/nnmultihead/my_run_name/router_plots
```

### 3) 评估：`mlbot nnmultihead eval`

对某段数据进行评估并生成报告（用于 OOS 复核、回归测试）。
用途：对某个时间窗（尤其 OOS）单独做 head 评估（A 层），生成 report.html/metrics.json
什么时候跑：你要回答“模型 head 有没有信息、是否漂移、阈值一致评估是否成立”时跑
不依赖主链路：不需要 rule/rl 任何产物

```bash
mlbot nnmultihead eval --no-docker \
  --config config/nnmultihead/path_primitives_4h_80h_min \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 240T \
  --start-date 2025-07-01 \
  --end-date 2025-12-31 \
  --data-path data/parquet_data \
  --model results/nnmultihead/my_run_name/model.pt \
  --output-dir results/nnmultihead/my_run_name/eval_2025H2
```

### 4) 重新渲染报告：`mlbot nnmultihead render-report`

适用于：你更新了报告模板/summary 逻辑，但不想重训。
用途：你改了报告模板/summary 逻辑后，对已有 run 重渲染 HTML/summary
什么时候跑：你更新了报告样式（比如我们刚做的结论化）但不想重训/重评估

主链路 vs 辅助链路（一句话）
主链路：为了产出 “Router + execution 假设 的 PnL 报告”
predict → mode-3action → build-logs → run-e2e
辅助链路：为了产出 “模型本身/报告渲染 的复盘材料”
eval（重新评估） / render-report（只重画报告）

```bash
mlbot nnmultihead render-report --no-docker \
  --run-dir results/nnmultihead/my_run_name
```

### 5)（可选前置）primitives 因子筛选（Pool B）：`mlbot nnmultihead factor-eval`

对候选特征做 “primitives 目标” 的单因子稳定性评估，并导出 Pool B YAML（后续用于快速收敛特征集）。

**它和 train 的先后关系**
- **通常在 train 之前**：当你还在“选特征/收敛特征集”阶段，先用 `factor-eval` 找出更稳定的候选特征（Pool B），再把它固化到 `features.yaml` 里去训练。
- **不是必跑**：如果你已经有一套确定要用的 `features.yaml`（例如已经固定为 PoolB/selected 版本），可以直接 `mlbot nnmultihead train`。

```bash
mlbot nnmultihead factor-eval --no-docker \
  --config-dir config/nnmultihead/path_primitives_4h_80h_min \
  --candidates-yaml config/strategies/sr_reversal_rr_reg_long/features_all.yaml \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 240T \
  --features-store-root feature_store \
  --features-store-layer features_83f12ecc5e \
  --start-date 2023-01-01 \
  --end-date 2024-12-31 \
  --min-samples-per-group 120
```

默认输出目录（约定）：
- `results/pools/path_primitives_4h_80h_min/pool_b_primitives/`
  - `primitives_factor_eval_metrics.csv`
  - `primitives_factor_eval_summary.json`
  - `features_pool_b_primitives.yaml`

下一步（推荐）：参考 `docs/strategies/NNMULTIHEAD_FEATURE_SEARCH_PLAYBOOK_CN.md` 把 Pool B 变成可迭代的 `features.yaml`（required/optional_blocks）并做 multi-run 对比。

---

## 6)（推荐）NN 特征集搜索：A/B/C（三阶段）`mlbot nnmultihead feature-group-search --run-abc`

用途：在候选特征组里，用 **A（快筛）→B（收敛）→C（验收）** 的实验编排方式，找到对 primitives 目标最有效的一套特征组合，并固化中间 shortlist 与结论，便于复跑/复盘。

你只需要记住一句话：
- **`--search-algo pipeline` 是“一次搜索的底层算法（halving→beam→prune）”**
- **A/B/C 是“三次搜索的预算编排（3 套参数）+ shortlist 固化”**  
  A/B/C 每一阶段通常都会运行 pipeline，只是预算不同（训练 epochs、halving/beam 的宽度、max_steps 等）。

### 6.1 一键推荐命令（只用这一种方式即可）

```bash
mlbot nnmultihead feature-group-search --no-docker \
  --run-abc \
  --base-config config/nnmultihead/path_primitives_4h_80h_min \
  --pool-b-yaml results/pools/tree_union_all/features_pool_b_all_feature_nodes.yaml \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT \
  --timeframe 240T \
  --start-date 2023-01-01 --end-date 2024-12-31 \
  --features-store-root feature_store \
  --features-store-layer nnmh_tree_union_all_240T_v1 \
  --objective dir_auc \
  --output-dir results/nn_feature_group_search/abc_tree_union_all_highcap6_2023_2024
```

说明：
- `--pool-b-yaml`：候选集合（Pool B）。这里建议用 *feature nodes*（`xxx_f`），也支持输出列名（细粒度）。
- `--run-abc`：会自动生成 `A/ B/ C/` 三个目录，阶段间会写出 shortlist groups yaml，并在根目录生成 `summary.md`。
- `--objective`：要最大化的 nn 训练指标（来自 metrics.json，比如 `dir_auc` / `roll_icir__dir` 等）。

### 6.2 产物（典型）

在 `--output-dir` 下你会看到：
- `A/nn_feature_group_search_result.json` + `A/groups_shortlist_A.yaml`
- `B/nn_feature_group_search_result.json` + `B/groups_shortlist_B.yaml`
- `C/nn_feature_group_search_result.json`
- `summary.md`（每阶段 survivors / selected_groups 的数量摘要）

### 6.3 如果你还想手动跑一次（不推荐，主要用于 debug）

你仍然可以单次运行（一次预算、一次结果）：
- `--search-algo pipeline`：只跑一次 pipeline（halving→beam→prune），不做 A/B/C 编排与 shortlist 固化。

---

## 可选模块（将来可用，不影响当前主路线）

### A) BC shadow（行为一致性门禁）

用途：验证“一个小模型能否稳定复现 rule 的 mode 行为分布/切换频率”，**不是 PnL 优化**。

```bash
mlbot nnmultihead shadow-eval-3action --no-docker \
  --logs <LOGS_PARQUET> \
  --out <OUT_DIR>
```

看：
- `<OUT_DIR>/shadow_report.html`

### B) RL / FSM（策略探索 + 上线控制）

用途：研究/探索 policy；FSM 是把 RL 当候选时的上线门禁（hard_sharpe/hard_sortino/drift 等）。
当前路线里你可以先忽略它。
> 说明：真正的 BC/RL “训练”命令目前不是主链路必备，
> 如果需要训练 BC/Offline RL，统一以 `logs_3action.parquet` 为输入，再单独训练。



