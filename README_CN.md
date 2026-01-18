# ML Trading Bot（中文）
 **Alpha不是收集的，是雕刻的。**
 
本仓库包含因子研究、模型训练、回测与数据管道的生产就绪组件。**本 README 保持尽量短**：只提供“命令 + 推荐流程 + 入口文档链接”。研究解释型内容已迁移到独立文档。

---

## 快速开始

1) 创建虚拟环境（conda、venv 等）并激活  
2) 以可编辑模式安装：

```bash
pip install -e .[dev]
```

3) （可选但推荐）安装 Git pre-commit 钩子：

```bash
make install-hooks
```

4) 查看命令：

```bash
mlbot --help
mlbot analyze --help
mlbot train --help
mlbot diagnose --help
mlbot optimize --help
mlbot data --help
```

---

## 数据管道（下载 → 转换 → 训练）

```bash
mlbot data download \
  --symbols BTCUSDT,ETHUSDT \
  --start-year 2021 \
  --start-month 1

mlbot data convert

# 或一次性跑完
mlbot data pipeline \
  --symbols BTCUSDT,ETHUSDT
```

### Universe 驱动（多币种批量：下载 + 转 parquet）

`pipeline-universe` 会按 universe 配置解析 symbol 列表，并 **下载后立刻转成 Parquet**：

```bash
mlbot data pipeline-universe \
  --universe-config config/download/crypto_4h_token_universe_groups.yaml \
  --universe-set starter_a \
  --start-year 2023 \
  --start-month 1 \
  --no-docker
```

### 扩展数据：市值（Market Cap）与资金费率（Funding Rate）

#### 1) 市值快照（CoinGecko）

```bash
export COINGECKO_API_KEY='...'

mlbot data update-market-cap \
  --config config/data/market_cap.yaml \
  --max-age-days 7 \
  --no-docker
```

结果默认落盘：
- `data/market_cap/<SYMBOL>.parquet`
- `data/market_cap/market_cap_manifest.json`

#### 2) Binance 资金费率（按月 ZIP → Parquet）

```bash
mlbot data download-funding-rate \
  --symbols BTCUSDT,ETHUSDT \
  --start-year 2024 \
  --start-month 1 \
  --progress-every 10 \
  --no-docker
```

结果默认落盘：
- ZIP：`data/funding_rate/zip/`
- Parquet：`data/funding_rate/parquet/`

---

## 推荐工作流（MVP：最小闭环）

> README 只保留“可复制的最小命令”。详细解释与扩展流程见：
> - `docs/guides/DEPLOYMENT_MVP_WORKFLOW_CN.md`
> - `docs/guides/CROSS_SECTIONAL_PIPELINE_CN.md`（CS：截面因子评估→筛选→回测→训练）
> - `docs/guides/CROSS_SECTIONAL_WORKFLOW_END2END_CN.md`（CS：端到端一张图 + 回测审计与产物）
> - `docs/guides/CS_VS_TS_PIPELINE_CN.md`（CS vs TS：两套 pipeline 差异与指标取舍）
> - `docs/architecture/CROSS_SECTIONAL_ALPHA101_FEATURESTORE_ARCH_CN.md`（CS Alpha101：为何不走 DAG + 缓存复用架构）
> - `docs/guides/LIVE_TRADING_ROADMAP_MULTI_ASSET_CN.md`（多资产合约实盘落地路线图）
> - `docs/architecture/NN_MULTI_ASSET_CONSTITUTIONAL_SYSTEM_DESIGN_CN.md`（NN 多资产系统：Task/Router/Gate/Execution 宪法与运维落地设计）
> - `docs/architecture/ARCH_UPGRADE_TASKSPEC_CONSTITUTION_V1_CN.md`（架构升级 V1：TaskSpec + Constitution + PCM）
> - `docs/architecture/archetype灭绝级回测.md`（Archetype 灭绝级回测：压力测试→生存评分→Router/Size 映射）
> - `docs/architecture/ood头的训练.md`（OOD/Survival Head：监督信号定义、loss、评估曲线、熄火/复燃验证）
> - `docs/architecture/LiveDashboard.md`（LiveDashboard：只盯 5 个数（含增强版），用于阻止系统犯蠢）
> - `docs/guides/RD_TO_LIVE_TIERED_WORKFLOW_V1_CN.md`（研发→上线分层工作流：Tier×Universe×TaskSpec）
> - `docs/guides/POOLB_INVERT_FEATURES_CN.md`（Pool‑B 反向特征：invert_features 处理规则）
> - `docs/strategies/树策略导出的可泛化规则.md`（tree 策略 if/else：语义规则模板 + 扫描汇总（含 VPIN/订单流规则））
> - `docs/guides/THRESHOLD_PLATEAU_TUNING_PROTOCOL_CN.md`（阈值调参：找“平坦高原”而非尖峰，Router/SLTP 通用）
> - `docs/live_stream/README.md`（实盘事件流/回放/对账/稳定性：Live 边缘系统入口）
> - `docs/guides/NNMULTIHEAD_CONFIG_FILES_CN.md`（nnmultihead 配置文件职责图：TaskSpec/FeaturePlan/features.yaml/labels.yaml/model.yaml）

### 0) 质量闸门（推荐）

```bash
make test-key-features-all
mlbot diagnose feature-contract --no-docker
```

### 1) 特征搜索（Pool‑B + 语义组，一键输出建议 features）

```bash
mlbot diagnose poolb-semantic-search \
  --strategies <strategy_dir_name> \
  --tag <TAG> \
  --symbol BTCUSDT --timeframe 240T \
  --start-date 2024-01-01 --end-date 2025-04-30 \
  --search-algo pipeline \
  --expand-semantic-singletons \
  --regen-poolb --rerun-search \
```

#### A/B/C 预算预设（推荐工作流：A → B → C）

这个命令固定执行 **最稳** 的 staged 工作流（你不需要再手动串 A/B/C）：

- **Pool‑B**：先生成候选因子池（factor-eval → `features_pool_b.yaml`）
- **Stage A**：preset A（快筛）→ 自动导出 shortlist
- **Stage B**：preset B（收敛）→ 自动导出 shortlist
- **Stage C**：preset C（最终验收）→ 输出最终 `features_suggested_*_C.yaml`

你只需要记住这一条命令即可；详细说明见：

- `docs/guides/FEATURE_GROUP_SEARCH_PRESETS_CN.md`
  - 性能加速（Fast Mode / 频谱拆分 / 月度并行）：见该文档第 6 节
  - 特征按计算复杂性分层（先易后难、逐层解锁）：`docs/guides/FEATURE_COMPLEXITY_LAYERS_CN.md`

> 重要补充：Stage B / C 写回的 `features_suggested_*.yaml` **是可直接喂给模型训练/回测的候选特征配置**；  
> 但“可用 ≠ 可上线”，合并/上线前建议用 holdout/rolling/多标的/Nautilus 做验收门禁。  
> 具体命令与解释见 `docs/guides/FEATURE_GROUP_SEARCH_PRESETS_CN.md` 的 “B/C 输出能不能直接用？（以及为什么还要验收/门禁）”。

### 2) 最终验收：6 个月 Holdout（只验收，不再调参）

```bash
mlbot diagnose holdout-eval \
  --config config/strategies/<strategy> \
  --symbol BTCUSDT --timeframe 240T \
  --train-start-date 2024-01-01 \
  --holdout-start-date 2025-05-01 \
  --holdout-end-date 2025-10-31 \
  --output-root results/holdout_eval \
  --deterministic \
  --no-docker
```

### 3) 上线产物：训练最终模型（全窗训练）

```bash
mlbot train final \
  --config config/strategies/<strategy> \
  --symbol BTCUSDT --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31 \
  --output-root models \
  --deterministic \
  --no-docker
```

### 4) （可选）rolling / Nautilus

- rolling：用于**跨月稳定性验证**与**上线后监控**（见 `docs/guides/DEPLOYMENT_MVP_WORKFLOW_CN.md`）
  - 说明：`poolb-semantic-search` 不是 rolling；它是在**单个训练窗 + 单个测试窗**（time split + 多 seed）上做特征搜索/收敛。
- Nautilus：用于“回测=实盘一致性验证”（事件驱动回放）（见 `docs/live_stream/reference/Nautilus_Trader_集成指南.md`、`docs/live_stream/07_与NautilusTrader对齐清单.md`）

**Nautilus MetaRouter 事件回测（本地数据 / 多币）**

```bash
mlbot backtest nautilus \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 240T \
  --start-date 2020-01-01 --end-date 2020-01-02 \
  --trade-size 0.001 \
  --output-dir results/backtest_smoke_multi \
  --max-files 1 \
  --no-docker
```

**Nautilus 事件抽样回测（vectorbt 选点 → Nautilus 只跑窗口）**

先把 vectorbt trades 导出为 json/csv（包含 `Entry Timestamp` 或 `entry_time` 列），然后生成窗口：

```bash
# 第一步：从 vectorbt 回测产物导出 trades.json（不用手工拼）
mlbot diagnose export-vectorbt-trades \
  --artifacts-dir results/strategies/sr_reversal_long/BTCUSDT \
  --out results/backtest/vectorbt_trades.json \
  --no-docker
```

```bash
# 第二步：把 trades 转成“回放窗口”列表（前后扩展 pre/post，必要时合并重叠窗口）
mlbot diagnose backtest-time-windows \
  --trades results/backtest/vectorbt_trades.json \
  --out results/backtest/time_windows.json \
  --pre-minutes 480 \
  --post-minutes 480 \
  --default-symbol BTCUSDT \
  --merge-overlap \
  --no-docker
```

然后用 Nautilus 只回放这些窗口：

```bash
mlbot backtest nautilus \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 240T \
  --start-date 2020-01-01 --end-date 2020-01-02 \
  --trade-size 0.001 \
  --time-windows-json results/backtest/time_windows.json \
  --output-dir results/backtest_smoke_windows \
  --no-docker
```
> 说明：即使提供了 `--time-windows-json`，当前实现仍要求给出 `--start-date/--end-date` 用于框定全局回测区间；
> 真正的数据过滤由窗口列表完成。`trade-size` 目前仍是 CLI 参数（后续可以迁移到 live config / yaml）。

**Nautilus Adapter 数据流（testnet）**

先把 testnet key 放到本地文件（已默认忽略）：

```bash
source config/local/env.testnet
```

然后运行：

```bash
mlbot backtest nautilus \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 240T \
  --start-date 2020-01-01 --end-date 2020-01-02 \
  --trade-size 0.001 \
  --output-dir results/backtest_smoke_adapter \
  --max-files 1 \
  --use-adapter-data \
  --adapter-testnet \
  --adapter-account-type USDT_FUTURES \
  --env-file config/local/env.testnet \
  --no-docker
```
> 说明：`--max-files` 用于本地 parquet 的“抽样加载”（smoke 级别），只读前 N 个文件，加速验证链路。

**Nautilus Adapter 数据流（live）**

```bash
source config/local/env.live
```

```bash
mlbot backtest nautilus \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 240T \
  --start-date 2020-01-01 --end-date 2020-01-02 \
  --trade-size 0.001 \
  --output-dir results/backtest_smoke_adapter_live \
  --max-files 1 \
  --use-adapter-data \
  --adapter-account-type USDT_FUTURES \
  --env-file config/local/env.live \
  --no-docker
```

---

## TaskSpec 驱动的 Tier0/Tier1 对比（nnmultihead）

你问的“Tier0/Tier1 会如何影响训练？是不是跑两次看报告？”——**是的**，但需要做到两点才能可复盘：  
1) 每个 Tier 生成一个**具体可执行的 config 目录**（不直接靠“标签”）  
2) 用各自 config 训练出 model，再用统一流程评估（A-layer + system/e2e）  

### 1) 先从 TaskSpec 生成派生 config（让 tiers 变成真实 features.yaml）

```bash
mlbot nnmultihead materialize-config-from-task-spec --no-docker \
  --task-spec config/tasks/task_spec.yaml \
  --base-config config/nnmultihead/path_primitives_4h_80h_min \
  --out-config results/derived_cfg/tier01
```

> `task_spec.yaml` 里通过 `feature_plan.tiers_enabled` + `tier_feature_files` 显式定义 Tier0/Tier1 的 feature nodes 列表。  

### 2) 训练（TaskSpec-only：命令会自动 materialize 派生 config）

```bash
mlbot nnmultihead train --no-docker \
  --task-spec config/tasks/task_spec.yaml \
  --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT
```

### 3) 跑主链路评估（predict → router → build-logs → e2e）（TaskSpec-only）

```bash
mlbot nnmultihead pipeline-3action-e2e --no-docker \
  --task-spec config/tasks/task_spec.yaml \
  --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT \
  --timeframe 240T \
  --start-date 2025-05-01 --end-date 2025-12-31 \
  --model <PATH_TO_MODEL_PT_FROM_TRAIN> \
  --returns-source rr_execution \
  --out results/nnmh_e2e/tier01
```

### 3.1) Router 阈值：用“平坦高原”协议做稳健调参（推荐）

> 目的：避免“找尖峰”导致的炼丹，优先选多窗口/bootstrapped 都稳的阈值组合。  
> 详细解释见：`docs/guides/THRESHOLD_PLATEAU_TUNING_PROTOCOL_CN.md`
>
> 说明：`mlbot nnmultihead pipeline-3action-e2e` 会在输出目录下自动写出
> `router_thresholds_baseline.json`（使用你传入的阈值覆盖 + 未传入则用 Router 默认值），
> 供 plateau 命令直接复用。
>
> 默认 tuned-threshold 流程已包含：**heuristic bounds**（防离谱阈值）与 **trend_rate 约束**（防 TREND 趋零）。

```bash
mlbot diagnose threshold-plateau --no-docker \
  --preds results/nnmh_e2e/tier01/preds \
  --logs  results/nnmh_e2e/tier01/logs_3action.parquet \
  --model <PATH_TO_MODEL_PT_FROM_TRAIN> \
  --baseline-json results/nnmh_e2e/tier01/router_thresholds_baseline.json \
  --out results/plateau/router3action_tier01_oos_v1 \
  --trend-rate-min 0.005 --trend-rate-penalty 2.0 \
  --heuristic-bounds --heuristic-qmin 0.05 --heuristic-qmax 0.95
```

**用法（推荐两步法）**

- **Step A：先跑一次 pipeline 产出 `preds/` + `logs_3action.parquet` + baseline thresholds**

```bash
mlbot nnmultihead pipeline-3action-e2e --no-docker \
  --task-spec config/tasks/task_spec.yaml \
  --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 --end-date 2024-06-30 \
  --feature-store-root feature_store \
  --feature-store-layer nnmh_tree_union_all_240T_v2 \
  --model <PATH_TO_MODEL_PT_FROM_TRAIN> \
  --returns-source rr_execution \
  --out results/nnmh_e2e/tier01
```

- **Step B：做 plateau tuning，得到 `router_thresholds_best.json`，再把它喂回 pipeline 重跑**
  - 输出位置：`results/plateau/router3action_tier01_oos_v1/router_thresholds_best.json`
  - 重要：**阈值调参只能用 train/oos（可调参）窗口**；`holdout`（只验收）不要用来调参。

```bash
# 1) tune -> best thresholds
mlbot diagnose threshold-plateau --no-docker \
  --preds results/nnmh_e2e/tier01/preds \
  --logs  results/nnmh_e2e/tier01/logs_3action.parquet \
  --model <PATH_TO_MODEL_PT_FROM_TRAIN> \
  --baseline-json results/nnmh_e2e/tier01/router_thresholds_baseline.json \
  --out results/plateau/router3action_tier01_oos_v1 \
  --trend-rate-min 0.005 --trend-rate-penalty 2.0 \
  --heuristic-bounds --heuristic-qmin 0.05 --heuristic-qmax 0.95

# 2) rerun pipeline with tuned thresholds (explicitly applied)
mlbot nnmultihead pipeline-3action-e2e --no-docker \
  --task-spec config/tasks/task_spec.yaml \
  --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 --end-date 2024-06-30 \
  --feature-store-root feature_store \
  --feature-store-layer nnmh_tree_union_all_240T_v2 \
  --model <PATH_TO_MODEL_PT_FROM_TRAIN> \
  --router-thresholds-json results/plateau/router3action_tier01_oos_v1/router_thresholds_best.json \
  --returns-source rr_execution \
  --out results/nnmh_e2e/tier01_tuned
```

### 3.2) 灭绝回放（Extinction Replay）：产出 survival labels（给 Survival Head / 熄火复燃验证）

> 目的：把 “在极端路径里会不会死” 变成可回放、可产物、可训练的标签（`labels.parquet`）。
> 对应长文：`docs/architecture/archetype灭绝级回测.md`、`docs/architecture/ood头的训练.md`

```bash
mlbot diagnose extinction-replay-3action --no-docker \
  --logs results/nnmh_e2e/tier01/logs_3action.parquet \
  --out  results/extinction_replay/tier01_v1
```

### 3.3) 训练 Survival Head（MLP）：产出 survival_prob（给 size cap / 熄火复燃）

> 输入：`logs_3action.parquet` + 上一步产出的 `labels.parquet`  
> 输出：`model.pt` + `survival_preds.parquet` + `report.html`（含 ROC/PR/Calibration 曲线）

```bash
mlbot diagnose survival-head-train --no-docker \
  --logs   results/nnmh_e2e/tier01/logs_3action.parquet \
  --labels results/extinction_replay/tier01_v1/labels.parquet \
  --out    results/survival_head/tier01_v1
```

### 3.4) Conditional Survival Table：学习 OOD → Archetype 生存权重（baseline）

> 目的：先用最稳的“表格基线”学习 `survival_rate(archetype | ood_bin)`，并导出可部署的 `weights.yaml`。  
> 备注：需要 `logs_3action.parquet` 中存在 `ood_score` 与 `active_archetype` 列（通常来自 LiveDashboard/Router 产物合并）。

```bash
mlbot diagnose ood-to-archetype-weights --no-docker \
  --logs   results/nnmh_e2e/tier01/logs_3action.parquet \
  --labels results/extinction_replay/tier01_v1/labels.parquet \
  --out    results/ood_to_archetype/tier01_v1
```

---

## Research Notes（近期结论）

- **Tier2 / Orderflow-only 对多头模型无增强**（HighCap6 / 2024H1）
  - Orderflow baseline：`results/runs/tier02_highcap6_2024H1_orderflow_20260115_041919/`
  - Orderflow tuned（plateau 后）：`results/runs/tier02_highcap6_2024H1_orderflow_tuned_20260115_044953/`
  - 对比报告：`results/compare/nnmh_runs/20260115_045141/report.md`
  - 结论：A-layer 提升不明显，系统层 Sharpe/收益无改善，trade_rate 下降；可暂时放弃该方向。

- **Tier2 / Spectrum+Math 对多头模型无增强**（HighCap6 / 2024H1）
  - Spectrum+Math baseline：`results/runs/tier02_highcap6_2024H1_spectrum_math_20260115_042103/`
  - Spectrum+Math warm3+extfill：`results/runs/tier02_highcap6_2024H1_spectrum_math_warm3_extfill_20260115_050924/`
  - Spectrum+Math nocache：`results/runs/tier02_highcap6_2024H1_spectrum_math_nocache_20260115_075757/`
  - 对比报告（同组内部）：`results/compare/nnmh_runs/20260115_095830/report.md`
  - 对比报告（与 Tier01 baseline）：`results/compare/nnmh_runs/20260115_100550/report.md`
  - 结论：系统层 Sharpe/收益无明显改善且 trade_rate 更低；无 plateau 报告产出，优先级降低。


对比方式：
- 跑两份 TaskSpec（Tier0-only vs Tier0+Tier1），生成两份 derived config / model / 报告
- 对比：A-layer（head eval）+ system（e2e counterfactual + KPI gate + snapshot）

---

## 文档入口（建议先读）

### 核心工作流文档

- **上线 MVP 闭环（最重要，先看这个）**：`docs/guides/DEPLOYMENT_MVP_WORKFLOW_CN.md`
  - Pool‑B + 语义组搜索 → 6 个月 holdout 验收 → 训练最终上线模型
  - rolling 与 Nautilus 的职责边界（OOS vs 实盘一致性）

- **多资产合约实盘落地路线图（从 1w→10w 的可执行路线，低维护）**：`docs/guides/LIVE_TRADING_ROADMAP_MULTI_ASSET_CN.md`
- **ETH 拖累处理与 Universe 演进（V1 交易 / V2 Shadow 监控）**：`docs/guides/ETH_DRAG_AND_UNIVERSE_EVOLUTION_CN.md`
- **特征搜索 Playbook（详细算法/命令/概念）**：`docs/strategies/FEATURE_SEARCH_PLAYBOOK_CN.md`
- **语义特征单列展开说明**：`docs/strategies/SEMANTIC_GROUPS_SINGLETON_EXPANSION.md`
- **归一化契约与检查**：`docs/architecture/NORMALIZATION_CONTRACT_AND_CHECKS.md`
- **“保留但不喂给模型”的列排除机制（exclude_columns）**：`docs/guides/FEATURE_PIPELINE_EXCLUDE_COLUMNS_CN.md`
- **Feature-group-search / pipeline 调参指南**：`docs/guides/FEATURE_GROUP_SEARCH_TUNING_GUIDE_CN.md`
  - nnmultihead 推荐顺序：search → train(primitives) → OOS predict → build-logs → Router 阈值调参 → BC/RL
- **特征测试设计与覆盖（4类测试 + 覆盖快照保存）**：`docs/tests/FEATURE_TEST_DESIGN_AND_COVERAGE_CN.md`
- **实盘特征契约与证据字段（缺失策略/has_orderflow/has_sr_quality）**：`docs/guides/LIVE_FEATURE_CONTRACT_AND_EVIDENCE_CN.md`
- **Archetype 上线前 Checklist（v0）**：`docs/architecture/ARCHETYPE_PRELIVE_CHECKLIST_CN.md`

- **项目 TODO / Roadmap**：`docs/architecture/ARCH_UPGRADE_TASKSPEC_CONSTITUTION_V1_CN.md`
  - TODO 已内聚到架构升级文档中（按 P0/P1/P2 分层）

### 架构文档

- **系统架构图（已更新到当前 CLI/工作流）**：`docs/ARCHITECTURE.md`

- **特征列表与归一化状态**：`docs/architecture/FEATURE_CATALOG.md`
  - 全部 208 个特征节点的归一化状态
  - 归一化方法说明（ATR 归一化、百分比归一化、相似度转换等）
  - 按类别分组的特征列表
  - 树模型 vs NN 模型的使用建议

- **特征归一化策略**：`docs/architecture/FEATURE_NORMALIZATION_POLICY.md`
  - Phase 1/2/3 归一化实现进度
  - 跨资产可比性验证
  - 因果性归一化方法（避免未来泄露）

- **工业化 Experiment Loop（Layer A/B/C、TaskSpec、Filter→Wrapper、稳定性口径）**：`docs/architecture/EXPERIMENT_LOOP_ARCHITECTURE.md`
- **NN 多头 Path Primitives + Router→Execution（NO/MEAN/TREND）**：`docs/时序模型/架构：NN多头路径原语（Path Primitives）+Router解耦升级.md`
- **研究 Playbook（标签/执行一致性、timeframe、仓位管理）**：`docs/guides/RESEARCH_PLAYBOOK_CN.md`
- **Policy 执行假设（intrabar vs close）**：`docs/strategies/POLICY_EXECUTION_ASSUMPTIONS_CN.md`
- **NN 多头 → 3-action → RL/BC e2e（长文档）**：`docs/guides/NNMULTIHEAD_3ACTION_E2E_CN.md`
- [多头NN和订单流的使用分类和评估](/workspaces/ml_trading_bot/docs/architecture/多头NN和订单流.md)
- [训练落地文档](docs/guides/FEATURE_COMPLEXITY_LAYERS_CN.md)
- [谁对sharp负责](docs/architecture/谁对sharp负责.md)
- [删除的策略(该不做什么)](docs/architecture/删除的策略该不做什么.md)
- [alpha可以更多吗](docs/architecture/alpha可以更多吗.md)
- [VolMean难在哪里](docs/architecture/VolMean难在哪里.md)
- [时间框架高级甜点区](docs/architecture/时间框架高级甜点区.md)
- [职责坍缩](docs/architecture/职责坍缩.md)

---

## 获取帮助

```bash
mlbot --help
```

---

## 实盘（Live：Nautilus + MetaRouterStrategy）

> 实盘入口与事件流/回放/对账等细节：见 `docs/live_stream/README.md`。

启动 MetaRouterStrategy（单策略 + 多 archetype 编排）：

```bash
python -m src.time_series_model.live.run_nautilus_strategy \
  --strategy-id meta_router \
  --live-config config/nnmultihead/live/meta_router_live_config.yaml \
  --symbol BTCUSDT-PERP \
  --timeframe 15T \
  --testnet
```


