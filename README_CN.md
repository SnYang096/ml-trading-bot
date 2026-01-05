# ML Trading Bot（中文）

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
  --no-docker
```

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

- rolling：用于跨月稳定性与上线后监控（见 `docs/guides/DEPLOYMENT_MVP_WORKFLOW_CN.md`）
- Nautilus：用于“回测=实盘一致性验证”（见 `docs/live_stream/reference/Nautilus_Trader_集成指南.md`、`docs/live_stream/07_与NautilusTrader对齐清单.md`）

---

## 文档入口（建议先读）

### 核心工作流文档

- **上线 MVP 闭环（最重要，先看这个）**：`docs/guides/DEPLOYMENT_MVP_WORKFLOW_CN.md`
  - Pool‑B + 语义组搜索 → 6 个月 holdout 验收 → 训练最终上线模型
  - rolling 与 Nautilus 的职责边界（OOS vs 实盘一致性）

- **特征搜索 Playbook（详细算法/命令/概念）**：`docs/strategies/FEATURE_SEARCH_PLAYBOOK_CN.md`
- **语义特征单列展开说明**：`docs/strategies/SEMANTIC_GROUPS_SINGLETON_EXPANSION.md`
- **归一化契约与检查**：`docs/architecture/NORMALIZATION_CONTRACT_AND_CHECKS.md`
- **“保留但不喂给模型”的列排除机制（exclude_columns）**：`docs/guides/FEATURE_PIPELINE_EXCLUDE_COLUMNS_CN.md`
- **Feature-group-search / pipeline 调参指南**：`docs/guides/FEATURE_GROUP_SEARCH_TUNING_GUIDE_CN.md`
  - nnmultihead 推荐顺序：search → train(primitives) → OOS predict → build-logs → Router 阈值调参 → BC/RL
- **特征测试设计与覆盖（4类测试 + 覆盖快照保存）**：`docs/tests/FEATURE_TEST_DESIGN_AND_COVERAGE_CN.md`

- **项目 TODO List**：`docs/TODO_LIST.md`
  - 所有待完成任务的详细说明
  - 按优先级和类别组织
  - 包含任务作用、命令示例、预期结果等

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

---

## 获取帮助

```bash
mlbot --help
```


