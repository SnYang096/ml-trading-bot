# 系统架构（总览，已更新到当前 CLI 与工作流）

> 这份文档回答两个问题：
> 1) 这个仓库“有哪些层”，分别负责什么？
> 2) 一个完整闭环（数据→特征→搜索→验收→上线）在代码层面怎么串起来？
>
> 细节（命令参数、算法说明、边界条件）请跳转到对应专题文档，避免在这里重复维护。

## 目录

1. [系统概述](#系统概述)
2. [架构分层（Tree / NN 双轨）](#架构分层tree--nn-双轨)
3. [核心目录与关键文件](#核心目录与关键文件)
4. [数据管道（Download→Convert→Parquet）](#数据管道downloadconvertparquet)
5. [特征系统（依赖图、归一化契约、缓存）](#特征系统依赖图归一化契约缓存)
6. [特征选择与搜索（Pool‑B + 语义组 + Pipeline）](#特征选择与搜索poolb--语义组--pipeline)
7. [训练/验收/上线（Rolling / Holdout / Final）](#训练验收上线rolling--holdout--final)
8. [回测与执行（研究 Backtest vs Nautilus 对齐）](#回测与执行研究-backtest-vs-nautilus-对齐)
9. [相关文档入口](#相关文档入口)

---

## 系统概述

ML Trading Bot 是一个**配置驱动**的量化研究与交易系统：

- **输入**：多资产 OHLCV +（可选）market cap、funding rate 等扩展数据
- **中间**：可复用的特征依赖图 + 归一化契约 + 特征缓存/feature store
- **输出**：
  - Tree 模型：直接产出开仓/风控所需的预测（策略配置驱动）
  - NN 多头：Path Primitives（路径原语）→ Router → Execution（NO/MEAN/TREND）

系统的“生产闭环”目标是：**一次搜索产出建议 YAML**，随后进入**冻结窗口的 holdout 验收**，最后**全窗训练产出上线工件（ModelArtifact）**。

---

## 架构分层（Tree / NN 双轨）

```
┌───────────────────────────────────────────────────────────────────────┐
│ CLI / Orchestration                                                   │
│  - mlbot data / analyze / diagnose / train                            │
│  - scripts/*（pipeline, feature_store, nn training, reports）         │
└───────────────────────────────────────────────────────────────────────┘
                           ↓
┌───────────────────────────────────────────────────────────────────────┐
│ Strategy Config（策略配置层）                                         │
│  - config/strategies/<strategy>/{features,labels,model,backtest}.yaml  │
│  - 可选：config/feature_groups_<strategy>_semantic.yaml               │
└───────────────────────────────────────────────────────────────────────┘
                           ↓
┌───────────────────────────────────────────────────────────────────────┐
│ Feature System（特征系统）                                            │
│  - config/feature_dependencies.yaml（依赖图 + output columns + 归一化）│
│  - StrategyFeatureLoader / FeatureComputer（计算 + 缓存）              │
│  - feature_store（可选：按 layer/version 的特征离线存储）              │
└───────────────────────────────────────────────────────────────────────┘
                           ↓
┌───────────────────────────────────────────────────────────────────────┐
│ Modeling（模型层，双轨）                                               │
│  - Tree：LightGBM/XGBoost/CatBoost（train_strategy_pipeline.py）       │
│  - NN：Path Primitives + Router + Execution（src/time_series_model/...）│
└───────────────────────────────────────────────────────────────────────┘
                           ↓
┌───────────────────────────────────────────────────────────────────────┐
│ Evaluation & Deployment（评估与上线）                                  │
│  - 研究 backtest（训练脚本内置）                                       │
│  - holdout-eval（冻结 6 个月）                                         │
│  - train final（全窗训练产出 ModelArtifact）                             │
│  - Nautilus（事件驱动一致性验证/实盘对齐）                               │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 核心目录与关键文件

### 1) 策略配置（强入口）

- `config/strategies/<strategy>/features.yaml`
  - `feature_pipeline.requested_features`: **模型输入的特征节点/列**
  - `feature_pipeline.exclude_columns`: **保留但不喂给模型**（见下文）
- `config/strategies/<strategy>/labels.yaml`
  - label 生成、过滤、样本权重等（Tree 策略通常在这里决定 trades 分布）
- `config/strategies/<strategy>/model.yaml` / `backtest.yaml`

### 2) 特征依赖图（最权威）

- `config/feature_dependencies.yaml`
  - 每个 feature node 的：dependencies / required_columns / output_columns / normalization contract
  - **注意**：feature-group-search 默认在 **node/group** 粒度工作；开启 singletons 才会细到 output column

### 3) 特征组（用于组合搜索）

- 优先：`config/feature_groups_<strategy_dir_name>_semantic.yaml`（存在则用）
- 否则：`config/feature_groups.yaml`

### 4) 训练/诊断入口

- Tree 训练：`scripts/train_strategy_pipeline.py`
- Feature group search：`src/time_series_model/diagnostics/feature_group_search.py`
- 一键（Pool‑B + Search + Report）：`scripts/run_poolb_semantic_search.py`（由 `mlbot diagnose poolb-semantic-search` 封装）

---

## 数据管道（Download→Convert→Parquet）

核心目标：把原始数据整理成训练可直接读取的 Parquet（例如 `data/parquet_data`）。

入口命令（详见 README_CN）：
- `mlbot data download / convert / pipeline`
- Universe 批量：`mlbot data pipeline-universe`

扩展数据：
- Market cap：`mlbot data update-market-cap`
- Funding rate：`mlbot data download-funding-rate`

---

## 特征系统（依赖图、归一化契约、缓存）

### 1) 依赖图驱动（Feature Node）

特征由 feature node（函数/计算单元）构成，node 可输出多个列（output columns）。

- node 定义：`config/feature_dependencies.yaml`
- 计算与装配：`src/features/loader/strategy_feature_loader.py`

### 2) 归一化契约（Normalization Contract）

所有特征的归一化“声明”以 `config/feature_dependencies.yaml` 为准，并通过合同检查工具/测试防止漂移。

相关文档：
- `docs/architecture/NORMALIZATION_CONTRACT_AND_CHECKS.md`
- `docs/architecture/FEATURE_CATALOG.md`

### 3) exclude_columns（保留但不喂给模型）

某些列用于 label/backtest 尺度或诊断，但不希望作为模型输入（例如 `atr` 这种 price-unit 列）。

- 配置入口：`feature_pipeline.exclude_columns`
- 说明文档：`docs/guides/FEATURE_PIPELINE_EXCLUDE_COLUMNS_CN.md`

### 4) 缓存与 feature_store

Tree 策略：通常依赖 FeatureComputer 的缓存与按需计算。  
NN 多头：更倾向先构建 `feature_store`（按 layer/version），再做训练/搜索。

---

## 特征选择与搜索（Pool‑B + 语义组 + Pipeline）

### 1) 为什么需要 Pool‑B？

全量特征空间太大：先通过 `factor-eval` 等机制生成候选池 Pool‑B，再与 semantic groups 合并，能显著降低组合爆炸。

Pool‑B 默认落盘（由一键工具生成）：
- `results/pools/<strategy>/pool_b/<TAG>/features_pool_b.yaml`

### 2) feature-group-search 的粒度

- 默认：**node/group** 粒度（一个 node 可能展开成多列）
- 开启 `--expand-semantic-singletons`：对 multi-output semantic nodes 做列级展开（更细，但预算要更大）

### 3) pipeline（SH → Beam → SFFS）

- **Successive Halving**：用小预算筛掉明显差的候选
- **Beam Search**：保留 top‑K 路径，捕捉“协同效应”
- **SFFS**：在最终组合上做去冗余（可选）

调参指南：
- `docs/guides/FEATURE_GROUP_SEARCH_TUNING_GUIDE_CN.md`

### 4) writeback（闭环）

feature-group-search 会输出建议的 features YAML（便于进入 rolling/holdout/final）：

- `config/strategies/<strategy>/features_suggested_<algo>_...yaml`

---

## 训练/验收/上线（Rolling / Holdout / Final）

推荐最小闭环（MVP）：

1) **特征搜索**：`mlbot diagnose poolb-semantic-search` → 写回建议 YAML
2) **冻结验收**：`mlbot diagnose holdout-eval`（固定 6 个月 holdout，不再调参）
3) **上线产物**：`mlbot train final`（全窗训练）→ 产出 `ModelArtifact`

详细命令与验收标准见：
- `docs/guides/DEPLOYMENT_MVP_WORKFLOW_CN.md`

---

## 回测与执行（研究 Backtest vs Nautilus 对齐）

仓库存在两种“回测语境”：

- **研究回测**：训练脚本内置的回测，用于快速迭代与一致口径比较
- **Nautilus 对齐**：事件驱动回测/实盘一致性验证（更接近真实执行）

相关文档：
- `docs/实时流计算/reference/Nautilus_Trader_集成指南.md`
- `docs/实时流计算/07_与NautilusTrader对齐清单.md`

---

## 相关文档入口

- README（最小可复制命令入口）：`README_CN.md`
- 上线闭环：`docs/guides/DEPLOYMENT_MVP_WORKFLOW_CN.md`
- 特征搜索 playbook：`docs/strategies/FEATURE_SEARCH_PLAYBOOK_CN.md`
- 语义单列展开：`docs/strategies/SEMANTIC_GROUPS_SINGLETON_EXPANSION.md`
- 归一化契约：`docs/architecture/NORMALIZATION_CONTRACT_AND_CHECKS.md`
- Experiment Loop（Layer A/B/C、TaskSpec）：`docs/architecture/EXPERIMENT_LOOP_ARCHITECTURE.md`
- NN 多头（Path Primitives + Router）：`docs/时序模型/架构：NN多头路径原语（Path Primitives）+Router解耦升级.md`

