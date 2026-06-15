# 系统架构（BPC 纯规则版）

**状态**: ✅ 当前版本  
**最后更新**: 2026-04-07

> 这份文档回答四个问题：
> 1) 系统分层与职责边界是什么？
> 2) 当前 BPC 决策管线如何组织？
> 3) 研发到上线的 pipeline 如何组织与验收？
> 4) 配套文档（工程指南、归档、历史树/ML 路径）去哪里找？
>
> 细节（命令参数、算法说明、边界条件）仍由专题文档承载，这里只做统一对齐与索引。全库导航见 **[docs/README.md](README.md)**；工程向长文索引见 **[docs/architecture/README.md](architecture/README.md)**。

## 一页速览

- **核心原则**：纯规则决策，无 ML 模型依赖。所有信号来自 BPC 结构特征。
- **决策管线（与代码一致）**：Prefilter → Direction → Gate → Entry Filter → **Execution** → TradeIntent（**无按分档 sizing**；`GenericLiveStrategy` 内未使用的分支保留为扩展钩子）
- **层级结构**：Data → FeatureStore → **GenericLiveStrategy** → Prefilter / Gate / Entry / Execution → PCM → Ops
- **PCM 定义**：slot rotation / add-on 是否允许 / portfolio 风险预算（不管 entry/exit 细节）
- **统一路径**：配置驱动 → plateau 优化 → 回测验证 → 实盘部署
- **总宣言**：规则管因果 → **Gate / Entry 管放行与时机** → PCM 管生存
- **文档入口**：`docs/README.md`（总索引）· `docs/architecture/README.md`（工程专题）· **树模型 / Pool-B 历史流程**已迁至 `docs/archive/guides/tree/`（非主线，仅对照）

## 目录

1. [系统概述](#系统概述)
2. [BPC 决策管线](#bpc-决策管线)
3. [Gate / Entry / Execution（配置）](#sec-gate-entry-exec)
4. [系统分层与职责边界](#系统分层与职责边界)
5. [端到端 Pipeline（统一路径）](#端到端-pipeline统一路径)
6. [PCM / Position & Slot Management](#pcm--position--slot-management)
7. [实盘架构](#实盘架构)
8. [核心目录与关键文件](#核心目录与关键文件)
9. [配套文档地图](#配套文档地图)
10. [Pipeline TODO](#pipeline-todo)
11. [相关文档入口](#相关文档入口)

---

## 系统概述

ML Trading Bot 是一个配置驱动的多资产交易系统，当前使用 **BPC（Breakout → Pullback → Continuation）纯规则架构**：

- **无 ML 模型依赖**：所有信号来自结构特征（breakout direction / pullback / score）
- **配置驱动**：Prefilter / Direction / Gate / Entry Filter / Execution 等参数 YAML 化（**无分档 sizing 配置**）
- **责任可分解（与当前产线对齐）**：主路径是 **Prefilter → 方向 → Gate → Entry → Execution**。默认 BPC turbo 流水线 **`execution_opt` 关闭**，rolling/事件侧常用 **`simple_execution` 固定 R**（见 `config/prod_train_pipeline_*_bpc_only.yaml`）
- **系统可回滚**：任何新增自由度都必须可开关、可审计、可降级

---

## BPC 决策管线

```
features (from IncrementalFeatureComputer；规模随 archetypes 汇总而变)
    │
    ├─ 0. Prefilter（prefilter.yaml，可选）
    │
    ├─ 1. Direction（direction.yaml：结构/带通/双信号等 → +1 / -1 / 0）
    │
    ├─ 2. Gate（gate.yaml：hard deny / soft 加权 / guardrail / quantile 规则）
    │
    ├─ 3. Entry Filter（entry_filters.yaml：入场时机 OR/组合）
    │
    ├─ 4. Execution（execution.yaml；产线侧常与 `simple_execution` 固定 R 对齐）
    │     └─ **不按分档选型**（未启用的代码分支保留为扩展钩子）
    │
    └─ 5. TradeIntent → OrderFlowListener / OrderManager 执行
```

> **与 `prod_train_pipeline_*_bpc_only.yaml` 对齐**：主线 rolling 侧重 **prefilter / gate / direction / entry_filter** 阈值与 KPI gate；**`execution_opt: false`**，验收常用 **`strategies.bpc.simple_execution` 固定 `sl_r` / `tp_r`**（**不做分档 execution 优化**）。

---

<a id="sec-gate-entry-exec"></a>

## Gate / Entry / Execution（配置）

当前实现以 **YAML 配置 + `GenericLiveStrategy`**（`scripts/run_live.py`）驱动：**Prefilter → Direction → Gate → Entry → Execution**。

### Gate 是可组合约束算子

Gate 的每条规则都是一个"约束算子"（谓词），通过 `deny_if / allow_if + allow_mode` 组合形成整体 Gate：

- **Hard Deny**：`deny_if` + 触发任一则 veto
- **Soft Filter**：加权评分，总分 < 阈值则否决
- **Guardrail**：安全边界检查

### Execution 层

- **当前主线**：止损/止盈/结构出场/加仓等以 **`execution.yaml` + 流水线 `simple_execution`** 为主；**`prod_train_pipeline_*_bpc_only.yaml` 中 `execution_opt: false`**（未跑 execution 网格）。
- **扩展钩子**：`generic_live_strategy.py` 内仍有未接 BPC archetypes 的评分→分档分支；**当前配置栈不启用**。旧版分层叙事见 `docs/architecture/` 专题长文，**不作为本文主线**。
- **持仓管理**：Time Stop → Breakeven → Trailing / 结构出场等（以配置为准，与 `OrderFlowListener` 持仓状态机对齐）

### 主要配置文件

- `config/strategies/bpc/archetypes/prefilter.yaml` — 前置环境过滤（与流水线 `has_prefilter` 一致）
- `config/strategies/bpc/archetypes/direction.yaml` — 方向规则（与 `direction_tuning` / macro epsilon 等标定联动）
- `config/strategies/bpc/archetypes/gate.yaml` — Gate 规则集
- `config/strategies/bpc/archetypes/entry_filters.yaml` — Entry Filter
- `config/strategies/bpc/archetypes/execution.yaml` — 执行与止损止盈/结构出场等
- **`config/strategies/<strategy>/archetypes/`** — 实盘默认由此目录自动汇总特征列与节点（`run_live.py` 传入 `archetypes_dir`，见 `IncrementalFeatureComputer`）
- `config/live/live_feature_plan.yaml` — **遗留路径**：仅当未设置 `archetypes_dir` 时回退（或 `MLBOT_LIVE_FEATURE_PLAN_YAML`）
- `config/feature_dependencies.yaml` — 特征 DAG 与归一化契约
- `config/prod_train_pipeline_*_bpc_only.yaml` — **当前 BPC turbo 主线**：`execution_opt: false`、`pcm_eval: false`，`simple_execution` 固定 R 等（以文件为准）

---

## 系统分层与职责边界

```
Layer0  DataCoverage           数据完整性/对齐/漂移检测
Layer1  FeatureStore           特征依赖图/归一化契约/BPC 结构特征
Layer2  GenericLiveStrategy    配置驱动决策引擎 (Prefilter → Direction → Gate → Entry → Execution)
Layer3  Gate                   规则 veto/allow/bias（hard deny + soft filter + guardrail）
Layer4  Execution              单仓执行参数与持仓状态机（当前主线常与 simple_execution 对齐）
Layer5  PCM_Portfolio          组合分配/风控（生产决策层）
Layer6  ReportsAndOps          KPI/审计/快照/回放
```

关键边界：

- **Gate** 只做 veto/allow/bias，不直接触发 entry
- **Execution** 只负责单仓生命周期，不管理 slot 与预算
- **PCM/Portfolio** 是唯一**生产决策**层

---

## 端到端 Pipeline（统一路径）

```
Data → FeatureStore → BPC特征 → Prefilter → Direction → Gate → Entry Filter → Execution → PCM
```

统一路径说明（**与 `prod_train_pipeline_*_bpc_only` 默认开关一致**）：

- YAML 冻结 **prefilter / direction / gate / entry** 等主路径参数
- **Gate / Entry** 等阈值可用 plateau / 诊断子命令标定；**execution 网格** 等为可选脚本，**默认 BPC-only turbo 配置中 `execution_opt: false`**
- Rolling / 事件回测验收常用 **`simple_execution` 固定 R**（**不做分档验收**）
- 回测验证以配置为准（bar / 事件驱动等）；细节见流水线 YAML 与 `docs/workflow/PIPELINE_WORKFLOW.md`
- **批量化研发/验收**：可用 `mlbot pipeline run` 与仓库内 `config/prod_train_pipeline_*.yaml` 驱动多阶段（如 gate、entry_filter、execution_opt、rolling_sim 等）；命令细节与历史链路说明见 **[docs/workflow/PIPELINE_WORKFLOW.md](workflow/PIPELINE_WORKFLOW.md)**，中文速查见根目录 **README_CN.md**

---

## PCM / Position & Slot Management

PCM = Portfolio / Capital / Meta Control 层，负责 "是否允许" 的决策，而不是 "怎么下单" 的细节。

### PCM 的职责边界（硬规则）
- **必须包含**：slot rotation / add-on 许可 / portfolio 风险预算
- **不包含**：entry/exit 价格、止损/止盈如何挂、微观 execution 细节

> **Execution 是单兵战术，PCM 是战役指挥。**

### v0 固定规则
- `max_slots = 2`（硬编码，不支持动态扩容）
- slot 是稀缺资源，不是趋势放大器

---

## 实盘架构

```
Binance market WebSocket
        │
        ▼
quant-feature-bus
  scripts/run_market_feature_publisher.py
  ticks → bars/features → live/shared_feature_bus
        │
        ├── quant-trend-swing
        │     scripts/run_live.py
        │     disk Feature Bus → GenericLiveStrategy → PCM → OrderManager
        │
        └── quant-hedge-multileg
              scripts/run_multi_leg_live.py --bar-source feature-store
              disk Feature Bus → hedge multi-leg daemon → execution adapter
```

生产边界：

- `quant-feature-bus` 是唯一监听 Binance 行情 WebSocket 的进程。
- `quant-trend-swing` 是方向性 B·Trend 账户消费者（`run_live.py`），只消费磁盘 Feature Bus。
- `quant-hedge-multileg` 是 hedge 账户的多腿策略进程，只消费磁盘 Feature Bus。
- `BinanceUserStream` 仅用于账户成交/订单回报，不代表行情 WebSocket。**Trend 与 Multi-leg 各进程、各子账户各建一条 User Data Stream**（`run_live.py` / `run_multi_leg_live.py`）；Multi-leg 只订阅订单/成交流，Trend 另订阅 `ACCOUNT_UPDATE`。详见 [multi_leg_user_stream_design.md](architecture/multi_leg_user_stream_design.md)。

本地启动趋势消费者示例：
```bash
MLBOT_FEATURE_SOURCE=bus MLBOT_FEATURE_BUS_ROOT=live/shared_feature_bus python scripts/run_live.py
```

---

## 核心目录与关键文件

### 配置文件

| 文件 | 说明 |
|------|------|
| `config/strategies/bpc/archetypes/prefilter.yaml` | 前置过滤 |
| `config/strategies/bpc/archetypes/direction.yaml` | 方向规则 |
| `config/strategies/bpc/archetypes/gate.yaml` | Gate 规则集 |
| `config/strategies/bpc/archetypes/entry_filters.yaml` | Entry Filter |
| `config/strategies/bpc/archetypes/execution.yaml` | 执行与止损/结构出场等 |
| `config/strategies/bpc/archetypes/*.yaml` | 其余 archetype 片段（以目录为准） |
| `config/live/live_feature_plan.yaml` | （可选/遗留）未使用 archetypes 自动检测时的特征清单 |
| `config/feature_dependencies.yaml` | 特征 DAG 与归一化契约 |
| `config/prod_train_pipeline_*_bpc_only.yaml` | BPC turbo 主线流水线（含 `simple_execution`、`execution_opt` 开关） |

### 代码文件

| 文件 | 职责 |
|------|------|
| `scripts/run_market_feature_publisher.py` | 唯一行情 WebSocket publisher，写入磁盘 Feature Bus |
| `scripts/run_live.py` | `quant-trend-swing` 消费者；装配 **GenericLiveStrategy** + LivePCM |
| `scripts/run_multi_leg_live.py` | `quant-hedge-multileg` 消费者；只读 Feature Bus / feature-store |
| `src/time_series_model/live/generic_live_strategy.py` | **配置驱动决策引擎**（Prefilter→…→Execution） |
| `src/live_data_stream/order_flow_listener.py` | publisher/consumer 共享的 bar/feature/状态机组件 |
| `src/live_data_stream/multi_symbol_manager.py` | 多币种管理 |
| `src/live_data_stream/websocket_client.py` | Binance 行情 WebSocket 客户端（仅 publisher 生产路径使用） |
| `src/time_series_model/live/incremental_feature_computer.py` | 增量特征计算 |
| `src/time_series_model/execution/entry_filter.py` | Entry Filter 公共模块 (batch + live) |
| `src/time_series_model/execution/tier.py` | 分档→执行参数映射（**BPC 当前不用**） |
| `src/time_series_model/execution/noise_penalty.py` | Execution 噪声惩罚（可选路径） |
| `src/time_series_model/live/tree_gate.py` | Gate 规则评估 |
| `src/time_series_model/evidence/bpc_evidence_calculator.py` | 评分/证据计算器（**未接入当前 BPC archetypes**） |
| `src/time_series_model/core/trade_intent.py` | TradeIntent 数据结构 |

### 优化脚本

| 文件 | 用途 |
|------|------|
| `scripts/optimize_gate_unified.py` | Gate 规则优化 |
| `scripts/optimize_evidence_plateau.py` | 评分 plateau 脚本（**非** BPC-only 主线默认环节） |
| `scripts/optimize_entry_filter_snotio.py` | Entry Filter 组合搜索 |
| `scripts/optimize_entry_filter_plateau.py` | Entry Filter 阈值 plateau 扫描 |
| `scripts/optimize_execution_grid.py` | Execution Grid Search（**可选**；与 `execution_opt` 联动） |
| `scripts/backtest_execution_layer.py` | 回测（bar-by-bar 模拟） |
| `scripts/eval_soft_gates.py` | Soft Gate 质量评估 |

---

## 配套文档地图

以下为 **与本文档（BPC 主线）并列** 的文档位置，避免在 `docs/` 下迷路。

| 区域 | 路径 | 说明 |
|------|------|------|
| 全库导航 | [docs/README.md](README.md) | 按角色/主题的入口汇总 |
| 工程专题索引 | [docs/architecture/README.md](architecture/README.md) | Gate、实验循环、数学特征等长文 |
| 当前工程指南 | [docs/architecture/guides/](architecture/guides/) | Plateau、Gate、BPC 注记、基线/归因、特征分层、`TREE_TRAINING_DATA_AND_CACHE` 等 |
| 树模型历史指南 | [docs/archive/guides/tree/](archive/guides/tree/) | 原 `architecture/guides/tree/`：**MVP 闭环**、特征组搜索预设/调参、研究 Playbook、`ModelArtifact` 等（**非当前产品主线**） |
| 策略与 Playbook | [docs/architecture/strategies/](architecture/strategies/) | 特征搜索协议、Policy 假设等；过程性笔记见 [docs/archive/strategies/](archive/strategies/) |
| 树模型策略报告 | [docs/architecture/树模型策略report/](architecture/树模型策略report/) | 特征筛选索引、归一化/订单流审计；中间结果见 [docs/archive/reports/](archive/reports/) |
| 实盘数据流 | [docs/architecture/live_stream/](architecture/live_stream/) | 契约、事件流、补全与对账；legacy 见 [docs/archive/live_stream/legacy/](archive/live_stream/legacy/) |
| 指标与评估 | [docs/architecture/metrics/](architecture/metrics/) | Sharpe、泄漏鉴别等 |
| 部署 | [.github/workflows/deploy.yml](../../.github/workflows/deploy.yml)（镜像构建 + 三 systemd 服务 + Grafana 同步） | 生产路径 |
| 总归档 | [docs/archive/](archive/) | NN 多头、旧架构长文、随笔等 |

占位目录（`docs/guides/`、`docs/live_stream/`、`docs/strategies/` 等）仅指向 **architecture/** 或 **archive/** 正文，见各目录下 `README.md`。

---

## Pipeline TODO

**v0（当前）**
- PCM / Slot State Machine 落地（rotation / add-on 许可 / 风险预算）
- 多 Archetype 扩展（CompositeDecisionHandler）
- KPI Journal 全链路覆盖

**v1（实盘稳定后）**
- 免疫/记仇/分情境 IC 验证模块
- `config/nnmultihead/` 整个目录清理

---

## 相关文档入口

### BPC 实验与实现说明

- **实验文档**：`z实验_001_bpc/实验_001.md`（命令速查） / `z实验_001_bpc/实验_001详细记录.md`（详细原理）
- **Entry Filter 设计**：`z实验_001_bpc/entry_filter_design.md`
- **实盘代码说明**：`src/time_series_model/live/README.md`

### 工作流与命令

- **流水线命令序列**：[docs/workflow/PIPELINE_WORKFLOW.md](workflow/PIPELINE_WORKFLOW.md)
- **中文快速开始**：根目录 `README_CN.md` / `README.md`

### 工程指南（调参 / 归因 / 特征）

- **阈值平坦高原**：`docs/architecture/guides/THRESHOLD_PLATEAU_TUNING_PROTOCOL_CN.md`
- **Plateau / 基线 / 实盘归因**：`docs/architecture/guides/PLATEAU_OPTIMIZATION_WORKFLOW.md`、`docs/architecture/guides/BASELINE_TESTING_WORKFLOW.md`、`docs/architecture/guides/PRODUCTION_ATTRIBUTION_WORKFLOW.md`
- **特征复杂度分层**：`docs/architecture/guides/FEATURE_COMPLEXITY_LAYERS_CN.md`
- **研发→上线分层（Tier×Universe×TaskSpec）**：`docs/architecture/guides/RD_TO_LIVE_TIERED_WORKFLOW_V1_CN.md`

### 实盘、仓位与执行语义

- **Live Stream（契约与数据流）**：`docs/architecture/live_stream/README.md`
- **生产部署**：`docs/deployment/LIVE_PRODUCTION_RUNBOOK_CN.md`、仓库根目录 `.github/workflows/deploy.yml`
- **仓位与 PCM**：`docs/architecture/仓位管理办法.md`、`docs/architecture/LivePCM_多archetype信号仲裁.md`
- **回测与实盘对比**：`docs/architecture/backtest_vs_live_execution.md`
- **数学特征分层（path2.5）**：`docs/architecture/path2.5_math_features.md`

### 历史与对照（非 BPC 主线）

- **树模型上线 MVP / 特征组搜索（已归档）**：`docs/archive/guides/tree/DEPLOYMENT_MVP_WORKFLOW_CN.md` 与同目录下预设、调参、Playbook
- **NN 多头 / Router / 旧架构**：`docs/archive/`、`docs/archive/architecture/`（随笔与长文研究）

### 索引

- **全库文档导航**：[docs/README.md](README.md)
- **架构专题列表**：[docs/architecture/README.md](architecture/README.md)
