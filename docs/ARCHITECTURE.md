# 系统架构（BPC 纯规则版）

**状态**: ✅ 当前版本  
**最后更新**: 2026-02-10

> 这份文档回答三个问题：
> 1) 系统分层与职责边界是什么？
> 2) 当前 BPC 决策管线如何组织？
> 3) 研发到上线的 pipeline 如何组织与验收？
>
> 细节（命令参数、算法说明、边界条件）仍由专题文档承载，这里只做统一对齐与索引。

## 一页速览

- **核心原则**：纯规则决策，无 ML 模型依赖。所有信号来自 BPC 结构特征。
- **决策管线**：Direction → Gate → Entry Filter → Evidence → Tier → Execution
- **层级结构**：Data → FeatureStore → BPCLiveStrategy → Gate → Evidence → Execution → PCM → Ops
- **PCM 定义**：slot rotation / add-on 是否允许 / portfolio 风险预算（不管 entry/exit 细节）
- **统一路径**：配置驱动 → plateau 优化 → 回测验证 → 实盘部署
- **总宣言**：规则管因果 → Evidence 管质量 → PCM 管生存

## 目录

1. [系统概述](#系统概述)
2. [BPC 决策管线](#bpc-决策管线)
3. [配置驱动的 Gate / Evidence / Execution](#配置驱动的-gate--evidence--execution)
4. [系统分层与职责边界](#系统分层与职责边界)
5. [端到端 Pipeline（统一路径）](#端到端-pipeline统一路径)
6. [PCM / Position & Slot Management](#pcm--position--slot-management)
7. [实盘架构](#实盘架构)
8. [核心目录与关键文件](#核心目录与关键文件)
9. [Pipeline TODO](#pipeline-todo)
10. [相关文档入口](#相关文档入口)

---

## 系统概述

ML Trading Bot 是一个配置驱动的多资产交易系统，当前使用 **BPC（Breakout → Pullback → Continuation）纯规则架构**：

- **无 ML 模型依赖**：所有信号来自结构特征（breakout direction / pullback / score）
- **配置驱动**：Gate / Evidence / Entry Filter / Execution 参数全部 YAML 化
- **责任可分解**：Gate 否决、Evidence 评分、Tier 分档、Execution 执行，职责清晰
- **系统可回滚**：任何新增自由度都必须可开关、可审计、可降级

---

## BPC 决策管线

```
features (from IncrementalFeatureComputer, 104 features, 47 nodes)
    │
    ├─ 1. bpc_breakout_direction → 方向 (+1=LONG, -1=SHORT, 0=不入场)
    │
    ├─ 2. Gate 检查
    │     ├─ Hard Deny (3条): direction_crowded / atr_floor / volatility_extreme
    │     ├─ Soft Filter (9条): 加权评分，权重 < 阈值则否决
    │     └─ Guardrail (2条): spread_guard / volume_guard
    │
    ├─ 3. Entry Filter (OR 逻辑)
    │     ├─ bb: Bollinger Band 压缩 (bpc_bb_squeeze_pct < 0.38)
    │     └─ liq_silence: 流动性沉默 (ef_liquidity_silence > 0.60)
    │
    ├─ 4. Evidence Score (9 特征 quantile mapping → 0~1)
    │     └─ Tier 选择: T1(≥0.55) / T2(≥0.40) / T3(≥0.25) / T4(<0.25)
    │
    └─ 5. 输出 TradeIntent (action/symbol/archetype/tier_params/...)
         → OrderFlowListener._execute_intent() 执行
```

---

## 配置驱动的 Gate / Evidence / Execution

当前实现以**配置驱动为主**，代码提供"加载、验证、执行"框架；规则与策略适配由配置承载。

### Gate 是可组合约束算子

Gate 的每条规则都是一个"约束算子"（谓词），通过 `deny_if / allow_if + allow_mode` 组合形成整体 Gate：

- **Hard Deny**：`deny_if` + 触发任一则 veto
- **Soft Filter**：加权评分，总分 < 阈值则否决
- **Guardrail**：安全边界检查

### Evidence 层设计原则

Evidence 层负责评估 alpha 质量，仅基于结构/订单流/规制特征：

- **输入特征**：仅使用结构、订单流、规制特征（9 条 quantile mapping）
- **输出**：alpha 质量评分（0-1）→ Tier 选择
- **目的**：评估"是否值得交易"以及"用多大仓位交易"
- **语义**："Price tells you WHAT, Volume tells you IF, Order flow tells you WHO"

### Execution 层

Execution 层负责单仓生命周期管理：

- **Tier 参数**：SL/TP/Size/Timeout 按 Evidence Score 分4档
- **Noise Penalty**：数学特征（WPT/Spectrum/Hilbert/Hurst/EVT）动态调整执行参数
- **持仓管理 7 步**：Time Stop → Breakeven Lock → Water Mark → Activation Trailing → SL Hit → TP Hit → Execute Close

### 主要配置文件

- `config/strategies/bpc/archetypes/gate.yaml` — Gate 规则 (3 hard + 9 soft + 2 guardrail + 1 safety)
- `config/strategies/bpc/archetypes/evidence.yaml` — Evidence 特征 (9 条, quantile mapping)
- `config/strategies/bpc/archetypes/entry_filters.yaml` — Entry Filter (bb OR liq_silence)
- `config/strategies/bpc/archetypes/execution.yaml` — Execution (3 tier + trailing SL + noise penalty)
- `config/live/live_feature_plan.yaml` — Live 特征计划 (104 features, 47 nodes)
- `config/strategies/bpc/archetypes/holding.yaml` — 持仓管理参数 (breakeven/trailing/time_stop)

---

## 系统分层与职责边界

```
Layer0  DataCoverage           数据完整性/对齐/漂移检测
Layer1  FeatureStore           特征依赖图/归一化契约/BPC 结构特征
Layer2  BPCLiveStrategy        纯规则决策引擎 (Direction → Gate → Entry → Evidence → Tier)
Layer3  Gate                   规则 veto/allow/bias（hard deny + soft filter + guardrail）
Layer4  Evidence               Alpha 质量评估（9 特征 quantile mapping）
Layer5  Execution              单仓生命周期管理（含 noise_penalty 调整 + 7步持仓管理）
Layer6  PCM_Portfolio          组合分配/风控（生产决策层）
Layer7  ReportsAndOps          KPI/审计/快照/回放
```

关键边界：

- **Gate** 只做 veto/allow/bias，不直接触发 entry
- **Evidence** 只评估质量和分档，不做收益优化
- **Execution** 只负责单仓生命周期，不管理 slot 与预算
- **PCM/Portfolio** 是唯一**生产决策**层

---

## 端到端 Pipeline（统一路径）

```
Data → FeatureStore → BPC特征 → Gate → Entry Filter → Evidence → Tier → Execution → PCM
```

统一路径说明：

- 配置文件（YAML）冻结所有规则参数
- Gate/Evidence/Entry Filter 的阈值用 "平坦高原" 协议（plateau 优化）
- Execution 参数用 Grid Search 优化
- 回测验证用 bar-by-bar 模拟（非 forward_rr）

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
┌─────────────────────────────────────────────────────────────────────┐
│                    Binance WebSocket (fstream)                       │
│    wss://fstream.binance.com/stream?streams=<symbol>@trade          │
└───────────────────────┬─────────────────────────────────────────────┘
                        │ raw trades
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                MultiSymbolManager                                    │
│    每个 symbol 一个 OrderFlowListener                               │
└───────────────────────┬─────────────────────────────────────────────┘
                        │ tick → bar (240T)
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│              OrderFlowListener                                       │
│    tick 聚合 → IncrementalFeatureComputer                           │
│    → 特征计算 (104 features, 47 nodes)                              │
│    → decision_handler.decide(features)                              │
│    → 7 步持仓管理                                                    │
└───────────────────────┬─────────────────────────────────────────────┘
                        │ features
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│              BPCLiveStrategy (纯规则决策引擎)                        │
│    Gate (3 hard + 9 soft + 2 guardrail)                             │
│    → Entry Filter (bb OR liq_silence)                               │
│    → Evidence Score (9 features → 0~1)                              │
│    → Tier → TradeIntent                                             │
│    → 无 ML 模型依赖                                                 │
└───────────────────────┬─────────────────────────────────────────────┘
                        │ TradeIntent
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    OrderManager                                      │
│    place_order() → Binance REST API                                 │
│    市价单 + 止损单 + 止盈单                                          │
└─────────────────────────────────────────────────────────────────────┘
```

启动命令：
```bash
MLBOT_LIVE_SYMBOLS=BTCUSDT python scripts/run_live.py
```

---

## 核心目录与关键文件

### 配置文件

| 文件 | 说明 |
|------|------|
| `config/strategies/bpc/archetypes/gate.yaml` | Gate 规则 (3 hard + 9 soft + 2 guardrail + 1 safety) |
| `config/strategies/bpc/archetypes/evidence.yaml` | Evidence 特征 (9 条, quantile mapping) |
| `config/strategies/bpc/archetypes/entry_filters.yaml` | Entry Filter (bb OR liq_silence) |
| `config/strategies/bpc/archetypes/execution.yaml` | Execution (3 tier + trailing SL + noise penalty) |
| `config/strategies/bpc/archetypes/holding.yaml` | 持仓管理参数 |
| `config/live/live_feature_plan.yaml` | Live 特征计划 (104 features, 47 nodes) |
| `config/feature_dependencies.yaml` | 特征 DAG 与归一化契约 |

### 代码文件

| 文件 | 职责 |
|------|------|
| `scripts/run_live.py` | 统一实盘入口 (BPC only) |
| `src/time_series_model/live/bpc_live_strategy.py` | BPC 纯逻辑决策引擎 |
| `src/live_data_stream/order_flow_listener.py` | Tick聚合 + 特征计算 + 决策路由 + 7步持仓管理 |
| `src/live_data_stream/multi_symbol_manager.py` | 多币种管理 |
| `src/live_data_stream/websocket_client.py` | Binance WebSocket 客户端 |
| `src/time_series_model/live/incremental_feature_computer.py` | 增量特征计算 |
| `src/time_series_model/execution/entry_filter.py` | Entry Filter 公共模块 (batch + live) |
| `src/time_series_model/execution/tier.py` | Evidence → Tier 映射 |
| `src/time_series_model/execution/noise_penalty.py` | Execution 噪声惩罚计算器 |
| `src/time_series_model/live/tree_gate.py` | Gate 规则评估 |
| `src/time_series_model/evidence/bpc_evidence_calculator.py` | BPC 证据计算器 |
| `src/time_series_model/core/trade_intent.py` | TradeIntent 数据结构 |

### 优化脚本

| 文件 | 用途 |
|------|------|
| `scripts/optimize_gate_unified.py` | Gate 规则优化 |
| `scripts/optimize_evidence_plateau.py` | Evidence 特征 plateau 优化 |
| `scripts/optimize_entry_filter_snotio.py` | Entry Filter 组合搜索 |
| `scripts/optimize_entry_filter_plateau.py` | Entry Filter 阈值 plateau 扫描 |
| `scripts/optimize_execution_grid.py` | Execution 参数 Grid Search |
| `scripts/backtest_execution_layer.py` | 回测（bar-by-bar 模拟） |
| `scripts/eval_soft_gates.py` | Soft Gate 质量评估 |

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

- **实验文档**：`z实验_001_bpc/实验_001.md`（命令速查） / `z实验_001_bpc/实验_001详细记录.md`（详细原理）
- **Entry Filter 设计**：`z实验_001_bpc/entry_filter_design.md`
- **实盘说明**：`src/time_series_model/live/README.md`
- **Live Stream**：`docs/live_stream/README.md`
- **仓位管理**：`docs/architecture/仓位管理办法.md`
- **阈值平坦高原**：`docs/guides/THRESHOLD_PLATEAU_TUNING_PROTOCOL_CN.md`
- **回测与实盘对比**：`docs/architecture/backtest_vs_live_execution.md`
- **数学特征分层**：`docs/architecture/path2.5_math_features.md`
- **归档文档**：`docs/archive/`（NN 多头 / Router / MetaRouter 等历史文档）
