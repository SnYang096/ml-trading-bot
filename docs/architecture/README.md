# 架构文档索引（工程专题）

**最后更新**: 2026-06-14  
**相关文档**: [主文档索引](../README.md) · [系统架构（统一版）](../ARCHITECTURE.md)

> **主线**：当前产品叙述以 **BPC 纯规则** 为主（见根目录 `ARCHITECTURE.md`）。长篇随笔、NN/Router 哲学对比等已迁至 **[docs/archive/architecture/](../archive/architecture/README.md)**。

---

## 必读与主线延伸

| 文档                                                                                            | 说明                                                                                    |
| ----------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| [../ARCHITECTURE.md](../ARCHITECTURE.md)                                                        | 分层、BPC 管线、配置入口、Pipeline TODO（权威入口）                                     |
| [RD控制台_研发监控管理对账_设计_CN.md](RD控制台_研发监控管理对账_设计_CN.md)                    | RD 控制台设计：替代老 rolling_dashboard，研发/监控/管理/对账一体化（对标 qlib，设计稿） |
| [backtest_vs_live_execution.md](backtest_vs_live_execution.md)                                  | 回测与实盘执行差异                                                                      |
| [path2.5_math_features.md](path2.5_math_features.md)                                            | 数学特征分层                                                                            |
| [仓位管理办法.md](仓位管理办法.md)                                                              | 仓位与 PCM 相关长篇说明                                                                 |
| [LivePCM_多archetype信号仲裁.md](LivePCM_多archetype信号仲裁.md)                                | 多 archetype 与 PCM                                                                     |
| [自由度限制.md](自由度限制.md) / [自由度限制-归因-仓位和加仓.md](自由度限制-归因-仓位和加仓.md) | 自由度与组合约束                                                                        |

---

## 实验循环与稳健性

| 文档                                                                           | 说明                                    |
| ------------------------------------------------------------------------------ | --------------------------------------- |
| [EXPERIMENT_LOOP_ARCHITECTURE.md](EXPERIMENT_LOOP_ARCHITECTURE.md)             | Layer A/B/C、TaskSpec、特征搜索与稳定性 |
| [walk_forward_validation.md](walk_forward_validation.md)                       | Walk-forward                            |
| [multi_seed_and_execution_stability.md](multi_seed_and_execution_stability.md) | 多 seed 与执行稳定性                    |
| [数据切分与look-ahead风险.md](数据切分与look-ahead风险.md)                     | 切分与泄漏风险                          |
| [Failure-first研究方法.md](Failure-first研究方法.md)                           | Failure-first                           |
| [FAILURE_TO_RETURN_PIPELINE.md](FAILURE_TO_RETURN_PIPELINE.md)                 | Failure→Return 管线                     |

---

## Live 执行 / multileg

| 文档                                                           | 说明                                                                        |
| -------------------------------------------------------------- | --------------------------------------------------------------------------- |
| [live_stream/multi_leg_live_daemon.md](live_stream/multi_leg_live_daemon.md) | C multileg 守护进程架构（Feature Bus、orchestrator、对账） |
| [live_stream/20260616_late_fill_infinite_loop_postmortem_CN.md](live_stream/20260616_late_fill_infinite_loop_postmortem_CN.md) | Jun 16 late-fill 事故复盘 |
| [account_safety_gate_CN.md](account_safety_gate_CN.md) | 账户级 Safety Gate：宪法 kill-switch 统一执行边界（TG / 禁开 / 可平） |
| [multi_leg_user_stream_design.md](multi_leg_user_stream_design.md) | Multi-leg **User-stream 现状 + 三层兜底 + 监控/并发 backlog** |
| [segment-lifecycle.md](segment-lifecycle.md)                   | C multileg **段生命周期专题**（ghost 根因 + P0–P4 实现 + 单测）；可观测性见 abc 文档 |
| [abc_execution_layer_issues_CN.md](abc_execution_layer_issues_CN.md) | ABC **执行层总账**：近期修复、TruthSync 术语、对账 metrics Phase 0–6、Live 观察 |
| [trend_position_state_and_truth_sync_CN.md](trend_position_state_and_truth_sync_CN.md) | **B·Trend 持仓三份状态**、2026-06 BNB 事故、补丁与 TrendPositionTruthSync 改进提案 |
| [backtest_vs_live_execution.md](backtest_vs_live_execution.md) | 回测与实盘执行差异（必读中包含）                                            |

---

## Evidence / Gate / 树与标签

| 文档                                                                 | 说明             |
| -------------------------------------------------------------------- | ---------------- |
| [EVIDENCE_ARCHITECTURE_V2.md](EVIDENCE_ARCHITECTURE_V2.md)           | Evidence 架构 v2 |
| [EVIDENCE_SCORING_ARCHITECTURE.md](EVIDENCE_SCORING_ARCHITECTURE.md) | Evidence 打分    |
| [OUTCOME_BASED_TREE_LABELING.md](OUTCOME_BASED_TREE_LABELING.md)     | 结果导向树标签   |
| [树模型规则导出与维护方法.md](树模型规则导出与维护方法.md)           | 规则导出与维护   |
| [规则重要性分析_vs_特征组搜索.md](规则重要性分析_vs_特征组搜索.md)   | 规则 vs 特征组   |
| [INTERACTION_SCREENING.md](INTERACTION_SCREENING.md)                 | 交互筛选         |
| [策略中dir的使用方式.md](策略中dir的使用方式.md)                     | dir 字段         |
| [标签在规则类的作用.md](标签在规则类的作用.md)                       | 标签与规则类     |

---

## 特征与执行细节

| 文档                                                                           | 说明                      |
| ------------------------------------------------------------------------------ | ------------------------- |
| [math_feature_separation_principle.md](math_feature_separation_principle.md)   | 数学特征分离原则          |
| [数学特征如何使用.md](数学特征如何使用.md)                                     | 数学特征使用              |
| [VOLUME_PROFILE_WPT_BOUNDARY_DESIGN.md](VOLUME_PROFILE_WPT_BOUNDARY_DESIGN.md) | Volume Profile / WPT 边界 |
| [订单流特征聚合频率的选择.md](订单流特征聚合频率的选择.md)                     | 订单流聚合频率            |
| [MAX_HOLDING_BARS_DESIGN_DECISION.md](MAX_HOLDING_BARS_DESIGN_DECISION.md)     | 最长持仓 bar 设计         |
| [币安交易成本计算.md](币安交易成本计算.md)                                     | 手续费与成本              |

---

## 策略与组合

| 文档                                                         | 说明                                      |
| ------------------------------------------------------------ | ----------------------------------------- |
| [BPC_FER_互补对冲设计.md](BPC_FER_互补对冲设计.md)           | BPC 与 FER 互补                           |
| [6种对称策略的启发式规则.md](6种对称策略的启发式规则.md)     | 对称策略启发式（与 archetype 设计对照用） |
| [archetype特征语义约束规范.md](archetype特征语义约束规范.md) | Archetype 特征语义                        |
| [最优参数寻找方法.md](最优参数寻找方法.md)                   | 参数搜索方法                              |

---

## 非平稳与方法论（仍偏工程）

| 文档                                                                                     | 说明                               |
| ---------------------------------------------------------------------------------------- | ---------------------------------- |
| [P5_非平稳性防护方案.md](P5_非平稳性防护方案.md)                                         | Regime/OOD/Alpha decay 方案        |
| [WHY_STATISTICAL_RULES_OVER_E2E_MODEL_CN.md](WHY_STATISTICAL_RULES_OVER_E2E_MODEL_CN.md) | 统计规则 vs 端到端模型（设计取舍） |

---

## 工程指南（`guides/` 子目录）

可操作工作流与调参协议见 **[guides/README.md](guides/README.md)**；以下为快速入口。

### 工作流与 Plateau

| 文档                                                                                             | 说明                                    |
| ------------------------------------------------------------------------------------------------ | --------------------------------------- |
| [guides/THRESHOLD_PLATEAU_TUNING_PROTOCOL_CN.md](guides/THRESHOLD_PLATEAU_TUNING_PROTOCOL_CN.md) | 阈值平坦高原协议                        |
| [guides/PLATEAU_OPTIMIZATION_METHODOLOGY.md](guides/PLATEAU_OPTIMIZATION_METHODOLOGY.md)         | Plateau 方法论                          |
| [guides/PLATEAU_OPTIMIZATION_WORKFLOW.md](guides/PLATEAU_OPTIMIZATION_WORKFLOW.md)               | Gate 高原优化工作流                     |
| [guides/BASELINE_TESTING_WORKFLOW.md](guides/BASELINE_TESTING_WORKFLOW.md)                       | 基线测试                                |
| [guides/PRODUCTION_ATTRIBUTION_WORKFLOW.md](guides/PRODUCTION_ATTRIBUTION_WORKFLOW.md)           | 实盘归因                                |
| [guides/RD_TO_LIVE_TIERED_WORKFLOW_V1_CN.md](guides/RD_TO_LIVE_TIERED_WORKFLOW_V1_CN.md)         | 研发→上线分层（Tier×Universe×TaskSpec） |

### Gate / BPC / 特征

| 文档                                                                                                               | 说明                       |
| ------------------------------------------------------------------------------------------------------------------ | -------------------------- |
| [guides/GATE_WHEN_THEN_EXECUTION_ORDER.md](guides/GATE_WHEN_THEN_EXECUTION_ORDER.md)                               | Gate when/then 与执行顺序  |
| [guides/HARD_GATE_SYSTEM.md](guides/HARD_GATE_SYSTEM.md)                                                           | Hard-Gate 协议             |
| [guides/MULTI_OBJECTIVE_GATE_OPTIMIZATION.md](guides/MULTI_OBJECTIVE_GATE_OPTIMIZATION.md)                         | 多目标 Gate 优化           |
| [guides/GATE_OPTIMIZATION_FEATURESTORE_IMPLEMENTATION.md](guides/GATE_OPTIMIZATION_FEATURESTORE_IMPLEMENTATION.md) | Gate×FeatureStore 实现说明 |
| [guides/GATE_OPTIMIZATION_FEATURESTORE_USAGE.md](guides/GATE_OPTIMIZATION_FEATURESTORE_USAGE.md)                   | FeatureStore 使用          |
| [guides/BPC_ADD_POSITION_LEVERAGE_ATR_NOTE.md](guides/BPC_ADD_POSITION_LEVERAGE_ATR_NOTE.md)                       | BPC 加仓与 ATR 注记        |
| [guides/FEATURE_COMPLEXITY_LAYERS_CN.md](guides/FEATURE_COMPLEXITY_LAYERS_CN.md)                                   | 特征复杂度分层             |
| [guides/FEATURE_PIPELINE_EXCLUDE_COLUMNS_CN.md](guides/FEATURE_PIPELINE_EXCLUDE_COLUMNS_CN.md)                     | `exclude_columns` 机制     |
| [guides/TREE_TRAINING_DATA_AND_CACHE.md](guides/TREE_TRAINING_DATA_AND_CACHE.md)                                   | 树训练数据与缓存           |

### 树模型 / Pool-B（已归档）

树模型 MVP、特征组搜索预设/调参、研究 Playbook、`ModelArtifact` 等已从 **`guides/tree/`** 迁至 **[docs/archive/guides/tree/](../archive/guides/tree/)**（主线已转向 BPC / 规则类；树路径仅作历史参考）。索引见 [archive/guides/README.md](../archive/guides/README.md)。

原 **`docs/guides/`** 仅保留 [占位索引](../guides/README.md)；历史 Gate 状态/截面 pipeline 等见 **[docs/archive/guides/](../archive/guides/README.md)**。

---

## 指标、报告、实盘、策略与部署

| 主题                                | 入口                                                                                                                                                        |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 指标与评估                          | [metrics/README.md](metrics/README.md)                                                                                                                      |
| 树模型策略报告（特征筛选 / 归一化） | [树模型策略report/README.md](树模型策略report/README.md)                                                                                                    |
| 实时流与实盘                        | [live_stream/README.md](live_stream/README.md)                                                                                                              |
| 策略 Playbook 与协议                | [strategies/README.md](strategies/README.md)                                                                                                                |
| 生产部署（Docker + GitHub Actions） | [.github/workflows/deploy.yml](../../.github/workflows/deploy.yml)、[deployment/LIVE_PRODUCTION_RUNBOOK_CN.md](../deployment/LIVE_PRODUCTION_RUNBOOK_CN.md) |

原 **`docs/metrics/`、`docs/reports/`、`docs/live_stream/`、`docs/strategies/`** 仅保留 [占位索引](../metrics/README.md) 等；过程性策略笔记与 legacy 实盘分析见 **[docs/archive/strategies/](../archive/strategies/README.md)**、**[docs/archive/live_stream/](../archive/live_stream/README.md)**、**[docs/archive/reports/](../archive/reports/README.md)**。

---

## 相关归档（NN 多头、旧索引等）

- [docs/archive/](../archive/)：NNMULTIHEAD、旧架构长文、横截面实验代码与说明等  
- [docs/archive/architecture/](../archive/architecture/)：已从本目录迁出的随笔与长文研究  
- [docs/archive/guides/](../archive/guides/)：已从 `docs/guides/` 迁出的历史与专项指南  
