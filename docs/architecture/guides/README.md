# 工程指南（`docs/architecture/guides/`）

本目录存放**可执行工作流、调参协议与实现注记**；与概念性长文区分，优先当作「按步骤做事」的入口。

**上级索引**：[架构文档 README — 工程指南表格](../README.md#工程指南guides-子目录)  
**归档**：树模型 / Pool-B 等已从 `guides/tree/` 迁至 [docs/archive/guides/tree/](../../archive/guides/tree/) · [archive/guides/README.md](../../archive/guides/README.md)  
**占位索引**：原 `docs/guides/` → [docs/guides/README.md](../../guides/README.md)

---

## 文档一览（本目录）

| 文档 | 说明 |
|------|------|
| [BASELINE_TESTING_WORKFLOW.md](./BASELINE_TESTING_WORKFLOW.md) | 基线 KPI / `mlbot gate` + `diagnose e2e-kpi` + 一键脚本 |
| [PRODUCTION_ATTRIBUTION_WORKFLOW.md](./PRODUCTION_ATTRIBUTION_WORKFLOW.md) | 实盘或滚动窗相对基线的退化检测与分层诊断入口 |
| [RD_TO_LIVE_TIERED_WORKFLOW_V1_CN.md](./RD_TO_LIVE_TIERED_WORKFLOW_V1_CN.md) | 研发→上线：Tier × Universe × TaskSpec 最短命令路径 |
| [PLATEAU_OPTIMIZATION_METHODOLOGY.md](./PLATEAU_OPTIMIZATION_METHODOLOGY.md) | Plateau 自由度、两阶段（锚点→高原）方法论 |
| [PLATEAU_OPTIMIZATION_WORKFLOW.md](./PLATEAU_OPTIMIZATION_WORKFLOW.md) | 高原优化**当前推荐**：`optimize_gate_unified.py`；`mlbot optimize gate-plateau` 状态说明 |
| [THRESHOLD_PLATEAU_TUNING_PROTOCOL_CN.md](./THRESHOLD_PLATEAU_TUNING_PROTOCOL_CN.md) | 阈值平坦高原长协议（含 legacy 脚本路径注意） |
| [GATE_WHEN_THEN_EXECUTION_ORDER.md](./GATE_WHEN_THEN_EXECUTION_ORDER.md) | Gate when/then：`tree_gate` 五段式 vs 实盘 `loader.apply_gate` 双路径 |
| [HARD_GATE_SYSTEM.md](./HARD_GATE_SYSTEM.md) | Hard-Gate 语义与统一优化入口 |
| [MULTI_OBJECTIVE_GATE_OPTIMIZATION.md](./MULTI_OBJECTIVE_GATE_OPTIMIZATION.md) | 多目标与稳健性权衡（与统一脚本对照） |
| [GATE_OPTIMIZATION_FEATURESTORE_USAGE.md](./GATE_OPTIMIZATION_FEATURESTORE_USAGE.md) | Gate×FeatureStore 使用（历史脚本名已标注） |
| [GATE_OPTIMIZATION_FEATURESTORE_IMPLEMENTATION.md](./GATE_OPTIMIZATION_FEATURESTORE_IMPLEMENTATION.md) | 同上实现归档说明 |
| [BPC_ADD_POSITION_LEVERAGE_ATR_NOTE.md](./BPC_ADD_POSITION_LEVERAGE_ATR_NOTE.md) | BPC 加仓、`fixed_multiplier` / `target_leverage_gap` 与杠杆直觉 |
| [FEATURE_COMPLEXITY_LAYERS_CN.md](./FEATURE_COMPLEXITY_LAYERS_CN.md) | 特征复杂度 Tier 约定 |
| [FEATURE_PIPELINE_EXCLUDE_COLUMNS_CN.md](./FEATURE_PIPELINE_EXCLUDE_COLUMNS_CN.md) | `exclude_columns` 与模型输入列 |
| [TREE_TRAINING_DATA_AND_CACHE.md](./TREE_TRAINING_DATA_AND_CACHE.md) | 树训练：DataHandler / FeatureStore / `cache/features` |
| [tree/README.md](./tree/README.md) | 子目录占位：树相关已迁 archive |

**事件回测（与实盘对齐）** 主文档在 sibling 目录：[event_drive_backtest/](../event_drive_backtest/)（`scripts/event_backtest.py`）。

---

## 与当前代码一致（维护说明）

1. **Gate 高原**：可跑入口为 **`python scripts/optimize_gate_unified.py --help`**。`mlbot optimize gate-plateau` / `gate-plateau-all` / `mlbot rule optimize-gate-plateau` 仍指向缺失的 **`scripts/optimize_gate_plateau.py`**，修复前勿依赖。  
2. **归因**：`mlbot diagnose outcome-attribution` 依赖的 **`scripts/diagnose_outcome_attribution.py` 当前缺失**；详见 [PRODUCTION_ATTRIBUTION_WORKFLOW.md](./PRODUCTION_ATTRIBUTION_WORKFLOW.md)。  
3. **`production-attribution` 告警 JSON**：`trade_count_drop` 为相对基线的比例（缩量时为负），建议显式传参，勿混用 CLI 默认字符串中的正数默认值与脚本语义。  
4. **归档链接**：TaskSpec 长文、树模型角色等见 **`docs/archive/`**，各 guide 内链接已逐步改为归档路径。
