# 使用指南索引

**状态**: ✅ 当前版本  
**最后更新**: 2026-01-25  
**相关文档**: [主文档索引](../README.md)

## 核心工作流指南

### ⭐ 最重要（先看这个）

- **[上线MVP闭环](tree/DEPLOYMENT_MVP_WORKFLOW_CN.md)** ⭐ 最重要
  - Pool-B + 语义组搜索 → 6个月holdout验收 → 训练最终模型
  - rolling与Nautilus的职责边界

### 工作流指南

- **[基线测试工作流](BASELINE_TESTING_WORKFLOW.md)** - 建立各archetype性能基准
- **[平坦高原优化工作流](PLATEAU_OPTIMIZATION_WORKFLOW.md)** - Gate规则参数优化方法
- **[实盘归因工作流](PRODUCTION_ATTRIBUTION_WORKFLOW.md)** - 分层诊断和上线评估

## 特征与模型指南

### 特征搜索

- **[特征搜索Playbook](../strategies/FEATURE_SEARCH_PLAYBOOK.md)** - Pool-B + 语义组搜索详细说明
- **[特征复杂度分层](FEATURE_COMPLEXITY_LAYERS_CN.md)** - 先易后难、逐层解锁
- **[特征Pipeline排除列机制](FEATURE_PIPELINE_EXCLUDE_COLUMNS_CN.md)** - exclude_columns说明
- **[特征组搜索调参指南](FEATURE_GROUP_SEARCH_TUNING_GUIDE_CN.md)** - 调参方法

### NNMULTIHEAD

- **[NNMULTIHEAD配置文件](NNMULTIHEAD_CONFIG_FILES_CN.md)** - TaskSpec/FeaturePlan配置说明
- **[NNMULTIHEAD命令总览](NNMULTIHEAD_COMMANDS_CN.md)** - 含RL/BC/FSM说明
- **[NNMULTIHEAD 3-action E2E](NNMULTIHEAD_3ACTION_E2E_CN.md)** - 完整端到端流程
- **[NNMULTIHEAD Returns Source](NNMULTIHEAD_RETURNS_SOURCE_CN.md)** - 收益来源说明

### 阈值与优化

- **[Plateau优化方法论](PLATEAU_OPTIMIZATION_METHODOLOGY.md)** ⭐ **关键** - 为什么Plateau搜索慢 + 怎么改（核心：分层冻结 + 子空间搜索，不允许同时调节超过3个参数）
- **[阈值平坦高原协议](THRESHOLD_PLATEAU_TUNING_PROTOCOL_CN.md)** - Router/SLTP通用调参方法
- **[平坦高原优化工作流](PLATEAU_OPTIMIZATION_WORKFLOW.md)** - Gate规则参数优化方法
- **[Plateau vs Optuna对比](PLATEAU_VS_OPTUNA_COMPARISON.md)** - 两种优化方法对比
- **[Gate优化实验总结](GATE_OPTIMIZATION_EXPERIMENTS_SUMMARY.md)** - Gate优化实验
- **[多目标Gate优化](MULTI_OBJECTIVE_GATE_OPTIMIZATION.md)** - 多目标优化方法
- **[硬Gate系统](HARD_GATE_SYSTEM.md)** - 硬Gate设计

## 实盘与部署

- **[多资产合约实盘路线图](LIVE_TRADING_ROADMAP_MULTI_ASSET_CN.md)** - 从1w→10w的可执行路线
- **[实盘特征契约与证据](LIVE_FEATURE_CONTRACT_AND_EVIDENCE_CN.md)** - 实盘特征契约
- **[研发到上线分层工作流](RD_TO_LIVE_TIERED_WORKFLOW_V1_CN.md)** - Tier×Universe×TaskSpec

## 评估与诊断

- **[E2E KPI说明](E2E_KPI_EXPLANATION_CN.md)** - 端到端KPI指标说明
- **[执行日志Schema](EXECUTION_LOG_SCHEMA_CN.md)** - 执行日志结构

## Gate优化相关

- **[Gate优化状态](GATE_OPTIMIZATION_STATUS.md)** - Gate优化当前状态
- **[Gate优化修复总结](GATE_OPTIMIZATION_FIXES_SUMMARY.md)** - 修复记录
- **[Gate优化FeatureStore实现](GATE_OPTIMIZATION_FEATURESTORE_IMPLEMENTATION.md)** - FeatureStore实现
- **[Gate优化FeatureStore使用](GATE_OPTIMIZATION_FEATURESTORE_USAGE.md)** - FeatureStore使用

## 文档分类

### 按用途分类

- **快速开始**: DEPLOYMENT_MVP_WORKFLOW_CN.md
- **特征研究**: FEATURE_SEARCH_PLAYBOOK.md, FEATURE_COMPLEXITY_LAYERS_CN.md
- **模型训练**: NNMULTIHEAD相关文档
- **参数优化**: THRESHOLD_PLATEAU_TUNING_PROTOCOL_CN.md, PLATEAU_OPTIMIZATION_WORKFLOW.md
- **实盘部署**: LIVE_TRADING_ROADMAP_MULTI_ASSET_CN.md, PRODUCTION_ATTRIBUTION_WORKFLOW.md

### 按重要性分类

- **⭐ 必读**: DEPLOYMENT_MVP_WORKFLOW_CN.md
- **📖 推荐**: BASELINE_TESTING_WORKFLOW.md, PLATEAU_OPTIMIZATION_WORKFLOW.md, PRODUCTION_ATTRIBUTION_WORKFLOW.md
- **📚 参考**: 其他专题指南

## 相关文档

- [主文档索引](../README.md)
- [系统架构](../ARCHITECTURE.md)
- [工作流文档](../workflow/PIPELINE_WORKFLOW.md)
