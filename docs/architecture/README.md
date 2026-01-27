# 架构文档索引

**状态**: ✅ 当前版本  
**最后更新**: 2026-01-25  
**相关文档**: [主文档索引](../README.md)

## 核心架构文档

### 系统架构（统一版）

- **[系统架构（统一版）](../ARCHITECTURE.md)** ✅ 当前版本
  - 系统分层与职责边界
  - v0/v1/v2 哲学划分
  - 端到端 Pipeline 统一路径
  - Tree → NN 知识迁移

- **[最终简化架构（2026-01）](FINAL_SIMPLIFIED_ARCHITECTURE_2026_01.md)** ✅ 当前版本
  - 工程收敛状态说明
  - 从树模型到分层架构的统一
  - 归因能力与设计目标
  - 具体实现细节

> **说明**: 两个架构文档互补，建议都阅读：
> - `ARCHITECTURE.md`: 系统分层、职责边界、Pipeline组织（高层设计）
> - `FINAL_SIMPLIFIED_ARCHITECTURE_2026_01.md`: 具体实现、设计目标、问题解决（详细设计）

## 专题架构文档

### 实验循环与Pipeline

- **[工业化实验循环](EXPERIMENT_LOOP_ARCHITECTURE.md)** - Layer A/B/C、TaskSpec、Filter→Wrapper、稳定性规则

### 特征系统

- **[特征目录](FEATURE_CATALOG.md)** - 全部208个特征节点的归一化状态
- **[特征归一化策略](FEATURE_NORMALIZATION_POLICY.md)** - Phase 1/2/3归一化实现进度
- **[归一化契约与检查](NORMALIZATION_CONTRACT_AND_CHECKS.md)** - 归一化契约说明
- **[NNMULTIHEAD特征契约](NNMULTIHEAD_FEATURE_CONTRACT_BLOCK_GATING.md)** - 特征契约与Block Gating

### 系统设计

- **[NN多资产系统设计](NN_MULTI_ASSET_CONSTITUTIONAL_SYSTEM_DESIGN_CN.md)** - Task/Router/Gate/Execution宪法与运维设计
- **[架构升级V1](ARCH_UPGRADE_TASKSPEC_CONSTITUTION_V1_CN.md)** - TaskSpec + Constitution + PCM
- **[Archetype架构](ARCHETYPE_BASED_ARCHITECTURE_2026_01.md)** - 基于Archetype的架构设计
- **[NN多头路径原语架构](架构：NN多头路径原语（Path Primitives）+Router解耦升级.md)** - Path Primitives + Router解耦

### Regime与Gate

- **[为什么用规则做Regime判断](WHY_REGIME_RULES_OVER_NN_CN.md)** - Regime规则优于NN的原因
- **[Regime Filter + Trade Quality方案](REGIME_FILTER_TRADE_QUALITY_PLAN_CN.md)** - Regime过滤与交易质量
- **[Gate一组可组合的约束算子](Gate一组可组合的约束算子.md)** - Gate设计原理

### 策略与执行

- **[树模型策略知识迁移](树模型策略知识迁移到多头模型.md)** - Tree→NN迁移方法
- **[树模型在多头模型下游的角色](树模型在多头模型下游的角色.md)** - Tree模型定位
- **[FR策略中dir的使用方式](FR策略中dir的使用方式.md)** - FR策略方向判断

### 风险与生存

- **[Archetype灭绝级回测](archetype灭绝级回测.md)** - 压力测试→生存评分→Router/Size映射
- **[OOD头的训练](ood头的训练.md)** - OOD/Survival Head监督信号定义
- **[LiveDashboard](LiveDashboard.md)** - 只盯5个数，用于阻止系统犯蠢

### 设计哲学

- **[谁对sharp负责](谁对sharp负责.md)** - Sharpe责任归属
- **[职责坍缩](职责坍缩.md)** - 职责边界设计
- **[仓位管理办法](仓位管理办法.md)** - 仓位管理策略
- **[自由度限制](自由度限制.md)** - 自由度控制原则
- **[为什么延迟RL](WHY_RL_IS_DELAYED_CN.md)** - RL延迟原因

### 其他专题

- **[可选块语义与实现](NNMULTIHEAD_FEATURE_CONTRACT_BLOCK_GATING.md)** - Optional Blocks完整说明（语义、需求、实现）
- **[Archetype上线前Checklist](ARCHETYPE_PRELIVE_CHECKLIST_CN.md)** - 上线前检查清单
- **[6种archetype简化成4种的原因](6种archetype简化成4种的原因.md)** - Archetype简化逻辑

## 文档分类

### 按主题分类

- **系统设计**: ARCHITECTURE.md, FINAL_SIMPLIFIED_ARCHITECTURE_2026_01.md, NN_MULTI_ASSET_CONSTITUTIONAL_SYSTEM_DESIGN_CN.md
- **特征系统**: FEATURE_CATALOG.md, FEATURE_NORMALIZATION_POLICY.md, NORMALIZATION_CONTRACT_AND_CHECKS.md
- **策略与执行**: 树模型相关文档, FR策略相关文档
- **风险控制**: archetype灭绝级回测.md, ood头的训练.md, LiveDashboard.md
- **设计哲学**: 谁对sharp负责.md, 职责坍缩.md, 自由度限制.md

### 按重要性分类

- **⭐ 必读**: ARCHITECTURE.md, FINAL_SIMPLIFIED_ARCHITECTURE_2026_01.md
- **📖 推荐**: EXPERIMENT_LOOP_ARCHITECTURE.md, FEATURE_CATALOG.md, NN_MULTI_ASSET_CONSTITUTIONAL_SYSTEM_DESIGN_CN.md
- **📚 参考**: 其他专题文档

## 相关文档

- [主文档索引](../README.md)
- [工作流文档](../workflow/PIPELINE_WORKFLOW.md)
- [使用指南](../guides/)
