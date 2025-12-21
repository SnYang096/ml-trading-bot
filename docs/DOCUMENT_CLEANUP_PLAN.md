# 文档清理计划

## 概述

本文档列出需要清理的过时和重复文档，以及保留的核心文档结构。

## 核心文档结构

### 必须保留的核心文档

```
docs/
├── ARCHITECTURE.md                    # 系统架构文档（新创建）
├── DEVELOPMENT_WORKFLOW.md            # 研发流程指南（新创建）
├── DEPLOYMENT_WORKFLOW.md             # 上线流程指南（新创建）
├── TEST_COVERAGE.md                   # 测试覆盖总结（新创建）
├── README.md                          # 项目主 README（根目录）
│
├── features/                          # 特征使用指南
│   ├── liquidity_void_price_impact_guide.md    # Price Impact 指南（新创建）
│   ├── lvn_improvements.md                     # LVN 改进说明（新创建）
│   ├── wpt_enhancements.md                     # WPT 增强说明（新创建）
│   ├── 特征计算流程改进总结.md                  # 保留：流程说明
│   └── ...（其他特征相关文档）
│
└── 时序模型/                          # 时序模型相关
    ├── 完整流程指南.md                # 保留：详细流程
    └── ...（其他时序模型文档）
```

## 建议归档的文档

### 1. 问题分析和修复文档（可归档）

这些文档记录了历史问题，对当前开发参考价值有限：

```
docs/
├── 重复列名根本原因分析.md
├── 重复列名问题分析.md
├── 已存在列复用方案.md
├── POC_HAL独立特征重构说明.md
├── feature_dependency_improvement_proposal.md
├── feature_pipeline_refactor.md
└── baseline_pure_migration_audit.md
```

**建议**: 移动到 `docs/archive/` 目录

### 2. 测试总结文档（可合并）

多个测试总结文档可以合并：

```
docs/
├── 测试总结.md
├── 测试准备完成总结.md
├── 关键特征测试完成总结.md
├── 复杂特征测试完成总结.md
├── 特征测试覆盖总结.md
└── tests/（测试相关文档）
```

**建议**: 保留最新的总结，其他归档

### 3. 策略优化文档（保留但整理）

这些文档记录了策略优化过程，应该保留但可以整理：

```
docs/策略优化/
├── Optuna优化实现总结.md
├── Optuna优化设计说明.md
├── 数据泄漏问题修复说明.md
└── ...（其他优化相关文档）
```

**建议**: 保留，但标记日期和版本

### 4. 临时分析文档（可归档）

临时的问题分析和诊断文档：

```
docs/
├── INF_ROOT_CAUSE_ANALYSIS.md
├── INF_ROOT_CAUSE_FIXES.md
├── INF_VALUE_FIXES.md
└── features/INF_*.md（INF 相关问题文档）
```

**建议**: 移动到 `docs/archive/`，如果问题已解决

### 5. Volume Profile 合并文档（可归档）

Volume Profile 相关的合并和对比文档：

```
docs/
├── VOLUME_PROFILE_COMPARISON.md
├── VOLUME_PROFILE_CONSOLIDATION_COMPLETE.md
├── VOLUME_PROFILE_CONSOLIDATION_SUMMARY.md
└── VOLUME_PROFILE_MERGE_SUMMARY.md
```

**建议**: 保留最新的总结文档，其他归档

## 文档清理步骤

### 步骤 1: 创建 archive 目录

```bash
mkdir -p docs/archive/{problems,refactoring,test_summaries}
```

### 步骤 2: 移动过时文档

```bash
# 问题分析文档
mv docs/重复列名*.md docs/archive/problems/
mv docs/已存在列复用方案.md docs/archive/problems/
mv docs/INF_*.md docs/archive/problems/
mv docs/features/INF_*.md docs/archive/problems/

# 重构相关文档
mv docs/POC_HAL独立特征重构说明.md docs/archive/refactoring/
mv docs/feature_*.md docs/archive/refactoring/
mv docs/baseline_pure_migration_audit.md docs/archive/refactoring/

# 测试总结文档（保留最新的）
# 移动旧的测试总结到 archive
```

### 步骤 3: 更新文档索引

在 `docs/README.md` 或主 README 中更新文档链接。

## 文档维护原则

1. **核心文档**: 保持最新，定期更新
2. **历史文档**: 归档但保留，便于追溯
3. **临时文档**: 问题解决后归档
4. **重复文档**: 合并或删除重复内容

## 注意事项

- 删除文档前先确认是否有其他地方引用
- 重要信息提取到核心文档后再删除
- 归档文档保留 git 历史，便于查找

