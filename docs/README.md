# 项目文档索引

**最后更新**: 2026-04-07  
**状态**: ✅ 当前版本

> 本文档是项目文档的统一入口，帮助您快速找到所需信息。

## 📚 文档导航

### 🚀 快速开始

- **[README.md](../README.md)** - 英文版快速开始（面向国际用户）
- **[README_CN.md](../README_CN.md)** - 中文版快速开始（面向中文用户）
- **[工作流文档](workflow/PIPELINE_WORKFLOW.md)** - 完整工作流命令序列

### 🏗️ 系统架构

#### 核心架构文档

- **[系统架构（统一版）](ARCHITECTURE.md)** ✅ 当前版本
  - 系统分层与职责边界
  - v0/v1/v2 哲学划分
  - 端到端 Pipeline 统一路径
  
- **[最终简化架构（2026-01）](archive/leagcy/FINAL_SIMPLIFIED_ARCHITECTURE_2026_01.md)**（归档）
  - 工程收敛状态说明
  - 从树模型到分层架构的统一
  - 归因能力与设计目标

> **说明**: 主架构见 `ARCHITECTURE.md`；`FINAL_SIMPLIFIED_ARCHITECTURE_2026_01.md` 已迁入 `docs/archive/leagcy/`，作历史参考。

#### 架构专题文档

- **[工业化实验循环](architecture/EXPERIMENT_LOOP_ARCHITECTURE.md)** - Layer A/B/C、TaskSpec、Filter→Wrapper
- **[NN多头路径原语架构](archive/leagcy/架构：NN多头路径原语（Path Primitives）+Router解耦升级.md)**（归档）- Path Primitives + Router 解耦
- **[Archetype架构](archive/leagcy/ARCHETYPE_BASED_ARCHITECTURE_2026_01.md)**（归档）- 基于 Archetype 的架构设计
- **[特征目录](archive/leagcy/FEATURE_CATALOG.md)**（归档）- 全部特征列表与归一化状态
- **[特征归一化策略](archive/leagcy/FEATURE_NORMALIZATION_POLICY.md)**（归档）- 归一化实现进度与方法
- **[NN多资产系统设计](archive/leagcy/NN_MULTI_ASSET_CONSTITUTIONAL_SYSTEM_DESIGN_CN.md)**（归档）- Task/Router/Gate/Execution 宪法设计
- **[架构升级V1](archive/leagcy/ARCH_UPGRADE_TASKSPEC_CONSTITUTION_V1_CN.md)**（归档）- TaskSpec + Constitution + PCM

### 📖 使用指南

#### 核心工作流

- **[上线MVP闭环](guides/tree/DEPLOYMENT_MVP_WORKFLOW_CN.md)** ⭐ 最重要，先看这个
  - Pool-B + 语义组搜索 → 6个月holdout验收 → 训练最终模型
  
- **[基线测试工作流](guides/BASELINE_TESTING_WORKFLOW.md)** - 建立各archetype性能基准
- **[平坦高原优化工作流](guides/PLATEAU_OPTIMIZATION_WORKFLOW.md)** - Gate规则参数优化方法
- **[实盘归因工作流](guides/PRODUCTION_ATTRIBUTION_WORKFLOW.md)** - 分层诊断和上线评估

#### 特征与模型

- **[特征搜索Playbook](strategies/FEATURE_SEARCH_PLAYBOOK.md)** - Pool-B + 语义组搜索详细说明
- **[NNMULTIHEAD配置文件](archive/NNMULTIHEAD_CONFIG_FILES_CN.md)**（归档）- TaskSpec/FeaturePlan 配置说明
- **[NNMULTIHEAD命令总览](archive/NNMULTIHEAD_COMMANDS_CN.md)**（归档）- 含 RL/BC/FSM 说明
- **[NNMULTIHEAD 3-action E2E](archive/NNMULTIHEAD_3ACTION_E2E_CN.md)**（归档）- 完整端到端流程
- **TaskSpec 示例**: `config/tasks/minimal_path_primitives_task_spec.yaml`（仓库内可引用路径）

#### 阈值与优化

- **[阈值平坦高原协议](guides/THRESHOLD_PLATEAU_TUNING_PROTOCOL_CN.md)** - Router/SLTP通用调参方法
- **[特征复杂度分层](guides/FEATURE_COMPLEXITY_LAYERS_CN.md)** - 先易后难、逐层解锁

### 🔬 实验报告

- **[实验索引](experiments/README.md)** - 实验报告入口与按主题导航（最后更新: 2026-04-07）
- **[实验结论导读](experiments/EXPERIMENTS_SUMMARY.md)** - 保留报告摘要与删除说明

#### 重要实验

- **[FR/ET优化实验](experiments/EXP_FR_ET_MEAN_REGIME_OPTIMIZATION_V2_2026_01.md)** - FR/ET 与 MEAN_REGIME（物理特征修复后）
- **[Regime Gate对比](experiments/EXP_REGIME_GATE_COMPARISON_V2_2026_01.md)** - Regime和Gate重要性分析
- **[实验结论汇总](experiments/EXPERIMENTS_CONCLUSIONS_2026_01.md)** - 实验总结

### 🔴 实盘相关

- **[实时流计算入口](live_stream/README.md)** - 事件回测与实盘数据流
  - 一致性原则与契约
  - 事件流与时间对齐
  - 特征计算、存储、补全、对账
  - 实盘稳定性运行手册

### 🧪 测试文档

- **[特征测试设计](tests/FEATURE_TEST_DESIGN_AND_COVERAGE_CN.md)** - 4类测试 + 覆盖快照
- **[Reflexivity测试](tests/REFLEXIVITY_ET_EFFECTIVENESS_TESTS.md)** - Reflexivity有效性测试

### 📋 策略文档

- **[树模型策略区分机制](strategies/树模型策略区分机制.md)** - 策略区分方法
- **[树策略导出规则](strategies/树策略导出的可泛化规则.md)** - 可泛化规则模板
- **[从标签设计到架构迁移](strategies/从标签设计到架构迁移的完整逻辑.md)** - 完整迁移逻辑

## 📂 文档目录结构

```
docs/
├── README.md                    # 本文档（主索引）
├── ARCHITECTURE.md              # 系统架构（统一版）✅
│
├── architecture/                # 架构文档（当前专题）
│   ├── EXPERIMENT_LOOP_ARCHITECTURE.md           # 实验循环架构
│   └── [其他架构专题文档]
├── archive/leagcy/              # 历史架构长文（FEATURE_CATALOG / FINAL_SIMPLIFIED_* 等）
│
├── guides/                      # 使用指南
│   ├── tree/DEPLOYMENT_MVP_WORKFLOW_CN.md      # MVP工作流 ⭐
│   ├── BASELINE_TESTING_WORKFLOW.md             # 基线测试
│   ├── PLATEAU_OPTIMIZATION_WORKFLOW.md         # 高原优化
│   └── [其他指南]
│
├── experiments/                 # 实验报告
│   ├── README.md                                 # 实验索引
│   └── [实验报告文件]
│
├── live_stream/                 # 实盘相关
│   ├── README.md                                 # 实时流入口
│   └── [实盘文档]
│
├── workflow/                    # 工作流文档
│   └── PIPELINE_WORKFLOW.md                      # 完整工作流
│
├── strategies/                  # 策略文档
│   └── [策略相关文档]
│
├── tests/                       # 测试文档
│   └── [测试相关文档]
│
└── archive/                     # 归档目录（历史文档）
    └── [已归档的过时文档]
```

## 🎯 按用户角色导航

### 👤 新手用户

1. 阅读 **[README_CN.md](../README_CN.md)** 或 **[README.md](../README.md)**
2. 了解 **[上线MVP闭环](guides/tree/DEPLOYMENT_MVP_WORKFLOW_CN.md)**
3. 参考 **[工作流文档](workflow/PIPELINE_WORKFLOW.md)** 执行命令

### 👨‍💻 开发者

1. 阅读 **[系统架构（统一版）](ARCHITECTURE.md)** 了解系统设计
2. 查看 **[最终简化架构（归档）](archive/leagcy/FINAL_SIMPLIFIED_ARCHITECTURE_2026_01.md)** 了解历史设计细节
3. 参考 **[特征目录（归档）](archive/leagcy/FEATURE_CATALOG.md)** 和 **[归一化策略（归档）](archive/leagcy/FEATURE_NORMALIZATION_POLICY.md)**
4. 查看 **[测试文档](tests/)** 了解测试方法

### 🔬 研究者

1. 阅读 **[实验结论导读](experiments/EXPERIMENTS_SUMMARY.md)** 与 **[实验索引](experiments/README.md)**
2. 参考 **[特征搜索Playbook](strategies/FEATURE_SEARCH_PLAYBOOK.md)** 进行特征研究
3. 查看 **[实验结论汇总](experiments/EXPERIMENTS_CONCLUSIONS_2026_01.md)** 了解研究进展

### 🚀 实盘部署

1. 阅读 **[实时流计算入口](live_stream/README.md)** 了解实盘架构
2. 参考 **[实盘归因工作流](guides/PRODUCTION_ATTRIBUTION_WORKFLOW.md)** 进行上线评估
3. 查看 **[Archetype上线前Checklist（归档）](archive/leagcy/ARCHETYPE_PRELIVE_CHECKLIST_CN.md)**

## 📌 文档状态说明

文档顶部可能包含以下状态标记：

- **✅ 当前版本** - 当前使用的权威文档
- **⚠️ 已过时** - 已废弃，请查看新文档
- **📝 草稿** - 正在编写中，内容可能不完整
- **🔄 待更新** - 需要更新，但当前仍可使用

## 🔗 重要链接

- **项目主页**: [README.md](../README.md) / [README_CN.md](../README_CN.md)
- **系统架构**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **工作流**: [workflow/PIPELINE_WORKFLOW.md](workflow/PIPELINE_WORKFLOW.md)
- **实盘入口**: [live_stream/README.md](live_stream/README.md)
- **实验索引**: [experiments/README.md](experiments/README.md)

## 📝 文档维护

- 文档最后更新日期标注在文档顶部
- 过时文档会标记为 ⚠️ 并指向新文档
- 历史文档归档到 `docs/archive/` 目录

---

**提示**: 如果您发现文档过时或存在矛盾，请提交Issue或PR。
