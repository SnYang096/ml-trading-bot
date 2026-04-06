# 归档：历史与专项指南

**说明**：此处文档多为一段时间内的**状态快照、实验总结、非主线 pipeline（如截面）或已边缘化的分析笔记**，不再放在 `docs/guides/` 根下。

**当前推荐使用的指南**请见：

- [docs/architecture/guides/](../../architecture/guides/)（工程工作流、Plateau、Gate、BPC 等；**不含**树模型专题）
- [docs/guides/README.md](../../guides/README.md)（迁移说明）

## 本目录

| 文件 | 备注 |
|------|------|
| `GATE_OPTIMIZATION_STATUS.md` | Gate 优化实现状态（时点快照） |
| `GATE_OPTIMIZATION_FIXES_SUMMARY.md` | 修复记录汇总 |
| `GATE_OPTIMIZATION_EXPERIMENTS_SUMMARY.md` | 实验总结 |
| `HIGHCAP6_COMPRESSION_OPTIMIZATION_CN.md` | HighCap6 压缩模式专项 |
| `LIVE_TRADING_ROADMAP_MULTI_ASSET_CN.md` | 多资产实盘路线图（长文随笔） |

## `tree/`（树模型指南，已归档）

> 主线叙述以 BPC / 规则类为主时，下列树模型流程文档**仅作历史与对照**，不再放在 `docs/architecture/guides/tree/`。

| 文件 | 备注 |
|------|------|
| `DEPLOYMENT_MVP_WORKFLOW_CN.md` | 树模型上线 MVP 闭环（holdout / rolling / Nautilus） |
| `RESEARCH_PLAYBOOK_CN.md` | 研究侧说明（标签、timeframe、仓位等） |
| `FEATURE_GROUP_SEARCH_PRESETS_CN.md` | Pool-B + 语义组搜索 A/B/C 预设 |
| `FEATURE_GROUP_SEARCH_TUNING_GUIDE_CN.md` | feature-group-search / pipeline 调参 |
| `MODEL_ARTIFACT_USAGE.md` | ModelArtifact 训练与部署 |
| `INVERT_FEATURES_SEARCH_ANALYSIS.md` | invert 特征搜索分析 |
| `POOLB_INVERT_FEATURES_CN.md` | Pool-B 反向特征说明 |
| `TREE_VS_NNMULTIHEAD_COMMANDS_CN.md` | 树 vs nnmultihead 命令对照（易随 CLI 过时） |

## `cs/`（截面 pipeline）

| 文件 | 备注 |
|------|------|
| `CROSS_SECTIONAL_PIPELINE_CN.md` | 截面 pipeline |
| `CROSS_SECTIONAL_WORKFLOW_END2END_CN.md` | 端到端工作流 |
| `CS_VS_TS_PIPELINE_CN.md` | 截面 vs 时序 |
