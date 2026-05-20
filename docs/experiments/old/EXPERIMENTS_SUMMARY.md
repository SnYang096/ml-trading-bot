# 实验文档结论摘要（保留报告导读）

**更新**: 2026-04-07  
**范围**: `docs/experiments/` 下经整理后仍保留的报告；过程性/重复文档已移除（见文末说明）。

---

## 1. 仍值得记住的结论（按主题）

### 1.1 FR / ET / MEAN_REGIME（2025-05～2025-10 等窗口）

- **问题**：FR/ET 在粗配置下极差（如 Sharpe 约 -2.4）；MEAN_REGIME 一度几乎无样本；物理特征曾未正确进 gated 日志。
- **措施**：放宽 MEAN_REGIME 阈值、为 FR/ET 增加基于 path_efficiency / deviation_z 等的 gate 规则、修复物理特征合并与脚本链路。
- **结果（摘录）**：MEAN_REGIME 样本从极少增至约 27；在该子集上 FR/ET 可出现正收益与较高 Sharpe（详见 `EXPERIMENTS_CONCLUSIONS_2026_01.md`）。整体仍受 **样本量小**、symbol 异质性制约。
- **Regime vs Gate**：Regime 过滤对 Sharpe 贡献最大；Gate 与 Semantic veto 为补充与底线（见 `EXP_REGIME_GATE_IMPORTANCE_2026_01.md`、`EXP_REGIME_GATE_COMPARISON_V2_2026_01.md`、`EXP_SUMMARY_REGIME_GATE_2026_01.md`）。

### 1.2 ET 专项（2024 数据 + Volume Profile）

- **2024 最终结果**：在完整订单流与 ET_REGIME / Gate 调优后，ET 由大幅负 Sharpe 转为 **正 Sharpe（约 1.94）**、胜率约 **48%**，但全样本占比仍低（约 27 笔量级），且 **ETH/SOL 明显好于 XRP/ADA**（`EXP_ET_2024_FINAL_RESULTS_2026_01.md`）。
- **2025 切片**：同年方法论在 2025 数据上曾表现为负 Sharpe，说明 **时段敏感**（`EXP_ET_2025_ANALYSIS_2026_01.md`）。
- **为何需要 Volume Profile**：`et_near_lvn`、VPVR/LVN 距离等与 ET 语义强相关；`has_volume_profile` 为关键 evidence（`EXP_ET_VOLUME_PROFILE_EFFECTIVENESS_2026_01.md`）。
- **语义与 Regime 设计**：MEAN_REGIME 与 ET 需求不一致，长期方向是 **独立 ET_REGIME** 及互斥分类逻辑（`EXP_ET_RULES_SEMANTIC_ANALYSIS_2026_01.md`）。

### 1.3 FR Evidences × Regime

- **has_orderflow（vpin quantile）过高** 会导致全 Regime 下通过数为 0；下调 quantile 后可通过但全样本绩效仍偏负，需与 **MEAN_REGIME 等过滤** 联用（`EXP_FR_EVIDENCES_REGIME_OPTIMIZATION_RESULTS_2026_01.md`）。

### 1.4 Gate / Archetype / PCM 工程

- **Plateau / 规则迭代**：移除不适配 archetype、补充价格轨迹类 gate 规则等（`EXP_GATE_PLATEAU_OPTIMIZATION_2026_01.md`）；多 archetype 选择与增强对比见 `EXP_MULTI_ARCHETYPE_SELECTION_2026_01.md`、`EXP_GATE_ENHANCEMENT_COMPARISON_2026_01.md`。
- **PCM 以 archetype 为调度单元**：Slot/Candidate/Add-on 带 archetype、兼容性矩阵、ET 与 TC/TE 等冲突处理；与 ET 2024 测试同周期记录（`EXP_PCM_REDESIGN_AND_ET_TEST_2026_01.md`）。
- **可选块与 Regime 架构**：optional blocks、特征块与 regime 管线关系（`EXP_OPTIONAL_BLOCKS_AND_REGIME_ARCHITECTURE_2026_01.md`）。
- **Auto-detect 特征依赖**：`any_key_contains` 与 tier 自动补齐等（`EXP_AUTO_DETECT_COMPUTE_REQUIREMENTS_2026_01.md`）。

### 1.5 NN Multi-head / 协议 / 消融（编号实验）

- **EXP_001 / 003 / 006 / 007**：BTC/ETH 与多 token 评估、OOS、阈值调优、分组消融、**NNMH 评估协议**——仍作方法学与复现入口（文件名以 `EXP_00*` 开头）。

### 1.6 组合与资产

- `EXP_PORTFOLIO_ASSETS_ANALYSIS_2026_01.md`：portfolio_assets 与 router 依赖等梳理。

---

## 2. 保留文件一览（按文件名）

| 文件 | 用途简述 |
|------|-----------|
| `EXPERIMENTS_CONCLUSIONS_2026_01.md` | 2026-01 综合结论（FR/ET/Regime/Gate 等） |
| `EXPERIMENTS_SUMMARY.md` | 本文：导读 + 删除说明 |
| `EXP_SUMMARY_REGIME_GATE_2026_01.md` | Regime/Gate KPI 对比表 |
| `EXP_REGIME_GATE_IMPORTANCE_2026_01.md` | Regime vs Gate 重要性 |
| `EXP_REGIME_GATE_COMPARISON_V2_2026_01.md` | Regime/Gate 对比 v2 |
| `EXP_FR_ET_DISTRIBUTION_ANALYSIS_2026_01.md` | FR/ET 分布分析 |
| `EXP_FR_ET_MEAN_REGIME_OPTIMIZATION_V2_2026_01.md` | FR/ET + MEAN_REGIME 优化（含物理特征修复验证） |
| `EXP_FR_ET_EVIDENCES_PERFORMANCE_2026_01.md` | FR/ET evidences 表现 |
| `EXP_MEAN_REGIME_FR_ET_DEEP_ANALYSIS_2026_01.md` | MEAN_REGIME 深度分析 |
| `EXP_FR_EVIDENCES_REGIME_OPTIMIZATION_RESULTS_2026_01.md` | FR evidences × regime 优化**结果** |
| `EXP_ET_2024_FINAL_RESULTS_2026_01.md` | ET 2024 最终量化结果 |
| `EXP_ET_2025_ANALYSIS_2026_01.md` | ET 2025 数据分析 |
| `EXP_ET_VOLUME_PROFILE_EFFECTIVENESS_2026_01.md` | ET 与 Volume Profile 必要性 |
| `EXP_ET_RULES_SEMANTIC_ANALYSIS_2026_01.md` | ET 规则语义与 ET_REGIME 设计 |
| `EXP_PCM_REDESIGN_AND_ET_TEST_2026_01.md` | PCM 改造 + ET 测试摘要 |
| `EXP_GATE_PLATEAU_OPTIMIZATION_2026_01.md` | Gate plateau 实施记录 |
| `EXP_GATE_ENHANCEMENT_COMPARISON_2026_01.md` | Gate 增强对比 |
| `EXP_MULTI_ARCHETYPE_SELECTION_2026_01.md` | 多 archetype 选择 |
| `EXP_OPTIONAL_BLOCKS_AND_REGIME_ARCHITECTURE_2026_01.md` | 可选块与架构 |
| `EXP_AUTO_DETECT_COMPUTE_REQUIREMENTS_2026_01.md` | Auto-detect 需求 |
| `EXP_PORTFOLIO_ASSETS_ANALYSIS_2026_01.md` | 组合资产配置分析 |
| `EXP_001_NN_MULTIHEAD_BTC_ETH_2024.md` | NNMH 基线实验 |
| `EXP_003_EVAL_BTC_ETH_2025H2.md` | 2025H2 评估 |
| `EXP_003_MULTI_TOKEN_4Y.md` | 多 token 长期 |
| `EXP_006_GROUPED_VS_MIXED_ABLATION_TOP9.md` | 分组 vs 混合消融 |
| `EXP_006_OOS_SINGLE_SYMBOL_EVAL_TOP10.md` | OOS 单币种 |
| `EXP_006_RULE_THRESHOLD_TUNING_TOP9.md` | 规则阈值调优 |
| `EXP_007_NNMH_EVAL_PROTOCOL.md` | NNMH 评估协议 |

---

## 3. 已删除文档（为何删）

- **ET 过程稿**（约 12 篇）：数据可用性、测试状态、timestamp 修复、缺失诊断、优化「分析/实施/完成/最终状态」重复篇、v3 layer 某次目录检查快照、与最终结论文重复的中间测试报告等；**结论已合并理解读** `EXP_ET_2024_FINAL_RESULTS_2026_01.md` 与 `EXP_ET_RULES_SEMANTIC_ANALYSIS_2026_01.md`。
- **空壳汇总**（5 篇）：`GATE_ANALYSIS_SUMMARY.md`、`MEAN_REGIME_ANALYSIS_SUMMARY.md`、`REGIME_ANALYSIS_SUMMARY.md`、`EVIDENCE_ANALYSIS_SUMMARY.md`、`FR_ET_OPTIMIZATION_SUMMARY.md` — 仅列表无「关键发现」正文。
- **过时索引与操作说明**：`EXPERIMENTS_INDEX.md`（含不存在文件的链接）、`STATUS_SUMMARY_2026_01.md`（元状态）、`NEXT_STEPS_VPIN_FEATURE_2026_01.md`（命令与当前 CLI 不一致）、`FR_ET_OPTIMIZATION_TEST.md`（旧训练命令）。
- **FR evidences 重复报告**：`EXP_FR_EVIDENCES_REGIME_OPTIMIZATION_2026_01.md` 与 `..._RESULTS_2026_01.md` 内容重叠，保留 **RESULTS** 篇。

若需从 Git 历史恢复某篇过程稿，可使用 `git log -- docs/experiments/` 与 `git show <commit>:path` 找回。
