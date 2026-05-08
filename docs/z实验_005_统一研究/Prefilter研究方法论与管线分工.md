# Prefilter 研究方法论与管线分工（正式结论）

> **状态**：制度化结论；与 `slow_realistic` 的 adoption 护栏（语义修复、prefilter drift、`adoption_gate`）互补。  
> **维护**：若未来在仓库内增加「专用 prefilter 研究子流程 / 报告物 / CI 闸门」，请同步更新本文 **§4 缺口** 与 [INDEX](./INDEX_管线与文档导航.md) 导航。

---

## 1. 正式结论

以下内容作为本仓库研究策略的**明确立场**（非临时讨论）：

1. **Prefilter 特征与规则的定义**需要在**研究阶段**完成充分证据建设，不能仅依赖单次全样本调参或直觉。
2. 下列研究机制是**必要且工业上可规范化的**（可与业务经验结合使用，但不应被经验单独替代）：
   - **基于树模型等的特征发现**（从特征池到进入 prefilter 候选集）；
   - **严格时间切分下的验证**（walk-forward、holdout、避免信息泄漏的切分约定，与本仓库 `rolling` / `calibration_months` / `end-date` 因果性一致）；
   - **边际与联合视角**（在全集或已有规则集合上，单条或子集的增量贡献；与「只跑单变量显著性」区分）；
   - **多重比较与过拟合风险意识**（试错次数、并行假设、报告 deflated 指标或等价纪律，按团队章程执行）；
   - **合约化输出**：将结论固化为 `archetypes/prefilter.yaml` 中的语义（如 `locked`、`skip_parquet_tune`）与可漂移项的边界。
3. **经验的作用**是与上述机制对齐：决定哪些维度进入候选、风控上是否保留「冗余保险」规则、以及在非平稳环境下的解读；**不负责替代** OOS 与切片协议。

---

## 2. 与现有管线的分工

| 层级 | 典型职责 | 与 Prefilter 研究的关系 |
|------|-----------|-------------------------|
| **前置 / 慢研究管线**（全量、`pipeline run`、`SHAP_*`、长窗树训练等） | 特征池、结构、首轮发现与较重的统计验证 | **主要承载** §1 中的发现与验证；算力与时间预算相对宽松 |
| **`slow_realistic` 滚动** | 低频结构刷新 + 月频校准 + 采纳门禁 | **在节奏上压缩**重做全量发现的频率；依托 `adoption_gate`（含 semantic / prefilter drift）降低「静默分叉」风险；**不等于**每期重跑完整特征发现流水线 |
| **`turbo_fixed_features`** | 固定特征语义下的阈值与下游标定 | 假定 prefilter **合约已由研究阶段定义**；适合做「近似实盘滚动」对标 |

换言之：**厚重的「边际 / 联合 + 多重比较 + 时间切分」证据链**，默认归属**前置研究**；滚动 slow 侧重**在有护栏的前提下延续或微调**已采纳合约。

---

## 3. 与代码中已实现能力的关系（简述）

- 树模型、SHAP、`enable_model_training` 等能力与 **研究 / `slow` yaml** 可组合使用，语义上服务于 §2 表格第一行。
- **`rolling.slow_realistic.adoption_gate` 内的 prefilter drift guard** 服务于**阈值/数值漂移门禁**与采纳回退，**不替代** §1 中整条「发现 + 多维验证」研究流程。
- **`locked` / `skip_parquet_tune`** 是将研究结论写成**运维合约**的机制，应在研究定稿阶段与_yaml_一致。

---

## 4. 当前框架缺口与后续加强方向（记录）

以下内容承认现状，便于排期：**不阻塞** §1 结论生效，但需在后续迭代中加强。

| 方向 | 说明 |
|------|------|
| **专用研究阶段编排** | 将「重度特征发现 → 定型 prefilter → 产物审查 → 写入 archetypes」与 `rolling_sim` 月循环解耦 clearer，避免职责混淆 |
| **可复现证据物** | 标准目录下保存消融 / 子集对比、分段 OOS、（按章程）多重校正或 deflated 类摘要 |
| **与 drift guard 对齐** | 显式列出「漂移监控关注的非 locked 特征 / 绑定规则」，与研究阶段 watchlist 一致 |
| **自动化闸门（可选）** | CI 或发布 checklist 上对「未定稿大范围改 prefilter」做人工 + 脚本双签 |

---

## 5. 相关文档与配置入口

- 管线总索引：[INDEX_管线与文档导航.md](./INDEX_管线与文档导航.md)
- 长窗滚动设计（模式对照）：[archive/rolling_long_horizon_pipeline.md](./archive/rolling_long_horizon_pipeline.md)
- 策略侧 slow 示例（adoption 与 drift 配置）：`config/strategies/bpc/research/slow.yaml` 内 `rolling.slow_realistic.adoption_gate`
