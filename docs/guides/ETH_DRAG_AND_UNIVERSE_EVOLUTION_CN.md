# ETH 拖累（Drag）处理与 Universe 演进：V1 交易 / V2 Shadow 监控（CN）

本文档回答两个核心问题：

- **我们现在发现 ETH 拖累，后面怎么处理？**
- **静态 YAML 分组（HighCap/Alt/Meme）是否“过时”？Version 2/3（动态分组/行为聚类）应该怎么引入才不把系统带沟里？**

> 结论先说：**你的理解大体正确**，但需要一个关键层级修正：  
> - **V1（静态 universe / 可回溯）可以且应该用于交易决策**  
> - **V2/V3（行为聚类/动态分组）现在可以跑，但必须先做 Shadow（旁路监控），不直接影响仓位/下单**  

---

## 1) 术语对齐：你在说的是“不同层级”的东西

### 1.1 V1：Research / Control Layer（研究与可控性层）

V1 的目标不是“永远自适应”，而是：

- **限制自由度**（变量少，才能归因）
- **可回溯、可复现**（同一套 universe/配置，未来可复盘）
- **出现回撤时能定位原因**（数据覆盖？特征？执行？阈值？单币异常？）

对应实现就是像 `config/download/crypto_4h_token_universe_groups.yaml` 这样的静态分组，以及固定的训练/评估窗口。

### 1.2 V2/V3：Production / Scaling / Regime-Aware Layer（生产与扩展层）

V2/V3 讨论的是：

- 市场行为漂移（drift）发生时，系统如何不“突然死掉”
- 如何引入 regime detector / 分组聚类 / 动态路由

**注意：这类机制一旦直接参与交易，会显著增加系统自由度与不可解释性。**

---

## 2) ETH “拖累”到底是什么？如何判断是真拖累还是假象？

### 2.1 先用“可证据化”的诊断定义拖累

我们建议把“ETH 拖累”拆成三个可量化问题（都能落盘）：

- **(A) ETH 单币指标明显劣后**：per-symbol Sharpe/DD/trade_rate 等显著差于组内平均  
- **(B) 组合对 ETH 敏感**：`portfolio_sharpe(with ETH) << portfolio_sharpe(without ETH)`  
- **(C) 行为差异**：ETH 的 mode 分布/换手/收益形态与组内其他币显著不同（例如 mode 塌缩、过度 MEAN 或过度 NO_TRADE）

项目里已存在相关实验产物（示例）：
- `docs/experiments/EXP_006_OOS_SINGLE_SYMBOL_EVAL_TOP10.md`
- `results/exp006_nnmh_highcap6_best_top10/e2e_top10/single_symbol_eval_rule_summary.csv`

### 2.2 先排除“数据不全导致的假拖累”

拖累在多币训练/评估里最常见的假象来源是：**不同 symbol 的 bars 覆盖不一致**。

你现在已经修复了：
- OOS（`2025-05..2025-10`）raw parquet + FeatureStore 覆盖
- Train（`2023-01..2025-04`）raw parquet + FeatureStore 覆盖

这一步非常关键：它把“ETH 拖累”从数据问题剥离出来，避免误判模型/执行。

---

## 3) V1 阶段：面对 ETH 拖累，推荐的低维护处理顺序（不要乱改 universe）

> 总原则：**优先在 Router/Execution 层解决（低自由度、可解释、可回滚），最后才动 universe / 动模型结构。**

### 3.1 Step 0：做一个 ETH Watchlist（只监控，不干预）

**Watchlist 的含义**：记录、告警、复盘入口，不直接触发交易动作。

建议固定输出（按月/按 OOS 窗口）：
- ETH 的 `rule_sharpe / dd / trade_rate / turnover`
- ETH mode 直方图（NO/MEAN/TREND）
- 组合 “leave-one-out” 指标（去掉 ETH 的变化量）

> 这一步不改变系统行为，但能把后续所有“改动”的收益贡献量化。

### 3.2 Step 1：优先加“低自由度”的 gating（推荐）

如果 ETH 的问题是“乱交易/错在某些状态”，先上 **gating v1**：

- 更严格的 tradeability（例如波动、流动性、极端行情过滤）
- cooldown（防止噪声连续触发）
- 更保守的开仓阈值（但只限 ETH 的 gating，而不是全局阈值）

对应 repo TODO（已存在）：
- `exp006_eth_gating_v1`

### 3.3 Step 2：再考虑 per-symbol RR profile（次选，控制自由度）

如果 gating 不够，才尝试 **ETH 专用 rr_execution profile**（例如最大持仓 bars、TP/SL 结构）。

对应 repo TODO（已存在）：
- `exp006_eth_rr_profile_v1`

> 建议尽量避免“每币一套 Router 阈值”，因为它自由度更大、可泛化更差、维护成本更高。

### 3.4 Step 3：最后才考虑“分组训练多个模型”

你提出的 “HighCap/Alt/Meme 三模型”在终局是合理方向，但在 V1 阶段要非常谨慎：

- **优点**：减少异质样本互相污染（heterogeneous contamination）
- **缺点**：模型数量上升，回测/上线/监控成本上升；且容易把“分组误差”引入决策链

推荐策略：
- **交易决策层**先用 gating/仓位宪法解决冲突  
- **训练侧**如果要做“三模型”，先以实验对照方式验证（同窗、同成本、同评估口径），不要直接上实盘决策链

---

## 4) V2 Shadow：行为聚类/动态分组“现在就能做”，但必须只做监控

你写的 V2 内容（行为特征、聚类、coherence、drift_score）本身大体是对的，关键是**定位**：

> V2 的正确身份是 **Market Structure Sensor（市场结构传感器）**，不是 Trading Decision Maker。

### 4.1 Shadow 的硬约束（必须满足）

- V2 **不能改仓位**
- V2 **不能切 universe**
- V2 **不能改变模型输入/选择哪个模型**
- V2 只允许：
  - 写 CSV/JSON
  - 画 drift 曲线
  - 产出“建议重分组”但仅用于复盘/对照实验

### 4.2 什么时候允许 V2 进入决策链？（证据链门槛）

只有当同时满足：

1. **V1 出现系统性退化**（多个窗口/多币都变差，而非单次噪声）
2. **V2 给出一致、可重复的 drift 信号**（不是一次聚类抖动）
3. **历史回放验证**：按 V2 分组或按 V2 regime 过滤后，指标显著改善，并在多个窗口成立

才允许把 V2 升级为 `version: 2` 的交易宇宙/分组。

---

## 5) “自由度限制”与“仓位宪法”：解决 ETH 抢仓位 / 信号冲突的工程答案

> 在多币系统里，最常见的不是“模型不够聪明”，而是“系统太容易自作主张”。  
> 所以需要 **Position Constitution（仓位宪法）**：用少量硬规则把自由度关在可解释的位置。

建议最低配就包含：

- **Per-symbol cap**：每个 symbol 最大风险预算（例如 ETH ≤ 20%）
- **Slots**：同一时间最多持有 N 个仓位（按 score 排序）
- **冲突裁决顺序**：信号冲突时如何淘汰、如何换仓
- **冷却/换手约束**：避免噪声期频繁切换

相关背景可参考：
- `docs/architecture/自由度限制.md`

---

## 6) 推荐的“下一步动作”（落地清单）

你现在已经完成了“数据覆盖修复”（Train + OOS）这一重大前置条件。下一步建议按以下顺序执行：

1. **重跑 Top10 全链路评估（数据修复后的口径）**  
   对应 TODO：`exp006_rerun_top10_eval_after_data_fix`
2. **针对 ETH 做 gating v1（低维护）并复验**  
   对应 TODO：`exp006_eth_gating_v1`
3. **若仍拖累，再做 ETH rr_profile v1**  
   对应 TODO：`exp006_eth_rr_profile_v1`
4. **并行跑 V2 Shadow（只监控）**：输出 drift/coherence 与 “建议重分组”（LOG ONLY）

---

## 7) 你那段材料里“对的点/混在一起的点”速查表

| 观点 | 是否正确 | 关键修正 |
| --- | --- | --- |
| V1 静态 YAML 分组是合理工程选择 | ✅ | 作为“可控/可回溯”基座非常正确 |
| 树模型对 regime 自适应弱 | ✅ | 但树模型的“僵化”也能成为优势（可解释/可治理） |
| NN 更平滑、更适合多任务表示 | ✅ | 但不代表可以把所有自由度交给 NN 决策 |
| 动态分组是终局方向 | ✅ | **现在先 shadow**，不要直接触达资金 |
| ETH 异常就立刻挪组 | ❌ | 先做证据链（多窗/回放）再升级 |
| “V2 现在就别搞” | ⚠️ | 更准确是：**V2 现在就可以跑，但只跑 shadow** |

