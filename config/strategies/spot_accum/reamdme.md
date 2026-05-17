# spot_accum — 设计思想与配置说明

> **定位**：`spot_accum` 属于 **A 系统 = 长期现货 beta 积累**（macro cycle / inventory），与 **B/C = 合约/趋势 alpha 抽取** 在架构上应 **彻底分离**。本文档说明每一层职责、推荐特征、简单性边界，并 **对照当前 `archetypes/` 实现** 做符合度评估。

---

## 1. portfolio 架构：A 与 B/C 分离（最重要的一点）

| 层级 | 职责 | 与 `spot_accum` 的关系 |
|------|------|------------------------|
| **A 系统** | 长周期 beta、熊市吸筹、周期末降风险；逻辑尽量 **简单、低换手下、难 overfit** | **仅** `spot_accum`（本目录） |
| **B 系统** | 中周期 swing / 趋势 alpha（BPC/TPC/ME 等） | **不要**在 A 里复制 breakout、紧止损、追赶高 Sharpe |
| **C 系统** | 短周期 inventory / 战术 alpha | 同上 |

**反模式（必须避免）**：把 A 做成「低 Sharpe 的 B」——在 accumulate 上堆 **breakout、ATR stop、trailing、复杂加仓金字塔**，会同时丢掉 **真凸性** 与 **accumulate 的简单性**，且与 B/C **高相关、regime 重叠**。

**当前配置**：`model.type: rule_based`，与 B/C 树模型/执行优化 **路径分离**；宪法使用独立 `spot` 顶级域（`spot.account` / `spot.accumulation` / `spot.strategy_limits`），与 trend/multi-leg 账户池隔离。

---

## 2. spot_accum vs spot_ft（本质区别）

| 维度 | **spot_accum**（A） | **spot_ft**（A 层另一支，偏趋势参与） |
|------|---------------------|----------------------------------------|
| **收益来源** | 长周期 **beta expansion**，弱市吸筹、强市持有 | 更偏 **大趋势确认后参与**，控制熊市暴露 |
| **心智模型** | **周期囤币 / 价值积累**；接受半山腰与浮亏 | **趋势参与**；偏机械确认，少抄「大底」 |
| **痛苦换什么** | 提前承担弱势区建仓的痛苦 → **换低成本筹码** | 放弃部分底部 → **换更少熊市回撤** |
| **优化陷阱** | 逻辑应极简单；**几乎不该**长期拟合 breakout/stop | 更容易滑向「优化突破、止损、regime」→ 复杂度膨胀 |

两者长期都可能吃到 crypto beta，但 **资金曲线、持仓心理、参数敏感度** 完全不同；**不要**把同一套微观执行（ATR 紧止损、频繁 trailing）搬到两边。

---

## 3. 分层职责（本策略应怎么分工）

| 层 | 职责（思想） | 当前 `config/strategies/spot_accum/archetypes/` | 符合度 |
|----|----------------|--------------------------------------------------|--------|
| **prefilter** | **Regime**：只允许在定义的「弱势/积累区」考虑建仓；用 **慢变量**，避免和 B/C 抢同一类入场触发 | `abc_macro_regime_score < 3`（弱侧/非「转换期以上」） | ✅ 与「熊市侧 accumulation」一致；阈值 3 与 `abc_macro_regime_score_f` 内 `score_threshold_transition` 对齐 |
| **gate** | 仅保留 **硬风控**（流动性、极端 evt、系统安全）；**宜少宜粗** | 空列表 | ✅ **简单**；若以后要加，应只加「否决灾难状态」类规则，忌微观 alpha gate |
| **entry_filters** | **极简单**：价格/结构「允许开仓」的窄条件，避免做成第二套 B 策略 | `ema_200_position` 在 **[-0.01, 0.03]**（靠近 EMA200 带） | ✅ 简单、非突破机；若过窄会导致样本极少，属 **调参/放宽** 问题而非方向错误 |
| **direction** | **固定多头**（囤现货 beta） | `fixed_direction: long` | ✅ |
| **execution** | **极简**：无日内 alpha 止损链；**出场**由 **宏观周期/结构** 定义，而非 ATR trailing | `initial_r: 0`、`trailing/tp 关闭`、`structural_exit: weekly_macro_cycle`、`execution_constraints.accumulate_same_archetype: true`、`allow_add_on: false`、`add_position.trigger=disabled`（**同账本多次买入并入持仓**，不使用子仓 add-leg） | ✅ 与「不要 ATR/trailing 当主退出」一致 |

---

## 4. 什么该复杂、什么必须简单

- **值得花时间（可复杂）**  
  - **Regime 定义**：`abc_macro_regime_score` 的合成与阈值（弱势 vs 转换期）。  
  - **熊市 accumulation 的定义**：用 **慢特征**（EMA 系、macro score、周级退出）表达「允许吸筹的宏观窗口」。  
  - **universe**：只做长期主义标的（如 `meta.symbol_include` 的 BTC/ETH/SOL），**避免**垃圾叙事币。

- **必须简单（忌过度设计）**  
  - **分批建仓**：与 B/C 的「子仓 add-leg / float ladder」不同；事件回测中同 symbol + 同 archetype 的重复信号可走 **`accumulate_same_archetype`**，把再买并入 **同一虚拟持仓账本**（VWAP、`pcm_scale_in` 审计），受 `min_order_interval_minutes` 与可选 `max_deploy_legs` 约束。宪法层 `spot.strategy_limits.spot_accum.allow_add_position: false`，避免再走子仓链路。  
  - **微观止损**：**不用** ATR/trailing 作为主要 risk-off；避免把 A 做成迷你 B。  
  - **gate**：不要为了「提 Sharpe」堆 micro gate。

---

## 5. 为什么不把 ATR / trailing stop 当主线？

- **Accumulate 的退出语义**应是 **「宏观周期结束 / 结构破坏」**（本仓库用 **`weekly_macro_cycle`** 作为 structural exit 表达），而不是 **「每根 K 线噪声下的跟踪止损」**。  
- ATR/trailing 更适合 **B/C 的战术持仓**；搬到 A 上会：  
  - 在熊市吸筹段被 **正常波动洗出**；  
  - 与 B/C **同构**，损害组合分散度；  
  - 引入大量 **可优化参数** → overfit。

**当前 `execution.yaml`**：`trailing.enabled: false`、`breakeven: false`、`take_profit: false`，主退出走 `structural_exit: weekly_macro_cycle`，**与上述一致**。

---

## 6. `abc_macro_regime_score` 在 spot_accum 里应怎么用？

- 该分数由 **`ema_1200_position`、`ema_1200_slope_10`、`atr_percentile`、`oi_zscore`、`funding_rate_zscore_50`** 等 **0–5 分** 合成（见 `abc_macro_regime_score_f` / `compute_abc_macro_regime_score_from_series`）。  
- **文档化语义（示意）**：高分偏 risk-on；**`score < 3`** 表示未到「转换期」阈值一侧，适合作为 **「仅弱势侧才建仓」** 的门。  
- **当前 prefilter**：`< 3` → 仅在 **相对弱势/未进入转换强势** 时进入漏斗，符合 **「熊市积累、不简单追涨 risk-on」** 的研究设定。  
- **可调但需记录**：若改为 `<=2` 或加入「连续 N 根」条件，属于 **明确假设变更**，应单变量回测。

---

## 7. 「熊市 accumulation」在本配置里如何定义（可操作层面）

1. **宏观**：`abc_macro_regime_score < 3`（当前实现）。  
2. **入手结构**：弱势下价格 **回到 EMA200 附近**（`ema_200_position` 窄带）——表达「不深追突破、在均带附近补」。  
3. **方向**：只做多优质标的（`meta.symbol_include`）。  
4. **退出**：**周级宏观周期破坏**（`weekly_macro_cycle`），而非短周期止损。  

这是一条 **可写进 YAML 的、与 B/C 解耦** 的 A 层定义；**不等于**「全市场任意时刻 DCA」，而是 **带 regime 与结构边界的 accumulation**。

---

## 8. 推荐特征（与 `features.yaml` 对照）

**思想上** A 层需要：**慢趋势位置、波动 regime、流动性/资金状态、宏观分数、周线级退出代理**。

当前 `feature_groups.baseline` 已包含：

- `abc_macro_regime_score_f` — regime  
- `ema_1200_position_f` / `ema_1200_slope_f`、`atr_percentile_f` — 分数输入与波动上下文  
- `oi_features_f`、`funding_rate_features_f`、`funding_oi_crowding_f` — 流动性/拥挤度  
- `ema_200_position_f` — entry 带  
- `weekly_macro_cycle_exit_f` — 与 `execution.structural_exit: weekly_macro_cycle` 配合  

**一般不必**：为 A 层单独塞 **微观 order-flow 点火**（除非研究明确需要）；那会向 B 靠拢。

---

## 9. 已知差距与后续可做事项（非强制）

| 项目 | 说明 |
|------|------|
| **入场可能过窄** | 仅 `ema_200_position` 窄带时，**事件回测里成交可能极少**；属 **样本量** 问题，可放宽带或加「OR 轨」但保持简单 |
| **Gate 全空** | 符合「极简」；若上生产可再议 **流动性/熔断级** 硬门 |
| **研究与实盘 symbol** | `meta.symbol_include` 限制标的；若回测 CLI 仍扫 6 标的，需在脚本层对齐，否则归因混淆 |
| **定投式加仓** | 当前为 **signal-driven 同账本累加**；若要严格日历 DCA，应新增 spot 专用触发器，避免抄 B/C 子仓加仓机 |

---

## 10. 账户与资金语义（新增）

- `spot` 是独立账户域：与 `resource_allocation`（trend/fat-tail）和 `multi_leg` 分开管理。  
- `spot.account.backtest_equity_usdt` 仅用于回测与离线报告锚点；**实盘资金必须来自远程账户同步**。  
- 以 `$10k` 为例：`spot.accumulation.tranche_count=4` + `unit_notional=2500` 表示「分四档逐步部署」，不是单笔满仓。  
- 若未显式传 `--initial-capital`，事件回测可从 `spot.account.backtest_equity_usdt` 自动取资金锚点。

---

## 11. 一句话 checklist（维护此策略时自问）

1. 这条改动是在 **叠 beta**，还是在 **偷偷做 alpha**？  
2. 是否 **加重**了与 B/C 的相关性？  
3. 是否新增 **可过拟合旋钮**（微观止损/突破/复杂 gate）？  
4. A 层是否仍 **比 B/C 简单一个数量级**？  

若 2–4 任一为「是」，优先停手或拆到 B/C 系统。

---

## 12. 原文档思想摘要（你提供的纲领，压缩版）

- **spot_accum**：熊市/弱市 **持续积累优质 beta**，牛市 **长持**，退出偏 **周期末**；核心是 **长期风险收益比**，不是短突破。  
- **spot_ft**：更偏 **趋势参与**，少抄大底、少熬熊，**成本更高、心理更轻**。  
- **与 B/C 共存**：你已有 swing/短周期 alpha，则 A **不必再做成第三套趋势机**；否则 **三套追趋势 → 相关性爆炸**。  
- **Accumulate 优势**：逻辑简单、**相对难 overfit**；危险：长期浮亏、标的质量、周期逆人性。  
- **致命错误**：用 breakout + ATR + trailing + 金字塔把 A 做成劣质 B。  

---

*文件路径：`config/strategies/spot_accum/reamdme.md` — 与 `archetypes/*.yaml` 同步维护。*
