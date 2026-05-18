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
| **prefilter** | **Regime**：只允许在定义的「弱势/积累区」考虑建仓；用 **慢变量**，避免和 B/C 抢同一类入场触发 | `abc_macro_regime_score < 2`（仅明确弱势；过渡/牛市 score≥2 不新开仓） | ✅ 与「熊市吸筹」一致；比 `<3` 更严，2023–2024 牛市段几乎不买 |
| **gate** | 仅保留 **硬风控**（流动性、极端 evt、系统安全）；**宜少宜粗** | 空列表 | ✅ **简单**；若以后要加，应只加「否决灾难状态」类规则，忌微观 alpha gate |
| **entry_filters** | **极简单**：价格/结构「允许开仓」的窄条件，避免做成第二套 B 策略 | `ema_200_position` **[-0.03, 0.03]** + **RSI < 35**（AND） | ✅ 简单、非突破机；**prebull 实验表明 entry 边际收益低**（见 §13） |
| **direction** | **固定多头**（囤现货 beta） | `fixed_direction: long` | ✅ |
| **execution** | **极简**：无日内 alpha 止损链；**出场**由 **regime lifecycle**（非 B 式 risk-off 洗盘）定义 | `structural_exit: abc_macro_regime_lifecycle`、`regime_lifecycle_exit`、`regime_deploy_scale`（低 score 加大档位）、`accumulate_same_archetype: true`、每日 1 档 + 限价偏移 | ⚠️ **identity 仍在演进**：当前 lifecycle 仍偏 B 思维，见 §13–§14 |

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

**当前 `execution.yaml`**：`trailing` / `breakeven` / `take_profit` 关闭；主退出为 **`abc_macro_regime_lifecycle`**（持仓内跟踪 peak score，bull 后 risk-off 才平；可选 `arm_risk_off_min_peak`）。`weekly_macro_cycle` 仍可作为 **Cycle Death** 候选，但不应在 transition 段替代 inventory 持有。

---

## 6. `abc_macro_regime_score` 在 spot_accum 里应怎么用？

- 该分数由 **`ema_1200_position`、`ema_1200_slope_10`、`atr_percentile`、`oi_zscore`、`funding_rate_zscore_50`** 等 **0–5 分** 合成（见 `abc_macro_regime_score_f` / `compute_abc_macro_regime_score_from_series`）。  
- **文档化语义（示意）**：高分偏 risk-on；**`score < 3`** 表示未到「转换期」阈值一侧，适合作为 **「仅弱势侧才建仓」** 的门。  
- **当前 prefilter**：`< 2` → 仅在 **明确弱势** 时新开仓；`2 ≤ score < 5` 为 transition/bull，**只允许持币/补仓语义，不应默认清仓**（见 §14）。  
- **分数上界为 5**：静态「`score >= 5` 才平仓」在急跌里常失效（peak 仅 4 即回落）；应用 **状态机**（先见过 bull，再 risk-off），见 `position_logic.regime_lifecycle_risk_off_exit`。

---

## 7. 「熊市 accumulation」在本配置里如何定义（可操作层面）

1. **宏观**：`abc_macro_regime_score < 2`（当前实现）。  
2. **入手结构**：弱势下 **EMA200 带 + RSI 超卖**（避免在均线上方追涨）。  
3. **方向**：只做多优质标的（`meta.symbol_include`）。  
4. **退出**：**regime lifecycle**（inventory 框架），而非 transition 段的 B 式 risk-off / floor。  

这是一条 **可写进 YAML 的、与 B/C 解耦** 的 A 层定义；**不等于**「全市场任意时刻 DCA」，而是 **带 regime 与结构边界的 accumulation**。

---

## 8. 推荐特征（与 `features.yaml` 对照）

**思想上** A 层需要：**慢趋势位置、波动 regime、流动性/资金状态、宏观分数、周线级退出代理**。

当前 `feature_groups.baseline` 已包含：

- `abc_macro_regime_score_f` — regime  
- `ema_1200_position_f` / `ema_1200_slope_f`、`atr_percentile_f` — 分数输入与波动上下文  
- `oi_features_f`、`funding_rate_features_f`、`funding_oi_crowding_f` — 流动性/拥挤度  
- `ema_200_position_f` — entry 带  
- `weekly_macro_cycle_exit_f` — 候选 **Cycle Death**（Regime 4），勿在 transition 单独作为主退出  

**一般不必**：为 A 层单独塞 **微观 order-flow 点火**（除非研究明确需要）；那会向 B 靠拢。

---

## 9. 已知差距与后续可做事项（非强制）

| 项目 | 说明 |
|------|------|
| **瓶颈在 persistence，不在 entry** | prebull 实验：去 RSI / 加快 deploy **未提高** 2023 初仓位与收益；主因是 **lifecycle 过早 liquidation**（§13–§14） |
| **Lifecycle vs identity** | `risk_off` / `floor exit` / transition 清仓是 **B 系统逻辑**；accumulate 应优先 **inventory retention**（§15–§16） |
| **Gate 全空** | 符合「极简」；若上生产可再议 **流动性/熔断级** 硬门 |
| **研究与实盘 symbol** | 宪法 `symbol_budgets_usdt` 与 `meta.symbol_include` 应对齐（BTC/BNB/SOL） |
| **核心 KPI** | 对 accumulate，首要指标是 **`bull market inventory exposure`**（牛市前囤了多少），不是 Sharpe |

---

## 10. 账户与资金语义（新增）

- `spot` 是独立账户域：与 `resource_allocation`（trend/fat-tail）和 `multi_leg` 分开管理。  
- `spot.account.backtest_equity_usdt` 仅用于回测与离线报告锚点；**实盘资金必须来自远程账户同步**。  
- 以 `$10k` 为例：`symbol_budgets_usdt`（BTC 5000 / BNB+SOL 各 2500）+ `tranches_per_symbol: 20` + 固定 `symbol_unit_notional_usdt` 表示「分档部署」，不是单笔满仓。  
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

## 13. 核心发现：不是「怎么买更多」，而是 lifecycle 阻止长期 accumulation

2022-01～2026-05 事件回测（`backtest_equity_usdt=10k`，BTC/BNB/SOL 预算 5k+2.5k+2.5k）表明：

**真正的问题不是「怎么买得更多」，而是「lifecycle 正在阻止长期 accumulation」。**

| 常见误区 | 实际情况 |
|----------|----------|
| 优化 deploy、限价、RSI、entry 带 | **持仓持久性（position persistence）** 决定能否囤到牛市 |
| 「买不到」 | 多数是 **「留不住」** — score 2–4 transition 被 risk_off / floor 洗掉 |
| 首要优化 Sharpe | accumulate 更应看 **bull market inventory exposure** |

**v4 盈利主要来自 BTC 长持 exposure**，而非高频 deploy。说明：真正赚钱的是 **长周期 exposure**；不是 deploy 优化、entry 微调、RSI、限价、execution 微调。

### Strategy identity conflict

- **目标**：accumulate（熊市增 inventory → 牛市持 inventory → 周期末清 inventory）。
- **实现残留**：risk_off、floor exit、lifecycle cleanup、transition 平仓 — **B 系统逻辑**。
- **Accumulate 核心**：熊市 **尽量保留 exposure**，而不是 equity smoothness。

Crypto 大收益来自 **长期持有极少数趋势资产**；过早 risk_off / cleanup 会 **永远拿不住超级周期**。牛熊转换区（score 2–4）最易被洗掉，也往往是大行情启动段。

---

## 14. 实验记录（同窗口 2022-01-01 → 2026-05-01）

**脚本**：`scripts/run_spot_accum_prebull_ablation.py`  
**汇总表**：`results/120T/spot_accum/prebull_ablation/comparison_table.csv`  
**基线**：`results/120T/spot_accum/retest_v4_lifecycle/`
**A-simple 落地回测**：`results/120T/spot_accum/retest_a_simple/`

### 14.1 牛市前库存快照（v4 基线，截止 2023-01-01）

| 币种 | 预算 | 仍持仓 | 占预算 |
|------|------|--------|--------|
| BTC | 5000 U | 5000 U | 100% |
| BNB | 2500 U | 375 U | 15% |
| SOL | 2500 U | 0 U | 0% |
| **合计** | **10000 U** | **5375 U** | **53.8%** |

### 14.2 单变量消融（相对 v4）

| 变体 | 2023-01 持仓占比 | 终值权益 | 总回报 | 结论 |
|------|------------------|----------|--------|------|
| **A-simple（当前落地）** | **85.6%** | **11,220** | **+12.2%** | transition 可补齐；risk_off=0；cycle death 延后 arm |
| **v4 基线** | **53.8%** | 10,632 | **+6.3%** | 2023 初库存最高 |
| lifecycle_relaxed | 2.0% | 10,269 | +2.7% | 未改善囤仓 |
| deploy_fast | 2.0% | 10,199 | +2.0% | 加快 deploy 无效 |
| entry_relaxed（去 RSI） | 15.0% | 10,123 | +1.2% | 信号 465→8985，更差 |
| hoard_all（三者叠加） | 3.0% | 10,350 | +3.5% | 仍未牛市前满仓 |

**实验结论**：

1. **Entry filtering 边际收益极低** — 瓶颈不在 signal generation。  
2. **Deploy 优化无法替代 inventory persistence**。  
3. 决定 2023 初仓位的，是 **lifecycle liquidation + transition 补齐权限**，不是「怎么进」。  
4. A-simple 把 2023-01 库存从 **53.8% → 85.6%**，risk_off exit 从 26 → 0。  
5. 优化方向：从 **trade optimization** → **cycle participation / inventory optimization**。

### 14.3 实现备注

- 事件回测须合并 `features.yaml` 与 `archetypes` 特征（含 `weekly_macro_cycle_exit_f`）。  
- A-simple 当前语义：`score < 2` 加速吸筹；`2 <= score < 5` 用剩余预算慢速补齐；`score >= 5` 后停止新增；见 bull 后至少 180 天才允许 `weekly_macro_cycle` 清仓。  
- 静态 `score>=5` risk-on exit 已移除；`score>=5` 仅用于 arm bull exposure，不直接平仓。

---

## 15. Inventory 框架（目标语义）

| 阶段 | 宏观（示意） | 目标 |
|------|--------------|------|
| **熊市 / deep bear** | score ≤ 2 | **增加 inventory** — 持续 deploy；**禁止** risk_off / floor / cleanup |
| **持有 / transition** | score 上升，2–5 | **补齐/维持 inventory** — **禁止主动减仓**；允许剩余预算慢速补仓 |
| **牛市** | score ≥ 5 / 已 `_regime_saw_bull` | **持有 inventory** — 停止新增，禁止短周期止盈 |
| **周期死亡** | bull 后足够久 + 周线结构破坏 | **清 inventory** — 允许 `weekly_macro_cycle` 清仓，reset lifecycle |

**核心 KPI**：`pre_bull_inventory_pct`、`bull_market_inventory_exposure` — **不是** Sharpe。

---

## 16. 建议下一版：彻底 regime 化 lifecycle（设计草案）

**A-simple 已部分落地**；后续优化必须继续以 inventory KPI 为主。

### Regime 1 — Deep Bear（score ≤ 2）

- 目标：accumulation only。  
- 规则：禁止 risk_off；禁止 floor exit；禁止 lifecycle cleanup；允许持续 deploy；不减仓。

### Regime 2 — Transition（2 < score < 5）

- 目标：maintain inventory。  
- 规则：**禁止主动减仓**；允许剩余预算慢速补仓；不允许清仓。

### Regime 3 — Bull（score ≥ 5 或已 `_regime_saw_bull`）

- 目标：ride convexity。  
- 规则：禁止短周期止盈；长期持有；停止新增；不主动减仓。

### Regime 4 — Cycle Death

- 触发：周线连续 LH/LL、周 EMA50 breakdown、breadth collapse 等（可组合 `weekly_macro_cycle_exit_f`）。  
- 目标：distribute inventory。  
- 规则：仅在 bull exposure 已 arm 且最短持有窗口满足后，允许 `weekly_macro_cycle` 清仓；reset lifecycle。

**已实现可复用**：`regime_lifecycle_exit.allow_regime_risk_off: false`、`cycle_exit_requires_bull`、`arm_cycle_exit_min_peak`、`cycle_exit_min_days_after_bull` — `position_logic.py`，`tests/unit/test_regime_lifecycle_exit.py`，`tests/unit/test_spot_accum_cycle_death_only.py`。

---

## 17. 维护者优先级（实验后）

1. **先改 lifecycle / inventory persistence**，再动 entry、限价、RSI。  
2. 每条退出规则问：是在 **叠 beta**，还是在做 **B 式 swing**？  
3. 回测必报：**2023-01-01 三币库存占比** + 全窗口 PnL。  
4. 勿做成「带 accumulation 风味的 swing」— 否则与 B/C 相关、长期凸性消失。

---

*文件路径：`config/strategies/spot_accum/reamdme.md` — 与 `archetypes/*.yaml`、§14 实验产物同步维护。最后更新：2026-05（prebull ablation + v4 lifecycle + inventory 框架）。*
